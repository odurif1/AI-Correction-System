from typing import List, Dict, Optional
from pathlib import Path
from rich.console import Console

from core.models import CopyDocument
from vision.pdf_reader import PDFReader, split_pdf_by_ranges

class DetectionService:
    """
    Service responsible for loading copies, analyzing structure,
    and detecting students/barème from PDFs.
    """

    def __init__(
        self,
        session,
        store,
        ai,
        pdf_paths: List[str],
        pages_per_copy: Optional[int] = None,
        grading_mode: str = "individual",
        comparison_mode: bool = False
    ):
        self.session = session
        self.store = store
        self.ai = ai
        self.pdf_paths = pdf_paths or []
        self._pages_per_copy = pages_per_copy
        self._grading_mode = grading_mode
        self._comparison_mode = comparison_mode
        self.structure_pre_detected = False

    async def analyze_only(self) -> Dict:
        """Phase 1: Analyze copies without grading."""
        await self._load_copies_phase()

        from core.models import SessionStatus, ClassAnswerMap
        self.session.transition_to(SessionStatus.DIAGNOSTIC)
        self.session.class_map = ClassAnswerMap()
        self.session.transition_to(SessionStatus.CORRECTION)

        detected_questions = {}
        detected_language = 'fr'

        if self.session.copies:
            first_copy = self.session.copies[0]
            detected_language = first_copy.language or 'fr'

            for key in first_copy.content_summary.keys():
                if key.startswith('_') or key.endswith('_points') or key.endswith('_points_unknown') or key.endswith('_confidence'):
                    continue
                if not (key.startswith('Q') and key[1:].isdigit()):
                    continue
                detected_questions[key] = f"Question {key}"

        if not detected_questions:
            questions_detected_during_grading = True
            detected_questions = {}
        else:
            questions_detected_during_grading = False

        detected_scale = {q: 1.0 for q in detected_questions.keys()}

        return {
            'questions': detected_questions,
            'scale': detected_scale,
            'scale_detected': False,
            'copies_count': len(self.session.copies),
            'language': detected_language,
            'questions_detected_during_grading': questions_detected_during_grading,
            'structure_pre_detected': self.structure_pre_detected
        }

    async def _load_copies_phase(self):
        console = Console()
        detection = self.store.load_detection()

        if detection and detection.students:
            if self._grading_mode == "batch":
                console.print(f"[cyan]Mode batch: {len(detection.students)} élève(s) attendu(s)[/cyan]")
                await self._load_copies_minimal()
            else:
                console.print(f"[bold cyan]📋 Utilisation de la détection: {len(detection.students)} élève(s) détecté(s)[/bold cyan]")
                await self._load_copies_from_detection(detection)
        elif self._pages_per_copy:
            await self._load_copies_individual_mode()
        else:
            await self._load_copies_minimal()

    async def _load_copies_minimal(self):
        console = Console()
        for pdf_path in self.pdf_paths:
            reader = PDFReader(pdf_path)
            page_count = reader.get_page_count()
            temp_dir = self.store.session_dir / "temp_images"
            temp_dir.mkdir(parents=True, exist_ok=True)

            page_images = []
            for page_num in range(page_count):
                image_bytes = reader.get_page_image_bytes(page_num)
                image_path = str(temp_dir / f"page_{page_num}.png")
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                page_images.append(image_path)

            copy = CopyDocument(
                pdf_path=pdf_path,
                page_count=page_count,
                student_name=None,
                content_summary={},
                page_images=page_images,
                language='fr'
            )
            self.session.copies.append(copy)

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            self.store.save_copy(copy, pdf_bytes)
            reader.close()
        self.store.cleanup_temp_images()

    async def _load_copies_from_detection(self, detection):
        console = Console()
        for pdf_path in self.pdf_paths:
            reader = PDFReader(pdf_path)
            ranges = []
            student_names = []
            for student in detection.students:
                start_page = student.start_page - 1
                end_page = student.end_page - 1
                ranges.append((start_page, end_page))
                student_names.append(student.name)
                console.print(f"  [dim]  • {student.name}: pages {student.start_page}-{student.end_page}[/dim]")

            split_dir = Path(self.store.session_dir) / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_paths = split_pdf_by_ranges(pdf_path, str(split_dir), ranges)

            for i, (split_path, student_name) in enumerate(zip(split_paths, student_names)):
                split_reader = PDFReader(split_path)
                split_page_count = split_reader.get_page_count()

                copy_dir = self.store.session_dir / "copies" / f"copy_{i+1}"
                copy_dir.mkdir(parents=True, exist_ok=True)
                page_images = []
                for page_num in range(split_page_count):
                    image_bytes = split_reader.get_page_image_bytes(page_num)
                    image_path = str(copy_dir / f"page_{page_num}.png")
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    page_images.append(image_path)
                split_reader.close()

                student_info = detection.students[i]
                copy = CopyDocument(
                    pdf_path=split_path,
                    page_count=split_page_count,
                    student_name=student_name,
                    content_summary={},
                    page_images=page_images,
                    language=detection.detected_language or 'fr',
                    start_page=student_info.start_page,
                    end_page=student_info.end_page
                )
                self.session.copies.append(copy)
                with open(split_path, 'rb') as f:
                    pdf_bytes = f.read()
                self.store.save_copy(copy, pdf_bytes, copy_number=i + 1)
            reader.close()
        self.structure_pre_detected = True
        console.print(f"[green]✓ {len(self.session.copies)} copie(s) créée(s) depuis la détection[/green]")

    async def _load_copies_individual_mode(self):
        console = Console()
        pages_per_copy = self._pages_per_copy
        copy_index = 0

        for pdf_path in self.pdf_paths:
            reader = PDFReader(pdf_path)
            total_pages = reader.get_page_count()
            reader.close()

            ranges = []
            for start in range(0, total_pages, pages_per_copy):
                end = min(start + pages_per_copy - 1, total_pages - 1)
                ranges.append((start, end))

            num_students = len(ranges)
            split_dir = Path(self.store.session_dir) / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_paths = split_pdf_by_ranges(pdf_path, str(split_dir), ranges)

            for i, split_path in enumerate(split_paths):
                copy_index += 1
                copy = await self._analyze_single_copy(split_path, copy_index, num_students)
                self.session.copies.append(copy)
                with open(split_path, 'rb') as f:
                    pdf_bytes = f.read()
                self.store.save_copy(copy, pdf_bytes)

    async def _analyze_single_copy(self, pdf_path: str, copy_index: int, total_copies: int) -> CopyDocument:
        from datetime import datetime
        reader = PDFReader(pdf_path)
        page_count = reader.get_page_count()
        temp_dir = self.store.session_dir / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)

        page_images = []
        for page_num in range(page_count):
            image_bytes = reader.get_page_image_bytes(page_num)
            image_path = str(temp_dir / f"page_{copy_index}_{page_num}.png")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            page_images.append(image_path)
        reader.close()

        return CopyDocument(
            id=f"copy_{copy_index}",
            pdf_path=pdf_path,
            page_count=page_count,
            student_name=None,
            page_images=page_images,
            language=None,
            created_at=datetime.now(),
            processed=False
        )
