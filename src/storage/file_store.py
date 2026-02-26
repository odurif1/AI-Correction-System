"""
Simplified storage system - single directory per session.

Architecture:
    data/
    ├── {session_id}/
    │   ├── session.json          # État complet session
    │   ├── policy.json           # Politique de correction
    │   ├── diagnostic.json       # Diagnostic PDF (structure, barème)
    │   ├── cache/                # Cache d'analyse
    │   ├── copies/
    │   │   └── {copy_number}/
    │   │       ├── original.pdf      # PDF original
    │   │       ├── annotation.json   # Infos essentielles pour annotation
    │   │       └── audit.json        # TOUT: échanges LLM, raisonnements, etc.
    │   ├── annotated/            # PDFs annotés (export)
    │   └── reports/              # Exports finaux (dérivés)
    └── _index.json               # Index des sessions

Fichiers par copie:
- annotation.json: Essentiel pour annoter (notes, feedbacks)
- audit.json: Complet pour debugging (échanges LLM, comparaisons, etc.)
"""

import csv
import fcntl
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.models import (
    GradingSession, GradedCopy, GradingPolicy,
    ClassAnswerMap, TeacherDecision, CopyDocument,
    PreAnalysisResult
)
from core.exceptions import SerializationError, ValidationFailedError
from config.constants import (
    DATA_DIR, SESSIONS_INDEX,
    SESSION_JSON, POLICY_JSON, CLUSTERS_JSON, AUDIT_LOG,
    GRADES_CSV, FULL_REPORT_JSON
)


class SessionStore:
    """
    Gestionnaire de stockage pour une session.

    Structure multi-tenant: data/{user_id}/{session_id}/
    Si user_id est None, utilise l'ancien chemin data/{session_id}/ (pour compatibilité)
    """

    def __init__(self, session_id: str, user_id: str = None, base_dir: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.base_dir = Path(base_dir or DATA_DIR)

        # Multi-tenant path: data/{user_id}/{session_id}/
        if user_id:
            self.session_dir = self.base_dir / user_id / session_id
        else:
            # Fallback for backward compatibility
            self.session_dir = self.base_dir / session_id

    # ==================== INITIALIZATION ====================

    def create(self) -> Path:
        """Crée la structure de dossiers pour la session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Créer les sous-dossiers
        (self.session_dir / "copies").mkdir(exist_ok=True)
        (self.session_dir / "cache").mkdir(exist_ok=True)
        (self.session_dir / "annotated").mkdir(exist_ok=True)
        (self.session_dir / "reports").mkdir(exist_ok=True)

        # Mettre à jour l'index
        self._update_index()

        return self.session_dir

    def exists(self) -> bool:
        """Vérifie si la session existe."""
        if (self.session_dir / SESSION_JSON).exists():
            return True
        # Vérifier aussi le chemin legacy pour les sessions créées avant la migration
        if self.user_id:
            return (self.base_dir / self.session_id / SESSION_JSON).exists()
        return False

    # ==================== SESSION STATE ====================

    def save_session(self, session: GradingSession) -> None:
        """Sauvegarde l'état complet de la session."""
        self.create()

        session_file = self.session_dir / SESSION_JSON
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session.model_dump(mode='json'), f, indent=2, ensure_ascii=False)

        self._log("session_saved", f"Status: {session.status}")

    def load_session(self) -> Optional[GradingSession]:
        """Charge l'état de la session avec validation."""
        session_file = self.session_dir / SESSION_JSON

        # Si le fichier n'existe pas dans le chemin user-specific, vérifier le chemin legacy
        if not session_file.exists():
            if self.user_id:
                # Essayer le chemin legacy: data/{session_id}/
                legacy_session_file = self.base_dir / self.session_id / SESSION_JSON
                if legacy_session_file.exists():
                    session_file = legacy_session_file
                else:
                    return None
            else:
                return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON in session file: {e}")

        # Convertir les dates
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])

        # Validate with Pydantic model
        try:
            session = GradingSession(**data)
        except Exception as e:
            raise ValidationFailedError(f"Session data validation failed: {e}", {"file": str(session_file)})

        # Vérifier l'appartenance si user_id est fourni (multi-tenant)
        # Si un user_id est fourni pour le chargement, la session DOIT appartenir à cet utilisateur
        if self.user_id:
            # La session doit avoir un user_id qui correspond
            if not session.user_id or session.user_id != self.user_id:
                return None  # Session n'appartient pas à cet utilisateur

        return session

    # ==================== COPIES (annotation.json) ====================

    def save_copy(self, copy: CopyDocument, original_pdf: bytes = None, copy_number: int = None) -> None:
        """
        Sauvegarde les infos essentielles d'une copie.

        Crée annotation.json avec les infos pour l'annotation future.
        Si la copie est déjà corrigée, les infos de notation seront
        ajoutées par save_graded_copy().

        Args:
            copy: Document de la copie
            original_pdf: Bytes du PDF original
            copy_number: Numéro de la copie (1, 2, 3...)
        """
        copy_dir = self.session_dir / "copies" / str(copy_number or copy.id)
        copy_dir.mkdir(parents=True, exist_ok=True)

        # Charger annotation.json existant ou créer nouveau
        annotation_file = copy_dir / "annotation.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
        else:
            annotation = {}

        # Mettre à jour les infos de la copie (sans écraser le grading)
        annotation.update({
            "copy_number": copy_number or copy.id,
            "student_name": copy.student_name,
            "student_id": copy.student_id,
            "page_count": copy.page_count,
            "page_images": copy.page_images,
            "created_at": copy.created_at.isoformat() if copy.created_at else None,
            "pdf_path": copy.pdf_path
        })

        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        # PDF original
        if original_pdf:
            with open(copy_dir / "original.pdf", 'wb') as f:
                f.write(original_pdf)

    def organize_copy_images(self, copy: CopyDocument, copy_number: int) -> CopyDocument:
        """
        Move images from temp folder to copy-specific folder.

        Args:
            copy: CopyDocument with page_images pointing to temp folder
            copy_number: Copy number (1, 2, 3...)

        Returns:
            CopyDocument with updated page_images paths
        """
        import shutil

        if not copy.page_images:
            return copy

        copy_dir = self.session_dir / "copies" / str(copy_number)
        images_dir = copy_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        new_page_images = []
        for i, old_path in enumerate(copy.page_images):
            old_path = Path(old_path)
            if old_path.exists():
                # New path in copy's images folder
                new_path = images_dir / f"page_{i}.png"
                # Move (not copy) to avoid duplicates
                shutil.move(str(old_path), str(new_path))
                new_page_images.append(str(new_path))
            else:
                # Keep original path if file doesn't exist in temp
                new_page_images.append(str(old_path))

        # Update copy's page_images
        copy.page_images = new_page_images

        # Update annotation.json with new paths
        annotation_file = copy_dir / "annotation.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            annotation["page_images"] = new_page_images
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)

        return copy

    def cleanup_temp_images(self):
        """Remove temp_images folder if empty."""
        temp_dir = self.session_dir / "temp_images"
        if temp_dir.exists() and temp_dir.is_dir():
            # Only remove if empty
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty, keep it

    def load_copy(self, copy_number: str) -> Optional[CopyDocument]:
        """
        Charge une copie par son numéro.

        Lit annotation.json.
        """
        copy_dir = self.session_dir / "copies" / str(copy_number)

        annotation_file = copy_dir / "annotation.json"
        if not annotation_file.exists():
            return None

        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ignorer les champs de notation (ils sont dans grading)
        data.pop('grading', None)
        data.pop('processed', None)

        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])

        return CopyDocument(**data)

    def load_annotation(self, copy_number: str) -> Optional[Dict[str, Any]]:
        """
        Charge les infos d'annotation d'une copie.

        Retourne le contenu complet d'annotation.json (infos copie + notation).
        """
        copy_dir = self.session_dir / "copies" / str(copy_number)
        annotation_file = copy_dir / "annotation.json"

        if not annotation_file.exists():
            return None

        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_copies(self) -> List[str]:
        """Liste les numéros de copies."""
        copies_dir = self.session_dir / "copies"
        if not copies_dir.exists():
            return []
        return [d.name for d in copies_dir.iterdir() if d.is_dir()]

    # ==================== GRADED COPIES (audit.json + annotation.json) ====================

    def save_graded_copy(self, graded: GradedCopy, copy_number: int = None) -> None:
        """
        Sauvegarde une copie corrigée.

        Écrit DEUX fichiers:
        - audit.json: TOUTES les données (échanges LLM, raisonnements, comparaisons)
        - annotation.json: Infos essentielles pour annotation (mis à jour)

        Args:
            graded: Données de correction complètes
            copy_number: Numéro de la copie
        """
        copy_dir = self.session_dir / "copies" / str(copy_number or graded.copy_id)
        copy_dir.mkdir(parents=True, exist_ok=True)

        # 1. audit.json - TOUTES les données
        audit_data = graded.model_dump(mode='json')
        with open(copy_dir / "audit.json", 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        # 2. annotation.json - Infos essentielles pour annotation
        annotation = self.load_annotation(str(copy_number or graded.copy_id)) or {}

        # Ajouter les infos de notation essentielles
        annotation["grading"] = {
            "graded_at": graded.graded_at.isoformat() if graded.graded_at else None,
            "total_score": graded.total_score,
            "max_score": graded.max_score,
            "percentage": round(graded.total_score / graded.max_score * 100, 1) if graded.max_score > 0 else 0,
            "confidence": graded.confidence,

            # Notes par question
            "questions": {
                q_id: {
                    "score": score,
                    "feedback": graded.student_feedback.get(q_id, "") if graded.student_feedback else ""
                }
                for q_id, score in graded.grades.items()
            },

            # Appréciation générale
            "feedback": graded.feedback
        }

        with open(copy_dir / "annotation.json", 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

    def load_graded_copy(self, copy_number: str) -> Optional[GradedCopy]:
        """
        Charge une copie corrigée (données complètes).

        Lit audit.json avec validation.
        """
        copy_dir = self.session_dir / "copies" / str(copy_number)
        audit_file = copy_dir / "audit.json"

        if not audit_file.exists():
            return None

        try:
            with open(audit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON in audit file: {e}")

        if 'graded_at' in data and isinstance(data['graded_at'], str):
            data['graded_at'] = datetime.fromisoformat(data['graded_at'])

        try:
            return GradedCopy(**data)
        except Exception as e:
            raise ValidationFailedError(f"GradedCopy validation failed: {e}", {"file": str(audit_file)})

    def is_graded(self, copy_number: str) -> bool:
        """Vérifie si une copie a été corrigée (audit.json existe)."""
        copy_dir = self.session_dir / "copies" / str(copy_number)
        return (copy_dir / "audit.json").exists()

    def get_copy_summary(self, copy_number: str) -> Optional[Dict[str, Any]]:
        """
        Retourne un résumé léger d'une copie (pour listage).

        Lit uniquement annotation.json (pas audit.json).
        """
        annotation = self.load_annotation(copy_number)
        if not annotation:
            return None

        summary = {
            "copy_number": copy_number,
            "student_name": annotation.get("student_name"),
            "graded": "grading" in annotation,
        }

        # Ajouter infos de notation si présentes
        grading = annotation.get("grading", {})
        if grading:
            summary["total_score"] = grading.get("total_score")
            summary["max_score"] = grading.get("max_score")
            summary["percentage"] = grading.get("percentage")
            summary["feedback"] = grading.get("feedback")

        return summary

    # ==================== CACHE ====================

    def get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère une analyse en cache."""
        cache_file = self.session_dir / "cache" / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def cache_analysis(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Met en cache une analyse."""
        cache_dir = self.session_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        data['cached_at'] = datetime.now().isoformat()

        cache_file = cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_cache(self) -> None:
        """Vide le cache."""
        cache_dir = self.session_dir / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()

    # ==================== DIAGNOSTIC PDF ====================

    def save_pre_analysis(self, result: PreAnalysisResult) -> None:
        """Sauvegarde le résultat du diagnostic PDF."""
        self.create()
        diagnostic_file = self.session_dir / "diagnostic.json"
        with open(diagnostic_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        self._log("diagnostic_saved", f"Analysis ID: {result.analysis_id}")

    def load_pre_analysis(self) -> Optional[PreAnalysisResult]:
        """Charge le résultat du diagnostic PDF."""
        diagnostic_file = self.session_dir / "diagnostic.json"

        if not diagnostic_file.exists():
            return None

        try:
            with open(diagnostic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON in diagnostic file: {e}")

        # Convert datetime strings
        if 'analyzed_at' in data and isinstance(data['analyzed_at'], str):
            data['analyzed_at'] = datetime.fromisoformat(data['analyzed_at'])

        try:
            return PreAnalysisResult(**data)
        except Exception as e:
            raise ValidationFailedError(f"PreAnalysisResult validation failed: {e}", {"file": str(diagnostic_file)})

    # ==================== JSON HELPERS ====================

    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to JSON file with proper encoding."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from JSON file with error handling."""
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise SerializationError(f"Failed to load JSON from {file_path}: {e}")

    # ==================== REPORTS ====================

    def save_report(self, data: Dict[str, Any], filename: str = None) -> Path:
        """Sauvegarde un rapport."""
        reports_dir = self.session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        filename = filename or FULL_REPORT_JSON
        report_file = reports_dir / filename

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return report_file

    def save_grades_csv(self, rows: List[List[str]]) -> Path:
        """Sauvegarde le CSV des notes."""
        reports_dir = self.session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        csv_file = reports_dir / GRADES_CSV

        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        return csv_file

    def get_reports_dir(self) -> Path:
        """Retourne le chemin du dossier reports."""
        return self.session_dir / "reports"

    # ==================== ANNOTATED PDFS ====================

    def save_annotated_pdf(self, copy_number: int, pdf_bytes: bytes) -> Path:
        """Sauvegarde un PDF annoté."""
        annotated_dir = self.session_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        output_file = annotated_dir / f"copie_{copy_number}_annote.pdf"
        with open(output_file, 'wb') as f:
            f.write(pdf_bytes)

        return output_file

    def get_annotated_dir(self) -> Path:
        """Retourne le chemin du dossier annotated."""
        return self.session_dir / "annotated"

    # ==================== POLICY ====================

    def save_policy(self, policy: GradingPolicy) -> None:
        """Sauvegarde la politique de correction."""
        policy_file = self.session_dir / POLICY_JSON
        with open(policy_file, 'w', encoding='utf-8') as f:
            json.dump(policy.model_dump(mode='json'), f, indent=2, ensure_ascii=False)

    def load_policy(self) -> Optional[GradingPolicy]:
        """Charge la politique de correction."""
        policy_file = self.session_dir / POLICY_JSON

        if not policy_file.exists():
            return None

        with open(policy_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return GradingPolicy(**data)

    # ==================== LOGGING ====================

    def _log(self, action: str, details: str) -> None:
        """Ajoute une entrée au log d'audit."""
        log_file = self.session_dir / AUDIT_LOG
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {action}: {details}\n"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(entry)

    def log_decision(self, decision: TeacherDecision) -> None:
        """Log une décision enseignant."""
        self._log(
            "teacher_decision",
            f"Q{decision.question_id}: {decision.teacher_guidance}"
        )

    def get_audit_log(self) -> List[str]:
        """Lit le log d'audit."""
        log_file = self.session_dir / AUDIT_LOG
        if not log_file.exists():
            return []
        with open(log_file, 'r', encoding='utf-8') as f:
            return f.readlines()

    # ==================== INDEX ====================

    def _update_index(self) -> None:
        """Met à jour l'index global des sessions avec verrouillage fichier."""
        # Use user-specific index file if user_id is set
        if self.user_id:
            index_file = self.base_dir / self.user_id / SESSIONS_INDEX
        else:
            index_file = self.base_dir / SESSIONS_INDEX

        # Ensure base directory exists
        index_file.parent.mkdir(parents=True, exist_ok=True)

        # Use a dedicated lock file (not the index file itself)
        lock_path = index_file.parent / ".index.lock"

        # Use context manager pattern for clean lock handling
        lock_fd = None
        try:
            # Open in append mode (atomic create on POSIX)
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                # Acquire exclusive lock (blocks until available)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                # Charger l'index existant
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                else:
                    index = {}

                # Mettre à jour l'entrée
                index[self.session_id] = {
                    'created_at': datetime.now().isoformat(),
                    'path': str(self.session_dir),
                    'user_id': self.user_id
                }

                # Sauvegarder atomiquement (write to temp, then rename)
                temp_file = index_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_file.replace(index_file)  # Atomic on POSIX

            finally:
                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            # Note: We don't delete the lock file - it's reused across calls
            # This avoids race conditions with lock file creation/deletion

    # ==================== CLEANUP ====================

    def delete(self) -> bool:
        """Supprime la session."""
        if not self.session_dir.exists():
            return False
        shutil.rmtree(self.session_dir)
        return True


class SessionIndex:
    """
    Gestionnaire de l'index des sessions.
    Supporte le multi-tenant avec user_id.
    """

    def __init__(self, user_id: str = None, base_dir: str = None):
        self.user_id = user_id
        self.base_dir = Path(base_dir or DATA_DIR)

        # User-specific index file if user_id is set
        if user_id:
            self.index_file = self.base_dir / user_id / SESSIONS_INDEX
        else:
            self.index_file = self.base_dir / SESSIONS_INDEX

    def list_sessions(self) -> List[str]:
        """Liste toutes les sessions de l'utilisateur."""
        # Si user_id est défini, on doit filtrer les sessions par propriétaire
        if self.user_id:
            # Vérifier d'abord l'index utilisateur
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                return list(index.keys())

            # Fallback: scanner le dossier utilisateur
            if self.index_file.parent.exists():
                return [
                    d.name for d in self.index_file.parent.iterdir()
                    if d.is_dir() and (d / SESSION_JSON).exists()
                ]

            # Dernier fallback: scanner toutes les sessions et filtrer par user_id
            # Cela gère le cas des sessions créées avant la migration multi-tenant
            global_index_file = self.base_dir / SESSIONS_INDEX
            if global_index_file.exists():
                with open(global_index_file, 'r', encoding='utf-8') as f:
                    global_index = json.load(f)

                user_sessions = []
                for session_id, info in global_index.items():
                    # Vérifier si la session appartient à cet utilisateur
                    session_user_id = info.get('user_id')
                    if session_user_id == self.user_id:
                        user_sessions.append(session_id)
                    elif session_user_id is None:
                        # Session sans user_id - vérifier dans le fichier session.json
                        session_file = self.base_dir / session_id / SESSION_JSON
                        if session_file.exists():
                            try:
                                with open(session_file, 'r', encoding='utf-8') as f:
                                    session_data = json.load(f)
                                if session_data.get('user_id') == self.user_id:
                                    user_sessions.append(session_id)
                            except (json.JSONDecodeError, IOError):
                                pass
                return user_sessions

            return []

        # Mode sans user_id (admin/debug) - retourne toutes les sessions
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            return list(index.keys())

        if not self.index_file.parent.exists():
            return []

        return [
            d.name for d in self.index_file.parent.iterdir()
            if d.is_dir() and (d / SESSION_JSON).exists()
        ]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Récupère les infos d'une session."""
        if not self.index_file.exists():
            return None

        with open(self.index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        return index.get(session_id)

    def cleanup_index(self) -> int:
        """Nettoie l'index (supprime les entrées sans dossier)."""
        if not self.index_file.exists():
            return 0

        with open(self.index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        cleaned = 0
        for session_id in list(index.keys()):
            session_dir = self.index_file.parent / session_id
            if not session_dir.exists() or not (session_dir / SESSION_JSON).exists():
                del index[session_id]
                cleaned += 1

        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        return cleaned
