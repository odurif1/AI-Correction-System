from collections.abc import Generator
import importlib

import fitz
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.app import (
    build_disagreement_response,
    confirm_session_documents,
    detect_document_type_from_pdf,
    delete_session_artifacts,
    get_disagreement_answer_excerpt,
    get_session_temp_dir,
    list_uploaded_pdfs,
    serialize_progress_for_client,
    verify_websocket_session_access,
)
from core.services.review_context_service import ReviewContextService
from audit.builder import extract_final_question_outputs
from api.auth import (
    create_access_token,
    decode_token,
    get_current_user,
    hash_password,
)
from api.middleware import verify_session_ownership
from config.settings import get_settings
from core.models import (
    AuditSummary,
    DocumentDecision,
    DocumentStatus,
    DocumentType,
    GradedCopy,
    GradingAudit,
    GradingPolicy,
    LLMResult,
    ProviderInfo,
    QuestionAudit,
    ResolutionInfo,
    GradingSession,
    SessionDocument,
    SessionStatus,
    CopyDocument,
)
from db import Base, SubscriptionTier, User
import db as db_module
import db.database as db_database
import storage.file_store as file_store
import config.constants as constants
import api.middleware as api_middleware

api_app_module = importlib.import_module("api.app")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch) -> Generator[sessionmaker, None, None]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    monkeypatch.setenv("AI_CORRECTION_JWT_SECRET", "x" * 32)
    monkeypatch.setenv("AI_CORRECTION_AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_CORRECTION_GEMINI_API_KEY", "dummy-key")
    monkeypatch.setenv("AI_CORRECTION_COMPARISON_MODE", "false")

    get_settings.cache_clear()

    monkeypatch.setattr(constants, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(file_store, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(api_middleware, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(db_database, "engine", engine)
    monkeypatch.setattr(db_database, "SessionLocal", testing_session_local)
    monkeypatch.setattr(db_database, "DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", testing_session_local)
    monkeypatch.setattr(api_app_module, "SessionLocal", testing_session_local)

    Base.metadata.create_all(bind=engine)

    try:
        yield testing_session_local
    finally:
        get_settings.cache_clear()


def create_user(session_factory: sessionmaker, email: str, name: str) -> User:
    db = session_factory()
    try:
        user = User(
            email=email,
            password_hash=hash_password("StrongPass123!"),
            name=name,
            subscription_tier=SubscriptionTier.FREE,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user
    finally:
        db.close()


def create_user_session(user_id: str, session_id: str) -> None:
    store = file_store.SessionStore(session_id=session_id, user_id=user_id)
    session = GradingSession(
        session_id=session_id,
        user_id=user_id,
        status=SessionStatus.DIAGNOSTIC,
    )
    store.save_session(session)


def create_graded_session(user_id: str, session_id: str) -> None:
    store = file_store.SessionStore(session_id=session_id, user_id=user_id)
    session = GradingSession(
        session_id=session_id,
        user_id=user_id,
        status=SessionStatus.COMPLETE,
    )
    session.graded_copies = [
        GradedCopy(copy_id="copy-1", total_score=8.0, max_score=10.0, grades={"Q1": 4.0, "Q2": 4.0}),
        GradedCopy(copy_id="copy-2", total_score=6.0, max_score=10.0, grades={"Q1": 3.0, "Q2": 3.0}),
    ]
    session.policy.question_weights = {"Q1": 5.0, "Q2": 5.0}
    store.save_session(session)


def create_pdf(path, text: str | list[str]) -> None:
    doc = fitz.open()
    pages = text if isinstance(text, list) else [text]
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text)
    doc.save(str(path))
    doc.close()


@pytest.mark.anyio
async def test_get_current_user_with_valid_and_invalid_token(isolated_env):
    session_factory = isolated_env
    user = create_user(session_factory, "alice@example.com", "Alice")

    db = session_factory()
    try:
        token = create_access_token(user.id)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        current_user = await get_current_user(credentials=credentials, db=db)
        assert current_user.id == user.id

        invalid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid-token",
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=invalid_credentials, db=db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Token invalide ou expiré"
    finally:
        db.close()

    assert decode_token(token)["sub"] == user.id
    assert decode_token("invalid-token") is None


@pytest.mark.anyio
async def test_verify_session_ownership_is_user_scoped(isolated_env):
    session_factory = isolated_env
    alice = create_user(session_factory, "alice@example.com", "Alice")
    bob = create_user(session_factory, "bob@example.com", "Bob")

    create_user_session(alice.id, "session-alice")

    owner_id = await verify_session_ownership("session-alice", current_user=alice)
    assert owner_id == alice.id

    with pytest.raises(HTTPException) as exc_info:
        await verify_session_ownership("session-alice", current_user=bob)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session non trouvée"


def test_verify_websocket_session_access_requires_token_and_ownership(isolated_env):
    session_factory = isolated_env
    alice = create_user(session_factory, "alice@example.com", "Alice")
    bob = create_user(session_factory, "bob@example.com", "Bob")

    create_user_session(alice.id, "session-alice")

    alice_token = create_access_token(alice.id)
    bob_token = create_access_token(bob.id)

    assert verify_websocket_session_access("session-alice", alice_token) == alice.id
    assert verify_websocket_session_access("session-alice", bob_token) is None
    assert verify_websocket_session_access("session-alice", None) is None
    assert verify_websocket_session_access("session-alice", "invalid-token") is None
    assert verify_websocket_session_access("missing-session", alice_token) is None


def test_analytics_can_be_generated_from_persisted_session(isolated_env):
    session_factory = isolated_env
    alice = create_user(session_factory, "alice@example.com", "Alice")
    create_graded_session(alice.id, "session-analytics")

    store = file_store.SessionStore(session_id="session-analytics", user_id=alice.id)
    session = store.load_session()

    from export.analytics import AnalyticsGenerator

    report = AnalyticsGenerator(session).generate()

    assert report.mean_score == 7.0
    assert report.max_score == 8.0
    assert report.question_stats["Q1"]["mean"] == 3.5


def test_get_disagreement_answer_excerpt_prefers_copy_summary():
    excerpt, source = get_disagreement_answer_excerpt(
        question_audit=QuestionAudit(
            llm_results={
                "LLM1": LLMResult(grade=1.0, reading="Lecture 1"),
                "LLM2": LLMResult(grade=1.0, reading="Lecture 2"),
            },
            resolution=ResolutionInfo(
                final_grade=1.0,
                final_max_points=2.0,
                method="consensus",
                phases=["initial"],
                agreement=True,
                final_reading="Reponse synthetique de la copie",
                final_reading_method="initial_consensus",
                reading_consensus=True,
            ),
        ),
        copy=None,
    )

    assert excerpt == "Reponse synthetique de la copie"
    assert source == "lecture_finale_consensuelle"


def test_build_disagreement_response_includes_prof_context():
    session = GradingSession(
        policy=GradingPolicy(question_names={"Q1": "Question 1 - Derivee"}),
    )
    copy = CopyDocument(
        id="copy-1",
        pdf_path="/tmp/copy.pdf",
        student_name="Alice",
        start_page=3,
        end_page=4,
        content_summary={"Q1": "L'eleve justifie correctement la derivee."},
    )
    audit = GradingAudit(
        mode="dual",
        grading_method="batch",
        verification_mode="none",
        providers=[
            ProviderInfo(id="LLM1", model="gemini-test"),
            ProviderInfo(id="LLM2", model="gpt-test"),
        ],
        questions={
            "Q1": QuestionAudit(
                llm_results={
                    "LLM1": LLMResult(grade=1.5, confidence=0.7, reasoning="Bonne methode", reading="Lecture A"),
                    "LLM2": LLMResult(
                        grade=0.5,
                        confidence=0.6,
                        question_text="Calculez la derivee de f(x) = x^2.",
                        reasoning="Justification insuffisante",
                        reading="Lecture B",
                    ),
                },
                resolution=ResolutionInfo(
                    final_grade=1.0,
                    final_max_points=2.0,
                    method="average",
                    phases=["initial"],
                    agreement=False,
                    final_reading=None,
                    final_reading_method=None,
                    reading_consensus=False,
                ),
            )
        },
        summary=AuditSummary(total_questions=1, agreed_initial=0, final_agreement_rate=0.0),
    )
    graded = GradedCopy(copy_id="copy-1", grading_audit=audit)

    disagreement = build_disagreement_response(
        session=session,
        graded=graded,
        copy=copy,
        copy_index=1,
        question_id="Q1",
        question_audit=audit.questions["Q1"],
        provider1=audit.providers[0],
        provider2=audit.providers[1],
        llm1_result=audit.questions["Q1"].llm_results["LLM1"],
        llm2_result=audit.questions["Q1"].llm_results["LLM2"],
        resolved=False,
    )

    assert disagreement.question_label == "Question 1 - Derivee"
    assert disagreement.question_text == "Calculez la derivee de f(x) = x^2."
    assert disagreement.disagreement_type == "grade+reading"
    assert disagreement.review_context is None
    assert disagreement.answer_excerpt is None
    assert disagreement.answer_excerpt_source is None
    assert disagreement.start_page == 3
    assert disagreement.end_page == 4
    assert disagreement.max_points == 2.0


def test_build_disagreement_response_includes_review_context_facts():
    session = GradingSession(
        policy=GradingPolicy(question_names={"Q6": "Question 6"}),
    )
    copy = CopyDocument(
        id="copy-6",
        pdf_path="/tmp/copy.pdf",
        student_name="Alice",
    )
    audit = GradingAudit(
        mode="dual",
        grading_method="batch",
        verification_mode="none",
        providers=[
            ProviderInfo(id="LLM1", model="gemini-test"),
            ProviderInfo(id="LLM2", model="gpt-test"),
        ],
        questions={
            "Q6": QuestionAudit(
                llm_results={
                    "LLM1": LLMResult(
                        grade=2.0,
                        confidence=0.8,
                        question_text="Calculer la masse de solute a prelever.",
                        reasoning="Le calcul est correct avec Cm = 40 g/L et V = 1 L.",
                        reading="m = 40 g/L x 1 L = 40 g",
                    ),
                    "LLM2": LLMResult(
                        grade=1.5,
                        confidence=0.7,
                        reasoning="La concentration donnee dans l'enonce est de 30 g/L et le volume est 250 mL.",
                        reading="m = 40g L-1 x 1L = 40g",
                    ),
                },
                resolution=ResolutionInfo(
                    final_grade=1.75,
                    final_max_points=2.0,
                    method="average",
                    phases=["initial"],
                    agreement=False,
                    final_reading="m = 40 g/L x 1 L = 40 g",
                    final_reading_method="initial_consensus",
                    reading_consensus=True,
                ),
            )
        },
        summary=AuditSummary(total_questions=1, agreed_initial=0, final_agreement_rate=0.0),
    )
    graded = GradedCopy(copy_id="copy-6", grading_audit=audit)

    disagreement = build_disagreement_response(
        session=session,
        graded=graded,
        copy=copy,
        copy_index=1,
        question_id="Q6",
        question_audit=audit.questions["Q6"],
        provider1=audit.providers[0],
        provider2=audit.providers[1],
        llm1_result=audit.questions["Q6"].llm_results["LLM1"],
        llm2_result=audit.questions["Q6"].llm_results["LLM2"],
        resolved=False,
    )

    assert disagreement.review_context is not None
    assert "Cm = 40 g/L" in disagreement.review_context.question_facts
    assert "30 g/L" in disagreement.review_context.question_facts
    assert "250 mL" in disagreement.review_context.question_facts


def test_review_context_service_extracts_compact_facts():
    service = ReviewContextService()

    context = service.build_context(
        question_text="Calculer la masse a prelever pour preparer 250 mL d'une solution de concentration 30 g/L.",
        llm_reasonings=["On utilise m = Cm x V."],
    )

    assert context is not None
    assert "250 mL" in context["question_facts"]
    assert "30 g/L" in context["question_facts"]
    assert "m = Cm x V" in context["question_facts"]


def test_extract_final_question_outputs_reads_resolution_fields():
    audit = GradingAudit(
        mode="dual",
        grading_method="batch",
        verification_mode="none",
        providers=[],
        questions={
            "Q1": QuestionAudit(
                llm_results={"LLM1": LLMResult(grade=1.0)},
                resolution=ResolutionInfo(
                    final_grade=1.5,
                    final_max_points=2.0,
                    final_confidence=0.72,
                    final_reasoning="Raisonnement final retenu",
                    final_feedback="Feedback final retenu",
                    method="average",
                    phases=["initial"],
                    agreement=False,
                ),
            )
        },
        summary=AuditSummary(total_questions=1, agreed_initial=0, final_agreement_rate=0.0),
    )

    outputs = extract_final_question_outputs(audit)

    assert outputs["Q1"]["confidence"] == 0.72
    assert outputs["Q1"]["reasoning"] == "Raisonnement final retenu"
    assert outputs["Q1"]["feedback"] == "Feedback final retenu"


def test_serialize_progress_for_client_hides_internal_fields():
    progress = {
        "status": "correction",
        "copies_uploaded": 3,
        "copies_graded": 1,
        "grading_mode": "dual",
        "user_id": "secret-user-id",
        "disagreements": ["internal-only"],
        "error": "boom",
    }

    assert serialize_progress_for_client(progress) == {
        "status": "correction",
        "copies_uploaded": 3,
        "copies_graded": 1,
        "grading_mode": "dual",
        "error": "boom",
    }


def test_delete_session_artifacts_removes_persistent_and_temp_files(isolated_env):
    session_factory = isolated_env
    alice = create_user(session_factory, "alice@example.com", "Alice")
    create_user_session(alice.id, "session-alice")

    temp_dir = get_session_temp_dir("session-alice")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "upload.pdf"
    temp_file.write_bytes(b"pdf")

    store = file_store.SessionStore(session_id="session-alice", user_id=alice.id)

    assert store.exists() is True
    assert temp_file.exists() is True

    assert delete_session_artifacts(store, "session-alice") is True
    assert store.exists() is False
    assert temp_dir.exists() is False


def test_list_uploaded_pdfs_is_sorted_and_scoped():
    temp_dir = get_session_temp_dir("session-alice")
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "b_file.pdf").write_bytes(b"pdf")
    (temp_dir / "a_file.pdf").write_bytes(b"pdf")
    (temp_dir / "ignore.txt").write_text("x", encoding="utf-8")

    pdfs = list_uploaded_pdfs("session-alice")

    assert [path.name for path in pdfs] == ["a_file.pdf", "b_file.pdf"]


def test_detect_document_type_from_pdf_uses_content(tmp_path):
    copies_pdf = tmp_path / "document.pdf"
    create_pdf(copies_pdf, "Nom: Dupont\nPrénom: Alice\nClasse: 4A\nMa reponse est 42.")

    detected_type, confidence, issues, page_count, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(copies_pdf),
        "document.pdf",
    )

    assert detected_type == DocumentType.STUDENT_COPIES
    assert confidence > 0.4
    assert page_count == 1
    assert "Peu ou pas de texte extractible" not in " ".join(issues)
    assert excerpt
    assert evidence
    assert len(page_classifications) == 1
    assert page_classifications[0].detected_type == DocumentType.STUDENT_COPIES
    assert len(segments) == 1
    assert segments[0].start_page == 1
    assert segments[0].end_page == 1


def test_detect_document_type_from_pdf_can_override_filename(tmp_path):
    subject_pdf = tmp_path / "copies_terminale.pdf"
    create_pdf(subject_pdf, "Sujet de mathematiques\nExercice 1\nConsignes: repondre sur la copie.")

    detected_type, confidence, issues, _, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(subject_pdf),
        "copies_terminale.pdf",
    )

    assert detected_type == DocumentType.SUBJECT_ONLY
    assert confidence > 0.4
    assert excerpt
    assert evidence
    assert not issues or all("ambigu" not in issue.lower() for issue in issues)
    assert len(page_classifications) == 1
    assert page_classifications[0].detected_type == DocumentType.SUBJECT_ONLY
    assert len(segments) == 1
    assert segments[0].detected_type == DocumentType.SUBJECT_ONLY


def test_detect_document_type_from_pdf_flags_weak_text_signal(tmp_path):
    weak_pdf = tmp_path / "scan_sujet.pdf"
    create_pdf(weak_pdf, ["Sujet", "", ""])

    detected_type, confidence, issues, page_count, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(weak_pdf),
        "scan_sujet.pdf",
    )

    assert detected_type == DocumentType.SUBJECT_ONLY
    assert confidence <= 0.55
    assert page_count == 3
    assert excerpt
    assert any("extraction textuelle partielle" in issue.lower() for issue in issues)
    assert len(page_classifications) == 3
    assert len(segments) >= 1


def test_detect_document_type_from_pdf_stays_conservative_when_ambiguous(tmp_path):
    ambiguous_pdf = tmp_path / "sujet_copies_mixte.pdf"
    create_pdf(
        ambiguous_pdf,
        "Sujet de mathematiques\nNom:\nPrenom:\nExercice 1\nBareme sur 20 points.",
    )

    ambiguous_type, confidence, issues, _, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(ambiguous_pdf),
        "sujet_copies_mixte.pdf",
    )

    assert ambiguous_type == DocumentType.UNCLEAR
    assert confidence < 0.5
    assert issues
    assert excerpt
    assert evidence
    assert page_classifications
    assert segments


def test_detect_document_type_from_pdf_builds_page_segments(tmp_path):
    mixed_pdf = tmp_path / "mixed.pdf"
    create_pdf(
        mixed_pdf,
        [
            "Sujet de mathematiques\nExercice 1\nConsignes generales.",
            "Nom: Dupont\nPrenom: Alice\nClasse: 4A\nMa reponse est 42.",
        ],
    )

    detected_type, confidence, issues, page_count, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(mixed_pdf),
        "mixed.pdf",
    )

    assert detected_type == DocumentType.UNCLEAR
    assert confidence < 0.5
    assert page_count == 2
    assert excerpt
    assert evidence
    assert any("ambigu" in issue.lower() for issue in issues)
    assert [page.detected_type for page in page_classifications] == [
        DocumentType.SUBJECT_ONLY,
        DocumentType.STUDENT_COPIES,
    ]
    assert [(segment.start_page, segment.end_page, segment.detected_type) for segment in segments] == [
        (1, 1, DocumentType.SUBJECT_ONLY),
        (2, 2, DocumentType.STUDENT_COPIES),
    ]


def test_detect_document_type_from_pdf_flags_filename_only_signal(tmp_path):
    filename_only_pdf = tmp_path / "Interro_courte_2copies.pdf"
    create_pdf(filename_only_pdf, ["", "", ""])

    detected_type, confidence, issues, page_count, excerpt, evidence, page_classifications, segments = detect_document_type_from_pdf(
        str(filename_only_pdf),
        "Interro_courte_2copies.pdf",
    )

    assert detected_type == DocumentType.STUDENT_COPIES
    assert confidence <= 0.3
    assert page_count == 3
    assert excerpt is None
    assert evidence == []
    assert any("principalement sur le nom du fichier" in issue.lower() for issue in issues)
    assert all(page.detected_type == DocumentType.UNCLEAR for page in page_classifications)
    assert len(segments) == 1
    assert segments[0].detected_type == DocumentType.UNCLEAR


def test_confirm_session_documents_requires_at_least_one_copy():
    session = GradingSession(
        session_id="session-docs",
        user_id="user-1",
        documents=[
            SessionDocument(
                filename="sujet.pdf",
                storage_path="/tmp/sujet.pdf",
                detected_type=DocumentType.SUBJECT_ONLY,
                confidence=0.8,
                status=DocumentStatus.CLASSIFIED,
            )
        ],
    )

    copy_ids, reference_ids, excluded_ids, issues = confirm_session_documents(session)

    assert copy_ids == []
    assert reference_ids == [session.documents[0].id]
    assert excluded_ids == []
    assert issues == ["Confirmez au moins un document comme copie élève avant de lancer la correction."]
    assert session.prepared_correction is not None
    assert session.prepared_correction.ready_to_grade is False


def test_confirm_session_documents_applies_decisions_and_flags_duplicates():
    copy_doc = SessionDocument(
        filename="copies.pdf",
        storage_path="/tmp/copies.pdf",
        detected_type=DocumentType.STUDENT_COPIES,
        confidence=0.8,
        status=DocumentStatus.CLASSIFIED,
    )
    subject_doc = SessionDocument(
        filename="sujet.pdf",
        storage_path="/tmp/sujet.pdf",
        detected_type=DocumentType.SUBJECT_ONLY,
        confidence=0.8,
        status=DocumentStatus.CLASSIFIED,
        user_decision=DocumentDecision.SUBJECT_ONLY,
    )
    extra_subject_doc = SessionDocument(
        filename="autre-sujet.pdf",
        storage_path="/tmp/autre-sujet.pdf",
        detected_type=DocumentType.SUBJECT_ONLY,
        confidence=0.8,
        status=DocumentStatus.CLASSIFIED,
        user_decision=DocumentDecision.SUBJECT_ONLY,
    )
    excluded_doc = SessionDocument(
        filename="bruit.pdf",
        storage_path="/tmp/bruit.pdf",
        detected_type=DocumentType.UNCLEAR,
        confidence=0.2,
        status=DocumentStatus.CLASSIFIED,
        user_decision=DocumentDecision.EXCLUDE,
    )
    session = GradingSession(
        session_id="session-docs",
        user_id="user-1",
        documents=[copy_doc, subject_doc, extra_subject_doc, excluded_doc],
    )

    copy_ids, reference_ids, excluded_ids, issues = confirm_session_documents(session)

    assert copy_ids == [copy_doc.id]
    assert reference_ids == [subject_doc.id, extra_subject_doc.id]
    assert excluded_ids == [excluded_doc.id]
    assert issues == ["Plusieurs documents sont marqués comme sujet. Gardez-en un seul ou excluez les doublons."]
    assert copy_doc.status == DocumentStatus.CONFIRMED
    assert copy_doc.usable is True
    assert excluded_doc.status == DocumentStatus.REJECTED
    assert session.prepared_correction is not None
    assert session.prepared_correction.ready_to_grade is False
    assert "Plusieurs documents de copies" not in session.prepared_correction.warnings


def test_confirm_session_documents_keeps_ambiguous_documents_blocking():
    session = GradingSession(
        session_id="session-docs",
        user_id="user-1",
        documents=[
            SessionDocument(
                filename="copies.pdf",
                storage_path="/tmp/copies.pdf",
                detected_type=DocumentType.STUDENT_COPIES,
                confidence=0.8,
                status=DocumentStatus.CLASSIFIED,
            ),
            SessionDocument(
                filename="ambigu.pdf",
                storage_path="/tmp/ambigu.pdf",
                detected_type=DocumentType.UNCLEAR,
                confidence=0.2,
                status=DocumentStatus.CLASSIFIED,
            ),
        ],
    )

    copy_ids, reference_ids, excluded_ids, issues = confirm_session_documents(session)

    assert copy_ids == [session.documents[0].id]
    assert reference_ids == []
    assert excluded_ids == []
    assert issues == ["Certains documents restent ambigus et doivent être confirmés ou exclus."]
    assert session.documents[1].status == DocumentStatus.CLASSIFIED
    assert session.documents[1].usable is False
    assert session.prepared_correction is not None
    assert session.prepared_correction.ready_to_grade is False


def test_confirm_session_documents_marks_ready_and_warns_for_multiple_copy_documents():
    session = GradingSession(
        session_id="session-docs",
        user_id="user-1",
        documents=[
            SessionDocument(
                filename="copies-a.pdf",
                storage_path="/tmp/copies-a.pdf",
                detected_type=DocumentType.STUDENT_COPIES,
                confidence=0.9,
                status=DocumentStatus.CLASSIFIED,
            ),
            SessionDocument(
                filename="copies-b.pdf",
                storage_path="/tmp/copies-b.pdf",
                detected_type=DocumentType.STUDENT_COPIES,
                confidence=0.85,
                status=DocumentStatus.CLASSIFIED,
            ),
            SessionDocument(
                filename="sujet.pdf",
                storage_path="/tmp/sujet.pdf",
                detected_type=DocumentType.SUBJECT_ONLY,
                confidence=0.8,
                status=DocumentStatus.CLASSIFIED,
            ),
        ],
    )

    copy_ids, reference_ids, excluded_ids, issues = confirm_session_documents(session)

    assert copy_ids == [session.documents[0].id, session.documents[1].id]
    assert reference_ids == [session.documents[2].id]
    assert excluded_ids == []
    assert issues == []
    assert session.prepared_correction is not None
    assert session.prepared_correction.ready_to_grade is True
    assert session.prepared_correction.primary_copy_document_id == session.documents[0].id
    assert session.prepared_correction.warnings == [
        "Plusieurs documents de copies sont confirmés. La correction utilisera tout le lot confirmé."
    ]
