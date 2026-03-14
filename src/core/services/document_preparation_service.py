"""
Prepare a grading session from uploaded documents.

This service keeps the pre-grading contract small:
- determine which documents are usable as copies or references
- surface only blocking questions for the teacher
- expose a ready / not-ready state for the grading pipeline
"""

from __future__ import annotations

from typing import List, Optional

from core.models import (
    DocumentDecision,
    DocumentStatus,
    DocumentType,
    GradingSession,
    PreparedCorrectionQuestion,
    PreparedCorrectionSession,
    SessionDocument,
)


def decision_to_document_type(decision: DocumentDecision) -> Optional[DocumentType]:
    mapping = {
        DocumentDecision.STUDENT_COPIES: DocumentType.STUDENT_COPIES,
        DocumentDecision.SUBJECT_ONLY: DocumentType.SUBJECT_ONLY,
        DocumentDecision.GRADING_SCHEME: DocumentType.GRADING_SCHEME,
    }
    return mapping.get(decision)


class DocumentPreparationService:
    """Build and persist a minimal prepared correction state on the session."""

    def apply_document_decisions(self, session: GradingSession) -> PreparedCorrectionSession:
        warnings: List[str] = []
        questions: List[PreparedCorrectionQuestion] = []
        copy_ids: List[str] = []
        reference_ids: List[str] = []
        excluded_ids: List[str] = []

        subject_ids: List[str] = []
        grading_ids: List[str] = []
        pending_ids: List[str] = []

        for document in session.documents:
            decision = document.user_decision
            if decision == DocumentDecision.PENDING:
                decision = {
                    DocumentType.STUDENT_COPIES: DocumentDecision.STUDENT_COPIES,
                    DocumentType.SUBJECT_ONLY: DocumentDecision.SUBJECT_ONLY,
                    DocumentType.GRADING_SCHEME: DocumentDecision.GRADING_SCHEME,
                }.get(document.detected_type, DocumentDecision.PENDING)
                document.user_decision = decision

            if decision == DocumentDecision.PENDING:
                document.usable = False
                document.status = DocumentStatus.CLASSIFIED
                pending_ids.append(document.id)
                continue

            if decision == DocumentDecision.EXCLUDE:
                document.usable = False
                document.status = DocumentStatus.REJECTED
                excluded_ids.append(document.id)
                continue

            document_type = decision_to_document_type(decision)
            if document_type is None:
                document.usable = False
                document.status = DocumentStatus.REJECTED
                excluded_ids.append(document.id)
                continue

            document.detected_type = document_type
            document.usable = True
            document.status = DocumentStatus.CONFIRMED

            if document_type == DocumentType.STUDENT_COPIES:
                copy_ids.append(document.id)
            else:
                reference_ids.append(document.id)
                if document_type == DocumentType.SUBJECT_ONLY:
                    subject_ids.append(document.id)
                if document_type == DocumentType.GRADING_SCHEME:
                    grading_ids.append(document.id)

        if not copy_ids:
            questions.append(
                PreparedCorrectionQuestion(
                    code="missing_copies",
                    message="Confirmez au moins un document comme copie élève avant de lancer la correction.",
                    document_ids=[doc.id for doc in session.documents],
                )
            )

        if pending_ids:
            questions.append(
                PreparedCorrectionQuestion(
                    code="pending_documents",
                    message="Certains documents restent ambigus et doivent être confirmés ou exclus.",
                    document_ids=pending_ids,
                )
            )

        if len(subject_ids) > 1:
            questions.append(
                PreparedCorrectionQuestion(
                    code="multiple_subjects",
                    message="Plusieurs documents sont marqués comme sujet. Gardez-en un seul ou excluez les doublons.",
                    document_ids=subject_ids,
                )
            )

        if len(grading_ids) > 1:
            questions.append(
                PreparedCorrectionQuestion(
                    code="multiple_grading_schemes",
                    message="Plusieurs documents sont marqués comme barème. Gardez-en un seul ou excluez les doublons.",
                    document_ids=grading_ids,
                )
            )

        if len(copy_ids) > 1:
            warnings.append(
                "Plusieurs documents de copies sont confirmés. La correction utilisera tout le lot confirmé."
            )

        prepared = PreparedCorrectionSession(
            copy_document_ids=copy_ids,
            reference_document_ids=reference_ids,
            excluded_document_ids=excluded_ids,
            warnings=warnings,
            questions_for_user=questions,
            ready_to_grade=not questions and bool(copy_ids),
            primary_copy_document_id=copy_ids[0] if copy_ids else None,
        )
        session.prepared_correction = prepared
        return prepared

    def build_from_existing_state(self, session: GradingSession) -> PreparedCorrectionSession:
        """Rebuild prepared state without mutating user decisions."""
        mutable_copy = session.model_copy(deep=True)
        return self.apply_document_decisions(mutable_copy)

    def get_copy_documents(self, session: GradingSession) -> List[SessionDocument]:
        prepared = session.prepared_correction
        if prepared and prepared.copy_document_ids:
            allowed = set(prepared.copy_document_ids)
            return [doc for doc in session.documents if doc.id in allowed and doc.usable]

        return [
            doc
            for doc in session.documents
            if doc.usable and doc.user_decision == DocumentDecision.STUDENT_COPIES
        ]

    def get_reference_documents(self, session: GradingSession) -> List[SessionDocument]:
        prepared = session.prepared_correction
        if prepared and prepared.reference_document_ids:
            allowed = set(prepared.reference_document_ids)
            return [doc for doc in session.documents if doc.id in allowed and doc.usable]

        return [
            doc
            for doc in session.documents
            if doc.usable and doc.user_decision in (DocumentDecision.SUBJECT_ONLY, DocumentDecision.GRADING_SCHEME)
        ]
