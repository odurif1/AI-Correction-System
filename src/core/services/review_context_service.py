"""
Post-correction enrichment for professor-facing review screens.

This service is intentionally separate from the grading pipeline. It derives
compact review context only when disagreements are displayed.
"""

from __future__ import annotations

import json
import re
import tempfile
from typing import Iterable

from config.settings import get_settings
from vision.pdf_reader import PDFReader


UNIT_PATTERN = r"(?:g\/L|mg\/L|kg\/L|g|kg|mg|mL|L|cL|dL|mol\/L|mmol\/L|cm3|cm³|m3|m³|%)"
QUANTITY_RE = re.compile(rf"\b\d+(?:[.,]\d+)?\s*{UNIT_PATTERN}\b", re.IGNORECASE)
ASSIGNMENT_RE = re.compile(
    rf"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*\d+(?:[.,]\d+)?\s*{UNIT_PATTERN}\b",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[A-Za-z][A-Za-z0-9_]*\s*[x×*]\s*[A-Za-z][A-Za-z0-9_]*\b"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


class ReviewContextService:
    """Build compact factual context for disagreement review."""

    def __init__(self, provider=None):
        self.settings = get_settings()
        self.provider = provider or self._create_review_context_provider()

    def build_context(
        self,
        question_text: str | None,
        llm_reasonings: list[str] | None = None,
    ) -> dict[str, object] | None:
        ai_context = self._build_ai_context(question_text, llm_reasonings or [])
        if ai_context:
            return ai_context

        sources: list[tuple[str, str]] = []
        if question_text and question_text.strip():
            sources.append(("question_text", question_text.strip()))
        for reasoning in llm_reasonings or []:
            if reasoning and reasoning.strip():
                sources.append(("llm_reasoning", reasoning.strip()))

        facts_with_source: list[tuple[str, str]] = []
        for source_name, text in sources:
            for fact in self._extract_facts(text):
                facts_with_source.append((fact, source_name))

        unique_facts = _dedupe_keep_order(fact for fact, _ in facts_with_source)
        if not unique_facts:
            return None

        fact_sources = {source for _, source in facts_with_source}
        if fact_sources == {"question_text"}:
            facts_source = "enonce"
        elif fact_sources == {"llm_reasoning"}:
            facts_source = "raisonnements_ia"
        else:
            facts_source = "mixte"

        return {
            "question_facts": unique_facts[:6],
            "facts_source": facts_source,
        }

    def extract_question_text_from_pdf(
        self,
        pdf_path: str | None,
        question_hint: str | None = None,
    ) -> str | None:
        if not pdf_path:
            return None

        try:
            with PDFReader(pdf_path) as reader:
                candidate_lines: list[str] = []
                for page_num in range(min(reader.get_page_count(), 2)):
                    text = reader.extract_text(page_num)
                    for raw_line in text.splitlines():
                        line = " ".join(raw_line.split()).strip()
                        if self._looks_like_question_line(line):
                            candidate_lines.append(line)

                if not candidate_lines:
                    return self._extract_question_text_with_vision(reader, question_hint)

                return " ".join(_dedupe_keep_order(candidate_lines[:2]))
        except Exception:
            return None

    def _extract_question_text_with_vision(
        self,
        reader: PDFReader,
        question_hint: str | None = None,
    ) -> str | None:
        if not self.provider or not hasattr(self.provider, "call_vision"):
            return None

        image_paths: list[str] = []
        temp_files: list[str] = []
        try:
            for page_num in range(min(reader.get_page_count(), 2)):
                image = reader.get_page_image(page_num)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(tmp.name, format="PNG")
                image_paths.append(tmp.name)
                temp_files.append(tmp.name)
                tmp.close()

            hint = question_hint or "la question concernee"
            prompt = (
                "Lis ces pages de copie et retrouve uniquement l'enonce imprime de "
                f"{hint}. Retourne seulement le texte de l'enonce, sans explication, "
                "sans markdown, sans prefixe. Si tu ne trouves pas un enonce lisible, retourne une chaine vide."
            )
            raw = self.provider.call_vision(prompt, image_path=image_paths).strip()
            if not raw:
                return None

            cleaned = " ".join(raw.split()).strip()
            if len(cleaned) < 12:
                return None
            return cleaned
        except Exception:
            return None
        finally:
            for path in temp_files:
                try:
                    import os
                    os.unlink(path)
                except OSError:
                    pass

    def _looks_like_question_line(self, line: str) -> bool:
        if len(line) < 20:
            return False
        lowered = line.lower()
        if re.fullmatch(r"[0-9\s.:;,+\-/*=()%]+", line):
            return False
        question_markers = (
            "?",
            "calcul",
            "détermin",
            "determiner",
            "exprimer",
            "justifier",
            "donner",
            "déduire",
            "deduire",
            "montrer",
            "expliquer",
            "compléter",
            "completer",
        )
        return any(marker in lowered for marker in question_markers)

    def _create_review_context_provider(self):
        from ai.provider_factory import create_ai_provider

        provider_name = self.settings.review_context_provider
        model = self.settings.review_context_model

        if not model:
            return None

        if not provider_name:
            provider_name = self.settings.ai_provider

        try:
            return create_ai_provider(provider_name, model=model)
        except Exception:
            return None

    def _build_ai_context(
        self,
        question_text: str | None,
        llm_reasonings: list[str],
    ) -> dict[str, object] | None:
        if not self.provider:
            return None

        prompt_parts = []
        if question_text and question_text.strip():
            prompt_parts.append(f"ENONCE:\n{question_text.strip()}")

        reasoning_block = "\n\n".join(
            reasoning.strip()
            for reasoning in llm_reasonings
            if reasoning and reasoning.strip()
        )
        if reasoning_block:
            prompt_parts.append(f"RAISONNEMENTS IA:\n{reasoning_block}")

        if not prompt_parts:
            return None

        prompt = (
            "Extrait au maximum 6 donnees utiles et factuelles pour aider un professeur a arbitrer un desaccord.\n"
            "Concentre-toi sur les valeurs, unites, formules, grandeurs et contraintes explicites.\n"
            "Ne deduis pas la bonne note. Ne paraphrase pas.\n"
            "Retourne UNIQUEMENT un JSON de la forme:\n"
            '{"question_facts":["..."],"facts_source":"enonce|raisonnements_ia|mixte"}\n\n'
            + "\n\n".join(prompt_parts)
        )

        try:
            raw = self.provider.call_text(prompt, response_format="json")
            data = json.loads(raw)
        except Exception:
            return None

        facts = data.get("question_facts", [])
        if not isinstance(facts, list):
            return None

        normalized_facts = _dedupe_keep_order(str(fact) for fact in facts)
        if not normalized_facts:
            return None

        facts_source = data.get("facts_source")
        if facts_source not in {"enonce", "raisonnements_ia", "mixte"}:
            facts_source = "mixte"

        return {
            "question_facts": normalized_facts[:6],
            "facts_source": facts_source,
        }

    def _extract_facts(self, text: str) -> list[str]:
        facts: list[str] = []

        facts.extend(match.group(0) for match in ASSIGNMENT_RE.finditer(text))
        facts.extend(match.group(0) for match in FORMULA_RE.finditer(text))
        facts.extend(match.group(0) for match in QUANTITY_RE.finditer(text))

        if facts:
            return _dedupe_keep_order(facts)

        candidate_sentences = []
        for sentence in SENTENCE_SPLIT_RE.split(text):
            stripped = sentence.strip()
            if not stripped:
                continue
            if QUANTITY_RE.search(stripped):
                candidate_sentences.append(stripped)

        return _dedupe_keep_order(candidate_sentences[:3])
