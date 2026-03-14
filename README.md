# AI Correction System

Backend open source pour l’audit et l’expérimentation d’un pipeline de correction assistée par IA sur copies PDF.

## Objet

Le dépôt se concentre sur la logique métier de correction:
- détection de structure des documents PDF
- préparation de session de correction
- correction mono ou dual-LLM
- résolution de désaccords
- calibration entre copies
- exports et annotation PDF

Le dépôt public ne contient pas de frontend, de billing, ni de packaging de déploiement.

## Pipeline métier

1. Ingestion de PDFs
2. Détection du type de document et de la structure
3. Préparation de la session et du barème
4. Correction par un ou deux modèles
5. Audit et résolution des désaccords
6. Exports CSV / JSON / PDF annoté / overlay

## Structure

```text
src/
├── api/            # API FastAPI minimale pour exposer le pipeline
├── ai/             # Providers et orchestration LLM
├── analysis/       # Détection et analyse des documents
├── core/           # Modèles métier, session, grading
├── db/             # Persistance SQLite
├── export/         # Analytics et annotation PDF
├── prompts/        # Prompts
├── storage/        # Stockage des sessions et artefacts
└── vision/         # Lecture PDF / extraction page par page

tests/              # Tests ciblés sur le métier
docs/               # Documentation technique
```

## Démarrage minimal

Prérequis:
- Python 3.10+
- une clé API pour au moins un provider

Installation:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Configuration minimale:

```env
AI_CORRECTION_AI_PROVIDER=
AI_CORRECTION_GEMINI_API_KEY=
# ou AI_CORRECTION_OPENAI_API_KEY=
# ou AI_CORRECTION_GLM_API_KEY=
# ou AI_CORRECTION_OPENROUTER_API_KEY=
```

Lancement API:

```bash
python src/main.py api --port 8000
```

Exécution CLI:

```bash
python -m src.main correct <pdfs...>
```

## Points d’intérêt pour l’audit

- [src/analysis/detection.py](src/analysis/detection.py)
- [src/core/session.py](src/core/session.py)
- [src/core/grading/grader.py](src/core/grading/grader.py)
- [src/export/annotation_pipeline.py](src/export/annotation_pipeline.py)
- [src/export/annotation_service.py](src/export/annotation_service.py)
- [src/export/pdf_annotator.py](src/export/pdf_annotator.py)

## Documentation

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- [docs/API.md](docs/API.md)
- [docs/annotation.md](docs/annotation.md)
- [docs/dual_llm_architecture.md](docs/dual_llm_architecture.md)
- [docs/AUDIT_STRUCTURE.md](docs/AUDIT_STRUCTURE.md)

## Tests

```bash
pytest tests/
```

## Licence

MIT. Voir [LICENSE](LICENSE).
