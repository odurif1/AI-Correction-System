# AI Correction System

Code fonctionnel pour corriger des copies PDF avec des modèles d’IA, produire un audit exploitable et générer des PDF annotés.

## Ce que fait le projet

Le projet prend des copies PDF et exécute un pipeline complet de correction:

1. détection de la structure du document
2. préparation du barème et de la session
3. correction par un ou deux modèles
4. détection et résolution des désaccords
5. export des résultats
6. annotation PDF

Le résultat n’est pas seulement une note:
- lecture structurée des réponses
- décisions de correction traçables
- exports CSV / JSON
- PDF annoté sur la copie
- PDF overlay pour surimpression

## Capacités principales

- correction mono ou dual-LLM
- détection automatique de structure sur PDF
- audit détaillé des décisions de correction
- calibration et contrôle de cohérence entre copies
- annotation PDF avec placement des feedbacks
- API et CLI pour piloter le pipeline

## Structure utile

```text
src/
├── ai/             # Providers et orchestration LLM
├── analysis/       # Détection et analyse des documents
├── api/            # Endpoints HTTP + WebSocket
├── core/           # Session, grading, modèles métier
├── db/             # Persistance SQL
├── export/         # Exports et annotation PDF
├── prompts/        # Prompts
├── services/       # Jobs persistants, worker, billing
├── storage/        # Stockage des sessions et artefacts
└── vision/         # Lecture PDF / extraction page par page

tests/              # Tests ciblés
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
AI_CORRECTION_SESSION_SECRET=
AI_CORRECTION_DATABASE_URL=
# ou AI_CORRECTION_OPENAI_API_KEY=
# ou AI_CORRECTION_GLM_API_KEY=
# ou AI_CORRECTION_OPENROUTER_API_KEY=
```

En production:
- définissez `AI_CORRECTION_ENVIRONMENT=production`
- laissez `AI_CORRECTION_SESSION_COOKIE_SECURE=true`
- utilisez `AI_CORRECTION_DATABASE_URL` vers une base partagée plutôt que SQLite local
- faites tourner au moins un worker dédié
- configurez `AI_CORRECTION_ADMIN_API_KEY` si vous exposez les endpoints d’administration

API:

```bash
python src/main.py api --port 8000
```

Worker:

```bash
python src/main.py worker --poll-interval 2
```

Docker Compose production-like stack:

```bash
cp .env.example .env
docker compose up -d --build
```

CLI:

```bash
python -m src.main correct <pdfs...>
```

## Fichiers à lire en priorité

- [src/analysis/detection.py](src/analysis/detection.py)
- [src/core/session.py](src/core/session.py)
- [src/core/grading/grader.py](src/core/grading/grader.py)
- [src/export/annotation_pipeline.py](src/export/annotation_pipeline.py)
- [src/export/annotation_service.py](src/export/annotation_service.py)
- [src/export/pdf_annotator.py](src/export/pdf_annotator.py)

## Documentation

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
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
