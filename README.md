# AI Correction System

Système de correction intelligent utilisant l'IA Vision-Language pour corriger des copies d'élèves avec analyse croisée, vérification de cohérence et calibration rétroactive.

## Fonctionnalités

### Double LLM avec Comparaison
- **Correction parallèle**: Deux LLM notent chaque question en parallèle
- **Vérification croisée**: En cas de désaccord, chaque LLM voit le raisonnement de l'autre
- **Détection de fausse convergence**: Identifie quand les LLM prétendent être d'accord mais ont des notes différentes
- **Round 2 "Ultimatum"**: Force une décision réelle si la vérification croisée échoue

### Consensus de Lecture (activé par défaut)
- **Phase 1 - Lecture**: Les LLM décrivent ce qu'ils voient avant de noter
- **Détection de contradictions**: Erlenmeyer vs fiole jaugée, couleurs, formes...
- **Validation utilisateur**: Si les lectures divergent, l'enseignant tranche
- **Désactivation**: Utiliser `--skip-reading` pour noter directement sans validation de lecture

### Autres fonctionnalités
- **IA Vision-Language**: "Voit" et comprend directement l'écriture manuscrite
- **Analyse croisée**: Regroupe les réponses similaires pour garantir l'équité
- **Jurisprudence**: Les décisions passées de l'enseignant influencent les corrections futures
- **Stockage local**: Toutes les données en fichiers JSON - pas de base de données externe

## Installation

```bash
# Cloner le repository
git clone https://github.com/votre-repo/ai-correction.git
cd ai-correction

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

## Configuration

### Variables d'environnement

```bash
# Provider principal
AI_CORRECTION_LLM_PROVIDER=gemini  # ou openai

# Clés API
AI_CORRECTION_GEMINI_API_KEY=votre_cle_gemini
AI_CORRECTION_OPENAI_API_KEY=votre_cle_openai

# Mode comparaison (double LLM)
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o
```

## Utilisation

### CLI

Corriger des copies:

```bash
python src/main.py correct copies/*.pdf
```

Avec options:

```bash
# Mode automatique (pas d'interaction)
python src/main.py correct copies/*.pdf --auto

# Utiliser un seul LLM (plus rapide, moins coûteux)
python src/main.py correct copies/*.pdf --single

# Ignorer le consensus de lecture (les LLM notent directement)
python src/main.py correct copies/*.pdf --skip-reading

# Spécifier le barème
python src/main.py correct copies/*.pdf --scale Q1=5,Q2=3,Q3=4
```

Voir le statut d'une session:

```bash
python src/main.py status <session_id>
```

Voir les analytiques:

```bash
python src/main.py analytics <session_id>
```

### API

Démarrer le serveur:

```bash
uvicorn src.api.app:app --reload
```

Endpoints:

- `POST /api/sessions` - Créer une session
- `POST /api/sessions/{id}/upload` - Upload des PDFs
- `POST /api/sessions/{id}/grade` - Démarrer la correction
- `POST /api/sessions/{id}/decisions` - Soumettre une décision enseignant
- `GET /api/sessions/{id}` - Statut de la session

## Architecture

```
src/
├── core/
│   ├── models.py           # Modèles Pydantic (GradingSession, GradedCopy, etc.)
│   └── session.py          # Orchestration des sessions
├── ai/
│   ├── gemini_provider.py  # Provider Google Gemini
│   ├── openai_provider.py  # Provider OpenAI
│   ├── comparison_provider.py  # Double LLM avec comparaison
│   └── response_parser.py  # Parsing des réponses structurées
├── vision/
│   └── pdf_reader.py       # Lecture des PDFs
├── analysis/
│   ├── cross_copy.py       # Analyse croisée des copies
│   └── clustering.py       # Clustering des réponses
├── grading/
│   ├── grader.py           # Moteur de notation
│   ├── uncertainty.py      # Gestion des incertitudes
│   └── feedback.py         # Génération de feedback
├── storage/
│   └── file_store.py       # Stockage JSON local
├── api/
│   └── app.py              # API FastAPI
└── main.py                 # Point d'entrée CLI
```

## Structure des données

```
data/
├── {session_id}/
│   ├── session.json        # État de la session
│   ├── policy.json         # Barème de correction
│   ├── copies/
│   │   └── {n}/
│   │       ├── original.pdf      # PDF original
│   │       ├── annotation.json   # Infos essentielles pour annotation
│   │       └── audit.json        # TOUT: échanges LLM, debug
│   ├── annotated/          # PDFs annotés (export)
│   └── reports/            # Rapports (export)
└── _index.json             # Index des sessions
```

### Fichiers par copie

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `annotation.json` | Nom élève, notes, feedbacks | Annotation du PDF |
| `audit.json` | Échanges LLM, comparaisons, raisonnements | Debugging, audit |

### Structure de l'audit

L'audit contient une traçabilité complète du processus de correction:

```
audit.json
├── initial              # Résultats avant discussion
├── reading_analysis     # Comparaison des lectures
├── confidence_evolution # Évolution de la confiance
├── timing               # Durées de chaque phase
├── decision_path        # Chemin de décision emprunté
├── after_verification   # Résultats après vérification croisée
│   └── prompt_sent      # Prompt exact envoyé à chaque LLM
├── after_round2         # Résultats après ultimatum (si applicable)
└── final                # Note finale et méthode
```

Voir [docs/AUDIT_STRUCTURE.md](docs/AUDIT_STRUCTURE.md) pour la documentation complète.

## Workflow de correction

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW DE CORRECTION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. IMPORT PDF ──→ Extraction des pages                         │
│         │                                                        │
│         ▼                                                        │
│  2. DÉTECTION NOM ──→ Consensus entre LLMs                      │
│         │                                                        │
│         ▼                                                        │
│  3. POUR CHAQUE QUESTION:                                       │
│         │                                                        │
│         ├─→ Phase 1: LECTURE (par défaut)                       │
│         │       ├─→ LLM1 décrit ce qu'il voit                   │
│         │       ├─→ LLM2 décrit ce qu'il voit                   │
│         │       └─→ Si désaccord → validation enseignant        │
│         │                                                        │
│         ├─→ Phase 2: NOTATION (double LLM)                      │
│         │       ├─→ LLM1 note en parallèle                      │
│         │       └─→ LLM2 note en parallèle                      │
│         │                                                        │
│         ├─→ COMPARAISON                                         │
│         │       └─→ Notes identiques? → OK                      │
│         │                                                        │
│         ├─→ VÉRIFICATION CROISÉE (si désaccord)                 │
│         │       └─→ Chaque LLM voit le raisonnement de l'autre  │
│         │                                                        │
│         ├─→ ROUND 2 ULTIMATUM (si fausse convergence)           │
│         │       └─→ Force une décision réelle                   │
│         │                                                        │
│         └─→ DÉCISION FINALE                                     │
│                 └─→ Accord → note finale                        │
│                 └─→ Désaccord → demande utilisateur             │
│                                                                  │
│  4. GÉNÉRATION APPRÉCIATION                                     │
│  5. EXPORT (PDF annoté, CSV, JSON)                              │
│                                                                  │
│  Options CLI:                                                    │
│    --skip-reading  : Ignorer Phase 1 (notation directe)         │
│    --single        : Utiliser un seul LLM                       │
│    --auto          : Mode automatique sans interaction          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Principes

| Principe | Description |
|----------|-------------|
| **Justesse** | L'IA demande de l'aide quand incertaine |
| **Équité** | Cohérence entre tous les élèves |
| **Souplexité** | L'IA généralise à partir du contexte |
| **Transparence** | Toutes les décisions sont tracées |

## Développement

### Tests

```bash
pytest tests/
```

### Format du code

```bash
black src/
isort src/
```

## Contribution

Les contributions sont les bienvenues! Veuillez:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

## Licence

MIT License - voir [LICENSE](LICENSE)

## Auteurs

Développé pour simplifier et améliorer la correction des copies d'élèves.
