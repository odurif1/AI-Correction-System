# AI Correction System

**Correction automatique de copies utilisant deux IA en parallèle avec vérification croisée pour garantir fiabilité et équité.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Fonctionnalités

- **Dual LLM**: Deux IA notent en parallèle avec vérification croisée et ultimatum
- **4 Providers supportés**: Gemini, OpenAI, GLM (z.ai), OpenRouter
- **3 modes de correction**: Individual, Batch, Hybrid
- **Context Caching**: Images mises en cache pour vérification/ultimatum (~10x moins cher)
- **Annotation PDF**: Génération automatique de copies annotées avec overlays
- **Protection Anti-Hallucination**: Figement des lectures, détection des flip-flops
- **Infrastructure Robuste**: Retry automatique avec backoff exponentiel, rate limiting

---

## Installation

```bash
# Cloner le repository
git clone <repo-url>
cd Correction

# Installer les dépendances
pip install -r requirements.txt

# Configurer les clés API
cp .env.example .env
# Éditer .env avec vos clés API
```

---

## Démarrage Rapide

```bash
# Correction dual LLM en mode batch (recommandé)
python -m src.main correct dual batch copies.pdf --auto-confirm

# Avec annotation des copies
python -m src.main correct dual batch copies.pdf --annotate --auto-confirm

# Mode single LLM (plus rapide, moins coûteux)
python -m src.main correct single batch copies.pdf --auto-confirm
```

---

## Syntaxe CLI

```
python -m src.main correct <LLM_MODE> <GRADING_METHOD> <PDF> [OPTIONS]
```

### Arguments positionnels

| Argument | Valeurs | Description |
|----------|---------|-------------|
| `LLM_MODE` | `single`, `dual` | Nombre de LLM utilisés |
| `GRADING_METHOD` | `individual`, `batch`, `hybrid` | Méthode de correction |
| `PDF` | chemin(s) | Fichier(s) PDF à corriger |

### Options principales

| Option | Description |
|--------|-------------|
| `--pages-per-copy N` | Découpe le PDF en copies de N pages |
| `--auto-confirm` | Mode automatique sans interaction |
| `--batch-verify MODE` | Vérification post-batch: `per-question` ou `grouped` (dual uniquement) |
| `--chat-continuation` | Active le context caching pour vérification/ultimatum (défaut: activé) |
| `--no-chat-continuation` | Désactive le context caching |
| `--second-reading` | 2ème lecture pour auto-correction |
| `--parallel N` | Copies en parallèle (défaut: 6, mode individual uniquement) |
| `--annotate` | Génère les PDFs annotés avec overlays |
| `--auto-detect-structure` | Pré-analyse le PDF pour détecter structure |
| `--output DIR` | Répertoire de sortie |
| `--export json,csv` | Formats d'export |

---

## Modes de Correction

### Matrice des combinaisons

```
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                      MODE DE CORRECTION                              │
                    │                                                                      │
                    │     INDIVIDUAL            BATCH                HYBRID              │
                    │     (défaut)              (--mode batch)       (--mode hybrid)      │
┌───────────────────┼─────────────────────────────────────────────────────────────────────┤
│         SINGLE    │  1 LLM × N copies         1 LLM × 1 appel      ✗ Non disponible    │
│  NOMBRE   (--single)│  = N appels API          = TOUTES en 1 réponse                    │
│  DE LLM           │                                                                      │
├───────────────────┼─────────────────────────────────────────────────────────────────────┤
│         DUAL      │  2 LLM × N copies         2 LLM × 1 appel     LLM1: batch          │
│         (défaut)  │  = 2N appels API          = 2 appels          LLM2: individual     │
│                   │  + vérification           + comparaison       + comparaison        │
└───────────────────┴─────────────────────────────────────────────────────────────────────┘
```

### Mode INDIVIDUAL (défaut)

Chaque copie est corrigée séparément, avec traitement parallèle possible.

**Avantages:**
- Focus LLM concentré sur une seule copie
- Parallélisable pour accélérer le traitement
- Chaque copie est indépendante (timeout sur une n'affecte pas les autres)

**Inconvénients:**
- N appels pour N copies (coût plus élevé)
- Cohérence entre copies non garantie

### Mode BATCH

Toutes les copies sont corrigées en UN SEUL appel API.

**Avantages:**
- **Cohérence absolue**: Le LLM voit toutes les réponses, garantit même réponse = même note
- **Économie**: 1-2 appels API au lieu de N
- **Pattern detection**: Le LLM identifie les réponses courantes, outliers, copiage potentiel

**Inconvénients:**
- Tout ou rien: si l'appel échoue, tout échoue
- Limite de tokens (grandes classes)

### Mode HYBRID (Dual LLM uniquement)

Combine le meilleur des deux mondes: LLM1 en batch, LLM2 en individual.

**Avantages:**
- Cohérence (via LLM1 batch) + Vérification indépendante (via LLM2 individual)
- Détection des erreurs par comparaison

---

## Workflow de Correction

```
┌─────────────────────────────────────────────────────────────────────┐
│  WORKFLOW DE CORRECTION (6 phases)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: INITIALIZATION                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Chargement du PDF                                            │  │
│  │ - Si --auto-detect-structure: analyse AI de la structure       │  │
│  │ - Sinon si --pages-per-copy: découpe mécanique                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 2: GRADING                                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Single-pass: les 2 LLM notent toutes les questions          │  │
│  │ - Détection des désaccords                                     │  │
│  │ - Token tracking par phase                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 3: VERIFICATION / ULTIMATUM                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Cross-verification (les LLMs voient le travail de l'autre)  │  │
│  │ - Ultimatum si désaccord persiste                              │  │
│  │ - Review manuelle si mode interactif                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 4: CALIBRATION                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Consistency check interne                                    │  │
│  │ - Détection des incohérences entre copies                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 5: EXPORT                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Export JSON (résultats détaillés + audit)                   │  │
│  │ - Export CSV (tableau des notes)                               │  │
│  │ - Analytics (statistiques, distribution)                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 6: ANNOTATION (optionnel, --annotate)                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Détection des coordonnées par LLM vision                    │  │
│  │ - Génération des PDFs annotés (annotated/)                     │  │
│  │ - Génération des overlays (overlays/)                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Architecture de Vérification (DUAL LLM)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SINGLE-PASS                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Les 2 LLM notent toutes les questions en parallèle       │    │
│  │ + détectent le nom de l'élève                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ANALYSE: Détecter les désaccords                               │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [Aucun désaccord]               [Au moins 1 désaccord]       │
│     → Résultat final                    │                        │
│                                         ▼                        │
│  PHASE 2: VÉRIFICATION                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ UN appel par LLM pour TOUTES les divergences:           │    │
│  │ → Chaque LLM réexamine et peut ajuster                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  [Toujours désaccord?]                                            │
│              │                                                    │
│              ▼                                                    │
│  PHASE 3: ULTIMATUM                                             │
│  Décision finale pour toutes les divergences restantes      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Réduction des appels API

| Scénario | Avant | Après | Économie |
|----------|-------|-------|----------|
| 3 questions en désaccord | 18 appels | 4 appels | **~78%** |
| 1 question + nom en désaccord | 10 appels | 4 appels | **~60%** |
| Tout OK (consensus) | 2 appels | 2 appels | - |

---

## Annotation PDF (--annotate)

L'option `--annotate` génère automatiquement des PDFs annotés avec le feedback de correction.

### Sorties générées

```
outputs/<session_id>/
├── annotated/                      # Copies complètes annotées
│   ├── Dupont_Marie_annotated.pdf
│   └── Martin_Luc_annotated.pdf
│
└── overlays/                       # Overlays (annotations uniquement)
    ├── Dupont_Marie_overlay.pdf
    └── Martin_Luc_overlay.pdf
```

**Copies annotées** (`annotated/`): PDF complet avec contenu original + annotations

**Overlays** (`overlays/`): PDF transparent avec uniquement les annotations, pour superposition

### Configuration

```bash
# .env - LLM pour annotation (optionnel, utilise le provider principal si non défini)
AI_CORRECTION_ANNOTATION_PROVIDER=glm
AI_CORRECTION_ANNOTATION_MODEL=glm-4.6v
```

---

## Exemples d'utilisation

```bash
# ═══════════════════════════════════════════════════════════════
# MODE BATCH - Recommandé pour la cohérence
# ═══════════════════════════════════════════════════════════════

# Dual LLM batch standard
python -m src.main correct dual batch copies.pdf --auto-confirm

# Dual LLM batch avec vérification groupée (défaut)
python -m src.main correct dual batch copies.pdf --batch-verify grouped --auto-confirm

# Dual LLM batch avec vérification par question (plus précis)
python -m src.main correct dual batch copies.pdf --batch-verify per-question --auto-confirm

# ═══════════════════════════════════════════════════════════════
# MODE INDIVIDUAL - Avec découpage
# ═══════════════════════════════════════════════════════════════

# Dual LLM + découpage 2 pages/copy
python -m src.main correct dual individual copies.pdf --pages-per-copy 2 --auto-confirm

# Single LLM + découpage
python -m src.main correct single individual copies.pdf --pages-per-copy 2 --auto-confirm

# ═══════════════════════════════════════════════════════════════
# AVEC ANNOTATION
# ═══════════════════════════════════════════════════════════════

# Correction + génération PDFs annotés
python -m src.main correct dual batch copies.pdf --annotate --auto-confirm

# ═══════════════════════════════════════════════════════════════
# OPTIONS AVANCÉES
# ═══════════════════════════════════════════════════════════════

# Avec 2ème lecture (auto-vérification)
python -m src.main correct single batch copies.pdf --second-reading --auto-confirm

# Pré-analyse de la structure
python -m src.main correct dual batch copies.pdf --auto-detect-structure --auto-confirm

# Parallélisme agressif (mode individual uniquement)
python -m src.main correct dual individual copies.pdf --parallel 10 --auto-confirm

# Mode HYBRID (dual uniquement)
python -m src.main correct dual hybrid copies.pdf --pages-per-copy 2 --auto-confirm

# ═══════════════════════════════════════════════════════════════
# AUTRES COMMANDES
# ═══════════════════════════════════════════════════════════════

# Voir le statut d'une session
python -m src.main status <session_id>

# Lister les sessions
python -m src.main list

# Exporter une session
python -m src.main export <session_id> --format json,csv

# Lancer l'API REST
python -m src.main api --port 8000
```

---

## Configuration

### Variables d'environnement (.env)

```bash
# ═══════════════════════════════════════════════════════════════
# Provider principal (REQUIRED)
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_AI_PROVIDER=gemini  # "gemini", "openai", "glm", "openrouter"

# ═══════════════════════════════════════════════════════════════
# API Keys (au moins une requise)
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_GEMINI_API_KEY=your_key
AI_CORRECTION_OPENAI_API_KEY=your_key
AI_CORRECTION_GLM_API_KEY=your_key
AI_CORRECTION_OPENROUTER_API_KEY=your_key

# ═══════════════════════════════════════════════════════════════
# Gemini Configuration
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_GEMINI_MODEL=gemini-2.5-flash
AI_CORRECTION_GEMINI_VISION_MODEL=gemini-2.5-flash
AI_CORRECTION_GEMINI_EMBEDDING_MODEL=text-embedding-004

# ═══════════════════════════════════════════════════════════════
# OpenAI Configuration
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_OPENAI_MODEL=gpt-4o
AI_CORRECTION_OPENAI_VISION_MODEL=gpt-4o
AI_CORRECTION_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ═══════════════════════════════════════════════════════════════
# Mode Comparaison (Double LLM)
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o

# ═══════════════════════════════════════════════════════════════
# Annotation PDF (optionnel)
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_ANNOTATION_PROVIDER=glm
AI_CORRECTION_ANNOTATION_MODEL=glm-4.6v

# ═══════════════════════════════════════════════════════════════
# Seuils de confiance
# ═══════════════════════════════════════════════════════════════
AI_CORRECTION_CONFIDENCE_AUTO=0.85
AI_CORRECTION_CONFIDENCE_FLAG=0.60
```

### Providers supportés

| Provider | Vision | Embeddings | Caractéristiques |
|----------|--------|------------|------------------|
| **Gemini** | ✓ | ✓ | Rapide, économique |
| **OpenAI** | ✓ | ✓ | Haute qualité |
| **GLM** (z.ai) | ✓ | - | Visual grounding, bounding boxes |
| **OpenRouter** | ✓ | - | Accès multi-modèles |

---

## Structure de l'Audit

L'audit contient les résultats finaux et le cheminement complet de chaque décision.

### Structure JSON

```json
{
  "graded_copies": [{
    "llm_comparison": {
      "options": {
        "mode": "batch",
        "providers": ["LLM1: gemini-2.5-flash", "LLM2: gpt-4o"],
        "total_copies": 2
      },
      "questions": {
        "Q1": {
          "max_points": 2.0,
          "LLM1: gemini-2.5-flash": {
            "grade": 1.5, "reading": "...", "reasoning": "..."
          },
          "LLM2: gpt-4o": {
            "grade": 1.0, "reading": "...", "reasoning": "..."
          },
          "final": {
            "grade": 1.25, "method": "average", "agreement": false
          },
          "verification": {
            "final_grade": 1.25,
            "llm1_new_grade": 1.5,
            "llm2_new_grade": 1.0,
            "method": "verification_average"
          },
          "ultimatum": {
            "final_grade": 1.25,
            "method": "ultimatum_average"
          }
        }
      }
    }
  }]
}
```

### Détection des problèmes

#### Désaccord sur les noms d'élèves
```
⚠️  PROBLÈME DE DÉTECTION DES ÉLÈVES
Les deux LLMs ont détecté des noms différents:
  Copie 1:
    LLM1: "Sophia Hanou"
    LLM2: "Sapio Nancy"
    Similarité: 45%
```

#### Désaccord sur le barème
```
⚠ Désaccord sur le barème pour 2 question(s)
  Copie 1, Q5: LLM1=2.0pts, LLM2=1.5pts → résolu à 2.0pts
```

---

## Architecture du Code

```
src/
├── ai/                        # Providers LLM
│   ├── base_provider.py           # Classe abstraite
│   ├── gemini_provider.py         # Provider Google Gemini
│   ├── openai_provider.py         # Provider OpenAI/OpenRouter
│   ├── comparison_provider.py     # Dual LLM avec vérification
│   ├── batch_grader.py            # Batch grading avec retry
│   ├── single_pass_grader.py      # Notation en 1 appel
│   ├── provider_factory.py        # Factory pour providers
│   └── disagreement_analyzer.py   # Détection désaccords
│
├── prompts/                   # Templates de prompts
│   ├── grading.py                 # Prompts de correction
│   ├── batch.py                   # Prompts batch (multi-élèves)
│   ├── verification.py            # Prompts de vérification
│   └── annotation.py              # Prompts d'annotation PDF
│
├── core/                      # Modèles et orchestration
│   ├── models.py                  # Dataclasses (GradingSession, GradedCopy...)
│   ├── exceptions.py              # Hiérarchie d'exceptions
│   ├── session.py                 # Orchestrateur de session
│   ├── workflow.py                # Workflow de correction
│   └── workflow_state.py          # État immutable du workflow
│
├── config/                    # Configuration
│   ├── settings.py                # Paramètres (env vars)
│   ├── constants.py               # Constantes centralisées
│   ├── providers.py               # Configuration des providers
│   └── logging_config.py          # Configuration logging
│
├── export/                    # Export et annotation
│   ├── pdf_annotator.py           # Annotation PDF
│   ├── annotation_service.py      # Service de détection
│   └── analytics.py               # Rapports et statistiques
│
├── utils/                     # Utilitaires
│   ├── sorting.py                 # Tri naturel (Q1, Q2, Q10)
│   ├── confidence.py              # Calcul de confiance
│   ├── rate_limiter.py            # Rate limiting + Circuit breaker
│   ├── retry.py                   # Retry avec backoff
│   ├── json_extractor.py          # Extraction JSON centralisée
│   └── name_matching.py           # Matching de noms
│
├── vision/                    # Lecture PDF
│   └── pdf_reader.py              # PDFReader avec context manager
│
├── storage/                   # Stockage JSON
│   └── file_store.py              # SessionStore avec file locking
│
├── interaction/               # Interface utilisateur
│   └── cli.py                     # CLI interactive (Rich)
│
├── api/                       # API REST (optionnel)
│   └── app.py                     # FastAPI application
│
└── main.py                    # Point d'entrée CLI
```

---

## Structure des Données

```
data/
└── {session_id}/
    ├── session.json           # État de la session + audit
    ├── policy.json            # Barème
    ├── copies/                # PDFs originaux
    ├── annotated/             # PDFs annotés (export)
    └── reports/               # CSV, JSON (export)
```

---

## Infrastructure et Robustesse

### Gestion des Erreurs

- Hiérarchie d'exceptions custom (`AICorrectionError`, `ProviderError`, `APIConnectionError`...)
- Retry automatique avec backoff exponentiel (3 tentatives)
- Détection des erreurs retryables (503, 429, 500, 502, 504)

### Rate Limiting et Circuit Breaker

- Token bucket rate limiter
- Protection contre APIs défaillantes
- Rate limits configurables par provider

### Token Tracking

Le système track les tokens par phase:
- **Correction**: Appels initiaux de notation
- **Vérification**: Cross-verification + ultimatum
- **Calibration**: Consistency check
- **Annotation**: Génération PDFs annotés

### Sécurité

- Protection contre path traversal
- Limite de taille des fichiers (10 MB par page, 50 MB total)
- Sanitisation des clés API dans les logs

---

## Développement

```bash
# Tests unitaires
pytest tests/

# Formatage
black src/ && isort src/

# Type checking
mypy src/
```

---

## Documentation

- `docs/dual_llm_architecture.md` - Architecture Dual LLM détaillée
- `docs/annotation.md` - Module d'annotation PDF
- `docs/AUDIT_STRUCTURE.md` - Structure de l'audit

---

## Licence

MIT License - voir [LICENSE](LICENSE)
