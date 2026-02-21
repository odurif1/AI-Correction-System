# AI Correction System

**Correction automatique de copies utilisant deux IA en parallèle pour garantir fiabilité et équité.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Caractéristiques

- **Dual LLM**: Deux IA notent en parallèle avec vérification croisée
- **Architecture Unifiée**: 60-80% d'appels API en moins
- **Dashboard Temps Réel**: Visualisation du traitement parallèle
- **Protection Anti-Hallucination**: Figement des lectures, détection des flip-flops
- **Infrastructure Robuste**: Circuit breaker, rate limiting, retry automatique

---

## Concepts clés

Le système combine deux dimensions indépendantes:

### 1. Nombre de LLM: SINGLE vs DUAL

| Mode | Option | Description |
|------|--------|-------------|
| **SINGLE** | `--single` | Un seul LLM corrige chaque copie |
| **DUAL** | *(défaut)* | Deux LLM notent en parallèle, avec phases de vérification si désaccord |

### 2. Mode de correction: INDIVIDUAL vs BATCH vs HYBRID

| Mode | Option | Description |
|------|--------|-------------|
| **INDIVIDUAL** | *(défaut)* | Chaque copie est corrigée séparément (N appels API) |
| **BATCH** | `--mode batch` | Toutes les copies corrigées en UN appel API |
| **HYBRID** | `--mode hybrid` | LLM1=batch, LLM2=individual, puis comparaison |

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

---

## Pourquoi ce système ?

| Problème | Solution |
|----------|----------|
| Une IA peut se tromper | **DUAL**: Deux IA notent et se confrontent |
| Erreurs de lecture | **Phases de vérification**: isolée → échange → ultimatum |
| Manque de traçabilité | **Audit complet**: chaque décision est documentée |
| Feedback trop "gentil" | **Retours professionnels**: sobres, adaptés à la difficulté |
| Erreurs d'étourderie | **--second-reading**: 2ème lecture pour auto-correction |

---

## Mode INDIVIDUAL (défaut)

Chaque copie est corrigée séparément, avec traitement parallèle possible.

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE INDIVIDUAL                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Le PDF est découpé en chunks de N pages (--pages-per-student)│
│  2. Chaque chunk est corrigé indépendamment                      │
│  3. Parallélisable (--parallel 6 copies simultanées)             │
│                                                                  │
│  PDF de 12 pages + --pages-per-student 2 → 6 copies             │
│  6 copies × 2 LLM = 12 appels API (ou 6 si --single)            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Avantages:**
- Focus LLM concentré sur une seule copie
- Parallélisable pour accélérer le traitement
- Chaque copie est indépendante (timeout sur une n'affecte pas les autres)

**Inconvénients:**
- N appels pour N copies (coût plus élevé)
- Cohérence entre copies non garantie

---

## Mode BATCH (--mode batch)

Toutes les copies sont corrigées en UN SEUL appel API.

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE BATCH                                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Toutes les images de toutes les copies envoyées en 1 appel  │
│  2. Le LLM voit TOUTES les copies et peut comparer              │
│  3. Cohérence garantie: même réponse = même note                │
│                                                                  │
│  PDF de 12 pages → 1 appel API → TOUTES les notes               │
│  (ou 2 appels si Dual LLM pour comparaison)                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Avantages:**
- **Cohérence absolue**: Le LLM voit toutes les réponses, garantit same réponse = same note
- **Économie**: 1-2 appels API au lieu de N
- **Pattern detection**: Le LLM identifie les réponses courantes, outliers, copiage potentiel
- **Contexte**: Aider la lecture manuscrite en comparant les réponses

**Inconvénients:**
- Tout ou rien: si l'appel échoue, tout échoue
- Limite de tokens (grandes classes)

---

## Mode HYBRID (--mode hybrid, Dual LLM uniquement)

Combine le meilleur des deux mondes: LLM1 en batch, LLM2 en individual.

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE HYBRID (Dual LLM uniquement)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  EN PARALLÈLE:                                          │    │
│  │                                                         │    │
│  │  LLM1 (BATCH)              LLM2 (INDIVIDUAL)            │    │
│  │  1 appel                   N appels (parallélisés)      │    │
│  │  Voit toutes les copies    Chaque copie indépendante    │    │
│  │  Cohérence garantie        Vérification indépendante    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPARAISON                                            │    │
│  │  Pour chaque copie: comparer les notes LLM1 vs LLM2     │    │
│  │  Si désaccord → moyenne (ou flag pour review)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Avantages:**
- **Cohérence** (via LLM1 batch) + **Vérification indépendante** (via LLM2 individual)
- Détection des erreurs par comparaison
- Le meilleur des deux modes

**Inconvénients:**
- Nécessite Dual LLM
- 1 + N appels API (plus que batch, moins que individual)

---

## Architecture de Vérification Unifiée (DUAL LLM)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SINGLE-PASS                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Les 2 LLM notent toutes les questions en parallèle       │    │
│  │ + détectent le nom de l'élève                           │    │
│  │                                                          │    │
│  │ Si --second-reading:                                     │    │
│  │   → Chaque LLM fait une 2ème lecture DANS son prompt    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ANALYSE: Détecter les désaccords (nom + questions)             │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [Aucun désaccord]               [Au moins 1 désaccord]       │
│     → Résultat final                    │                        │
│                                         ▼                        │
│  PHASE 2: VÉRIFICATION UNIFIÉE (2 appels max)              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ UN SEUL appel par LLM pour TOUTES les divergences:       │    │
│  │ - Nom: LLM1='Jean' vs LLM2='Marie'                      │    │
│  │ - Q1: note 1.0 vs 0.5                                    │    │
│  │ - Q3: lecture différente                                 │    │
│  │ → Chaque LLM réexamine tout et répond                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  [Toujours désaccord?]                                            │
│              │                                                    │
│              ▼                                                    │
│  PHASE 3: ULTIMATUM UNIFIÉ (2 appels max)                   │
│  Décision finale pour toutes les divergences restantes      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Réduction des appels API

| Scénario | Avant | Après | Économie |
|----------|-------|-------|----------|
| 3 questions en désaccord | 18 appels | 4 appels | **~78%** |
| 1 question + nom en désaccord | 10 appels | 4 appels | **~60%** |
| Tout OK (consensus) | 0 appels | 0 appels | - |

### Mode SINGLE LLM (--single)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLE LLM - 1 ou 2 APPELS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  APPEL 1: CORRECTION INITIALE                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LLM: Analyse copie + Corrige toutes questions           │    │
│  │ → Retourne grades, readings, feedbacks                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│    [--second-reading OFF]          [--second-reading ON]         │
│    Finaliser directement          APPEL 2: DEUXIÈME LECTURE     │
│                                   ┌─────────────────────────┐   │
│                                   │ MÊME images + résultats  │   │
│                                   │ de l'appel 1             │   │
│                                   │ → Peut ajuster les notes │   │
│                                   └─────────────────────────┘   │
│                                              │                   │
│                                              ▼                   │
│                                      RÉSULTAT FINAL             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Démarrage rapide

```bash
# 1. Installer
pip install -r requirements.txt

# 2. Configurer les clés API
cp .env.example .env
# Éditer .env avec vos clés Gemini et/ou OpenAI

# 3. Lancer une correction
python -m src.main correct dual individual copies.pdf --auto-confirm
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

### Options

| Option | Description |
|--------|-------------|
| `--pages-per-copy N` | Découpe le PDF en copies de N pages. Si omis, PDF envoyé entier au LLM. |
| `--auto-confirm` | Mode automatique sans interaction |
| `--batch-verify MODE` | Vérification post-batch: `per-question` ou `grouped` (dual batch uniquement) |
| `--second-reading` | 2ème lecture (2 passes en Single, intégrée en Dual) |
| `--parallel N` | Copies en parallèle (défaut: 6, mode INDIVIDUAL uniquement) |
| `--output DIR` | Répertoire de sortie |
| `--export json,csv` | Formats d'export |

---

## Option --batch-verify

En mode **dual batch**, cette option est **obligatoire**. Elle détermine comment les désaccords entre les deux LLM sont résolus:

### Logique de vérification (identique au mode individual)

Les deux modes suivent la même logique:
1. **Vérification**: Les deux LLMs voient le travail de l'autre et peuvent ajuster leur note
2. **Ultimatum**: Si désaccord persiste, un round final est exécuté

### `--batch-verify per-question`

Un appel API **par désaccord** pour **chaque LLM** (les deux LLMs voient le travail de l'autre).

```
Batch → 2 LLMs → 3 désaccords → 3 appels × 2 LLMs = 6 appels de vérification
Si désaccord persiste → 6 appels d'ultimatum
Total max: 2 + 6 + 6 = 14 appels API
```

**Avantage**: Plus précis, chaque désaccord est traité isolément.

### `--batch-verify grouped`

Un **seul appel** par LLM regroupant **tous les désaccords** (les deux LLMs voient le travail de l'autre).

```
Batch → 2 LLMs → 3 désaccords → 1 appel × 2 LLMs = 2 appels de vérification
Si désaccord persiste → 2 appels d'ultimatum
Total max: 2 + 2 + 2 = 6 appels API
```

**Avantage**: Plus efficace, moins d'appels API.

**Exemple:**
```bash
# Vérification par question (plus précis)
python -m src.main correct dual batch copies.pdf --batch-verify per-question

# Vérification groupée (plus rapide)
python -m src.main correct dual batch copies.pdf --batch-verify grouped
```

---

## Option --pages-per-copy

Cette option est **optionnelle**. Elle détermine comment le PDF est traité:

### AVEC --pages-per-copy N (découpage)

Le PDF est découpé en copies de N pages chacune avant d'être envoyé au LLM.

```
PDF 8 pages + --pages-per-copy 2 → 4 copies de 2 pages
```

**Usage:**
- PDF multi-élèves avec structure fixe (chaque copie fait N pages)
- Contrôle précis du découpage

### SANS --pages-per-copy (PDF entier)

Le PDF est envoyé entier au LLM qui détecte automatiquement les copies.

**Usage:**
- PDF pré-découpé (1 fichier = 1 copie élève)
- Documents avec structure variable
- Laisser le LLM analyser la structure

---

## Exemples

```bash
# ═══════════════════════════════════════════════════════════════
# Avec découpage (--pages-per-copy)
# ═══════════════════════════════════════════════════════════════

# DUAL LLM + INDIVIDUAL + découpage 2 pages/copy
python -m src.main correct dual individual copies.pdf --pages-per-copy 2 --auto-confirm

# SINGLE LLM + BATCH + découpage 2 pages/copy (1 seul appel API!)
python -m src.main correct single batch copies.pdf --pages-per-copy 2 --auto-confirm

# ═══════════════════════════════════════════════════════════════
# Sans découpage (PDF envoyé entier)
# ═══════════════════════════════════════════════════════════════

# PDF pré-découpé (1 fichier = 1 élève)
python -m src.main correct dual batch eleve_dupont.pdf --auto-confirm

# Laisser le LLM analyser la structure
python -m src.main correct single individual copies.pdf --auto-confirm

# ═══════════════════════════════════════════════════════════════
# Autres options
# ═══════════════════════════════════════════════════════════════

# Avec 2ème lecture (auto-vérification)
python -m src.main correct single individual copies.pdf --second-reading --auto-confirm

# Parallélisme agressif (mode individual uniquement)
python -m src.main correct dual individual copies.pdf --parallel 10 --auto-confirm

# Mode HYBRID (dual uniquement)
python -m src.main correct dual hybrid copies.pdf --pages-per-copy 2 --auto-confirm
```

---

## Parallélisme (--parallel)

En mode INDIVIDUAL, les copies peuvent être traitées en parallèle:

```
AVANT (--parallel 1, séquentiel):
Copie 1 ────────►
                  Copie 2 ────────►
                                  Copie 3 ────────►
Temps: 3 × temps_par_copie

APRÈS (--parallel 6, défaut):
Copie 1 ────────►
Copie 2 ────────►
Copie 3 ────────►
... (max 6 simultanés)
Temps: ~temps_par_copie
```

**Note**: Le parallélisme est limité par les rate limits des APIs (Gemini: ~15 RPM).

---

## Structure de l'audit

L'audit contient les résultats finaux et le cheminement complet de chaque décision.

### Mode DUAL LLM

```json
{
  "results": {
    "Q1": { "grade": 1.0, "max_points": 1.0, "reading": "fiole jaugée", "feedback": "Correct." },
    "Q2": { "grade": 1.0, "max_points": 1.0, "reading": "balance", "feedback": "Correct." }
  },
  "options": {
    "mode": "dual_llm",
    "providers": ["gemini-2.5-flash", "gemini-3-flash-preview"],
    "second_reading": false
  },
  "llm_comparison": {
    "Q1": {
      "llm1": { "grade": 0, "reading": "bécher", "reasoning": "..." },
      "llm2": { "grade": 1.0, "reading": "fiole jaugée", "reasoning": "..." },
      "verification": {
        "method": "isolated",
        "evolution": { "initial": [0, 1.0], "after_isolated": [1.0, 1.0] }
      },
      "final": { "grade": 1.0, "method": "consensus" }
    }
  }
}
```

### Mode SINGLE LLM

```json
{
  "results": {
    "Q1": { "grade": 1.0, "max_points": 1.0, "reading": "ballon", "feedback": "Correct." }
  },
  "options": {
    "mode": "single_llm",
    "provider": "gemini-2.5-flash",
    "second_reading": true
  },
  "llm_comparison": {
    "second_reading": {
      "changes": {
        "Q3": {
          "grade_changed": true,
          "grade_before": 1.5,
          "grade_after": 2.0,
          "reasoning": "Relisant la réponse..."
        }
      }
    }
  }
}
```

---

## Configuration

### Variables d'environnement (.env)

```bash
# Clés API (au moins une requise)
AI_CORRECTION_GEMINI_API_KEY=your_key
AI_CORRECTION_OPENAI_API_KEY=your_key

# Mode comparaison (défaut: true avec les deux clés)
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o
```

---

## Structure des données

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

## Architecture du code

```
src/
├── ai/                    # Providers LLM
│   ├── __init__.py             # Exports avec lazy imports
│   ├── base_provider.py        # Classe abstraite + error handling
│   ├── gemini_provider.py      # Provider Google Gemini
│   ├── openai_provider.py      # Provider OpenAI
│   ├── comparison_provider.py  # DUAL LLM avec vérification unifiée
│   ├── single_pass_grader.py   # Notation toutes questions en 1 appel
│   ├── provider_factory.py     # Factory pour providers
│   └── disagreement_analyzer.py  # Détection désaccords + position swaps + reading anchors
├── core/                  # Modèles et orchestration
│   ├── __init__.py             # Exports publics
│   ├── models.py               # Dataclasses (GradingSession, GradedCopy...)
│   ├── exceptions.py           # Hiérarchie d'exceptions custom
│   ├── session.py              # Orchestrateur de session
│   ├── workflow.py             # Workflow de correction
│   └── workflow_state.py       # État immutable du workflow
├── config/                # Configuration
│   ├── __init__.py             # Exports publics
│   ├── settings.py             # Paramètres (env vars)
│   ├── constants.py            # Constantes centralisées
│   ├── prompts.py              # Templates de prompts
│   └── logging_config.py       # Configuration logging
├── utils/                 # Utilitaires
│   ├── __init__.py             # Exports publics
│   ├── sorting.py              # Tri naturel (Q1, Q2, Q10)
│   ├── confidence.py           # Calcul de confiance
│   ├── rate_limiter.py         # Rate limiting + Circuit breaker
│   ├── retry.py                # Retry avec backoff exponentiel
│   └── type_guards.py          # Validation runtime des types
├── grading/               # Moteur de notation
│   ├── grader.py               # IntelligentGrader
│   ├── feedback.py             # Génération de feedback
│   └── uncertainty.py          # Calcul d'incertitude
├── vision/                # Lecture PDF
│   ├── __init__.py
│   └── pdf_reader.py           # PDFReader avec context manager
├── storage/               # Stockage JSON
│   ├── __init__.py
│   └── file_store.py           # SessionStore avec file locking
├── export/                # Export résultats
│   ├── __init__.py
│   ├── pdf_annotator.py        # Annotation PDF
│   └── analytics.py            # Rapports et statistiques
├── analysis/              # Analyse cross-copies
│   ├── __init__.py
│   ├── clustering.py           # Clustering embeddings
│   └── cross_copy.py           # Détection similarités
├── calibration/           # Calibration et cohérence
│   ├── __init__.py
│   ├── retroactive.py          # Application décisions rétroactives
│   └── consistency.py          # Détection incohérences
├── interaction/           # Interface utilisateur
│   ├── __init__.py
│   ├── cli.py                  # CLI interactive
│   └── live_progress.py        # Dashboard temps réel
├── api/                   # API REST (optionnel)
│   ├── __init__.py
│   └── app.py                  # FastAPI application
└── main.py                # Point d'entrée CLI
```

---

## Infrastructure et Robustesse

### Gestion des Erreurs

```python
from core.exceptions import (
    AICorrectionError,      # Exception de base
    ProviderError,          # Erreur de provider
    APIConnectionError,     # Erreur de connexion
    APITimeoutError,        # Timeout
    ParsingError,           # Erreur de parsing
)
```

### Rate Limiting et Circuit Breaker

```python
from utils import (
    RateLimiter,            # Token bucket rate limiter
    CircuitBreaker,         # Protection contre APIs défaillantes
    get_gemini_rate_limiter,
    get_openai_circuit_breaker,
)
```

### Retry avec Backoff

```python
from utils import retry_with_backoff, API_RETRY_CONFIG

@retry_with_backoff(max_attempts=3, base_delay=1.0)
async def call_api():
    ...
```

### Logging Centralisé

```python
from config import setup_logging, get_logger

setup_logging(level="INFO", log_file="app.log")
logger = get_logger(__name__)
```

### Validation des Types

```python
from utils import is_grading_result, ensure_dict, ensure_float

if is_grading_result(response):
    grade = ensure_float(response.get('grade'))
```

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

## Améliorations Récentes

### Analyse des Désaccords

**Logique de Flag Simplifiée**

Le système détecte les désaccords entre les deux LLMs pour décider si une vérification est nécessaire:

```
┌─────────────────────────────────────────────────────────────────┐
│  DÉTECTION DES DÉSACCORDS                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Barème différent (max_points LLM1 ≠ max_points LLM2)        │
│     → FLAG: SCALE_DIFFERENCE                                    │
│                                                                  │
│  2. Un LLM trouve la réponse, l'autre non                       │
│     → FLAG: NOT_FOUND_CONFLICT                                  │
│                                                                  │
│  3. Notes différentes (différence >= 10% du barème)             │
│     → FLAG: READING_DIFFERENCE ou GRADE_DIFFERENCE              │
│                                                                  │
│  Lecture considérée "similaire" si:                              │
│  - SequenceMatcher ratio >= 0.8, OU                             │
│  - Une lecture contient l'autre (partial)                       │
│                                                                  │
│  Si notes identiques ET lecture similaire → PAS de flag         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Pourquoi ces règles?**
- Si les notes diffèrent → les LLMs ont des jugements différents → vérification nécessaire
- Si les notes sont identiques mais les lectures très différentes → les LLMs ont convergé, pas de flag
- "partial" capture les cas où un LLM inclut plus de contexte (ex: "V = 100 mL m = VxCm" vs "m = VxCm")

**Exemples concrets:**

| Lecture LLM1 | Lecture LLM2 | Ratio | Partial | Notes | Flag? |
|--------------|--------------|-------|---------|-------|-------|
| "fiole jaugée" | "bécher" | 0.0 | Non | Différentes | OUI |
| "m = VxCm" | "V = 100 mL m = VxCm" | 0.59 | Oui | Identiques | NON |
| "m = 40g. L¹ x 10L" | "m = 40g.L⁻¹ x 10L" | 0.96 | Non | Différentes | OUI (grade) |

### Détection Flip-Flop

Quand les LLMs échangent leurs positions après vérification (LLM1 était plus haut, devient plus bas), le système le signale. Seuil: 10% du barème.

### Prompts de Vérification

- Lecture indépendante de la copie (ne pas copier l'autre correcteur)
- Identifier d'abord la bonne réponse sur l'image, puis comparer
- Considérer les deux lectures pour auto-raisonnement
- Feedback sobre et professionnel

### Infrastructure

**Dashboard Temps Réel**
- Visualisation de toutes les copies pendant le traitement parallèle
- Statut par copie: en attente / en cours / terminé / erreur

**Protection Anti-Hallucination**
- Reading anchors: figement des lectures en cas d'accord initial
- Détection flip-flop: signalement des échanges de position
- Prompts anti-suggestion: règles explicites

**Sécurité**
- Protection contre path traversal
- Limite de taille des fichiers (50 MB)
- Sanitisation des clés API dans les logs

**Performance**
- Cache LRU pour conversion base64
- Embeddings batchés
- I/O non-bloquant avec `asyncio.to_thread`

**Robustesse**
- Circuit breaker pour APIs défaillantes
- Rate limiting (token bucket)
- Retry avec backoff exponentiel
- File locking atomique

**Architecture**
- Hiérarchie d'exceptions custom
- État de workflow immutable (`CorrectionState`)
- Module exports centralisés
- Type guards pour validation runtime

---

## Licence

MIT License - voir [LICENSE](LICENSE)
