# AI Correction System

**Correction automatique de copies utilisant deux IA en parallèle pour garantir fiabilité et équité.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Concepts clés

Le système combine deux dimensions indépendantes:

### 1. Nombre de LLM: SINGLE vs DUAL

| Mode | Option | Description |
|------|--------|-------------|
| **SINGLE** | `--single` | Un seul LLM corrige chaque copie |
| **DUAL** | *(défaut)* | Deux LLM notent en parallèle, avec phases de vérification si désaccord |

### 2. Mode de lecture: ENSEMBLE vs INDIVIDUAL

| Mode | Option | Description |
|------|--------|-------------|
| **ENSEMBLE** | *(défaut)* | Un PDF contient tous les élèves, l'IA détecte les copies |
| **INDIVIDUAL** | `--pages-per-student N` | PDF pré-découpé, N pages par élève |

### Matrice des combinaisons

```
                    ┌─────────────────────────────────────────────────────┐
                    │              MODE DE LECTURE                         │
                    │                                                      │
                    │     ENSEMBLE              INDIVIDUAL                  │
                    │     (détection IA)        (--pages-per-student N)     │
┌───────────────────┼─────────────────────────────────────────────────────┤
│         SINGLE    │  1 LLM détecte            1 LLM par copie           │
│  NOMBRE   (--single)│  et corrige              (découpage fixe)         │
│  DE LLM           │  toutes les copies                                  │
├───────────────────┼─────────────────────────────────────────────────────┤
│         DUAL      │  2 LLM détectent          2 LLM par copie           │
│         (défaut)  │  et corrigent             (découpage fixe)          │
│                   │  ensemble                 + vérification croisée    │
└───────────────────┴─────────────────────────────────────────────────────┘
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

## Mode INDIVIDUAL (--pages-per-student N)

Le système découpe le PDF en chunks de N pages, chaque chunk étant une copie d'élève.

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE INDIVIDUAL                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Le PDF est découpé en chunks de N pages                     │
│  2. Chaque chunk est analysé comme une copie indépendante       │
│  3. Analyse faite directement pendant la phase de correction    │
│                                                                  │
│  PDF de 12 pages + --pages-per-student 2 → 6 copies de 2 pages  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Avantages:**
- Focus LLM concentré sur une seule copie
- Aucune contamination entre copies
- Pas d'appel IA supplémentaire pour l'analyse préalable
- Parallélisable (--parallel)

**Inconvénients:**
- N appels pour N copies (coût plus élevé)
- Nécessite de connaître le nombre de pages par élève

---

## Mode ENSEMBLE (défaut)

L'IA analyse le PDF complet et détecte automatiquement les élèves.

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE ENSEMBLE                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. L'IA scanne tout le PDF                                      │
│  2. Elle identifie les zones de chaque élève                    │
│  3. Elle extrait et corrige chaque copie                        │
│                                                                  │
│  PDF de 12 pages → IA détecte 6 élèves → 6 copies               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Avantages:**
- Un seul appel pour détecter tous les élèves
- Pas besoin de connaître le nombre de pages par élève

**Inconvénients:**
- Dépend de la qualité de détection de l'IA
- Risque de confusion si copies mal délimitées

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
python -m src.main correct copies/*.pdf --pages-per-student 2 --auto
```

---

## Options CLI

### Options principales

| Option | Description |
|--------|-------------|
| `--pages-per-student N` | Active mode INDIVIDUAL: N pages par élève |
| `--single` | Mode SINGLE LLM (un seul LLM) |
| `--auto` | Mode automatique (pas d'interaction) |
| `--second-reading` | 2ème lecture (2 passes en Single, intégrée en Dual) |

### Options avancées

| Option | Défaut | Description |
|--------|--------|-------------|
| `--parallel N` | 6 | Nombre de copies traitées en parallèle (mode INDIVIDUAL) |
| `--skip-reading` | - | Ignorer le consensus de lecture |
| `--scale Q1=5,Q2=3` | - | Définir le barème manuellement |
| `--output DIR` | outputs | Répertoire de sortie |
| `--subject TEXT` | - | Matière/domaine pour la correction |
| `--annotate` | - | Générer les PDFs annotés |
| `--export json,csv` | json,csv | Formats d'export |

---

## Exemples

```bash
# ═══════════════════════════════════════════════════════════════
# Mode INDIVIDUAL (recommandé pour copies pré-découpées)
# ═══════════════════════════════════════════════════════════════

# DUAL LLM + INDIVIDUAL (défaut recommandé)
python -m src.main correct copies.pdf --pages-per-student 2 --auto

# SINGLE LLM + INDIVIDUAL (plus rapide)
python -m src.main correct copies.pdf --pages-per-student 2 --single --auto

# SINGLE LLM + INDIVIDUAL + 2ème lecture (auto-vérification)
python -m src.main correct copies.pdf --pages-per-student 2 --single --second-reading --auto

# DUAL LLM + INDIVIDUAL + parallélisme agressif
python -m src.main correct copies.pdf --pages-per-student 2 --parallel 10 --auto

# ═══════════════════════════════════════════════════════════════
# Mode ENSEMBLE (PDF contenant tous les élèves)
# ═══════════════════════════════════════════════════════════════

# DUAL LLM + ENSEMBLE (IA détecte les élèves)
python -m src.main correct classe_complete.pdf --auto

# SINGLE LLM + ENSEMBLE
python -m src.main correct classe_complete.pdf --single --auto
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
│   ├── base_provider.py        # Classe abstraite
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   ├── comparison_provider.py  # DUAL LLM avec 3 phases
│   ├── single_pass_grader.py   # Notation toutes questions en 1 appel
│   └── disagreement_analyzer.py
├── core/                  # Modèles et orchestration
│   ├── models.py               # Dataclasses
│   └── session.py              # Orchestrateur
├── grading/               # Moteur de notation
├── vision/                # Lecture PDF
├── storage/               # Stockage JSON
└── main.py                # CLI
```

---

## Développement

```bash
# Tests unitaires
pytest tests/

# Formatage
black src/ && isort src/
```

---

## Licence

MIT License - voir [LICENSE](LICENSE)
