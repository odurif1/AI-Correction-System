# AI Correction System

**Correction automatique de copies utilisant deux IA en parallèle pour garantir fiabilité et équité.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Pourquoi ce système ?

| Problème | Solution |
|----------|----------|
| Une IA peut se tromper | **Deux IA notent en parallèle** et se confrontent |
| Erreurs de lecture | **Lecture isolée + échange** pour éviter l'ancrage |
| Manque de traçabilité | **Audit complet** : chaque décision est documentée |
| Feedback trop "gentil" | **Retours professionnels** : sobres, adaptés à la difficulté |

---

## Architecture à 3 phases

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SINGLE-PASS                                            │
│  Les 2 LLM notent toutes les questions en parallèle              │
│                              │                                   │
│                              ▼                                   │
│  ANALYSE: Détecter les désaccords                               │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [Questions OK]                  [Questions flaggées]         │
│     Finaliser directement                                 │
│                                                                      │
│  PHASE 2a: VÉRIFICATION ISOLÉE (re-lecture fraîche)          │
│  Chaque LLM relit indépendamment, sans voir l'autre             │
│                              │                                     │
│                              ▼                                     │
│  PHASE 2b: ÉCHANGE (si désaccord persiste)                      │
│  Les LLM voient les lectures de l'autre et peuvent ajuster      │
│                              │                                     │
│                              ▼                                     │
│  PHASE 3: ULTIMATUM (si toujours désaccord)                     │
│  Décision finale avec évolution des notes                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Pourquoi cette architecture?

| Problème | Solution |
|----------|----------|
| Ancrage sur la première lecture | Phase 2a: re-lecture isolée sans influence |
| Erreur OCR propagée | Phase 2b: échange des lectures pour correction |
| Désaccord persistant | Phase 3: ultimatum avec historique |

---

## Démarrage rapide

```bash
# 1. Installer
pip install -r requirements.txt

# 2. Configurer les clés API
cp .env.example .env
# Éditer .env avec vos clés Gemini et/ou OpenAI

# 3. Lancer une correction
python -m src.main correct copies/*.pdf --auto
```

---

## Options CLI

| Option | Description |
|--------|-------------|
| `--auto` | Mode automatique (pas d'interaction) |
| `--single` | Un seul LLM (plus rapide, moins coûteux) |
| `--skip-reading` | Ignorer le consensus de lecture |
| `--scale Q1=5,Q2=3` | Définir le barème |
| `--annotate` | Générer les PDFs annotés |
| `--export json,csv` | Formats d'export |

---

## Structure de l'audit

```json
{
  "Q2": {
    "method": "exchange_consensus",
    "agreement": true,
    "single_pass": {
      "llm1": {"grade": 0.0, "reading": "Bicole Jaugée"},
      "llm2": {"grade": 1.0, "reading": "Erle.meijer"}
    },
    "after_isolated_verification": {
      "llm1": {"grade": 0.0, "reading": "fiole jaugée"},
      "llm2": {"grade": 1.0, "reading": "Fiole Jaugée"}
    },
    "after_exchange": {
      "llm1": {"grade": 1.0, "reading": "Fiole Jaugée"},
      "llm2": {"grade": 1.0, "reading": "Fiole Jaugée"}
    },
    "final": {"grade": 1.0, "agreement": true}
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
│   ├── comparison_provider.py  # Double LLM avec 3 phases
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
