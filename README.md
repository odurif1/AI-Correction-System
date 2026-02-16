# AI Correction System

**Correction automatique de copies utilisant deux IA en parallèle pour garantir fiabilité et équité.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Pourquoi ce système ?

| Problème | Solution |
|----------|----------|
| Une IA peut se tromper | **Deux IA notent en parallèle** et se confrontent |
| Les IA peuvent "inventer" | **Consensus de lecture** : les IA décrivent ce qu'elles voient avant de noter |
| Manque de traçabilité | **Audit complet** : chaque décision est documentée |
| Feedback trop "gentil" | **Retours professionnels** : sobres, adaptés à la difficulté |

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

## Exemple de sortie

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Session: sess_20240115_143052
🤖 Modèles: gemini-2.5-flash + gpt-4o
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ Copie 1/3 ━━━ Martin Jean ━━━
  Q1 (1/6) ▪ gemini: 1.0 ┃ gpt-4o: 1.0
  ✓ Q1: 1.0/1.0                          ← Accord immédiat (vert)
  Q2 (2/6) ▪ gemini: 0.5 ┃ gpt-4o: 1.0
  ✓ Q2: 0.75/2.0                         ← Accord après vérification (jaune)
  Q3 (3/6) ▪ gemini: 2.0 ┃ gpt-4o: 0.0
  ⚠ Q3: 1.0/3.0                          ← Moyenne (rouge)

  Total: 2.75/6.0 (46%) conf: 85%

━━━ Résumé ━━━
  Copie 1: Martin Jean     2.75/6.0  (46%)
  Copie 2: Dupont Marie    4.50/6.0  (75%)
  Copie 3: Bernard Luc     3.00/6.0  (50%)

📊 Token Usage:
  Total: 45,230 tokens
  gemini-2.5-flash: 23,500 tokens (15 calls)
  gpt-4o: 21,730 tokens (15 calls)
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

## Workflow de correction

```
┌─────────────────────────────────────────────────────────────┐
│  PDF → Extraction pages → Détection nom                     │
│                                                              │
│  Pour chaque question:                                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Phase 1: LECTURE (par défaut)                           ││
│  │   LLM1 décrit → LLM2 décrit → Validation si désaccord   ││
│  ├─────────────────────────────────────────────────────────┤│
│  │ Phase 2: NOTATION                                        ││
│  │   LLM1 note ║ LLM2 note (parallèle)                      ││
│  ├─────────────────────────────────────────────────────────┤│
│  │ Si désaccord:                                            ││
│  │   → Vérification croisée (chaque LLM voit l'autre)       ││
│  │   → Ultimatum si fausse convergence                      ││
│  │   → Demande utilisateur si persistant                    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  → Génération appréciation → Export                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Fonctionnalités clés

### Double LLM avec confrontation
- Deux IA notent indépendamment chaque réponse
- En cas de désaccord, elles doivent se justifier face à l'autre
- Détection des "fausses convergences" (prétendent être d'accord mais notes différentes)

### Consensus de lecture
- Les IA décrivent ce qu'elles voient **avant** de noter
- Détecte les erreurs d'interprétation (ex: erlenmeyer vs fiole jaugée)
- Désactivable avec `--skip-reading` pour gagner du temps

### Feedback professionnel
- Ton sobre, pas de "bravo" ou "continue comme ça"
- Adapté à la difficulté (question facile = retour minimal)
- Max 25 mots

### Audit complet
- Chaque décision est tracée
- Prompts exacts envoyés aux IA conservés
- Évolution de la confiance documentée

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
    ├── session.json           # État de la session
    ├── policy.json            # Barème
    ├── copies/
    │   └── {n}/
    │       ├── original.pdf   # PDF original
    │       ├── annotation.json # Notes, feedbacks (léger)
    │       └── audit.json     # Tout: échanges LLM (complet)
    ├── annotated/             # PDFs annotés (export)
    └── reports/               # CSV, JSON (export)
```

---

## Architecture

```
src/
├── ai/                    # Providers LLM
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   └── comparison_provider.py  # Double LLM
├── core/                  # Modèles et orchestration
├── grading/               # Moteur de notation
├── vision/                # Lecture PDF
├── storage/               # Stockage JSON
└── main.py                # CLI
```

---

## Développement

```bash
# Tests
pytest tests/

# Formatage
black src/ && isort src/
```

---

## Licence

MIT License - voir [LICENSE](LICENSE)
