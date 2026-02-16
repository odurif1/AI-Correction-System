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

## Philosophie d'architecture

### Ce que fait le LLM

Le LLM **lit et note en même temps**. C'est naturel pour lui: il regarde l'image, comprend la réponse, et l'évalue en un seul appel.

### Ce que fait le programme

Le programme **orchestre la confrontation** entre deux LLM:

```
┌─────────────────────────────────────────────────────────────┐
│  LLM1: lit + note                                           │
│  LLM2: lit + note                                           │
│                     ↓                                       │
│  Le programme détecte les désaccords et les résout          │
└─────────────────────────────────────────────────────────────┘
```

### Pourquoi ne pas séparer lecture et notation?

Séparer ces étapes serait:
- **Artificiel**: Ce n'est pas ainsi que fonctionne un LLM
- **Coûteux**: Double les appels API
- **Pas plus fiable**: Un LLM séparé pour la lecture ferait les mêmes erreurs

### Ce qu'apporte le programme

| Fonction | Description |
|----------|-------------|
| **Confrontation** | Deux LLM notent indépendamment |
| **Détection** | Identifier les désaccords (lecture OU note) |
| **Re-vérification** | Si lectures différentes: relire + réévaluer ensemble |
| **Cross-verification** | Si notes différentes: confronter les raisonnements |
| **Ultimatum** | Dernière chance d'accord avant intervention utilisateur |
| **Audit** | Tracer chaque décision pour diagnostic |

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
┌─────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE COMPLÈTE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PDF → Extraction pages → Détection nom élève                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    POUR CHAQUE QUESTION                            │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  PHASE 1: NOTATION INITIALE (parallèle)                            │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                │ │
│  │  │ LLM1:               │    │ LLM2:               │                │ │
│  │  │ - Lit la réponse    │    │ - Lit la réponse    │                │ │
│  │  │ - Note + reasoning  │    │ - Note + reasoning  │                │ │
│  │  │ - student_answer_read   │    student_answer_read │                │ │
│  │  └─────────────────────┘    └─────────────────────┘                │ │
│  │            │                          │                             │ │
│  │            └──────────┬───────────────┘                             │ │
│  │                       ▼                                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐│ │
│  │  │ ANALYSE: lecture1 vs lecture2 identiques?                       ││ │
│  │  │   → identique / accent / partielle / substantielle              ││ │
│  │  └─────────────────────────────────────────────────────────────────┘│ │
│  │                       │                                              │ │
│  │          ┌────────────┴────────────┐                                │ │
│  │          ▼                         ▼                                │ │
│  │   [Lectures OK]            [Lectures DIFFÉRENTES]                  │ │
│  │          │                         │                                │ │
│  │          │                         ▼                                │ │
│  │          │              ┌──────────────────────────────────────┐   │ │
│  │          │              │ PHASE 1.5: RE-VÉRIFICATION LECTURE   │   │ │
│  │          │              │ avec réévaluation de la note         │   │ │
│  │          │              │                                      │   │ │
│  │          │              │ LLM1 voit lecture LLM2 → ajuste?     │   │ │
│  │          │              │ LLM2 voit lecture LLM1 → ajuste?     │   │ │
│  │          │              └──────────────────────────────────────┘   │ │
│  │          │                         │                                │ │
│  │          └────────────┬────────────┘                                │ │
│  │                       ▼                                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐│ │
│  │  │ COMPARAISON: grade1 == grade2 ?                                 ││ │
│  │  └─────────────────────────────────────────────────────────────────┘│ │
│  │                       │                                              │ │
│  │          ┌────────────┴────────────┐                                │ │
│  │          ▼                         ▼                                │ │
│  │   [ACCORD] ✓              [DÉSACCORD] ⚠                            │ │
│  │   Note finale            │                                          │ │
│  │   = grade1               ▼                                          │ │
│  │              ┌──────────────────────────────────────────────┐      │ │
│  │              │ PHASE 2: VÉRIFICATION CROISÉE               │      │ │
│  │              │                                              │      │ │
│  │              │ Chaque LLM voit le reasoning de l'autre     │      │ │
│  │              │ "Un autre correcteur a noté X parce que..." │      │ │
│  │              │ → Réexamen indépendant                      │      │ │
│  │              └──────────────────────────────────────────────┘      │ │
│  │                       │                                          │ │
│  │          ┌────────────┴────────────┐                            │ │
│  │          ▼                         ▼                            │ │
│  │   [Accord après] ✓        [Toujours désaccord]                 │ │
│  │                          │                                      │ │
│  │                          ▼                                      │ │
│  │              ┌──────────────────────────────────────────────┐   │ │
│  │              │ PHASE 3: ULTIMATUM                          │   │ │
│  │              │                                              │   │ │
│  │              │ "Désaccord persistant - décision finale"    │   │ │
│  │              │ → Évolution des notes affichée              │   │ │
│  │              │ → Avertissement si LLM a changé             │   │ │
│  │              └──────────────────────────────────────────────┘   │ │
│  │                       │                                          │ │
│  │          ┌────────────┴────────────┐                            │ │
│  │          ▼                         ▼                            │ │
│  │   [Accord final] ✓        [Désaccord persistant]              │ │
│  │                          │                                      │ │
│  │                          ▼                                      │ │
│  │              ┌──────────────────────────────────────────────┐   │ │
│  │              │ INTERVENTION UTILISATEUR                    │   │ │
│  │              │ ou moyenne automatique                       │   │ │
│  │              └──────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  → Génération appréciation → Export (JSON, CSV, PDF annoté)             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Structure de l'audit

Chaque question dispose d'un audit complet permettant de retracer toutes les étapes:

```json
{
  "Q1": {
    "initial": {
      "llm1": {
        "provider": "gemini-2.5-flash",
        "grade": 1.0,
        "confidence": 0.9,
        "internal_reasoning": "...",
        "student_answer_read": "m = V x Cm"
      },
      "llm2": { ... },
      "difference": 0.0
    },

    "reading_analysis": {
      "llm1_read": "m = V x Cm",
      "llm2_read": "m = V X Cm",
      "identical": false,
      "difference_type": "accent"
    },

    "reading_reverification": {
      "llm1": {
        "initial_reading": "Mm. = V X Cm",
        "final_reading": "m = V X Cm",
        "reading_changed": true,
        "initial_grade": 1.0,
        "final_grade": 2.0,
        "grade_changed": true,
        "justification": "...",
        "prompt_sent": "...",
        "raw_response": "..."
      },
      "llm2": { ... }
    },

    "after_cross_verification": {
      "llm1": {
        "grade": 2.0,
        "prompt_sent": "─── CONTESTATION ───\n..."
      },
      "llm2": { ... }
    },

    "after_ultimatum": { ... },

    "decision_path": {
      "initial_agreement": false,
      "reading_reverification_triggered": true,
      "verification_triggered": true,
      "ultimatum_triggered": false,
      "final_method": "verification_consensus"
    },

    "final": {
      "grade": 2.0,
      "agreement": true,
      "method": "verification_consensus"
    }
  }
}
```

### Champs clés de l'audit

| Champ | Description |
|-------|-------------|
| `initial` | Résultats du premier passage (lecture + notation) |
| `reading_analysis` | Comparaison des lectures des deux LLM |
| `reading_reverification` | Re-vérification avec réévaluation (si lectures différentes) |
| `after_cross_verification` | Résultats après confrontation des raisonnements |
| `after_ultimatum` | Résultats après l'ultimatum (si désaccord persiste) |
| `decision_path` | Chemin de décision emprunté |
| `final` | Résultat final (note, accord, méthode) |
| `timing` | Durée de chaque phase en ms |

---

## Fonctionnalités clés

### Double LLM avec confrontation
- Deux IA notent indépendamment chaque réponse
- En cas de désaccord, elles doivent se justifier face à l'autre
- **Ultimatum**: phase finale avec évolution des notes et avertissements

### Re-vérification de lecture avec réévaluation
- Si les lectures diffèrent substantiellement → re-vérification automatique
- Chaque LLM voit la lecture de l'autre et peut **ajuster sa note**
- Résout le problème: "bonne lecture mais mauvaise note"

### Consensus de lecture
- Les IA décrivent ce qu'elles voient **avant** de noter
- Détecte les erreurs d'interprétation (ex: erlenmeyer vs fiole jaugée)
- Désactivable avec `--skip-reading` pour gagner du temps

### Feedback professionnel
- Ton sobre, pas de "bravo" ou "continue comme ça"
- Adapté à la difficulté (question facile = retour minimal)
- Max 25 mots

### Audit complet et traçable
- Chaque décision est tracée séquentiellement
- **Prompts exacts** envoyés aux IA conservés
- **Réponses brutes** des LLM conservées
- Évolution de la confiance documentée
- Timing de chaque phase

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
