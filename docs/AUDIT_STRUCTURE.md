# Structure de l'Audit - AI Correction System

> Voir aussi: [README principal](../README.md) | [Architecture Dual LLM](dual_llm_architecture.md)

Ce document décrit la structure complète de l'audit généré pour chaque question corrigée.

## Vue d'ensemble

L'audit est stocké dans `data/{session_id}/session.json` et contient toutes les informations sur le processus de correction, y compris les échanges avec les LLMs.

L'audit permet de:
- **Tracer** chaque décision de notation
- **Comprendre** les désaccords entre LLMs
- **Déboguer** les erreurs de lecture
- **Auditer** le processus complet

## Structure hiérarchique

```
session.json
├── graded_copies[]              # Liste des copies corrigées
│   ├── copy_id                  # Identifiant de la copie
│   ├── student_name             # Nom de l'élève
│   ├── total_score              # Score total
│   ├── max_score                # Score maximum
│   ├── grades                   # Notes par question
│   │   └── {Q1: {grade, feedback, reading}, Q2: {...}}
│   └── llm_comparison           # Audit détaillé du Dual LLM
│       ├── options              # Configuration utilisée
│       ├── student_detection    # Détection du nom
│       └── questions            # Détail par question
│
└── policy                       # Barème et critères
```

### Structure détaillée d'une question

```
questions/
└── {Q1, Q2, ...}/
    ├── max_points               # Barème de la question
    │
    ├── LLM1: {model}            # Résultat LLM1
    │   ├── grade                # Note attribuée
    │   ├── reading              # Lecture de la réponse
    │   ├── reasoning            # Justification
    │   └── feedback             # Feedback pour l'élève
    │
    ├── LLM2: {model}            # Résultat LLM2
    │   └── ...
    │
    ├── final                    # Résultat final
    │   ├── grade                # Note finale
    │   ├── method               # Méthode de résolution
    │   └── agreement            # Accord atteint?
    │
    ├── verification             # Phase de vérification (si désaccord)
    │   ├── llm1_new_grade       # Nouvelle note LLM1
    │   ├── llm2_new_grade       # Nouvelle note LLM2
    │   └── method               # Méthode utilisée
    │
    ├── ultimatum                # Phase ultimatum (si persiste)
    │   ├── llm1_final_grade     # Note finale LLM1
    │   ├── llm2_final_grade     # Note finale LLM2
    │   ├── llm1_decision        # maintained/changed
    │   ├── llm2_decision        # maintained/changed
    │   └── method               # Méthode utilisée
    │
    └── max_points_disagreement  # Désaccord sur le barème (si applicable)
        ├── llm1_max_points      # Barème LLM1
        ├── llm2_max_points      # Barème LLM2
        └── resolved_max_points  # Barème résolu
```

---

## Exemple complet

```json
{
  "graded_copies": [{
    "copy_id": "abc-123",
    "student_name": "Jean Dupont",
    "total_score": 5.5,
    "max_score": 8.0,
    "grades": {
      "Q1": {"grade": 1.0, "max_points": 1.0, "feedback": "Correct.", "reading": "fiole jaugée"},
      "Q2": {"grade": 1.0, "max_points": 1.0, "feedback": "Correct.", "reading": "balance"},
      "Q3": {"grade": 1.5, "max_points": 2.0, "feedback": "Partiellement correct.", "reading": "m = C × V"},
      "Q4": {"grade": 0.0, "max_points": 1.0, "feedback": "Incorrect.", "reading": "bécher"},
      "Q5": {"grade": 1.0, "max_points": 2.0, "feedback": "Calcul incomplet.", "reading": "..."},
      "Q6": {"grade": 1.0, "max_points": 1.0, "feedback": "Correct.", "reading": "..."}
    },
    "llm_comparison": {
      "options": {
        "mode": "batch",
        "providers": ["LLM1: gemini-2.5-flash", "LLM2: gpt-4o"],
        "total_copies": 2
      },
      "student_detection": {
        "copy_index": 1,
        "student_name": "Jean Dupont",
        "llm1_student_name": "Jean Dupont",
        "llm2_student_name": "J. Dupont",
        "name_disagreement": null
      },
      "questions": {
        "Q1": {
          "max_points": 1.0,
          "LLM1: gemini-2.5-flash": {
            "grade": 1.0,
            "reading": "fiole jaugée",
            "reasoning": "L'élève a correctement identifié...",
            "feedback": "Réponse correcte."
          },
          "LLM2: gpt-4o": {
            "grade": 1.0,
            "reading": "fiole jaugée",
            "reasoning": "Identification correcte...",
            "feedback": "Correct."
          },
          "final": {
            "grade": 1.0,
            "method": "consensus",
            "agreement": true
          }
        },
        "Q3": {
          "max_points": 2.0,
          "LLM1: gemini-2.5-flash": {
            "grade": 2.0,
            "reading": "m = C × V = 40 × 0.1 = 4g",
            "reasoning": "Calcul complet et correct",
            "feedback": "Excellent travail."
          },
          "LLM2: gpt-4o": {
            "grade": 1.0,
            "reading": "m = C × V",
            "reasoning": "Formule correcte mais pas de calcul numérique",
            "feedback": "Il manque l'application numérique."
          },
          "final": {
            "grade": 1.5,
            "method": "average",
            "agreement": false
          },
          "verification": {
            "final_grade": 1.5,
            "llm1_new_grade": 2.0,
            "llm2_new_grade": 1.0,
            "llm1_reasoning": "Je maintiens ma note...",
            "llm2_reasoning": "Je maintiens ma note...",
            "method": "verification_average"
          },
          "ultimatum": {
            "final_grade": 1.5,
            "llm1_final_grade": 2.0,
            "llm2_final_grade": 1.0,
            "llm1_decision": "maintained",
            "llm2_decision": "maintained",
            "method": "ultimatum_average"
          }
        }
      }
    }
  }]
}
```

---

## Méthodes de résolution

| Méthode | Description |
|---------|-------------|
| `consensus` | Accord initial entre les deux LLMs |
| `verification_consensus` | Accord après cross-verification |
| `ultimatum_consensus` | Accord après ultimatum |
| `average` | Moyenne des notes (désaccord persistant) |
| `user_choice` | Choix utilisateur (mode interactif) |

---

## Détection des problèmes

### Désaccord sur les noms d'élèves

```json
"student_detection": {
  "copy_index": 1,
  "student_name": "Jean Dupont",
  "llm1_student_name": "Sophia Hanou",
  "llm2_student_name": "Sapio Nancy",
  "name_disagreement": {
    "llm1_name": "Sophia Hanou",
    "llm2_name": "Sapio Nancy",
    "similarity": 0.45,
    "resolved_name": "Jean Dupont"
  }
}
```

### Désaccord sur le barème

```json
"max_points_disagreement": {
  "llm1_max_points": 2.0,
  "llm2_max_points": 1.5,
  "resolved_max_points": 2.0,
  "persisted_after_ultimatum": false
}
```

---

## Utilisation pour le debugging

### Pourquoi LLM1 a-t-il changé sa note?

1. Comparer `LLM1.grade` initial vs `verification.llm1_new_grade`
2. Lire `verification.llm1_reasoning` pour comprendre le changement
3. Vérifier si `ultimatum.llm1_decision` = "changed" ou "maintained"

### Pourquoi y a-t-il un désaccord?

1. Comparer les `reading` des deux LLMs
2. Vérifier si l'un a lu plus de contexte que l'autre
3. Regarder les `reasoning` pour comprendre les divergences

### Quel a été le chemin de décision?

1. `final.agreement = true` → Accord (consensus)
2. `final.agreement = false` + `verification` présent → Vérification effectuée
3. `final.agreement = false` + `ultimatum` présent → Ultimatum effectué
4. `method = "average"` → Désaccord persistant, moyenne appliquée

---

## Exemple CLI

```bash
# Après une correction, consulter l'audit
cat data/{session_id}/session.json | jq '.graded_copies[0].llm_comparison'

# Voir les désaccords
cat data/{session_id}/session.json | jq '.graded_copies[].llm_comparison.questions[] | select(.final.agreement == false)'

# Voir les méthodes de résolution utilisées
cat data/{session_id}/session.json | jq '.graded_copies[].llm_comparison.questions[].final.method' | sort | uniq -c
```

---

## Token Tracking

Le système track également l'utilisation des tokens par phase:

```json
"token_usage": {
  "grading": {"prompt": 15000, "completion": 3000},
  "verification": {"prompt": 5000, "completion": 1000},
  "calibration": {"prompt": 0, "completion": 0},
  "annotation": {"prompt": 2000, "completion": 500}
}
```

Cette information est affichée en fin de correction via la CLI.
