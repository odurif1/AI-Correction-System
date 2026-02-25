# Structure de l'Audit - AI Correction System

> Voir aussi: [README principal](../README.md) | [Architecture Dual LLM](dual_llm_architecture.md)

Ce document décrit la structure unifiée de l'audit généré pour chaque copie corrigée.

## Vue d'ensemble

L'audit est stocké dans `data/{session_id}/copies/{copy_number}/audit.json` et contient toutes les informations sur le processus de correction.

L'audit permet de:
- **Tracer** chaque décision de notation
- **Comparer** facilement les résultats LLM1 vs LLM2
- **Comprendre** les désaccords et leur résolution
- **Suivre** l'évolution par phase (initial → verification → ultimatum)

---

## Structure Unifiée `grading_audit`

La nouvelle structure `grading_audit` fonctionne pour tous les modes:
- **Single LLM** et **Dual LLM**
- Tous les modes de vérification (`grouped`, `per-copy`, `per-question`, `none`)

```json
{
  "grading_audit": {
    "mode": "single" | "dual",
    "grading_method": "batch" | "individual" | "hybrid",
    "verification_mode": "grouped" | "per-copy" | "per-question" | "none",

    "providers": [
      {"id": "LLM1", "model": "gemini-2.5-flash", "tokens": {"prompt": 15000, "completion": 3000}},
      {"id": "LLM2", "model": "gpt-4o", "tokens": {"prompt": 15000, "completion": 3000}}
    ],

    "questions": {
      "Q1": {
        "llm_results": {
          "LLM1": {
            "grade": 1.0,
            "max_points": 1.0,
            "reading": "fiole jaugée",
            "reasoning": "L'élève a correctement identifié...",
            "feedback": "Réponse correcte.",
            "confidence": 0.9
          },
          "LLM2": {
            "grade": 1.0,
            "max_points": 1.0,
            "reading": "fiole jaugée",
            "reasoning": "Identification correcte...",
            "feedback": "Correct.",
            "confidence": 0.95
          }
        },

        "resolution": {
          "final_grade": 1.0,
          "final_max_points": 1.0,
          "method": "consensus",
          "phases": ["initial"],
          "agreement": true
        }
      },

      "Q3": {
        "llm_results": {
          "LLM1": {"grade": 2.0, "max_points": 2.0, "reading": "...", "reasoning": "...", "feedback": "...", "confidence": 0.9},
          "LLM2": {"grade": 1.0, "max_points": 1.5, "reading": "...", "reasoning": "...", "feedback": "...", "confidence": 0.85}
        },

        "resolution": {
          "final_grade": 1.5,
          "final_max_points": 2.0,
          "method": "ultimatum_average",
          "phases": ["initial", "verification", "ultimatum"],
          "agreement": false
        },

        "phases": {
          "initial": {
            "llm1_grade": 2.0,
            "llm1_max_points": 2.0,
            "llm2_grade": 1.0,
            "llm2_max_points": 1.5,
            "agreement": false
          },
          "verification": {
            "llm1_grade": 2.0,
            "llm1_max_points": 2.0,
            "llm2_grade": 1.0,
            "llm2_max_points": 1.5,
            "llm1_reasoning": "Je maintiens...",
            "llm2_reasoning": "Je maintiens...",
            "agreement": false
          },
          "ultimatum": {
            "llm1_grade": 2.0,
            "llm1_max_points": 2.0,
            "llm2_grade": 1.0,
            "llm2_max_points": 1.5,
            "llm1_decision": "maintained",
            "llm2_decision": "maintained",
            "agreement": false
          }
        }
      }
    },

    "student_detection": {
      "final_name": "Jean Dupont",
      "llm_results": {
        "LLM1": "Jean Dupont",
        "LLM2": "J. Dupont"
      },
      "resolution": {
        "method": "consensus",
        "phases": ["initial"],
        "agreement": true
      }
    },

    "summary": {
      "total_questions": 6,
      "agreed_initial": 4,
      "required_verification": 2,
      "required_ultimatum": 1,
      "final_agreement_rate": 0.833
    }
  }
}
```

---

## Structure hiérarchique

```
audit.json
└── grading_audit                    # Structure unifiée d'audit
    ├── mode                         # "single" ou "dual"
    ├── grading_method               # "batch", "individual", "hybrid"
    ├── verification_mode            # "grouped", "per-copy", "per-question", "none"
    │
    ├── providers[]                  # Liste des LLMs utilisés
    │   ├── id                       # "LLM1" ou "LLM2"
    │   ├── model                    # Nom du modèle
    │   └── tokens                   # Usage des tokens
    │
    ├── questions{}                  # Détail par question
    │   └── {Q1, Q2, ...}/
    │       ├── llm_results{}        # Résultats par LLM
    │       │   └── {LLM1, LLM2}/
    │       │       ├── grade        # Note attribuée
    │       │       ├── max_points   # Barème détecté
    │       │       ├── reading      # Lecture de la réponse
    │       │       ├── reasoning    # Justification
    │       │       ├── feedback     # Feedback pour l'élève
    │       │       └── confidence   # Confiance (0-1)
    │       │
    │       ├── resolution           # Résolution finale
    │       │   ├── final_grade      # Note finale
    │       │   ├── final_max_points # Barème retenu
    │       │   ├── method           # Méthode de résolution
    │       │   ├── phases[]         # ["initial"] ou ["initial", "verification", "ultimatum"]
    │       │   └── agreement        # Accord atteint?
    │       │
    │       └── phases{}             # Détail par phase (optionnel)
    │           ├── initial          # Phase initiale
    │           ├── verification     # Phase de vérification
    │           └── ultimatum        # Phase ultimatum
    │
    ├── student_detection            # Détection du nom
    │   ├── final_name               # Nom final retenu
    │   ├── llm_results{}            # Noms détectés par LLM
    │   └── resolution               # Résolution
    │
    └── summary                      # Résumé statistique
        ├── total_questions          # Nombre total de questions
        ├── agreed_initial           # Questions en accord initial
        ├── required_verification    # Questions nécessitant vérification
        ├── required_ultimatum       # Questions nécessitant ultimatum
        └── final_agreement_rate     # Taux d'accord final
```

---

## Méthodes de résolution

| Méthode | Description |
|---------|-------------|
| `consensus` | Accord initial entre les deux LLMs |
| `verification_consensus` | Accord après cross-verification |
| `ultimatum_consensus` | Accord après ultimatum |
| `average` | Moyenne simple (désaccord initial sans vérification) |
| `verification_average` | Moyenne après vérification (désaccord persiste) |
| `ultimatum_average` | Moyenne après ultimatum (désaccord persiste) |
| `single_llm` | Mode LLM unique (pas de comparaison) |

---

## Avantages de la Nouvelle Structure

1. **Uniforme**: Même structure pour single/dual, tous les modes de vérification
2. **Comparable**: `llm_results.LLM1` vs `llm_results.LLM2` directement
3. **Traçable**: `phases` montre l'évolution complète (initial → verification → ultimatum)
4. **Queryable**: Structure prévisible pour jq/analyses
5. **Extensible**: Facile d'ajouter de nouveaux modes ou phases

---

## Exemple CLI

```bash
# Voir la structure d'audit unifiée
cat data/{session}/copies/1/audit.json | jq '.grading_audit'

# Voir le mode et les providers
cat data/{session}/copies/1/audit.json | jq '.grading_audit | {mode, verification_mode, providers}'

# Comparer les résultats LLM pour une question
cat data/{session}/copies/1/audit.json | jq '.grading_audit.questions.Q1.llm_results'

# Voir les questions avec désaccord
cat data/{session}/copies/1/audit.json | jq '.grading_audit.questions | to_entries[] | select(.value.resolution.agreement == false)'

# Voir le résumé
cat data/{session}/copies/1/audit.json | jq '.grading_audit.summary'

# Voir les méthodes de résolution utilisées
cat data/{session}/copies/1/audit.json | jq '.grading_audit.questions | to_entries[].value.resolution.method' | sort | uniq -c
```

---

## Accès à l'Audit

L'audit unifié `grading_audit` est la seule structure d'audit. Accédez-y directement:

```bash
# Voir l'audit complet
cat data/{session}/copies/1/audit.json | jq '.grading_audit'

# Voir le résumé
cat data/{session}/copies/1/audit.json | jq '.grading_audit.summary'
```

---

## Utilisation pour le debugging

### Pourquoi LLM1 a-t-il changé sa note?

1. Vérifier `phases.verification.llm1_grade` vs `phases.initial.llm1_grade`
2. Lire `phases.verification.llm1_reasoning` pour comprendre le changement
3. Vérifier `phases.ultimatum.llm1_decision` = "maintained" ou "changed"

### Pourquoi y a-t-il un désaccord?

1. Comparer `llm_results.LLM1.reading` vs `llm_results.LLM2.reading`
2. Vérifier les `reasoning` respectifs
3. Regarder l'évolution dans `phases`

### Quel a été le chemin de décision?

Regarder `resolution.phases`:
- `["initial"]` → Accord immédiat
- `["initial", "verification"]` → Vérification effectuée
- `["initial", "verification", "ultimatum"]` → Ultimatum nécessaire
