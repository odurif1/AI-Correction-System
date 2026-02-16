# Structure de l'Audit - AI Correction System

Ce document décrit la structure complète de l'audit généré pour chaque question corrigée.

## Vue d'ensemble

L'audit est stocké dans `copies/{n}/audit.json` et contient toutes les informations sur le processus de correction, y compris les échanges avec les LLMs.

## Structure hiérarchique

```
audit.json
├── initial                    # Phase 1: Correction initiale
│   ├── llm1                   # Résultat LLM1
│   ├── llm2                   # Résultat LLM2
│   └── difference             # Écart entre les notes
│
├── reading_analysis           # Analyse des lectures
│   ├── llm1_read              # Ce que LLM1 a lu
│   ├── llm2_read              # Ce que LLM2 a lu
│   ├── identical              # Lectures identiques?
│   └── difference_type        # Type de différence (accent/partial/substantial)
│
├── confidence_evolution       # Évolution de la confiance
│   ├── initial                # Confiance initiale
│   ├── after_verification     # Après vérification croisée
│   └── after_round2           # Après ultimatum (si applicable)
│
├── timing                     # Informations temporelles
│   ├── initial                # Durée phase initiale
│   ├── verification           # Durée vérification
│   ├── round2                 # Durée round 2
│   └── total_ms               # Durée totale
│
├── decision_path              # Chemin de décision
│   ├── initial_agreement      # Accord initial?
│   ├── verification_triggered # Vérification déclenchée?
│   ├── round2_triggered       # Round 2 déclenchée?
│   └── final_method           # Méthode finale (consensus/average/user_choice)
│
├── images                     # Images utilisées
│   ├── count                  # Nombre d'images
│   └── paths                  # Chemins des images
│
├── after_verification         # Phase 2: Après vérification croisée
│   ├── llm1                   # Résultat LLM1 + prompt envoyé
│   ├── llm2                   # Résultat LLM2 + prompt envoyé
│   └── difference             # Nouvel écart
│
├── after_round2               # Phase 3: Après ultimatum (optionnel)
│   ├── llm1                   # Résultat LLM1 + prompt envoyé
│   ├── llm2                   # Résultat LLM2 + prompt envoyé
│   └── difference             # Nouvel écart
│
└── final                      # Résultat final
    ├── grade                  # Note finale
    ├── agreement              # Accord atteint?
    └── method                 # Méthode utilisée
```

---

## Détail des champs

### 1. `initial` - Phase initiale

Résultats des deux LLMs **avant** toute discussion.

```json
"initial": {
  "llm1": {
    "provider": "gemini-2.5-flash",
    "grade": 0.0,
    "confidence": 1.0,
    "internal_reasoning": "Note: 0.0/1.0. L'image montre...",
    "student_feedback": "Ta réponse est incorrecte...",
    "student_answer_read": "Fiole jaugé",
    "duration_ms": 3500.0
  },
  "llm2": {
    "provider": "gemini-flash-latest",
    "grade": 1.0,
    "confidence": 1.0,
    "internal_reasoning": "Note: 1.0/1.0. L'élève a correctement...",
    "student_feedback": "Votre réponse est parfaitement correcte...",
    "student_answer_read": "Fiole jaugée",
    "duration_ms": 3200.0
  },
  "difference": 1.0
}
```

| Champ | Description |
|-------|-------------|
| `provider` | Nom du modèle utilisé |
| `grade` | Note attribuée |
| `confidence` | Confiance (0.0 - 1.0) |
| `internal_reasoning` | Raisonnement technique |
| `student_feedback` | Feedback pédagogique |
| `student_answer_read` | Ce que le LLM a lu de la réponse |
| `duration_ms` | Durée de l'appel API |

---

### 2. `reading_analysis` - Analyse des lectures

Compare ce que chaque LLM a lu de la réponse de l'élève.

```json
"reading_analysis": {
  "llm1_read": "Fiole jaugé",
  "llm2_read": "Fiole jaugée",
  "identical": false,
  "difference_type": "accent"
}
```

| `difference_type` | Description |
|-------------------|-------------|
| `null` | Lectures identiques |
| `"accent"` | Différence d'accent/orthographe |
| `"partial"` | Une lecture incluse dans l'autre |
| `"substantial"` | Différence significative |

---

### 3. `confidence_evolution` - Évolution de la confiance

```json
"confidence_evolution": {
  "initial": {"llm1": 1.0, "llm2": 1.0},
  "after_verification": {"llm1": 0.8, "llm2": 1.0},
  "after_round2": {"llm1": 0.9, "llm2": 0.9}
}
```

Permet de voir si la vérification croisée a:
- Augmenté la confiance (convergence)
- Diminué la confiance (doute)
- Laissé inchangée

---

### 4. `timing` - Informations temporelles

```json
"timing": {
  "initial": {
    "llm1": 3500.0,
    "llm2": 3200.0
  },
  "verification": {
    "total_ms": 4200.0
  },
  "round2": {
    "total_ms": 3800.0
  },
  "total_ms": 15000.0
}
```

Toutes les durées sont en millisecondes.

---

### 5. `decision_path` - Chemin de décision

```json
"decision_path": {
  "initial_agreement": false,
  "verification_triggered": true,
  "round2_triggered": false,
  "final_method": "average"
}
```

| `final_method` | Description |
|----------------|-------------|
| `"consensus"` | Accord initial |
| `"verification_consensus"` | Accord après vérification |
| `"average"` | Moyenne (désaccord persistant) |
| `"user_choice"` | Choix utilisateur |

---

### 6. `after_verification` - Après vérification croisée

Contient les résultats **après** que chaque LLM a vu le raisonnement de l'autre.

```json
"after_verification": {
  "llm1": {
    "provider": "gemini-2.5-flash",
    "grade": 0.75,
    "confidence": 0.8,
    "internal_reasoning": "L'autre correcteur a mal interprété...",
    "student_feedback": "...",
    "student_answer_read": "Fiole jaugé",
    "prompt_sent": "─── AUTRE CORRECTEUR ───\nNote: 1.0/1.0\nRaisonnement: ..."
  },
  "llm2": {...},
  "difference": 0.75
}
```

**Nouveau champ important**: `prompt_sent` contient le prompt exact qui a été envoyé au LLM.

---

### 7. `final` - Résultat final

```json
"final": {
  "grade": 0.38,
  "agreement": false,
  "method": "average"
}
```

Ou si accord:

```json
"final": {
  "grade": 1.0,
  "agreement": true,
  "method": "verification_consensus"
}
```

---

## Exemple complet

```json
{
  "Q1": {
    "initial": {
      "llm1": {
        "provider": "gemini-2.5-flash",
        "grade": 0.0,
        "confidence": 1.0,
        "internal_reasoning": "Note: 0.0/1.0. L'image montre un ballon...",
        "student_feedback": "L'identification est incorrecte...",
        "student_answer_read": "Fiole jaugé",
        "duration_ms": 3500.0
      },
      "llm2": {
        "provider": "gemini-flash-latest",
        "grade": 1.0,
        "confidence": 1.0,
        "internal_reasoning": "Note: 1.0/1.0. L'élève a correctement identifié...",
        "student_feedback": "Votre réponse est correcte...",
        "student_answer_read": "Fiole jaugée",
        "duration_ms": 3200.0
      },
      "difference": 1.0
    },

    "reading_analysis": {
      "llm1_read": "Fiole jaugé",
      "llm2_read": "Fiole jaugée",
      "identical": false,
      "difference_type": "accent"
    },

    "confidence_evolution": {
      "initial": {"llm1": 1.0, "llm2": 1.0},
      "after_verification": {"llm1": 0.8, "llm2": 1.0}
    },

    "timing": {
      "initial": {"llm1": 3500.0, "llm2": 3200.0},
      "verification": {"total_ms": 4200.0},
      "round2": null,
      "total_ms": 11000.0
    },

    "decision_path": {
      "initial_agreement": false,
      "verification_triggered": true,
      "round2_triggered": false,
      "final_method": "average"
    },

    "images": {
      "count": 1,
      "paths": ["data/session/copies/1/page_0.png"]
    },

    "after_verification": {
      "llm1": {
        "provider": "gemini-2.5-flash",
        "grade": 0.75,
        "confidence": 0.8,
        "internal_reasoning": "Je reconnais que l'autre lecture est correcte...",
        "prompt_sent": "─── AUTRE CORRECTEUR ───\nNote: 1.0/1.0\n..."
      },
      "llm2": {
        "provider": "gemini-flash-latest",
        "grade": 0.0,
        "confidence": 1.0,
        "internal_reasoning": "Je maintiens ma note car...",
        "prompt_sent": "─── AUTRE CORRECTEUR ───\nNote: 0.0/1.0\n..."
      },
      "difference": 0.75
    },

    "after_round2": null,

    "final": {
      "grade": 0.38,
      "agreement": false,
      "method": "average"
    }
  }
}
```

---

## Utilisation pour le debugging

### Pourquoi LLM1 a changé sa note?

1. Regarder `initial.llm1.grade` → `after_verification.llm1.grade`
2. Lire `after_verification.llm1.prompt_sent` pour voir ce qui a influencé
3. Lire `after_verification.llm1.internal_reasoning` pour comprendre le changement

### Pourquoi y a-t-il un désaccord?

1. Regarder `reading_analysis.difference_type`
2. Si `substantial`, les LLMs voient des choses différentes
3. Regarder les `student_answer_read` respectifs

### Combien de temps a pris la correction?

1. Regarder `timing.total_ms`
2. Divisé par phase: `timing.initial`, `timing.verification`, `timing.round2`
