# Architecture Stateless avec Contexte Explicite

## Principe

L'architecture stateless traite chaque appel comme **indépendant**. Les images sont re-envoyées à chaque phase pour un regard frais, et le contexte est explicite dans les prompts.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE STATELESS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: SINGLE-PASS                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Appel frais: Images + "Corrige toutes les questions"    │    │
│  │ Réponse: {Q1: {grade, reasoning, ...}, Q2: {...}, ...}  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ANALYSE: Détecter les désaccords                               │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [Questions OK]                  [Questions flaggées]         │
│     Finaliser directement           PHASE 2: Vérification        │
│                                                                      │
│  PHASE 2: VÉRIFICATION CIBLÉE (APPEL FRAIS PAR QUESTION)          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Prompt EXPLICITE:                                        │      │
│  │ - Question originale + critères                          │      │
│  │ - TON premier jugement: grade X, raison: "..."           │      │
│  │ - L'autre correcteur: grade Y, raison: "..."             │      │
│  │ - "Re-examine objectivement"                             │      │
│  │ + Images (re-envoyées)                                   │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                     │
│                              ▼                                     │
│  PHASE 3: ULTIMATUM (si toujours désaccord)                       │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Prompt EXPLICITE:                                        │      │
│  │ - Question + critères                                    │      │
│  │ - Évolution: TON grade: X→Y, Autre: A→B                  │      │
│  │ - "Décision finale"                                      │      │
│  │ + Images (re-envoyées)                                   │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Avantages de l'Architecture Stateless

| Aspect | Avantage |
|--------|----------|
| **Ancrage** | Éliminé - chaque appel est frais, sans mémoire du précédent |
| **Images** | Regard neuf à chaque vérification |
| **Contexte** | Explicite et complet dans le prompt |
| **Debugging** | Plus facile - chaque appel est indépendant et autonome |
| **Complexité** | Réduite - pas de gestion d'état entre les appels |

---

## Pourquoi Re-envoyer les Images?

Les APIs Gemini et OpenAI sont **stateless**. Le concept de "conversation" est simulé côté client en renvoyant tout l'historique à chaque appel. L'architecture stateless:

1. **Évite l'ancrage**: Le LLM ne peut pas être ancré sur sa première décision puisqu'il a un regard frais
2. **Simplifie le code**: Pas de gestion d'état, pas de sessions
3. **Améliore la qualité**: Le LLM peut ré-examiner objectivement les images

---

## Critères de Flagging

Une question est flaggée si:

| Condition | Description |
|-----------|-------------|
| `|grade1 - grade2| > 0.5` | Différence de note significative |
| Lecture substantiellement différente | Similarité Jaccard < 30% |
| Conflit "trouvé/non trouvé" | Un LLM trouve la réponse, l'autre non |

---

## Structure JSON (Single-Pass)

```json
{
  "student_name": "Dupont Marie",
  "questions": {
    "Q1": {
      "location": "page 1, haut",
      "student_answer_read": "m = V × Cm",
      "grade": 1.0,
      "confidence": 0.9,
      "reasoning": "Formule correcte",
      "feedback": "Exact."
    },
    "Q2": {
      "location": "page 1, milieu",
      "student_answer_read": "Cm = m/V",
      "grade": 2.0,
      "confidence": 0.85,
      "reasoning": "Formule inversée correctement",
      "feedback": "Correct."
    }
  }
}
```

---

## Configuration

L'architecture stateless s'active automatiquement quand les 2 LLM sont configurés:

```bash
# .env
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o
```

---

## Fallback automatique

Si le single-pass échoue (JSON invalide, erreur API), le système bascule automatiquement sur l'architecture per-question classique.

---

## Audit

```json
{
  "Q1": {
    "method": "single_pass_consensus",
    "agreement": true,
    "llm1": {"grade": 2.0, "confidence": 0.9, "reading": "..."},
    "llm2": {"grade": 2.0, "confidence": 0.85, "reading": "..."}
  },
  "Q2": {
    "method": "stateless_verification_consensus",
    "agreement": true,
    "single_pass": {
      "llm1": {"grade": 1.5},
      "llm2": {"grade": 0.5},
      "flagged_reason": "Différence de note: 1.0 points"
    },
    "verification": {
      "initial": {"llm1": {"grade": 1.5}, "llm2": {"grade": 0.5}},
      "after_verification": {"llm1": {"grade": 1.0}, "llm2": {"grade": 1.0}},
      "final": {"grade": 1.0, "agreement": true}
    }
  }
}
```

---

## Prompt de Vérification (Exemple Français)

```
─── QUESTION À VÉRIFIER ───
[Texte de la question]

Critères: [Critères de notation]
Note maximale: 5 points

─── TON PREMIER JUGEMENT ───
Tu as initialement noté: 1.5/5
Ton raisonnement: [Raisonnement initial]
Lecture de la réponse: [Ce que tu as lu]

─── L'AUTRE CORRECTEUR ───
Note: 0.5/5
Raisonnement: [Raisonnement de l'autre LLM]
Lecture: [Ce que l'autre a lu]

─── INSTRUCTION ───
1. Re-examine OBJECTIVEMENT la réponse de l'élève (images ci-jointes)
2. Considère les deux points de vue sans favoriser le tien
3. Décide de ta note finale

RÈGLES:
- Ne maintiens ta note que si tu peux justifier objectivement
- Change si tu identifies une erreur dans ton analyse initiale
- Si incertain: abaisse ta confiance (< 0.5)

─── FORMAT DE RÉPONSE ───
GRADE: [note]/5
CONFIDENCE: [0.0-1.0]
INTERNAL_REASONING: [analyse]
STUDENT_FEEDBACK: [feedback]
```

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `src/ai/single_pass_grader.py` | Grade toutes les questions en 1 appel |
| `src/ai/disagreement_analyzer.py` | Détecte les désaccords |
| `src/ai/comparison_provider.py` | Orchestre le workflow stateless |
| `src/config/prompts.py` | Prompt multi-questions JSON |
