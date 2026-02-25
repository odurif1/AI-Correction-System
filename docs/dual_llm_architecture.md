# Architecture Dual LLM avec Vérification Croisée

> Voir aussi: [README principal](../README.md) | [Structure de l'Audit](AUDIT_STRUCTURE.md) | [Annotation PDF](annotation.md)

## Principe

L'architecture utilise **deux LLMs en parallèle** avec un workflow de **cross-verification** pour garantir fiabilité et équité dans la correction.

### Avantages

| Aspect | Bénéfice |
|--------|----------|
| **Fiabilité** | Deux jugements indépendants → moins d'erreurs |
| **Équité** | Same réponse = same note (cohérence) |
| **Traçabilité** | Audit complet de chaque décision |
| **Robustesse** | Détection des erreurs de lecture |

## Providers supportés

| Provider | Modèles populaires | Caractéristiques |
|----------|-------------------|------------------|
| **Gemini** | gemini-2.5-flash, gemini-3-pro | Rapide, économique |
| **OpenAI** | gpt-4o, gpt-4o-mini | Haute qualité |
| **GLM** (z.ai) | glm-4-flash, glm-4.6v | Visual grounding |
| **OpenRouter** | google/gemini-2.0-flash-exp | Accès multi-modèles |

## Workflow Global

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WORKFLOW COMPLET (6 phases)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: INITIALIZATION                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Si --auto-detect-structure:                                    │  │
│  │   → LLM1 + LLM2 analysent le PDF entier                       │  │
│  │   → Détection: élèves, pages, noms, barème                    │  │
│  │   → Cross-verification si désaccord                           │  │
│  │ Sinon si --pages-per-copy:                                     │  │
│  │   → Split mécanique                                           │  │
│  │ Sinon:                                                          │  │
│  │   → Structure détectée pendant grading                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 2: GRADING                                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ SINGLE-PASS: Les 2 LLM notent toutes les questions            │  │
│  │ Réponse: {Q1: {grade, reading, feedback}, Q2: {...}, ...}     │  │
│  │ + détection désaccords                                         │  │
│  │ + Token tracking                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 3: VERIFICATION / ULTIMATUM                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Si structure détectée pendant grading:                         │  │
│  │   → Cross-verify noms + barème                                │  │
│  │ Pour chaque désaccord (notes):                                │  │
│  │   → Cross-verification (LLMs voient le travail de l'autre)    │  │
│  │   → Ultimatum si désaccord persiste                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 4: CALIBRATION → Consistency check                           │
│  Phase 5: EXPORT → JSON, CSV, analytics                             │
│  Phase 6: ANNOTATION → PDFs annotés (optionnel, --annotate)         │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Stateless (Phase GRADING)

Chaque appel est **indépendant**. Les images sont re-envoyées à chaque phase.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE GRADING: SINGLE-PASS                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
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
│     [Aucun désaccord]              [Au moins 1 désaccord]        │
│     → Résultat final                       │                     │
│                                             ▼                     │
│  PHASE VERIFICATION: Cross-verification                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Prompt EXPLICITE:                                        │    │
│  │ - Question originale + critères                          │    │
│  │ - TON premier jugement: grade X, raison: "..."           │    │
│  │ - L'autre correcteur: grade Y, raison: "..."             │    │
│  │ - "Re-examine objectivement"                             │    │
│  │ + Images (re-envoyées)                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  [Toujours désaccord?]                                            │
│              │                                                    │
│              ▼                                                    │
│  PHASE ULTIMATUM                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ - Évolution: TON grade: X→Y, Autre: A→B                  │    │
│  │ - "Décision finale"                                      │    │
│  │ + Images (re-envoyées)                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
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
| **Debugging** | Plus facile - chaque appel est indépendant |
| **Optimisation API** | Skip cross-verification si accord initial ou si un LLM échoue |

---

## Option --auto-detect-structure

### Principe

Analyse le PDF **entier** avec les **2 LLMs** AVANT la correction pour détecter:
- Nombre d'élèves
- Pages par élève
- Noms des élèves
- Barème (si visible)

### Workflow de Cross-verification

```
┌─────────────────────────────────────────────────────────────────┐
│  --auto-detect-structure (Phase INITIALIZATION)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ÉTAPE 1: Analyse parallèle                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LLM1: Analyse PDF → {students: [...], bareme: {...}}    │    │
│  │ LLM2: Analyse PDF → {students: [...], bareme: {...}}    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ÉTAPE 2: Comparaison                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Accord sur le nombre d'élèves?                          │    │
│  │ Accord sur les noms?                                     │    │
│  │ Accord sur le barème?                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│     [Accord total]                  [Au moins 1 désaccord]       │
│     → Structure validée                    │                     │
│     → Passer à GRADING                     ▼                     │
│                                  ÉTAPE 3: Cross-verification     │
│                                  ┌─────────────────────────┐    │
│                                  │ Chaque LLM voit le      │    │
│                                  │ résultat de l'autre     │    │
│                                  │ et peut ajuster         │    │
│                                  └─────────────────────────┘    │
│                                              │                   │
│                                              ▼                   │
│                                  [Toujours désaccord?]           │
│                                              │                   │
│                                              ▼                   │
│                                  ÉTAPE 4: Ultimatum              │
│                                  ┌─────────────────────────┐    │
│                                  │ Décision finale         │    │
│                                  └─────────────────────────┘    │
│                                              │                   │
│                                              ▼                   │
│                                  [Toujours désaccord?]           │
│                                              │                   │
│                                              ▼                   │
│                                  ÉTAPE 5: Demande utilisateur   │
│                                  (ou auto-moyenne si --auto)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Combinaisons avec --pages-per-copy

| Options | Comportement |
|---------|-------------|
| `--auto-detect-structure` seul | AI détecte tout (élèves, pages, noms) |
| `--auto-detect-structure` + `--ppc N` | AI détecte noms, puis split mécanique N pages |
| `--ppc N` seul | Split mécanique, noms détectés pendant grading |

---

## Détection Automatique en Mode BATCH

Le mode BATCH supporte automatiquement trois scénarios sans configuration supplémentaire:

| Mode | Commande | Détection | Description |
|------|----------|-----------|-------------|
| **Multi-élèves** | `batch copies.pdf` | `detect_students=True` | 1 PDF contenant plusieurs élèves |
| **Pré-découpé (mécanique)** | `batch copies.pdf --ppc N` | `detect_students=False` | 1 PDF découpé en copies de N pages |
| **Pré-découpé (fichiers)** | `batch a.pdf b.pdf c.pdf` | `detect_students=False` | X PDFs (1 fichier = 1 élève) |

### Logique de détection automatique

```python
def _should_detect_students(copies):
    """
    Retourne True si:
    - 1 seul "copy" (1 PDF sans pré-découpage)
    - Pas de nom d'élève pré-détecté

    Cela indique un PDF unique pouvant contenir plusieurs élèves.
    """
    if len(copies) != 1:
        return False  # Cas 2 ou 3: déjà découpé
    return not copies[0].get('student_name')  # Cas 1
```

---

## Critères de Flagging (Désaccords)

Une question est flaggée si:

| Condition | Description |
|-----------|-------------|
| `|grade1 - grade2| > seuil` | Différence de note significative (10% du barème) |
| Lecture substantiellement différente | Similarité < 80% |
| Conflit "trouvé/non trouvé" | Un LLM trouve la réponse, l'autre non |
| Barème différent | `max_points` LLM1 ≠ `max_points` LLM2 |

---

## Optimisation des Appels API

### Skip automatique de la cross-verification

```python
# Dans comparison_provider.py

# Skip si un LLM a échoué (quota, erreur)
if not llm1_result and llm2_result:
    return llm2_result  # Pas de cross-verification

# Skip si les deux LLMs sont d'accord
if normalized(llm1_name) == normalized(llm2_name):
    return llm1_result  # Consensus direct
```

### Réduction des appels

| Scénario | Sans optimisation | Avec optimisation |
|----------|-------------------|-------------------|
| Accord initial | 2 appels | 2 appels |
| Désaccord + cross-verify | 4 appels | 4 appels |
| Un LLM échoue | 4 appels | **2 appels** (skip) |
| Accord noms | 4 appels | **2 appels** (skip) |

### Réduction avec vérification groupée (--batch-verify grouped)

| Mode | API Calls/LLM | Granularité | Coût | Précision |
|------|---------------|-------------|------|-----------|
| `grouped` | 1 | Basse | Min | Moins précis |
| `per-copy` | N copies | Moyenne | Moyen | Équilibré |
| `per-question` | M désaccords | Haute | Max | Plus précis |

**Exemple avec 2 copies, 8 total désaccords:**
| Mode | API Calls/LLM | Tokens (approx) |
|------|---------------|-----------------|
| `grouped` | 1 | ~2,000 |
| `per-copy` | 2 | ~4,000 |
| `per-question` | 8 | ~16,000 |

| Scénario | Avant | Après | Économie |
|----------|-------|-------|----------|
| 3 questions en désaccord | 18 appels | 4 appels | **~78%** |
| 1 question + nom en désaccord | 10 appels | 4 appels | **~60%** |

---

## Configuration

```bash
# .env - Mode Dual LLM
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o
```

---

## Fichiers Clés

| Fichier | Rôle |
|---------|------|
| `src/ai/comparison_provider.py` | Orchestre le workflow dual LLM, cross-verification |
| `src/ai/batch_grader.py` | Batch grading avec détection automatique multi-élèves + retry |
| `src/ai/single_pass_grader.py` | Grade toutes les questions en 1 appel |
| `src/ai/disagreement_analyzer.py` | Détecte les désaccords, flip-flops |
| `src/core/session.py` | Orchestrateur de session, phases |
| `src/core/workflow_state.py` | État immutable du workflow |
| `src/prompts/batch.py` | Prompts de batch grading |
| `src/prompts/verification.py` | Prompts de vérification et ultimatum |

---

## Exemples d'utilisation

```bash
# ═══════════════════════════════════════════════════════════════
# Mode BATCH (recommandé)
# ═══════════════════════════════════════════════════════════════

# Dual LLM batch standard
python -m src.main correct dual batch copies.pdf --auto-confirm

# Dual LLM batch avec vérification groupée (défaut)
python -m src.main correct dual batch copies.pdf --batch-verify grouped --auto-confirm

# Dual LLM batch avec vérification par copie (équilibre coût/précision)
python -m src.main correct dual batch copies.pdf --batch-verify per-copy --auto-confirm

# Dual LLM batch avec vérification par question (plus précis)
python -m src.main correct dual batch copies.pdf --batch-verify per-question --auto-confirm

# ═══════════════════════════════════════════════════════════════
# Mode INDIVIDUAL
# ═══════════════════════════════════════════════════════════════

# Dual LLM individual avec pré-analyse
python -m src.main correct dual individual copies.pdf \
  --auto-detect-structure \
  --auto-confirm

# Dual LLM individual avec split mécanique
python -m src.main correct dual individual copies.pdf \
  --pages-per-copy 2 \
  --auto-confirm

# ═══════════════════════════════════════════════════════════════
# Options avancées
# ═══════════════════════════════════════════════════════════════

# Avec annotation des copies
python -m src.main correct dual batch copies.pdf --annotate --auto-confirm

# Avec 2ème lecture
python -m src.main correct dual batch copies.pdf --second-reading --auto-confirm

# Mode HYBRID (LLM1=batch, LLM2=individual)
python -m src.main correct dual hybrid copies.pdf --pages-per-copy 2 --auto-confirm
```

---

## Robustesse

### Retry automatique

Le système inclut un mécanisme de retry avec backoff exponentiel:

- **3 tentatives** maximum par appel API
- **Erreurs retryables**: 503, 429, 500, 502, 504
- **Délai exponentiel**: 1s, 2s, 4s...

### Token Tracking

Les tokens sont trackés par phase:
- **Correction**: Appels initiaux
- **Vérification**: Cross-verification + ultimatum
- **Calibration**: Consistency check
- **Annotation**: Génération PDFs annotés
