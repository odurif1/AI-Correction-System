# Module d'Annotation PDF

> Voir aussi: [README principal](../README.md) | [Architecture Dual LLM](dual_llm_architecture.md)

Ce module gère le placement intelligent des annotations de feedback sur les copies PDF des élèves.

## Principe

Le système utilise un LLM vision pour:
1. Analyser la page PDF et identifier les zones de réponse
2. Déterminer les coordonnées optimales pour placer le feedback
3. Générer des annotations visuelles (couleur selon le score)

## Architecture

```
src/
├── prompts/
│   └── annotation.py          # Prompts LLM pour détection coordonnées
└── export/
    ├── annotation_pipeline.py # Pipeline d'export des 2 artefacts PDF
    ├── annotation_service.py  # Service de détection et placement
    └── pdf_annotator.py       # Application des annotations sur PDF
```

## Configuration

Variables d'environnement (.env):

```bash
# LLM pour annotations (optionnel)
# Si non défini, le pipeline génère quand même les 2 PDF
# avec un placement heuristique des annotations.
# AI_CORRECTION_ANNOTATION_PROVIDER=<provider>
# AI_CORRECTION_ANNOTATION_MODEL=<vision-model>
```

### Providers supportés

| Provider | Exemple de modèle | Caractéristiques |
|----------|-------------------|------------------|
| **GLM** | `glm-4.6v` | Visual grounding, bounding boxes |
| **Gemini** | modèle vision configuré | Vision multimodale |
| **OpenAI** | modèle vision configuré | Vision multimodale |
| **OpenRouter** | modèle vision configuré | Accès multi-modèles |

## Utilisation CLI

L'annotation est intégrée au workflow principal via l'option `--annotate`:

```bash
# Correction avec annotation automatique
python -m src.main correct dual batch copies.pdf --annotate --auto-confirm

# Mode individual avec annotation
python -m src.main correct dual individual copies.pdf --pages-per-copy 2 --annotate --auto-confirm

# Mode single LLM avec annotation
python -m src.main correct single batch copies.pdf --annotate --auto-confirm
```

## Sorties générées

L'option `--annotate` produit **systématiquement deux types de fichiers**:

```
outputs/<session_id>/
├── annotated/                      # Copies complètes annotées
│   ├── student_001_annotated.pdf
│   ├── student_002_annotated.pdf
│   └── ...
│
└── overlays/                       # Overlays (annotations uniquement)
    ├── student_001_overlay.pdf
    ├── student_002_overlay.pdf
    └── ...
```

### Copies annotées (`annotated/`)
- PDF complet avec le contenu original + annotations
- Inclut une page de garde avec résumé
- Prêt à distribuer aux élèves

### Overlays (`overlays/`)
- PDF transparent avec **uniquement les annotations**
- Même dimensions que les copies originales
- Usage:
  - Superposition sur les copies scannées
  - Impression séparée sur calque
  - Application non-destructive

### Comportement sans modèle d'annotation

Si `AI_CORRECTION_ANNOTATION_MODEL` n'est pas configuré:
- le pipeline d'annotation n'est pas désactivé
- les deux PDF sont quand même générés
- le placement des annotations utilise le fallback heuristique

### Superposition manuelle

Pour appliquer un overlay sur une copie:

```python
import fitz  # PyMuPDF

# Ouvrir l'original et l'overlay
original = fitz.open("copie_originale.pdf")
overlay = fitz.open("overlay.pdf")

# Fusionner page par page
for i in range(len(original)):
    original[i].show_pdf_page(
        original[i].rect,
        overlay,
        overlay[i]
    )

original.save("copie_finale.pdf")
```

## Types de données

### AnnotationPlacement

Représente une annotation individuelle sur le PDF.

| Champ | Type | Description |
|-------|------|-------------|
| `question_id` | str | Identifiant de la question (ex: "Q1") |
| `feedback_text` | str | Texte du feedback à afficher |
| `page_number` | int | Numéro de page **1-based** (1 = première page) |
| `x_percent` | float | Position horizontale gauche (0-100%) |
| `y_percent` | float | Position verticale haute (0-100%) |
| `width_percent` | float | Largeur de la zone (défaut: 30%) |
| `height_percent` | float | Hauteur de la zone (défaut: 5%) |
| `placement` | str | Indication de position ("below_answer", etc.) |
| `confidence` | float | Confiance LLM (0.0-1.0) |

### CopyAnnotations

Contient toutes les annotations pour une copie.

```python
@dataclass
class CopyAnnotations:
    copy_id: str
    student_name: Optional[str]
    placements: List[AnnotationPlacement]
```

## Système de coordonnées

Les coordonnées sont exprimées en **pourcentage de la page**:

```
┌─────────────────────────────────────┐ y=0%
│                                     │
│    ┌─────────────────┐              │
│    │  Annotation     │ ← y_percent  │
│    │  (feedback)     │              │
│    └─────────────────┘              │
│         ↑                           │
│       x_percent                     │
│                                     │
└─────────────────────────────────────┘ y=100%
x=0%                            x=100%
```

**Exemple de calcul:**
- Page A4: 595 x 842 points
- `x_percent=15%` → x = 595 × 0.15 = 89 points
- `y_percent=45%` → y = 842 × 0.45 = 379 points

## API Programmatique

### Annotation automatique avec placement intelligent

```python
from export.pdf_annotator import PDFAnnotator
from core.models import CopyDocument, GradedCopy

# Créer l'annotateur
annotator = PDFAnnotator(session=session)

# Annoter une copie avec placement intelligent (LLM vision)
output_path = annotator.annotate_copy(
    copy=copy_document,
    graded=graded_copy,
    smart_placement=True,   # Active la détection LLM
    language='fr'
)
print(f"PDF annoté: {output_path}")
```

### Détection des coordonnées uniquement

```python
from export.annotation_service import AnnotationCoordinateDetector

detector = AnnotationCoordinateDetector()
annotations = detector.detect_annotations(
    pdf_path="/path/to/copy.pdf",
    graded_copy=graded,
    language='fr'
)

for placement in annotations.placements:
    print(f"Q{placement.question_id}: page {placement.page_number}")
    print(f"  Position: ({placement.x_percent}%, {placement.y_percent}%)")
    print(f"  Feedback: {placement.feedback_text}")
```

## Flux de traitement

```
┌─────────────────┐
│  GradedCopy     │  (résultats de correction)
│  + feedback     │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│ AnnotationExportService  │  Orchestration unique
│ 1 calcul de placement    │  pour 2 sorties PDF
└────────┬─────────────────┘
         │
         ▼
┌─────────────────┐
│ CopyAnnotations │  Liste des placements
└──────┬─────┬────┘
       │     │
       ▼     ▼
┌───────────────┐   ┌────────────────┐
│ PDF annoté    │   │ Overlay PDF    │
│ contenu+notes │   │ annotations    │
└───────────────┘   └────────────────┘
```

## Fallback heuristique

Si le LLM n'est pas disponible ou échoue, le système utilise un placement heuristique:

- Annotations placées dans la marge droite (x=70%)
- Distribution verticale des questions
- Confiance réduite (0.5)

## Couleurs d'annotation

| Score | Couleur | Signification |
|-------|---------|---------------|
| ≥ 80% | Vert | Correct |
| 50-79% | Orange | Partiel |
| < 50% | Rouge | Incorrect |

## Prompts disponibles

| Fonction | Description |
|----------|-------------|
| `build_direct_annotation_prompt()` | Prompt pour placement direct (recommandé) |
| `build_annotation_coordinate_prompt()` | Prompt avec zones numérotées |
| `build_zone_detection_prompt()` | Prompt pour détecter les zones vides |
| `parse_annotation_response()` | Parse la réponse JSON du LLM |

## Exemple de réponse LLM

```json
{
  "annotations": [
    {
      "question_id": "Q1",
      "page": 1,
      "feedback_text": "Identification correcte.",
      "x_percent": 15.0,
      "y_percent": 25.0,
      "placement": "below_answer",
      "confidence": 0.95
    },
    {
      "question_id": "Q2",
      "page": 1,
      "feedback_text": "Revoir la définition de la dilution.",
      "x_percent": 15.0,
      "y_percent": 45.0,
      "placement": "below_answer",
      "confidence": 0.90
    }
  ]
}
```

## Notes importantes

1. **Numérotation des pages**: Les pages sont numérotées à partir de 1 (1-based), plus intuitif pour les humains et les LLMs.

2. **Conversion interne**: `create_annotation_boxes()` convertit automatiquement les numéros de pages 1-based en index 0-based pour PyMuPDF.

3. **LLM séparé**: Le LLM d'annotation peut être différent des LLMs de correction (configuré via `.env`).

4. **Post-traitement**: L'annotation est une étape post-correction (Phase 6), ce qui permet:
   - D'utiliser un modèle différent
   - De batch plusieurs annotations
   - D'ajuster sans recorriger

5. **Token Tracking**: Les tokens utilisés pour l'annotation sont trackés séparément dans le workflow.
