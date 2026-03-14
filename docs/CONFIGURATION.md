# Configuration Reference

This document describes the configuration actually used by the application.

All values come from environment variables with the prefix `AI_CORRECTION_`.

## Minimal setup

Required for a working backend:

```env
AI_CORRECTION_AI_PROVIDER=<gemini|openai|glm|openrouter>
# Match the provider above
AI_CORRECTION_GEMINI_API_KEY=
# or AI_CORRECTION_OPENAI_API_KEY=
# or AI_CORRECTION_GLM_API_KEY=
# or AI_CORRECTION_OPENROUTER_API_KEY=
```

Recommended: start from `.env.example`.

## Core variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_CORRECTION_AI_PROVIDER` | Yes | Primary provider: `gemini`, `openai`, `glm`, or `openrouter` |
| `AI_CORRECTION_DATA_DIR` | No | Local storage directory, default `data` |

## Provider API keys

| Variable | Required when... |
|----------|------------------|
| `AI_CORRECTION_GEMINI_API_KEY` | `AI_CORRECTION_AI_PROVIDER=gemini` |
| `AI_CORRECTION_OPENAI_API_KEY` | `AI_CORRECTION_AI_PROVIDER=openai` |
| `AI_CORRECTION_GLM_API_KEY` | `AI_CORRECTION_AI_PROVIDER=glm` |
| `AI_CORRECTION_OPENROUTER_API_KEY` | `AI_CORRECTION_AI_PROVIDER=openrouter` |

## Provider models

All provider-specific model variables are optional. If you use them, set only the ones you need.

| Variable |
|----------|
| `AI_CORRECTION_GEMINI_MODEL` |
| `AI_CORRECTION_GEMINI_VISION_MODEL` |
| `AI_CORRECTION_GEMINI_EMBEDDING_MODEL` |
| `AI_CORRECTION_OPENAI_MODEL` |
| `AI_CORRECTION_OPENAI_VISION_MODEL` |
| `AI_CORRECTION_OPENAI_EMBEDDING_MODEL` |
| `AI_CORRECTION_GLM_MODEL` |
| `AI_CORRECTION_GLM_VISION_MODEL` |
| `AI_CORRECTION_OPENROUTER_MODEL` |
| `AI_CORRECTION_OPENROUTER_VISION_MODEL` |

## Dual-LLM grading

| Variable | Description |
|----------|-------------|
| `AI_CORRECTION_COMPARISON_MODE` | Enable dual-LLM grading |
| `AI_CORRECTION_LLM1_PROVIDER` | Provider for first grader |
| `AI_CORRECTION_LLM1_MODEL` | Model for first grader |
| `AI_CORRECTION_LLM2_PROVIDER` | Provider for second grader |
| `AI_CORRECTION_LLM2_MODEL` | Model for second grader |

## Annotation

| Variable | Description |
|----------|-------------|
| `AI_CORRECTION_ANNOTATION_PROVIDER` | Optional provider dedicated to annotation placement |
| `AI_CORRECTION_ANNOTATION_MODEL` | Optional model dedicated to annotation placement |

If `AI_CORRECTION_ANNOTATION_MODEL` is not set, annotated PDFs and overlays are still generated with heuristic placement.

## Review context enrichment

| Variable | Description |
|----------|-------------|
| `AI_CORRECTION_REVIEW_CONTEXT_PROVIDER` | Optional provider for disagreement context |
| `AI_CORRECTION_REVIEW_CONTEXT_MODEL` | Optional model for disagreement context |

## Confidence and caching

| Variable | Default |
|----------|---------|
| `AI_CORRECTION_CONFIDENCE_AUTO` | `0.85` |
| `AI_CORRECTION_CONFIDENCE_FLAG` | `0.60` |
| `AI_CORRECTION_GRADE_AGREEMENT_THRESHOLD` | `0.10` |
| `AI_CORRECTION_FLIP_FLOP_THRESHOLD` | `0.0` |
| `AI_CORRECTION_ENABLE_PROMPT_CACHING` | `true` |
| `AI_CORRECTION_CACHE_TTL_SECONDS` | `300` |
| `AI_CORRECTION_USE_EXPLICIT_CACHE` | `true` |

## Observability and API

| Variable | Default |
|----------|---------|
| `AI_CORRECTION_SENTRY_DSN` | empty |
| `AI_CORRECTION_SENTRY_ENVIRONMENT` | `development` |
| `AI_CORRECTION_SENTRY_TRACES_SAMPLE_RATE` | `0.1` |
| `AI_CORRECTION_CORS_ORIGINS` | `[]` |

## Example

```env
AI_CORRECTION_AI_PROVIDER=gemini
AI_CORRECTION_GEMINI_API_KEY=<api-key>

# Optional
AI_CORRECTION_GEMINI_MODEL=<configured-model>
AI_CORRECTION_GEMINI_VISION_MODEL=<configured-vision-model>

AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=<model-1>
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=<model-2>

```
