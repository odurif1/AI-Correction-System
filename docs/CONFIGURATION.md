# Configuration Reference

This document describes all configuration options for the AI Correction System.

## Environment Variables

All configuration is done via environment variables. You can set them in a `.env` file in the project root.

### API Keys

| Variable | Description | Required |
|----------|-------------|----------|
| `AI_CORRECTION_GEMINI_API_KEY` | Google Gemini API key | For Gemini |
| `AI_CORRECTION_OPENAI_API_KEY` | OpenAI API key | For OpenAI |
| `AI_CORRECTION_API_KEY` | API key for authenticating API requests | Optional |

### Provider Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_PROVIDER` | `gemini` | Primary AI provider (`gemini` or `openai`) |
| `AI_CORRECTION_MODEL` | `gemini-2.5-flash` | Model for single-LLM mode |
| `AI_CORRECTION_VISION_MODEL` | Same as MODEL | Vision model for image processing |
| `AI_CORRECTION_EMBEDDING_MODEL` | `text-embedding-004` | Model for embeddings |

### Gemini-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `AI_CORRECTION_GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Gemini vision model |
| `AI_CORRECTION_GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Gemini embedding model |

### Dual-LLM Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_COMPARISON_MODE` | `true` | Enable dual-LLM verification |
| `AI_CORRECTION_LLM1_PROVIDER` | `gemini` | First LLM provider |
| `AI_CORRECTION_LLM1_MODEL` | `gemini-2.5-flash` | First LLM model |
| `AI_CORRECTION_LLM2_PROVIDER` | `openai` | Second LLM provider |
| `AI_CORRECTION_LLM2_MODEL` | `gpt-4o` | Second LLM model |

### Annotation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_ANNOTATION_PROVIDER` | Same as PROVIDER | Provider for PDF annotation |
| `AI_CORRECTION_ANNOTATION_MODEL` | Same as VISION_MODEL | Model for annotation placement |

### Confidence Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_CONFIDENCE_AUTO` | `0.85` | Auto-grade above this confidence |
| `AI_CORRECTION_CONFIDENCE_FLAG` | `0.60` | Flag for review below this |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_API_HOST` | `127.0.0.1` | API server host |
| `AI_CORRECTION_API_PORT` | `8000` | API server port |
| `AI_CORRECTION_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CORRECTION_DEBUG_CAPTURE` | `false` | Enable debug capture to files |
| `AI_CORRECTION_LOG_LEVEL` | `INFO` | Logging level |

## Configuration File Example

```env
# .env file

# API Keys
AI_CORRECTION_GEMINI_API_KEY=your-gemini-api-key-here
AI_CORRECTION_OPENAI_API_KEY=your-openai-api-key-here

# Primary Provider (single-LLM mode when comparison is off)
AI_CORRECTION_PROVIDER=gemini
AI_CORRECTION_MODEL=gemini-2.5-flash

# Dual-LLM Configuration
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o

# Confidence Thresholds
AI_CORRECTION_CONFIDENCE_AUTO=0.85
AI_CORRECTION_CONFIDENCE_FLAG=0.60

# API Configuration (optional)
AI_CORRECTION_API_KEY=secure-api-key-here
AI_CORRECTION_CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Debug
AI_CORRECTION_DEBUG_CAPTURE=false
AI_CORRECTION_LOG_LEVEL=INFO
```

## Constants

System-wide constants are defined in `src/config/constants.py`:

### Grading Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `CONFIDENCE_THRESHOLD_AUTO` | 0.85 | Auto-grade threshold |
| `CONFIDENCE_THRESHOLD_FLAG` | 0.60 | Flag for review threshold |
| `GRADE_DIFFERENCE_THRESHOLD` | 0.5 | Difference to trigger verification |
| `ANSWER_SIMILARITY_THRESHOLD` | 0.8 | For answer clustering |

### Processing

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_PARALLEL_COPIES` | 6 | Copies processed in parallel |
| `PDF_DPI` | 200 | DPI for PDF rendering |
| `MAX_PAGE_SIZE` | 10MB | Maximum page image size |

### Retry Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_RETRIES` | 3 | Maximum API retry attempts |
| `RETRY_DELAY_MS` | 1000 | Base retry delay |
| `RETRY_BASE_DELAY` | 1.0 | Exponential backoff base |

## CLI Options

### `correct` command

```
python -m src.main correct [mode] [input] [options]

Modes:
  single    Single-LLM grading
  dual      Dual-LLM verification (default)

Input:
  Path to PDF file or directory

Options:
  --debug           Enable debug output
  --auto-confirm    Skip confirmation prompts
  --batch-size N    Number of copies per batch (default: 10)
  --lang LANG       Language for prompts (fr, en)
```

### `list` command

```
python -m src.main list
```

Lists all sessions with their status.

### `export` command

```
python -m src.main export <session_id> [options]

Options:
  --format FORMAT   Export format (csv, json)
  --output DIR      Output directory
```
