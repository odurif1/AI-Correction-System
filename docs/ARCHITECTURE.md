# Architecture Overview

This document describes the architecture of the AI Correction System.

## System Overview

The AI Correction System is an intelligent grading platform that uses dual-LLM verification to grade student assignments consistently and fairly.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI / API Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   CLI (cli.py)  │  │  API (app.py)   │  │  WebSocket (ws.py)  │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Core Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Session Manager │  │  Models (Pyd)   │  │  Workflow State     │  │
│  │   (session.py)  │  │  (models.py)    │  │ (workflow_state.py) │  │
│  └────────┬────────┘  └─────────────────┘  └─────────────────────┘  │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          AI Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Base Provider   │  │ Gemini Provider │  │ OpenAI Provider     │  │
│  │(base_provider)  │  │(gemini_provider)│  │(openai_provider)    │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│           │                    │                      │             │
│           ▼                    ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Comparison Provider (Dual LLM)                     ││
│  │              (comparison_provider.py)                           ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Storage Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Session Store   │  │   File Store    │  │   Audit Builder     │  │
│  │(session_store)  │  │ (file_store.py) │  │  (audit/builder.py) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Export Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ PDF Annotator   │  │ Data Exporter   │  │ Annotation Service  │  │
│  │(pdf_annotator)  │  │ (analytics.py)  │  │(annotation_service) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Session Management (`src/core/session.py`)

The `GradingSessionOrchestrator` is the main entry point for grading operations:

- Manages the complete grading workflow
- Coordinates between AI providers, storage, and export
- Handles teacher decisions and jurisprudence

### 2. AI Providers (`src/ai/`)

#### Base Provider
Abstract base class defining the provider interface:
- `call_vision()` - Image-based AI calls
- `call_text()` - Text-only AI calls
- `get_embedding()` - Text embeddings for similarity

#### Comparison Provider
Implements dual-LLM verification:
- Sends same request to two different LLMs
- Compares results and detects disagreements
- Resolves conflicts through verification phases

### 3. Storage (`src/storage/`)

#### Session Store
Manages session persistence:
```
data/
├── {session_id}/
│   ├── session.json          # Session state
│   ├── policy.json           # Grading policy
│   ├── cache/                # Analysis cache
│   ├── copies/
│   │   └── {copy_number}/
│   │       ├── original.pdf
│   │       ├── annotation.json
│   │       └── audit.json
│   ├── annotated/            # Annotated PDFs
│   └── reports/              # Export reports
└── _index.json               # Session index
```

### 4. Grading Workflow

The grading process follows these phases:

1. **Initialization** - Load PDFs, detect questions, extract text
2. **Grading** - Dual LLM grading with confidence scoring
3. **Verification** - Cross-verification for disagreements
4. **Ultimatum** - Final resolution for persistent disagreements
5. **Calibration** - Consistency check across all copies
6. **Export** - Generate reports and annotated PDFs

## Dual-LLM Architecture

See [dual_llm_architecture.md](./dual_llm_architecture.md) for details on the dual-LLM verification system.

## Data Flow

```
PDF Upload → Page Extraction → Question Detection →
    ↓
Dual LLM Grading (Parallel)
    ↓
Result Comparison → Agreement?
    ↓                    ↓
   Yes              No → Verification Phase
    ↓                    ↓
Store Results    Ultimatum Phase (if needed)
    ↓                    ↓
Generate Reports ← Final Resolution
    ↓
Annotated PDFs
```

## Key Design Decisions

### Immutable State
`CorrectionState` uses immutable patterns - all modifications return new instances.

### Token Tracking
Running totals are maintained for O(1) token usage queries.

### Atomic Writes
Index updates use file locking and atomic rename for crash safety.

### API Authentication
API key authentication via `X-API-Key` header (optional in development mode).

## Error Handling

- Custom exceptions in `src/core/exceptions.py`
- Automatic retry with exponential backoff for transient errors
- Graceful fallback to heuristic methods on LLM failure
