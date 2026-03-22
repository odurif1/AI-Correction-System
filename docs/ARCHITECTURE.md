# Architecture Overview

This document describes the runtime architecture actually used by the application.

## System Overview

The production path is split between an HTTP API and a dedicated grading worker:

```text
Browser / CLI
    |
    v
FastAPI API
    |
    +--> Session storage on shared filesystem (`AI_CORRECTION_DATA_DIR`)
    |
    +--> Persistent job queue in SQL (`session_jobs`, `session_job_events`)
              |
              v
        Grading worker
              |
              +--> LLM providers
              +--> session artifacts / reports / annotations
```

## Main Runtime Components

### 1. API process

The API in `src/api/app.py` is responsible for:

- authentication and browser session cookies
- upload, detection, review, export, and admin endpoints
- enqueuing grading jobs
- exposing progress through WebSocket by replaying persisted job events

The API no longer performs grading in memory with `BackgroundTasks`.

### 2. Worker process

The worker in `src/services/job_runner.py` is responsible for:

- claiming queued jobs from the database
- reconstructing the grading session from persisted files
- executing analysis and grading
- writing progress events back to the database
- finalizing token deduction and job status

You can start it with:

```bash
python src/main.py worker --poll-interval 2
```

### 3. Persistent job layer

The queue is defined in `src/db/models.py` and managed by `src/services/job_service.py`.

- `session_jobs` stores the durable job state
- `session_job_events` stores the ordered progress stream

This gives:

- crash recovery
- API/worker separation
- multi-process safety for queued work
- reconnection-friendly WebSocket progress

### 4. Shared filesystem storage

Session artifacts still live under `AI_CORRECTION_DATA_DIR`:

```text
data/
└── sessions/
    └── {user_id}/
        └── {session_id}/
            ├── session.json
            ├── policy.json
            ├── cache/
            ├── copies/
            ├── annotated/
            ├── overlays/
            ├── reports/
            └── debug/
```

The API and the worker must therefore share the same persistent volume.

## Grading Flow

1. The API receives uploads and stores session metadata.
2. Detection runs synchronously from the API.
3. `/confirm-detection` or `/grade` creates a durable grading job.
4. The worker claims the job and runs the grading pipeline.
5. Progress events are persisted in SQL.
6. WebSocket clients replay the event stream and stay in sync after reconnects.
7. Reports and annotated outputs are written into the shared session directory.

## Deployment Shape

For a serious deployment, the minimum topology is:

- 1+ API instances
- 1+ worker instances
- 1 shared SQL database, preferably PostgreSQL
- 1 shared persistent volume for `AI_CORRECTION_DATA_DIR`
- HTTPS in front of the API

SQLite is acceptable for development, not for staging/production.
