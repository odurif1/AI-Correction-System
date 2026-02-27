---
phase: 03-core-grading-experience
plan: 02
subsystem: api
tags: [websocket, progress-events, grading-mode, fastapi]

# Dependency graph
requires:
  - phase: 03-01
    provides: [pre-analysis, multi-scale detection, session confirmation]
provides:
  - Grading mode selection (single/dual LLM) via POST /grade endpoint
  - WebSocket progress events with typed event structure
  - Progress sync on WebSocket reconnection
affects: [03-03, 03-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Progress event constants for type-safe WebSocket communication
    - Request body for grading mode configuration
    - Progress sync on connect for reconnection support

key-files:
  created: []
  modified:
    - src/api/app.py - Grade endpoint with mode selection, WebSocket progress events
    - src/api/schemas.py - StartGradingRequest, updated GradeResponse

key-decisions:
  - "Single mode uses force_single_llm parameter for faster grading"
  - "Dual mode (default) uses comparison mode from settings for verification"
  - "Progress events defined as constants for type safety and documentation"

patterns-established:
  - "Progress event constants: PROGRESS_EVENT_* prefix for all event types"
  - "Progress sync on connect: Send current state immediately when client connects"
  - "Request body pattern: Use Pydantic schema for POST endpoint configuration"

requirements-completed: [GRAD-03, GRAD-04, GRAD-05, GRAD-06]

# Metrics
duration: 1min
completed: 2026-02-27
---

# Phase 3 Plan 2: Grading Execution Summary

**Single/dual LLM grading mode selection with WebSocket progress events and reconnection support**

## Performance

- **Duration:** 1 min (102 seconds)
- **Started:** 2026-02-27T17:54:13Z
- **Completed:** 2026-02-27T17:55:55Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Implemented grading mode selection (single for speed, dual for verification) at `/api/sessions/{id}/grade` endpoint
- Defined WebSocket progress event structure with typed constants (copy_start, question_done, copy_done, copy_error, session_complete, session_error)
- Added progress sync event on WebSocket connect for seamless reconnection after network issues

## Task Commits

Each task was committed atomically:

1. **Task 1: Add grading mode selection to grade endpoint** - `98e1561` (feat)
2. **Task 2: Define WebSocket progress event structure** - `87dde9d` (feat)
3. **Task 3: Add progress sync for WebSocket reconnection** - `a20db5c` (feat)

**Plan metadata:** (to be added in final commit)

## Files Created/Modified

- `src/api/schemas.py` - Added StartGradingRequest schema with grading_mode field (Literal["single", "dual"]), updated GradeResponse to include grading_mode
- `src/api/app.py` - Added progress event constants (PROGRESS_EVENT_*), updated start_grading to accept request body, modified grade_task to use constants, added progress_sync on WebSocket connect

## Decisions Made

- **Single vs Dual mode mapping**: Single mode sets `force_single_llm=True` on orchestrator, dual mode uses default comparison mode from settings
- **Progress event structure**: Defined 6 event types with consistent data fields (copy_start, question_done, copy_done, copy_error, session_complete, session_error)
- **Reconnection handling**: Send progress_sync event immediately on WebSocket connect with current session state (status, copies_uploaded, copies_graded, grading_mode)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Grading endpoint accepts single/dual mode selection and passes to orchestrator via force_single_llm parameter
- WebSocket sends 6 event types with consistent data structure for progress tracking
- Reconnecting WebSocket receives progress_sync immediately with current state
- Ready for Phase 3 Plan 03 (grading results and review)

---
*Phase: 03-core-grading-experience*
*Completed: 2026-02-27*
