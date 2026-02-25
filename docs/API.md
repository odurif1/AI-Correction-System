# API Documentation

This document describes the REST API endpoints for the AI Correction System.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints (except `/health`) require an API key when `AI_CORRECTION_API_KEY` is set in the environment.

```http
X-API-Key: your-api-key
```

If no API key is configured, the API operates in development mode without authentication.

## Endpoints

### Health Check

```http
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Create Session

```http
POST /api/sessions
```

Create a new grading session.

**Request Body:**
```json
{
  "subject": "Mathematics",
  "topic": "Algebra",
  "question_weights": {
    "Q1": 5.0,
    "Q2": 3.0
  }
}
```

**Response:**
```json
{
  "session_id": "abc12345",
  "status": "created",
  "created_at": "2024-01-15T10:30:00",
  "copies_count": 0,
  "graded_count": 0,
  "subject": "Mathematics",
  "topic": "Algebra"
}
```

---

### Upload Copies

```http
POST /api/sessions/{session_id}/upload
```

Upload PDF copies to a session.

**Request:** `multipart/form-data` with PDF files

**Response:**
```json
{
  "session_id": "abc12345",
  "uploaded_count": 3,
  "paths": ["/temp/abc12345/file1.pdf", "/temp/abc12345/file2.pdf"]
}
```

---

### Get Session

```http
GET /api/sessions/{session_id}
```

Get detailed session information.

**Response:**
```json
{
  "session_id": "abc12345",
  "status": "grading",
  "created_at": "2024-01-15T10:30:00",
  "copies_count": 10,
  "graded_count": 5,
  "average_score": 14.5,
  "subject": "Mathematics",
  "topic": "Algebra",
  "copies": [...],
  "graded_copies": [...],
  "question_weights": {"Q1": 5.0, "Q2": 3.0}
}
```

---

### List Sessions

```http
GET /api/sessions
```

List all sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "abc12345",
      "status": "complete",
      "created_at": "2024-01-15T10:30:00",
      "copies_count": 10,
      "graded_count": 10,
      "average_score": 14.5,
      "subject": "Mathematics"
    }
  ]
}
```

---

### Delete Session

```http
DELETE /api/sessions/{session_id}
```

Delete a session and all its data.

**Response:**
```json
{
  "success": true
}
```

---

### Start Grading

```http
POST /api/sessions/{session_id}/grade
```

Start the grading process for a session.

**Response:**
```json
{
  "success": true,
  "session_id": "abc12345",
  "graded_count": 0,
  "total_count": 10,
  "pending_review": 0
}
```

Progress updates are sent via WebSocket at `/api/sessions/{session_id}/ws`.

---

### Submit Teacher Decision

```http
POST /api/sessions/{session_id}/decisions
```

Submit a teacher's grading decision for a specific question.

**Request Body:**
```json
{
  "copy_id": "copy_1",
  "question_id": "Q1",
  "teacher_guidance": "Accept partial credit for method",
  "original_score": 2.0,
  "new_score": 3.5
}
```

**Response:**
```json
{
  "success": true,
  "updated_count": 3,
  "extracted_rule": "IF student shows correct method with calculation error THEN award partial credit"
}
```

---

### Get Disagreements

```http
GET /api/sessions/{session_id}/disagreements
```

Get all LLM disagreements for a session.

**Response:**
```json
[
  {
    "copy_id": "copy_1",
    "copy_index": 1,
    "student_name": "John Doe",
    "question_id": "Q2",
    "max_points": 5.0,
    "llm1": {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "grade": 3.0,
      "confidence": 0.85,
      "reasoning": "Partial understanding shown",
      "reading": "3.5"
    },
    "llm2": {
      "provider": "openai",
      "model": "gpt-4o",
      "grade": 4.5,
      "confidence": 0.90,
      "reasoning": "Mostly correct with minor errors",
      "reading": "4.5"
    },
    "resolved": true
  }
]
```

---

### Resolve Disagreement

```http
POST /api/sessions/{session_id}/disagreements/{question_id}/resolve
```

Manually resolve a disagreement.

**Request Body:**
```json
{
  "action": "accept_llm1",
  "final_grade": 3.0,
  "reasoning": "LLM1 more accurately reflects student understanding"
}
```

**Response:**
```json
{
  "success": true,
  "question_id": "Q2",
  "action": "accept_llm1"
}
```

---

### Get Analytics

```http
GET /api/sessions/{session_id}/analytics
```

Get analytics for a graded session.

**Response:**
```json
{
  "total_copies": 10,
  "graded_copies": 10,
  "average_score": 14.5,
  "median_score": 15.0,
  "std_dev": 2.3,
  "score_distribution": {
    "0-5": 1,
    "5-10": 2,
    "10-15": 3,
    "15-20": 4
  },
  "question_stats": {
    "Q1": {"mean": 3.5, "median": 4.0, "difficulty": 0.3},
    "Q2": {"mean": 2.0, "median": 2.0, "difficulty": 0.6}
  },
  "common_errors": ["Calculation error", "Unit conversion"],
  "exceptional_answers": ["Student 3 showed creative approach"]
}
```

---

### Export Session

```http
GET /api/sessions/{session_id}/export/{format}
```

Export session results.

**Formats:** `csv`, `json`

**Response:** File download

---

### Get Settings

```http
GET /api/settings
```

Get current application settings.

**Response:**
```json
{
  "ai_provider": "gemini",
  "comparison_mode": true,
  "llm1_provider": "gemini",
  "llm1_model": "gemini-2.5-flash",
  "llm2_provider": "openai",
  "llm2_model": "gpt-4o",
  "confidence_auto": 0.85,
  "confidence_flag": 0.60
}
```

---

### Update Settings

```http
PUT /api/settings
```

Update application settings (runtime only, not persistent).

**Request Body:**
```json
{
  "comparison_mode": false,
  "confidence_auto": 0.90
}
```

---

### List Providers

```http
GET /api/providers
```

List available LLM providers based on configuration.

**Response:**
```json
[
  {
    "id": "gemini",
    "name": "Google Gemini",
    "type": "gemini",
    "models": [
      {"id": "gemini-2.5-flash", "name": "gemini-2.5-flash"}
    ],
    "configured": true
  },
  {
    "id": "openai",
    "name": "OpenAI",
    "type": "openai",
    "models": [
      {"id": "gpt-4o", "name": "gpt-4o"}
    ],
    "configured": true
  }
]
```

---

## WebSocket

### Progress Updates

Connect to `/api/sessions/{session_id}/ws` for real-time progress updates.

**Events:**
- `copy_start` - Started grading a copy
- `question_done` - Finished grading a question
- `copy_done` - Finished grading a copy
- `copy_error` - Error grading a copy
- `session_complete` - Session grading complete
- `session_error` - Session error

**Message Format:**
```json
{
  "event": "question_done",
  "data": {
    "question_id": "Q1",
    "grade": 4.5,
    "max_points": 5.0,
    "confidence": 0.92
  }
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

**Common Status Codes:**
- `400` - Bad Request
- `401` - Unauthorized (invalid API key)
- `404` - Not Found
- `413` - Payload Too Large
- `500` - Internal Server Error
