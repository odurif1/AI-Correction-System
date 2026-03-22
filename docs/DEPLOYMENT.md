# Deployment Guide

This project is ready to deploy as:

- one or more API containers
- one or more worker containers
- one PostgreSQL database
- one shared persistent volume for `AI_CORRECTION_DATA_DIR`

The repository ships with a `docker-compose.yml` stack for that layout.

## 1. Prepare the environment file

Start from `.env.example`:

```bash
cp .env.example .env
```

Minimum production values to set:

```env
AI_CORRECTION_AI_PROVIDER=openai
AI_CORRECTION_OPENAI_API_KEY=...

AI_CORRECTION_ENVIRONMENT=production
AI_CORRECTION_SESSION_SECRET=replace-with-a-long-random-secret
AI_CORRECTION_SESSION_COOKIE_SECURE=true

POSTGRES_DB=ai_correction
POSTGRES_USER=ai_correction
POSTGRES_PASSWORD=replace-me

AI_CORRECTION_DATABASE_URL=postgresql+psycopg://ai_correction:replace-me@db:5432/ai_correction
AI_CORRECTION_CORS_ORIGINS=["https://your-domain.example"]
```

Notes:

- do not use SQLite in staging or production
- the API and worker must share the same `AI_CORRECTION_DATA_DIR`
- terminate TLS in front of the stack, or replace the provided nginx config with your own HTTPS setup

## 2. Build and start the stack

```bash
docker compose up -d --build
```

Services started:

- `db`: PostgreSQL
- `api`: FastAPI + Gunicorn
- `worker`: persistent grading worker
- `nginx`: reverse proxy on port `80`

## 3. Verify the deployment

Health:

```bash
curl http://localhost/health
```

Logs:

```bash
docker compose logs -f api
docker compose logs -f worker
```

Quick worker check:

- create a session
- upload a PDF
- confirm detection
- verify that a job is created and then consumed by the worker

## 4. Scale the services

API instances:

```bash
docker compose up -d --scale api=2
```

Worker instances:

```bash
docker compose up -d --scale worker=2
```

Requirements when scaling:

- keep PostgreSQL shared
- keep the session data volume shared
- keep the same `AI_CORRECTION_SESSION_SECRET` on every container

## 5. Backups

You should back up:

- PostgreSQL
- the persistent session volume mounted as `app-data`

Without the filesystem volume, reports, annotated PDFs, and cached session artifacts are lost.

## 6. Recommended production hardening

- put HTTPS in front of nginx or replace nginx with your existing ingress
- restrict `AI_CORRECTION_CORS_ORIGINS`
- set `AI_CORRECTION_ADMIN_API_KEY` if admin endpoints are exposed
- monitor `/health`, API logs, and worker logs
- rotate `AI_CORRECTION_SESSION_SECRET` only with a coordinated logout plan
