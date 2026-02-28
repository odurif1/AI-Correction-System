# La Corrigeuse

## What This Is

La Corrigeuse est un assistant de correction automatique pour enseignants. Elle lit les copies manuscrites (PDF) et les corrige via LLM avec double-vérification IA. Le prof upload, l'IA corrige, le prof valide et exporte vers Pronote ou Excel.

Produit commercial destiné aux professeurs français, avec une interface web moderne.

## Core Value

**Qualité des corrections.** Si tout le reste échoue, les notes doivent être justes, fiables et consistantes. La confiance du prof dans les résultats prime sur la vitesse.

## Requirements

### Validated

Fonctionnalités livrées en v1.0 MVP (2026-02-28) :

**Security & Auth:**
- ✓ JWT secret from environment with startup validation — v1.0
- ✓ Multi-tenant data isolation (user-scoped sessions) — v1.0
- ✓ File upload security (PDF validation, size limits) — v1.0
- ✓ Rate limiting, CORS, security headers — v1.0
- ✓ Password reset via email (SendGrid) — v1.0

**Observability:**
- ✓ Structured JSON logging with correlation IDs — v1.0
- ✓ Sentry error tracking integration — v1.0
- ✓ Health check endpoint (/health) — v1.0
- ✓ Request and business metrics — v1.0

**Core Grading:**
- ✓ Multi-PDF upload with automatic scale detection — v1.0
- ✓ LLM vision reads handwritten content directly — v1.0
- ✓ Single and dual-LLM grading modes — v1.0
- ✓ Real-time WebSocket progress updates — v1.0
- ✓ Grade review with inline editing — v1.0
- ✓ Multi-format export (CSV, JSON, Excel) — v1.0

**UI:**
- ✓ Dashboard with session cards and quick actions — v1.0
- ✓ Multi-file upload workflow with drag-drop — v1.0
- ✓ Grading progress screen with French messages — v1.0
- ✓ Review screen with inline editing — v1.0
- ✓ Responsive design (tablets, laptops) — v1.0

**Production:**
- ✓ Docker multi-stage build with nginx — v1.0
- ✓ Security scanning (pip-audit, bandit) — v1.0
- ✓ Token cost tracking with prompt caching — v1.0
- ✓ Stripe subscription webhooks — v1.0

### Active

Objectifs pour v1.1 :

- [ ] Docker deployment (blocked: PyMuPDF Python 3.13+ support pending)
- [ ] Calibration across copies (GRAD-07)
- [ ] Error states with user-friendly messages (UI-07)
- [ ] CI/CD pipeline (OPS-03 - deferred)

### Out of Scope

- Mobile app — web-first, mobile plus tard
- OAuth/Social login — email/password suffisant pour v1
- Real-time chat support — pas core
- Video content — stockage/bandwidth, defer
- Free tier — AI costs non-trivial; trial instead
- Model tiering (COST-03) — single model per user decision

## Context

**v1.0 MVP shipped** with 34k Python LOC, 100k TypeScript LOC.

Architecture en couches : Core (session orchestration), AI providers (Gemini, OpenAI, GLM), Grading, Vision (PDF), Storage, API, Export.

**Known Issues:**
- Docker deployment blocked by PyMuPDF Python 3.13+ compatibility
- 11 transitive dependency CVEs identified (pip-audit)
- 2 LOW bandit findings (MD5 usage in non-crypto context)

Stack : Python 3.14, FastAPI, Pydantic, SQLAlchemy, Next.js 16, TailwindCSS, Radix UI, Docker, nginx, Stripe.

## Constraints

- **Timeline** : v1.0 MVP shipped 2026-02-28 (12 days)
- **Standards** : Produit commercial = qualité prod, pas prototype
- **Sécurité** : Audit complet obligatoire avant mise en ligne
- **Budget tokens** : Optimiser les coûts LLM pour rester rentable
- **Licence** : PyMuPDF en AGPL — vérifier compatibilité SaaS

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Dual-LLM comparison | Fiabilité accrue par vérification croisée | ✓ Good |
| Vision directe (pas OCR) | LLM lit directement le manuscrit | ✓ Good |
| SQLite pour auth | Simple, suffisant pour début | ✓ Good |
| Next.js standalone | Déploiement simple | ✓ Good |
| Cost display AFTER grading | User decision - avoid friction | ✓ Good |
| Single model (no tiering) | User decision - simplify UX | ✓ Good |
| CI/CD deferred | Local scanning sufficient for pilot | ⚠️ Revisit for scale |
| Prompt caching enabled | Reduce costs 50-90% | ✓ Good |

---
*Last updated: 2026-02-28 after v1.0 MVP milestone*
