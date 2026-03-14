<h1 align="center">AI Correction System</h1>

<p align="center">
  <strong>Correction automatique de copies par IA pour les professeurs de collège et lycée</strong>
</p>

<p align="center">
  <a href="#fonctionnalités">Fonctionnalités</a> •
  <a href="#démarrage-rapide">Démarrage</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#licence">Licence</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/status-production%20ready-brightgreen.svg" alt="Production Ready">
</p>

---

## 🎯 Présentation

**Cette application automatise une partie du travail de correction tout en laissant la décision finale au correcteur.**

Le correcteur, c'est vous. L'IA sert d'assistant de correction.

Gagnez 90% de temps sur la correction de vos copies. Deux IA analysent chaque copie indépendamment avec une précision de 95%. Les désaccords sont automatiquement détectés pour vous permettre de trancher.

### Pour qui ?

- **Professeurs de collège et lycée** en France
- **Correcteurs** préparant le bac, le brevet, ou des examens
- **Établissements** cherchant à optimiser leurs corrections

### Comment ça marche ?

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Uploadez   │ ──▶ │  Diagnostic │ ──▶ │    2 IA     │ ──▶ │  Exportez   │
│  vos PDF    │     │   automat.  │     │ corrigent   │     │  vos notes  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ Désaccords  │
                                       │  signalés   │
                                       └─────────────┘
```

---

## ✨ Fonctionnalités

### 🤖 Double Validation IA

Deux modèles d'IA différents analysent chaque copie indépendamment :
- **Gemini** (Google) + **GPT-4** (OpenAI)
- Détection automatique des désaccords
- Interface de résolution intuitive

### 📊 Analytics en temps réel

- Distribution des notes
- Statistiques par question
- Détection des réponses atypiques
- Score moyen, écart-type, médiane

### 📤 Exports complets

- **CSV** : Tableau des notes importable dans Excel/Pronote
- **JSON** : Données complètes avec audit
- **PDF annotés** : Copies avec corrections et feedback

### 🔒 API multi-utilisateur

- Comptes utilisateurs individuels
- Données isolées par utilisateur
- Authentification JWT

---

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.10+
- Clés API Gemini et/ou OpenAI

### Installation

```bash
# Cloner le repository
git clone https://github.com/<organization>/<repository>.git
cd <repository>

# Backend Python
pip install -r requirements.txt
cp .env.example .env
# Éditer .env avec vos clés API
```

### Lancement

```bash
# Backend API
python src/main.py api --port 8000
```

---

## 💻 Interface CLI

L'application peut également être utilisée en ligne de commande pour une correction rapide.

### Commande de base

```bash
python -m src.main correct [options] <pdfs...>
```

### Options disponibles

| Option | Défaut | Description |
|--------|--------|-------------|
| `--mode {single,dual}` | `single` | Mode de correction (1 LLM ou double validation) |
| `--grading-mode {individual,batch,hybrid}` | `batch` | Mode de notation |
| `--pre-analysis` | activé | Détection automatique du barème |
| `--no-pre-analysis` | - | Désactive la pré-analyse (saisie manuelle requise) |
| `--chat-continuation` | activé | Context caching pour économiser les tokens (Gemini) |
| `--no-chat-continuation` | - | Désactive le context caching |
| `--auto-confirm` | désactivé | Confirme automatiquement le barème détecté |
| `--language {fr,en}` | `fr` | Langue des prompts |
| `--debug` | désactivé | Active le mode debug (logs détaillés) |

### Exemples

```bash
# Correction standard avec détection automatique du barème
python -m src.main correct devoir.pdf

# Double validation IA pour plus de précision
python -m src.main correct --mode dual devoir.pdf

# Sans pré-analyse (saisie manuelle du barème)
python -m src.main correct --no-pre-analysis devoir.pdf

# Mode automatique (sans confirmation)
python -m src.main correct --auto-confirm devoir.pdf

# Debug pour voir les appels API
python -m src.main correct --debug devoir.pdf
```

### Rapport de tokens

À la fin de chaque session, un rapport détaillé de l'utilisation des tokens est affiché :

```
📊 Token Usage par Phase:
  Détection: 1,234 tokens
  Correction: 45,678 tokens
  Vérification: 12,345 tokens
  ─────────────────────────
  Total: 59,257 tokens
```

---

## 🏗️ Architecture

### Stack Technique

```
┌─────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Auth JWT  │  │  SQLite DB  │  │    WebSocket Progress   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AI PROVIDERS                            │
│  ┌─────────────────────┐          ┌─────────────────────┐       │
│  │ Provider 1          │          │ Provider 2          │       │
│  │ configured model    │          │ configured model    │       │
│  └─────────────────────┘          └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Structure du Code

```
project/
├── 📁 src/                         # Backend Python
│   ├── 📁 api/                     # API REST
│   │   ├── app.py                  # Application FastAPI
│   │   ├── auth.py                 # Routes d'authentification
│   │   ├── schemas.py              # Modèles Pydantic
│   │   └── websocket.py            # WebSocket manager
│   │
│   ├── 📁 db/                      # Base de données
│   │   ├── database.py             # Connexion SQLite
│   │   └── models.py               # Modèles SQLAlchemy
│   │
│   ├── 📁 ai/                      # Providers LLM
│   │   ├── gemini_provider.py      # Google Gemini
│   │   ├── openai_provider.py      # OpenAI
│   │   ├── comparison_provider.py  # Dual LLM
│   │   └── batch_grader.py         # Correction batch
│   │
│   ├── 📁 analysis/                # Diagnostic PDF
│   │   ├── pre_analysis.py         # Analyseur de documents
│   │   ├── pre_analysis_prompts.py # Prompts de diagnostic
│   │   └── pre_analysis_translations.py  # Traductions FR/EN
│   │
│   ├── 📁 core/                    # Cœur métier
│   │   ├── models.py               # Modèles Pydantic
│   │   ├── session.py              # Orchestrateur
│   │   └── workflow.py             # Workflow de correction
│   │
│   ├── 📁 storage/                 # Stockage
│   │   └── file_store.py           # Gestion fichiers
│   │
│   ├── 📁 export/                  # Export
│   │   ├── analytics.py            # Rapports
│   │   └── pdf_annotator.py        # Annotation PDF
│   │
│   └── 📁 prompts/                 # Prompts IA
│       ├── grading.py              # Prompts de correction
│       └── batch.py                # Prompts batch
│
├── 📁 data/                        # Données utilisateur
│   └── {user_id}/                  # Isolation par utilisateur
│       └── {session_id}/           # Sessions de correction
│
└── 📄 .env                         # Configuration
```

---

## ⚙️ Configuration

### Variables d'environnement

```bash
# Setup minimal
AI_CORRECTION_AI_PROVIDER=<provider>
AI_CORRECTION_<PROVIDER>_API_KEY=<api-key>
AI_CORRECTION_JWT_SECRET=<long-random-secret>

# Modèles optionnels
# AI_CORRECTION_<PROVIDER>_MODEL=<configured-model>
# AI_CORRECTION_<PROVIDER>_VISION_MODEL=<configured-vision-model>

# Double LLM optionnel
AI_CORRECTION_COMPARISON_MODE=true
# AI_CORRECTION_LLM1_PROVIDER=<provider-1>
# AI_CORRECTION_LLM1_MODEL=<model-1>
# AI_CORRECTION_LLM2_PROVIDER=<provider-2>
# AI_CORRECTION_LLM2_MODEL=<model-2>

# Observabilité optionnelle
# AI_CORRECTION_SENTRY_DSN=
```

Consultez aussi `.env.example` pour le template complet à jour.

---

## 🔄 Workflow de Correction

```
┌─────────────────────────────────────────────────────────────────────┐
│  WORKFLOW DE CORRECTION (6 phases)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: DÉTECTION                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Upload du PDF                                                │  │
│  │ - Diagnostic automatique (type de document, structure)         │  │
│  │ - Détection du barème et du nombre d'élèves                    │  │
│  │ - ⚠️ CONFIRMATION: Le barème est FIGÉ définitivement           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 2: GRADING (Double IA)                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Les 2 LLM notent toutes les questions en parallèle          │  │
│  │ - Le barème figé est utilisé (pas de re-détection)            │  │
│  │ - Détection des désaccords (notes et lectures)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 3: VERIFICATION                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Cross-verification (les LLMs voient le travail de l'autre)  │  │
│  │ - Ultimatum si désaccord persiste                              │  │
│  │ - Barème reste figé (pas de négociation)                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 4: CALIBRATION                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Vérification de cohérence entre copies                       │  │
│  │ - Détection des incohérences                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 5: EXPORT                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Export CSV, JSON                                             │  │
│  │ - Analytics et statistiques                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  Phase 6: ANNOTATION (optionnel)                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ - Génération des PDFs annotés                                  │  │
│  │ - Feedback individualisé                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

### 💡 Barème Figé (Grading Scale)

Une fois le diagnostic confirmé (Phase 1), **le barème est figé** et ne peut plus changer:
- ✅ **Simplification** : Les LLMs ne détectent plus le barème pendant la correction
- ✅ **Cohérence** : Toutes les copies sont notées sur le même barème
- ✅ **Pas de désaccord de barème** : Les LLMs se concentrent uniquement sur les notes et lectures
- ✅ **Barème fourni dans les prompts** : Les LLMs reçoivent le barème figé explicitement
```

---

## 🔌 API REST

### Authentification

```http
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

### Sessions

```http
POST   /api/sessions                        # Créer une session
GET    /api/sessions                        # Lister les sessions
GET    /api/sessions/{id}                   # Détails d'une session
DELETE /api/sessions/{id}                   # Supprimer une session
POST   /api/sessions/{id}/upload            # Uploader des PDFs
POST   /api/sessions/{id}/pre-analyze       # Diagnostic du PDF (structure, barème)
POST   /api/sessions/{id}/confirm-pre-analysis  # Confirmer le diagnostic
POST   /api/sessions/{id}/grade             # Lancer la correction
```

### WebSocket

Utilisez l'endpoint `/api/sessions/{id}/ws` pour le suivi temps réel.

---

## 🧪 Développement

### Tests

```bash
# Tests unitaires
pytest tests/

# Tests avec couverture
pytest tests/ --cov=src --cov-report=html
```

### Qualité du code

```bash
# Formatage
black src/ && isort src/

# Type checking
mypy src/

# Linting
ruff check src/
```

### Déploiement

```bash
# Variables d'environnement production
export AI_CORRECTION_JWT_SECRET="production-secret-key"
export AI_CORRECTION_CORS_ORIGINS='["https://app.example.com"]'
```

---

## 📊 Performance

### Temps de correction

| Copies | Temps estimé | Appels API |
|--------|--------------|------------|
| 10 | ~2 min | 2-4 |
| 50 | ~5 min | 2-4 |
| 100 | ~10 min | 2-4 |

### Économies réalisées

| Scénario | Sans l'outil | Avec l'outil |
|----------|-------------------|-------------------|
| 100 copies | ~10h | ~15 min |
| Bac blanc (120 copies) | ~12h | ~20 min |

---

## 🛡️ Sécurité

- ✅ Authentification JWT avec expiration
- ✅ Mots de passe hachés (bcrypt)
- ✅ Isolation des données par utilisateur
- ✅ Protection contre path traversal
- ✅ Limite de taille des fichiers (50 MB)
- ✅ Validation des entrées utilisateur

---

## 📚 Documentation

- [Architecture Dual LLM](docs/dual_llm_architecture.md)
- [Module d'annotation PDF](docs/annotation.md)
- [Structure de l'audit](docs/AUDIT_STRUCTURE.md)

---

## 🤝 Contribution

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git switch -c feature/amelioration`)
3. Commit (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

---
