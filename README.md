<h1 align="center">AI Correction System</h1>

<p align="center">
  <strong>Backend open source de correction assistée par IA pour copies PDF</strong>
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
</p>

---

## 🎯 Présentation

**Ce dépôt contient un backend Python pour analyser des copies PDF, orchestrer une correction assistée par LLM, produire des exports et générer des annotations PDF.**

Le correcteur reste responsable de la validation finale. Les modèles servent d’assistants de lecture, de notation et de structuration.

### Pour qui ?

- **Développeurs** qui veulent auditer ou étendre un pipeline de correction
- **Équipes produit ou recherche** travaillant sur l’évaluation de copies PDF
- **Utilisateurs avancés** qui préfèrent un backend scriptable à une interface fermée

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

### 🤖 Correction mono ou dual-LLM

Deux modèles peuvent analyser une même copie indépendamment :
- configuration mono ou dual-LLM
- détection automatique des désaccords
- audit structuré des décisions

### 📊 Exports et audit

- **CSV** : notes et synthèses tabulaires
- **JSON** : audit détaillé de la session
- **PDF annotés** : rendu annoté et overlay de surimpression
- **Analytics** : agrégats de session et statistiques par question

### 🔒 API multi-utilisateur

- Protection optionnelle par clé API
- Sessions de correction persistées côté backend
- Backend utilisable sans couche frontend

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
│  │ API Endpoints│  │  SQLite DB  │  │    WebSocket Progress   │  │
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
├── src/
│   ├── api/                        # FastAPI, auth, websocket, schémas
│   ├── ai/                         # Providers et orchestration LLM
│   ├── analysis/                   # Détection et analyse de documents
│   ├── core/                       # Modèles métier, session, grading
│   ├── db/                         # Persistance SQLite et modèles SQLAlchemy
│   ├── export/                     # Exports analytics et annotation PDF
│   ├── prompts/                    # Prompts et traductions associées
│   ├── storage/                    # Stockage des sessions et fichiers
│   └── vision/                     # Lecture PDF / extraction page par page
├── tests/                          # Tests unitaires et d’intégration ciblés
├── docs/                           # Documentation technique
└── .env.example                    # Exemple de configuration
```

---

## ⚙️ Configuration

### Variables d'environnement

```bash
# Setup minimal
AI_CORRECTION_AI_PROVIDER=<provider>
AI_CORRECTION_<PROVIDER>_API_KEY=<api-key>
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
export AI_CORRECTION_CORS_ORIGINS='["https://app.example.com"]'
```

---

## 🛡️ Sécurité

- ✅ Clé API optionnelle pour restreindre l’accès
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
