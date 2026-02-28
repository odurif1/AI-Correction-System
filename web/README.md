# AI Correction System - Web Frontend

> **Partie de**: [AI Correction System](../README.md)

Interface web moderne pour l'AI Correction System, construite avec Next.js 16, React 19, et Tailwind CSS v4.

## Fonctionnalités

- **Dashboard** - Visualisation et gestion des sessions de correction avec recherche, filtres et actions groupées
- **Gestion des Sessions** - Création de sessions, upload de PDFs, suivi de progression
- **Progression Temps Réel** - Intégration WebSocket pour les mises à jour live
- **Résolution de Désaccords** - Comparaison côte-à-côte pour les conflits de notes LLM
- **Analytics** - Graphiques de distribution des scores et statistiques
- **Export** - Téléchargement des résultats en CSV ou JSON
- **Mode Sombre** - Support complet des thèmes
- **Responsive** - Interface mobile-friendly avec vues carte/grille

## Stack Technique

| Technologie | Version | Usage |
|-------------|---------|-------|
| Next.js | 16.1.6 | Framework React avec App Router |
| React | 19.2.4 | Bibliothèque UI |
| TypeScript | 5.9.3 | Type safety |
| Tailwind CSS | 4.2.0 | Styling |
| shadcn/ui | - | Composants UI |
| React Query | 5.90.21 | Data fetching |
| Sonner | 2.0.7 | Notifications toast |
| Recharts | 3.7.0 | Graphiques |
| Zod | 4.3.6 | Validation de formulaires |

## Prérequis

- Node.js 20+
- npm ou pnpm
- Backend API fonctionnant sur le port 8000

## Installation

```bash
# Installer les dépendances
npm install

# Serveur de développement
npm run dev

# Build production
npm run build
npm start
```

### Variables d'environnement

Créer `.env.local` pour une configuration personnalisée:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Structure du Projet

```
web/
├── app/                      # Next.js App Router
│   ├── layout.tsx            # Layout racine avec providers
│   ├── page.tsx              # Page d'accueil
│   ├── dashboard/            # Page dashboard
│   ├── sessions/
│   │   ├── new/              # Créer nouvelle session
│   │   └── [id]/             # Détail session
│   └── settings/             # Page paramètres
│
├── components/
│   ├── ui/                   # Composants shadcn/ui
│   ├── grading/              # Composants spécifiques grading
│   ├── layout/               # Header, Footer
│   ├── pdf-preview.tsx       # Dialog preview PDF
│   ├── export-button.tsx     # Export avec progression
│   ├── empty-states.tsx      # États vides/erreur
│   └── loading-skeletons.tsx # Placeholders de chargement
│
├── lib/
│   ├── api.ts                # Client API
│   ├── types.ts              # Types TypeScript
│   ├── utils.ts              # Fonctions utilitaires
│   ├── validations.ts        # Schémas Zod
│   └── websocket.ts          # Hook WebSocket
│
├── hooks/                    # Hooks React personnalisés
│
└── public/                   # Assets statiques
```

## Endpoints API

Le frontend communique avec l'API backend:

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/sessions` | GET, POST | Lister/créer sessions |
| `/api/sessions/:id` | GET, DELETE | Récupérer/supprimer session |
| `/api/sessions/:id/upload` | POST | Uploader PDFs |
| `/api/sessions/:id/grade` | POST | Démarrer correction |
| `/api/sessions/:id/analytics` | GET | Obtenir analytics |
| `/api/sessions/:id/disagreements` | GET | Lister désaccords |
| `/api/sessions/:id/export/:format` | GET | Export (csv/json) |
| `/api/sessions/:id/ws` | WebSocket | Progression temps réel |
| `/api/providers` | GET | Lister providers LLM |
| `/api/settings` | GET, PUT | Paramètres app |

## Composants

### Composants de Correction

- **DisagreementCard** - Comparaison côte-à-côte LLM avec options de résolution
- **ProgressGrid** - Visualisation progression correction temps réel
- **ScoreDistribution** - Histogramme de distribution des scores
- **SessionStatus** - Badge de statut avec icônes
- **FileUploader** - Upload PDF drag & drop
- **LLMGradeCard** - Affichage note LLM individuelle

### Composants UI (shadcn/ui)

```
Button, Card, Input, Label, Badge, Progress, Tabs,
Table, Dialog, AlertDialog, Checkbox, DropdownMenu,
Skeleton, ScrollArea
```

## Développement

### Scripts

```bash
npm run dev      # Démarrer serveur développement
npm run build    # Build production
npm run start    # Démarrer serveur production
npm run lint     # Exécuter ESLint
```

### Style de Code

- TypeScript strict mode activé
- ESLint avec config Next.js
- Composants fonctionnels avec hooks
- Colocaliser composants avec features

## Déploiement

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Vercel

```bash
# Installer CLI Vercel
npm i -g vercel

# Déployer
vercel
```

## Support Navigateurs

- Chrome/Edge 111+
- Firefox 111+
- Safari 16.4+

## Licence

MIT

---

## Voir aussi

- [README principal](../README.md) - Documentation complète du projet
- [Architecture Dual LLM](../docs/dual_llm_architecture.md) - Architecture détaillée
- [Annotation PDF](../docs/annotation.md) - Module d'annotation
- [Structure Audit](../docs/AUDIT_STRUCTURE.md) - Format de l'audit
