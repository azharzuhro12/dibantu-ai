# DibantuAI Frontend

Next.js (App Router) + TypeScript + Tailwind CSS dashboard for
[DibantuAI](../README.md), the AI agent for business operations.

The frontend talks **only** to the existing FastAPI backend — no
authentication, no database, no extra LLM APIs, and no invented
endpoints.

## Pages

| Page | Backend calls |
| --- | --- |
| `/chat` | `POST /api/chat` |
| `/inventory` | `POST /api/chat` (no dedicated inventory endpoint yet) |
| `/orders` | `POST /api/chat` (no dedicated orders endpoint yet) |
| `/reports` | `POST /api/chat` (no dedicated report endpoint yet) |
| `/approvals` | `GET /api/approvals`, `POST /api/approvals/{id}/approve`, `POST /api/approvals/{id}/reject` |
| header badge | `GET /health` (polled) |

Inventory, Orders, and Reports intentionally show assistant-backed
answers and clearly-labeled placeholders instead of fabricated
business data — those endpoints do not exist on the backend yet.

## Getting started

```bash
# Backend first (from the repository root):
docker compose up          # or: uvicorn app.main:app --reload --port 8090

# Then the frontend:
npm install
npm run dev
```

Open http://localhost:3000 (backend on http://localhost:8090, Swagger
on http://localhost:8090/docs).

## Configuration

Copy the example env file and point it at your backend:

```bash
cp .env.local.example .env.local
```

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8090` | Base URL of the FastAPI backend (Docker maps host 8090 -> container 8000). |

The backend must allow the frontend origin via CORS (default allows
`http://localhost:3000`; see `CORS_ALLOW_ORIGINS` in the root README).

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the dev server. |
| `npm run build` | Production build. |
| `npm run start` | Serve the production build. |
| `npm run lint` | ESLint. |

## Structure

```
src/
├── app/            # Routes: /, /chat, /inventory, /orders, /reports, /approvals
├── components/     # AppShell (sidebar + header), views, shared UI
├── hooks/          # useBackendHealth, usePendingApprovalCount
└── lib/            # api.ts (single API client + types), format.ts
```

Every network call goes through `src/lib/api.ts`, which reads
`NEXT_PUBLIC_API_URL` — the backend URL is never hardcoded elsewhere.
