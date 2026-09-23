# DibantuAI

**AI Agent for Business Operations** — asisten AI untuk operasional bisnis kecil (warung, kafe, toko): cek stok, buat pesanan, catat customer, dan laporan penjualan, semuanya lewat percakapan.

Masalah yang dipecahkan: pemilik usaha kecil tidak butuh aplikasi kasir kompleks — mereka butuh asisten yang bisa diajak chat dan langsung mengoperasikan data bisnis mereka dengan aman (stok tidak pernah bocor/oversell, aksi sensitif butuh persetujuan manusia).

## Fitur Utama

- **Chat agent dengan tool calling** — agent memanggil business tools (stok, order, customer, laporan) lewat loop GLM; tidak pernah mengarang data bisnis.
- **PostgreSQL persistence** — data bisnis tersimpan di Postgres (SQLAlchemy + Alembic migration); restart container tidak menghilangkan data.
- **Human-in-the-loop approval** — aksi sensitif (refund, cancel order, bulk stock update) tidak dieksekusi agent; menunggu persetujuan manusia lewat API/frontend.
- **WhatsApp webhook simulator** — endpoint webhook ber-format WhatsApp untuk pengujian lokal (bukan integrasi Meta sungguhan).
- **Dashboard Next.js** — halaman chat, inventory, orders, reports, dan approvals.
- **Evaluation suite deterministik** — 34 skenario offline yang memverifikasi tool selection, task completion, error handling, dan grounding jawaban.

## Arsitektur & Alur Agent

```
Pesan user → FastAPI /api/chat → Agent loop
  → GLM (endpoint Anthropic-compatible z.ai, model glm-5.3)
  → tool_use → registry → business tools → PostgreSQL
  → tool_result → GLM → jawaban final
```

- Agent loop maksimal 5 eksekusi tool per request (mencegah loop tanpa batas).
- Tool yang tidak dikenal atau gagal mengembalikan `{"error": ...}` ke model — bukan crash.
- Aksi sensitif diintersepsi **sebelum** eksekusi dan menjadi pending approval.
- Jawaban di-grounding ke hasil tool: skenario evaluasi memverifikasi setiap fakta dalam jawaban benar-benar berasal dari tool result.

## Tech Stack

| Komponen | Teknologi |
| --- | --- |
| Bahasa | Python 3.10+ / TypeScript |
| Backend | FastAPI + Uvicorn + Pydantic |
| LLM | GLM API (endpoint Anthropic-compatible z.ai, model `glm-5.3`) — custom agent loop, tanpa LangChain/LangGraph |
| Database | PostgreSQL 16 + SQLAlchemy 2 + Alembic |
| Frontend | Next.js (App Router) + TypeScript + Tailwind CSS |
| Infra | Docker / Docker Compose |
| Testing | pytest + evaluation suite deterministik |

## Menjalankan dengan Docker (cara utama)

```bash
cp .env.example .env
# isi GLM_API_KEY di .env

docker compose up -d --build
```

- **Backend:** http://localhost:8090 (host) → container 8000
- **Swagger:** http://localhost:8090/docs
- **PostgreSQL:** host port 5433 → container 5432. Port ini hanya untuk inspeksi lokal (`psql`), pytest, dan uvicorn host — bukan untuk diexpose publik.
- Saat start, container otomatis menjalankan `alembic upgrade head` lalu seed data bisnis (idempotent — restart tidak pernah menduplikasi baris atau menimpa order nyata).
- API key hanya masuk lewat `env_file` saat runtime — tidak pernah di-bake ke image (`.env` ada di `.dockerignore` dan `.gitignore`).

## Menjalankan Frontend

```bash
cd frontend
npm install
npm run dev
```

Buka **http://localhost:3000**. Jika port 3000 dipakai project lain, `next dev` otomatis pindah ke port berikutnya (mis. 3001/3002) — backend sudah mengizinkan CORS untuk 3000 dan 3002, dan bisa ditambah lewat `CORS_ALLOW_ORIGINS`.

URL backend diambil dari `NEXT_PUBLIC_API_URL` (`frontend/.env.local`, lihat `.env.local.example`):

```bash
cp frontend/.env.local.example frontend/.env.local
```

| Halaman | Data |
| --- | --- |
| **Chat** | `POST /api/chat` — loading, error, dan retry per pesan. |
| **Inventory** | Lewat assistant (`/api/chat`); placeholder untuk endpoint khusus masa depan. |
| **Orders** | Lewat assistant; placeholder endpoint masa depan. |
| **Reports** | Laporan on-demand dari assistant; placeholder API tersimpan. |
| **Approvals** | `GET /api/approvals` + approve/reject, lengkap penanganan 404/409. |

## Menjalankan Backend Tanpa Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # isi GLM_API_KEY + DATABASE_URL (localhost:5433)
docker compose up -d postgres # Postgres tetap via compose

uvicorn app.main:app --reload --port 8090
```

## Environment Variables

| Variable | Wajib | Deskripsi |
| --- | --- | --- |
| `GLM_API_KEY` | Ya | API key GLM dari z.ai. |
| `GLM_BASE_URL` | Tidak | Default: `https://api.z.ai/api/anthropic`. |
| `GLM_MODEL` | Tidak | Default: `glm-5.3`. |
| `DATABASE_URL` | Ya (tanpa compose) | `postgresql+psycopg://...`. Di dalam compose diisi otomatis; untuk uvicorn host: `...@localhost:5433/dibantu_ai`. |
| `CORS_ALLOW_ORIGINS` | Tidak | Origin browser yang boleh memanggil API. Default: localhost/127.0.0.1 × 3000/3002. |

> **Keamanan:** jangan pernah menulis API key asli ke file yang di-commit. Gunakan `.env` (di-ignore `.gitignore` dan `.dockerignore`).

## API

| Method | Path | Deskripsi |
| --- | --- | --- |
| GET | `/health` | Health check / liveness probe. |
| GET | `/docs` | Swagger UI (OpenAPI). |
| POST | `/api/chat` | Chat dengan agent (tool calling). |
| POST | `/webhook/whatsapp` | Simulator webhook WhatsApp (lokal, tanpa Meta). |
| GET | `/api/approvals` | Daftar approval pending. |
| POST | `/api/approvals/{id}/approve` | Setujui approval. |
| POST | `/api/approvals/{id}/reject` | Tolak approval. |

## Database

Skema (migration `alembic/versions/0001`): `products`, `customers`, `orders`, `order_items` — uang memakai `Numeric(12,2)` (bukan float), FK + index lengkap, nama produk/customer unik case-insensitive.

```bash
# inspeksi database
docker exec -it dibantu-postgres psql -U dibantu -d dibantu_ai

# reset data bisnis (truncate + seed ulang; volume tetap ada)
docker compose exec dibantu-ai python -c "from app.db.database import session_scope; from app.db.seed import reset_database
with session_scope() as s: reset_database(s)"
```

Keamanan transaksi: `create_order` mengunci baris produk (`FOR UPDATE`), memvalidasi stok kumulatif, lalu menulis order + item + potongan stok + agregat customer dalam **satu transaksi** — stok kurang berarti tidak ada order yang tersimpan.

## Testing & Evaluation

```bash
docker compose up -d postgres   # test database butuh Postgres
source .venv/bin/activate
pytest -q                       # 165 test
python -m evaluation.evaluator  # 34 skenario
```

- Test bisnis/DB berjalan di **database scratch terpisah** (`dibantu_ai_test`, dibuat & di-drop otomatis) — tidak pernah menyentuh data bisnis utama. Tanpa Postgres, test DB di-skip dengan alasan eksplisit.
- Evaluation **deterministik dan offline** (client script, tanpa API key): 34 skenario, 100% pass rate — memverifikasi tool selection, jumlah tool call, task completion, error handling, dan grounding jawaban. Ini bukan klaim akurasi model.

## Struktur Project

```
dibantu-ai/
├── app/
│   ├── agent/          # Agent loop + GLM client
│   ├── api/            # Routes (chat, webhook, approvals)
│   ├── approval/       # Human-in-the-loop approval store
│   ├── db/             # Engine/session, models, repository, seed
│   ├── models/         # Schema Pydantic
│   └── tools/          # Business tools + registry
├── alembic/            # Migration database
├── docker/             # entrypoint.sh (migrate + seed + uvicorn)
├── evaluation/         # Dataset + evaluator deterministik
├── frontend/           # Dashboard Next.js
├── tests/              # 165 test pytest
├── docker-compose.yml
└── Dockerfile
```

## Status & Limitations

🚧 **Step 11: PostgreSQL persistence — selesai.** Backend lengkap (agent + tool calling + approval + webhook simulator + 165 test), evaluation 34/34, dashboard Next.js, dan data bisnis tersimpan di PostgreSQL.

Belum diimplementasikan: WhatsApp Cloud API sungguhan (Meta auth + verifikasi signature), tunnel ngrok, integrasi Google Sheets, autentikasi, dan deployment. Approval store masih in-memory (single-process MVP). Endpoint inventory/orders/reports khusus belum ada — halaman frontend terkait masih lewat assistant.
