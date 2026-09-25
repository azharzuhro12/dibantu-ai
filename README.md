# DibantuAI

**AI Agent for Business Operations** — asisten AI untuk operasional bisnis kecil (warung, kafe, toko): cek stok, buat pesanan, catat customer, dan laporan penjualan, semuanya lewat percakapan.

Masalah yang dipecahkan: pemilik usaha kecil tidak butuh aplikasi kasir kompleks — mereka butuh asisten yang bisa diajak chat dan langsung mengoperasikan data bisnis mereka dengan aman (stok tidak pernah bocor/oversell, aksi sensitif butuh persetujuan manusia).

## Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Arsitektur & Alur Agent](#arsitektur--alur-agent)
3. [Tech Stack](#tech-stack)
4. [Menjalankan dengan Docker](#menjalankan-dengan-docker-cara-utama)
5. [Menjalankan Frontend](#menjalankan-frontend)
6. [Menjalankan Backend Tanpa Docker](#menjalankan-backend-tanpa-docker)
7. [Contoh Penggunaan](#contoh-penggunaan)
8. [Environment Variables](#environment-variables)
9. [API](#api)
10. [Database](#database)
11. [RAG / Knowledge Base (Step 12)](#rag--knowledge-base-step-12)
12. [Approval Persisten (Step 13)](#approval-persisten-step-13)
13. [Human-in-the-Loop Approval Execution (Step 16)](#human-in-the-loop-approval-execution-step-16)
14. [Agent Memory (Step 14)](#agent-memory-step-14)
15. [Observability (Step 15)](#observability-step-15)
16. [Testing & Evaluation](#testing--evaluation)
17. [Struktur Project](#struktur-project)
18. [Status & Limitations](#status--limitations)

## Fitur Utama

- **Chat agent dengan tool calling** — agent memanggil business tools (stok, order, customer, laporan) lewat loop GLM; tidak pernah mengarang data bisnis.
- **RAG knowledge base (Step 12)** — agent menjawab pertanyaan kebijakan/prosedur (refund, stok, order) dari dokumen basis pengetahuan lokal (ChromaDB + embeddings lokal), lengkap dengan sitasi sumber; data bisnis live tetap di PostgreSQL dan keduanya tidak pernah tertukar.
- **PostgreSQL persistence** — data bisnis tersimpan di Postgres (SQLAlchemy + Alembic migration); restart container tidak menghilangkan data.
- **Human-in-the-loop approval (Step 8 → 16)** — aksi sensitif (refund, cancel order, bulk stock update) tidak pernah dieksekusi agent; ia membuat approval persisten, manusia memutuskan, lalu **eksekusi eksplisit** menjalankan aksi tepat sekali dari snapshot payload immutable (allowlist executor, claim atomik di database).
- **Agent memory persisten (Step 14)** — fakta eksplisit (preferensi, konteks customer/bisnis, instruksi) tersimpan per owner di PostgreSQL dan di-inject sebagai konteks kecil di prompt; ada layar rahasia dan isolasi antar owner.
- **Observability (Step 15)** — setiap eksekusi agent ditrace end-to-end di PostgreSQL (LLM, tool, memory, RAG, approval) dengan satu `run_id` stabil yang dikembalikan ke pemanggil; metadata aman by construction (tanpa prompt/respon/rahasia).
- **WhatsApp webhook simulator** — endpoint webhook ber-format WhatsApp untuk pengujian lokal (bukan integrasi Meta sungguhan).
- **Dashboard Next.js** — halaman chat, inventory, orders, reports, approvals, memory, dan observability.
- **Evaluation suite deterministik** — 52 skenario offline yang memverifikasi tool selection, task completion, error handling, dan grounding jawaban (34 bisnis + 4 RAG + 4 memori + 8 approval + 2 observability).

## Arsitektur & Alur Agent

```
Pesan user → FastAPI /api/chat / webhook WhatsApp → DibantuAgent
  → memori owner: ≤3 fakta relevan di-inject ke system prompt (Step 14)
  → Agent loop (maks 5 eksekusi tool per request)
      → GLM (endpoint Anthropic-compatible z.ai, model glm-5.3)
      → tool_use ─┬→ business tools ────────→ PostgreSQL    (stok/order/customer/laporan)
      │           ├→ memory tools ─────────→ agent_memories (fakta per owner)
      │           ├→ search_knowledge_base → ChromaDB       (kebijakan, ber-sitasi)
      │           └→ aksi sensitif ─────────→ INTERSEP: buat approval pending (Step 8/13)
      → tool_result → GLM → jawaban final (+ run_id)
  → manusia approve → klik Execute → executor allowlist → satu transaksi (Step 16)
  → semua operasi ditrace → agent_runs / agent_events, metadata aman (Step 15)
```

- Agent loop maksimal 5 eksekusi tool per request (mencegah loop tanpa batas).
- Tool yang tidak dikenal atau gagal mengembalikan `{"error": ...}` ke model — bukan crash.
- Aksi sensitif diintersepsi **sebelum** eksekusi dan menjadi pending approval — tidak pernah dispatchable.
- Jawaban di-grounding ke hasil tool: skenario evaluasi memverifikasi setiap fakta dalam jawaban benar-benar berasal dari tool result.

## Tech Stack

| Komponen | Teknologi |
| --- | --- |
| Bahasa | Python 3.10+ / TypeScript |
| Backend | FastAPI + Uvicorn + Pydantic |
| LLM | GLM API (endpoint Anthropic-compatible z.ai, model `glm-5.3`) — custom agent loop, tanpa LangChain/LangGraph |
| Database | PostgreSQL 16 + SQLAlchemy 2 + Alembic |
| RAG | ChromaDB (vector store) + sentence-transformers `all-MiniLM-L6-v2` (embeddings lokal) — pipeline retrieval custom |
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
- Basis pengetahuan RAG perlu di-ingest sekali setelah build (downloads model embedding saat pertama kali, lalu disimpan di volume):

```bash
docker compose exec dibantu-ai python -m app.rag.ingest
```

- Vector store ChromaDB + cache model HuggingFace tersimpan di named volume `rag_data` — rebuild/restart tidak menghilangkannya, dan artefak embeddings tidak pernah masuk git/image.

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
| **Chat** | `POST /api/chat` — loading, error, dan retry per pesan; badge status koneksi backend. |
| **Inventory / Orders / Reports** | Lewat assistant (`/api/chat`); placeholder untuk endpoint khusus masa depan. |
| **Approvals** | Lifecycle penuh: `GET /api/approvals?status=all`, approve/reject, tombol **Execute** (Step 16), badge 6 status, panel hasil/error eksekusi, penanganan 404/409 + polling 30 detik. |
| **Memory** (Step 14) | `GET/DELETE /api/memory` per owner — daftar, hapus, ganti owner. |
| **Observability** (Step 15) | `GET /api/observability/runs` + timeline event per run, filter status. |

## Menjalankan Backend Tanpa Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
# torch CPU dulu supaya pip tidak menarik build CUDA (multi-GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env          # isi GLM_API_KEY + DATABASE_URL (localhost:5433)
docker compose up -d postgres # Postgres tetap via compose

uvicorn app.main:app --reload --port 8090
python -m app.rag.ingest      # ingest knowledge base (sekali; download model pertama kali)
```

## Contoh Penggunaan

```bash
# 1. Chat: buat pesanan (agent memanggil create_order → PostgreSQL, stok terpotong atomik)
curl -s -X POST http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Budi Santoso pesan 2 kopi susu."}'

# 2. Chat: minta aksi sensitif → TIDAK dieksekusi, menjadi approval pending
curl -s -X POST http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Refund pesanan ORD-0004."}'

# 3. Manusia memutuskan, lalu mengeksekusi — tepat sekali
curl -X POST http://localhost:8090/api/approvals/apr-XXXXXXXXXXXX/approve
curl -X POST http://localhost:8090/api/approvals/apr-XXXXXXXXXXXX/execute   # → executed
curl -X POST http://localhost:8090/api/approvals/apr-XXXXXXXXXXXX/execute   # → 409 conflict

# 4. Lihat trace run eksekusi tadi (event APPROVAL_EXECUTING → APPROVAL_EXECUTED)
curl -s "http://localhost:8090/api/observability/runs?source=approval_executor&limit=1"

# 5. Refund parsial: kirim amount lewat chat ("Refund 20000 dari ORD-0003")
#    → uang saja (refunded_amount kumulatif), stok tidak kembali, order tetap completed
```

Semua alur di atas juga tersedia dari halaman **Approvals** di frontend (approve → tombol Execute → hasil/error tampil di kartu).

## Environment Variables

| Variable | Wajib | Deskripsi |
| --- | --- | --- |
| `GLM_API_KEY` | Ya | API key GLM dari z.ai. |
| `GLM_BASE_URL` | Tidak | Default: `https://api.z.ai/api/anthropic`. |
| `GLM_MODEL` | Tidak | Default: `glm-5.3`. |
| `DATABASE_URL` | Ya (tanpa compose) | `postgresql+psycopg://...`. Di dalam compose diisi otomatis; untuk uvicorn host: `...@localhost:5433/dibantu_ai`. |
| `CORS_ALLOW_ORIGINS` | Tidak | Origin browser yang boleh memanggil API. Default: localhost/127.0.0.1 × 3000/3002. |
| `KNOWLEDGE_DIR` | Tidak | Direktori dokumen Markdown knowledge base. Default: `data/knowledge`. |
| `RAG_STORE_DIR` | Tidak | Direktori vector store ChromaDB. Default: `data/rag/chroma` (di compose: `/app/data/rag/chroma`). |
| `RAG_EMBEDDINGS` | Tidak | `local` (sentence-transformers, default) atau `hashing` (deterministik offline — untuk test/eval). |
| `RAG_EMBEDDING_MODEL` | Tidak | Default: `sentence-transformers/all-MiniLM-L6-v2`. |
| `RAG_TOP_K` | Tidak | Jumlah passage per query. Default: `3`. |

> **Keamanan:** jangan pernah menulis API key asli ke file yang di-commit. Gunakan `.env` (di-ignore `.gitignore` dan `.dockerignore`).

## API

| Method | Path | Deskripsi |
| --- | --- | --- |
| GET | `/health` | Health check / liveness probe. |
| GET | `/docs` | Swagger UI (OpenAPI). |
| POST | `/api/chat` | Chat dengan agent (tool calling; `owner_key` opsional di body untuk memori per-owner, default `"default"`). |
| POST | `/webhook/whatsapp` | Simulator webhook WhatsApp (lokal, tanpa Meta). |
| GET | `/api/approvals` | Daftar approval (default: pending; `?status=approved\|rejected\|executing\|executed\|failed\|all` untuk lifecycle penuh). |
| POST | `/api/approvals/{id}/approve` | Setujui approval (opsional `?reason=...`, maks 500 karakter) — hanya mencatat keputusan. |
| POST | `/api/approvals/{id}/reject` | Tolak approval (opsional `?reason=...`, tercatat sebagai `decision_reason`) — terminal, tidak bisa dieksekusi. |
| POST | `/api/approvals/{id}/execute` | **(Step 16)** Eksekusi approval yang approved — tepat sekali, dari payload snapshot di server; endpoint tidak menerima argumen. 409 untuk status lain. |
| GET | `/api/knowledge/search?q=...` | Cari passage di knowledge base (retrieval saja, tanpa LLM). |
| POST | `/api/knowledge/ingest` | (Re)ingest dokumen knowledge base (idempotent; tanpa parameter path — selalu direktori terkonfigurasi). |
| GET | `/api/memory?owner_key=...` | Daftar memori agent satu owner, terbaru dulu (opsional `memory_type`, `q`, `limit`). |
| POST | `/api/memory` | Buat memori (`owner_key` opsional, `memory_type`, `content`). |
| DELETE | `/api/memory/{id}?owner_key=...` | Hapus memori milik owner tersebut (id owner lain = 404). |
| GET | `/api/observability/runs` | Daftar run agent ter-trace, terbaru dulu (filter opsional `status`, `source`, `owner_key`, `limit`). |
| GET | `/api/observability/runs/{run_id}` | Satu run berdasarkan `run_id` stabil. |
| GET | `/api/observability/runs/{run_id}/events` | Timeline event run tersebut (filter opsional `event_type`, `limit`). |

## Database

Skema (migration `alembic/versions/0001`): `products`, `customers`, `orders`, `order_items` — uang memakai `Numeric(12,2)` (bukan float), FK + index lengkap, nama produk/customer unik case-insensitive. Migration `0002` menambah tabel `approvals` (lihat [Approval Persisten](#approval-persisten-step-13)), `0003` tabel `agent_memories` ([Agent Memory](#agent-memory-step-14)), `0004` tabel `agent_runs` + `agent_events` ([Observability](#observability-step-15)), dan `0005` kolom eksekusi approval + `orders.refunded_amount` ([Approval Execution](#human-in-the-loop-approval-execution-step-16)).

```bash
# inspeksi database
docker exec -it dibantu-postgres psql -U dibantu -d dibantu_ai

# reset data bisnis (truncate + seed ulang; volume tetap ada)
docker compose exec dibantu-ai python -c "from app.db.database import session_scope; from app.db.seed import reset_database
with session_scope() as s: reset_database(s)"
```

Keamanan transaksi: `create_order` mengunci baris produk (`FOR UPDATE`), memvalidasi stok kumulatif, lalu menulis order + item + potongan stok + agregat customer dalam **satu transaksi** — stok kurang berarti tidak ada order yang tersimpan.

## RAG / Knowledge Base (Step 12)

**Kenapa RAG?** Data bisnis live (stok, order, customer) dan pengetahuan bisnis (kebijakan, prosedur, aturan) adalah dua dunia berbeda. Stok bisa di-query dari database, tapi "berapa lama batas waktu refund?" adalah pertanyaan dokumen. RAG membuat agent menjawab pertanyaan kebijakan dari **dokumen resmi** dengan sitasi sumber — bukan dari ingatan model (yang bisa mengarang), dan tetap memakai tools database untuk angka live.

**Alur retrieval:**

```
pertanyaan kebijakan
  → agent memanggil tool search_knowledge_base(query)
  → query di-embed (sentence-transformers, lokal)
  → ChromaDB cosine similarity search → top-k passage
  → passage + metadata (source, section, similarity) kembali ke GLM
  → GLM menjawab HANYA dari passage tersebut + sitasi sumber
```

Komponen (semuanya lokal & gratis, tanpa API embedding berbayar, tanpa LangChain/LangGraph):

| Tahap | Implementasi |
| --- | --- |
| Dokumen | Markdown di `data/knowledge/` — 4 dokumen kebijakan demo |
| Chunking | Deterministik, sadar-heading Markdown: split per section `#`/`##`, pack paragraf ≤ 800 karakter, overlap 150 karakter |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (384-dim, on-device; alternatif `hashing` untuk test offline) |
| Vector store | ChromaDB persistent, collection tunggal, cosine space |
| Retrieval | top-k (default 3, 1–10) passage + metadata source/section/chunk_id/similarity |
| Agent | Tool `search_knowledge_base` di registry yang sama dengan business tools |

**Ingestion (idempotent):**

```bash
python -m app.rag.ingest           # host; di Docker: docker compose exec dibantu-ai python -m app.rag.ingest
# atau: POST /api/knowledge/ingest
```

Setiap dokumen di-hash; dokumen yang tidak berubah di-skip, dokumen berubah diganti seluruh chunk-nya (tidak ada duplikat/chunk basi). Output: jumlah dokumen dibuat/di-update/di-skip, chunk dibuat, total chunk, lokasi store.

**Grounding & pemisahan sumber data.** System prompt agent memisahkan tiga kasus secara eksplisit: (1) data bisnis → tools PostgreSQL; (2) kebijakan/prosedur → `search_knowledge_base` dan jawab hanya dari passage ber-sitasi (mis. "Menurut refund_policy.md, ..."); (3) informasi yang tidak ada di knowledge base → katakan tidak tersedia, jangan pernah mengarang kebijakan atau sitasi. Pertanyaan campuran (kebijakan + stok) dieksekusi kedua tool dalam satu turn dan jawabannya diberi label jelas.

**Catatan keamanan:** ingestion selalu terbatas pada direktori `KNOWLEDGE_DIR` yang dikonfigurasi di server — endpoint ingest tidak menerima path dari user. Artefak vector store (`data/rag/`) dan cache model HuggingFace di-ignore dari git dan tidak di-bake ke image (mereka hidup di volume `rag_data`).

> Dokumen di `data/knowledge/` adalah **kebijakan bisnis demo/contoh** untuk portfolio — bukan kebijakan perusahaan nyata. Menambah dokumen baru: taruh file `.md` di `data/knowledge/`, jalankan ulang ingestion.

## Approval Persisten (Step 13)

Human-in-the-loop approval untuk aksi sensitif (`refund_order`, `cancel_order`, `bulk_stock_update`) kini disimpan di tabel PostgreSQL `approvals` (migration `0002`), bukan lagi dict in-memory — **pending approval dan keputusannya selamat dari restart backend/container**.

**Lifecycle:**

```
agent minta aksi sensitif
  → intersepsi di Agent._run_tool (tool TIDAK dieksekusi)
  → baris `approvals` dibuat dengan status pending
  → agent memberi tahu user approval_id
  → manusia approve/reject via API (opsional ?reason=...)
  → keputusan + timestamp + alasan dipersistenkan
```

Jaminan keamanan yang dipertahankan:

- Aksi sensitif **tidak pernah dieksekusi** hanya karena ada approval — approve hanya mencatat keputusan manusia; tidak ada jalur eksekusi otomatis setelahnya.
- Keputusan dijalankan sebagai satu `UPDATE ... WHERE status = 'pending'` (atomic di level database): dua request bersamaan tidak bisa keduanya berhasil; yang kedua mendapat HTTP 409.
- Transisi status invalid ditolak: `approved → rejected`, `rejected → approved`, dan keputusan ulang semuanya 409; id yang tidak ada 404.
- Payload approval disimpan untuk direview manusia — berisi parameter aksi bisnis, bukan kredensial.

**Eksekusi:** approve memang hanya mencatat keputusan — eksekusi eksplisit ditambahkan di [Step 16](#human-in-the-loop-approval-execution-step-16) (executor allowlist, claim atomik, idempoten). Endpoint approval juga belum memakai autentikasi.

## Human-in-the-Loop Approval Execution (Step 16)

Approval kini **benar-benar bisa melanjutkan ke eksekusi** — tapi hanya lewat aksi manusia eksplisit, tepat sekali, dari snapshot payload yang tersimpan. Tiga aksi sensitif punya implementasi bisnis riil dan transaksional di `app/tools/sensitive_actions.py` (refund/cancel mengembalikan stok + menyesuaikan agregat; bulk update multi-produk dalam satu transisi), dieksekusi **hanya** oleh executor setelah manusia approve.

**State machine (migration `0005`, CHECK di database):**

```
pending ──→ approved ──→ executing ──→ executed
   │                        └──→ failed
   └──→ rejected   (terminal — tidak bisa dieksekusi)
```

Transisi di luar tabel ini ditolak `validate_transition`; `executing`/`executed`/`failed` terminal. Alur lengkap:

```
user minta aksi sensitif
  → agent memanggil (skema request aksi sensitif diiklankan; TIDAK dispatchable)
  → intersepsi: approval pending dibuat dengan snapshot payload immutable
  → agent memberi tahu user approval_id (aksi TIDAK dijalankan)
  → manusia approve (hanya mencatat keputusan)
  → manusia klik Execute / POST /api/approvals/{id}/execute
  → executor: claim atomik approved→executing → eksekusi → executed/failed
```

**Jaminan desain:**

- **Immutable execution snapshot** — eksekusi memakai payload yang tersimpan saat approval dibuat; endpoint execute **tidak menerima argumen apa pun**. "Approve request A tapi eksekusi request B" mustahil by construction; client tidak bisa mengirim ulang/mengubah payload.
- **Allowlist eksplisit** — executor hanya menjalankan 3 aksi di `APPROVED_EXECUTABLE_TOOLS` (nama → fungsi); bukan generic runner. Nama tool arbitrer/non-allowlist ditolak 422 tanpa perubahan state; validasi struktur payload (per-item untuk bulk) juga pre-claim.
- **Atomic concurrency claim** — `approved → executing` adalah satu `UPDATE ... WHERE status='approved'`: dari dua execute bersamaan tepat satu mendapat rowcount 1; yang kalah menerima 409. Database (bukan lock in-process) adalah autoritas concurrency.
- **Idempotensi** — execute atas `pending`/`rejected`/`executing`/`executed`/`failed` semuanya 409 tanpa efek samping; double refund/double stock update karena browser refresh/duplicate request mustahil. Lapisan bisnis juga menolak refund/cancel order yang sudah terminal (defense in depth).
- **Refund parsial vs penuh** — tanpa `amount` (atau `amount` == sisa): refund penuh → stok dikembalikan, agregat customer dibatalkan, order `refunded` (keluar dari laporan — `orders_since` kini hanya menghitung order `completed`). `amount` lebih kecil: refund parsial → uang saja (`refunded_amount` kumulatif di-cap total order, cek antar-approval), barang dianggap tetap dipegang, order tetap `completed`.
- **Execution result** — hasil tool disimpan di kolom `execution_result` (JSON) dan dikembalikan di response; kegagalan menyimpan pesan bisnis ber-bound (`execution_error` ≤ 500 karakter — tanpa traceback, tanpa kredensial).
- **Observability** — tiap eksekusi membentuk satu trace run (`source=approval_executor`): `APPROVAL_EXECUTING` → `APPROVAL_EXECUTED`/`APPROVAL_FAILED` (+ RUN_STARTED/COMPLETED), metadata aman hanya `{approval_id, action, status}` — payload tidak pernah masuk trace. Kegagalan tracing ditelan; tidak pernah menggagalkan eksekusi.
- **Persisten** — status, timestamps, result, dan error eksekusi tersimpan di PostgreSQL; selamat dari restart (diverifikasi live).

**Batasan (jujur):** approval `executing` bisa **stuck** jika proses mati setelah claim sebelum tulis status terminal (belum ada mekanisme recovery/re-scan); `failed` terminal tanpa explicit retry design (buat approval baru untuk mencoba lagi); endpoint approval **belum ada autentikasi/otorisasi** (siapa pun yang bisa memanggil API bisa approve/execute); eksekusi **tidak pernah otomatis oleh agent** (by design); batas aksi sensitif Step 8 yang masih berlaku: perubahan stok beberapa produk bisa "melewati" approval `bulk_stock_update` lewat beberapa kali `update_stock` reguler — agent dilarang lewat deskripsi tool, tapi tidak dicegah secara struktural; dan edge case yang diketahui dari audit final: **cancel atas order yang sudah pernah direfund parsial** mengurangi `total_spent` sebesar total penuh (kredit ganda pada agregat customer saja — stok/laporan tidak terpengaruh; belum diperbaiki, fix-nya menunggu follow-up).

## Agent Memory (Step 14)

Agent bisa mengingat **fakta eksplisit** lintas percakapan — preferensi, konteks customer, konteks bisnis, instruksi — di tabel PostgreSQL `agent_memories` (migration `0003`), sehingga selamat dari restart. Ini bukan riwayat percakapan: hanya fakta terstruktur yang diminta disimpan.

**Tipe memori** (tertutup, di-CHECK di database): `preference`, `customer_context`, `business_context`, `instruction`.

**Ownership:** `owner_key` — identitas level aplikasi, **bukan autentikasi**. Webhook WhatsApp memakai nomor sender; `/api/chat` menerima `owner_key` opsional (default `"default"`, client lama tetap jalan). Semua operasi memori (service, API, tool) di-scope per owner; update/delete memakai `UPDATE/DELETE ... WHERE id = ? AND owner_key = ?` sehingga owner lain tidak bisa mengubah — dan tidak bisa mem-probe keberadaan id owner lain (404 sama seperti id tak dikenal).

**Alur:**

```
request (owner_key)
  → retrieval deterministik: keyword overlap + recency (TANPA embedding — itu urusan RAG)
  → maksimal 3 memori relevan ditambahkan ke system prompt sebagai "Known facts"
  → tanpa memori, prompt & perilaku agent identik seperti sebelum Step 14
```

**Tulisan memori dikendalikan ketat** — model tidak bisa menulis semaunya: tool `save_memory` hanya dipakai atas permintaan eksplisit user (aturan prompt #9), input divalidasi aplikasi sebelum persist (tipe tertutup, konten ≤ 2000 karakter), dan **layar rahasia** menolak konten mirip password/API key/token/nomor kartu. Tool memori (`save_memory`, `search_memory`, `delete_memory`) otomatis terikat owner request saat itu — model tidak bisa memilih owner lain.

**Keamanan:** memori murni konteks — tidak ada jalur dari memori ke eksekusi aksi bisnis, modifikasi order/stok, RAG, atau approval. Memori dan knowledge base tetap konsep terpisah.

**Batasan:** belum ada autentikasi (owner_key bisa diklaim siapa saja yang bisa memanggil API — sama seperti endpoint lain di proyek ini); retrieval keyword saja (bukan semantik); tidak ada UI untuk mengedit memori (hanya lihat/hapus); dan layar rahasia menangkap pola umum (password/api key/token/nomor kartu) tetapi belum semua bentuk rahasia — mis. prefiks `sk-`/`ghp_`/`AKIA` atau passphrase tanpa label masih lolos (temuan audit final, follow-up).

## Observability (Step 15)

Setiap eksekusi agent ditrace **end-to-end** di dua tabel PostgreSQL (migration `0004`): satu baris `agent_runs` per eksekusi, satu baris `agent_events` per operasi yang ditrace — call LLM, eksekusi tool (diklasifikasikan: TOOL/MEMORY/RAG/APPROVAL), dengan nomor iterasi loop dan durasi. Semua diikat satu `run_id` stabil yang ikut dikembalikan di response `/api/chat` dan webhook, sehingga keluhan user bisa langsung dicari run-nya. **PostgreSQL-only, tanpa platform observability eksternal.**

**Aman by construction** — bukan sekadar "kami hati-hati":

- Tidak ada prompt, respon mentah, chain-of-thought, rahasia, atau stack trace — hanya nama kelas exception (`error_type`), maksimal 160 karakter preview request, dan metadata kecil (≤ 12 kunci, ≤ 400 karakter per nilai, shallow copy).
- Preview request yang mengandung penanda rahasia (`api key`, `password`, `token`, `sk-`, dst.) atau nomor kartu disimpan sebagai NULL, bukan dipotong.
- Kegagalan tracing **ditelan** oleh recorder — observability tidak pernah bisa merusak request bisnis; error bisnis tetap dipropagasi setelah run ditandai `failed`.
- `agent_events` sengaja tanpa foreign key ke `agent_runs` — setiap tulisan independen, kegagalan parsial tidak pernah cascade.

**Usage token** (`input_tokens`/`output_tokens`) diambil dari respons provider saat ada (tidak pernah dikarang) dan hanya masuk metadata observability. Keputusan approve/reject juga ditrace (informasional — tidak mengeksekusi apa pun).

**Batasan:** tracing level operasi (bukan baris-per-baris audit log); belum ada agregat/metric (mis. token per hari); retention/retensi data run belum diatur (tabel tumbuh sampai dibersihkan manual); endpoint observability belum memakai autentikasi; dan bila penyimpanan run gagal total (DB down), response tetap membawa `run_id` yang tak pernah tersimpan — pencarian run tersebut akan 404 (temuan audit final, follow-up).

## Testing & Evaluation

```bash
docker compose up -d postgres   # test database butuh Postgres
source .venv/bin/activate
pytest -q                       # 350 test (349 lulus + 1 skip kondisional)
python -m evaluation.evaluator  # 52 skenario
```

- Test bisnis/DB berjalan di **database scratch terpisah** (`dibantu_ai_test`, dibuat & di-drop otomatis) — tidak pernah menyentuh data bisnis utama. Tanpa Postgres, test DB di-skip dengan alasan eksplisit.
- Test & evaluasi RAG memakai **embeddings `hashing` deterministik + vector store scratch di tmp** — tanpa download model, tanpa network, dan `data/rag` milik developer tidak pernah disentuh (test model lokal men-skip dirinya jika model tak bisa di-download).
- Evaluation **deterministik dan offline** (client script, tanpa API key): 52 skenario (34 bisnis + 4 RAG + 4 memori + 8 approval + 2 observability), 100% pass rate — memverifikasi tool selection, jumlah tool call, task completion, error handling, dan grounding jawaban. Skenario approval menjalankan siklus hidup penuh (create → approve/reject → execute → conflict → persist) lewat manager/executor sungguhan; skenario observability memverifikasi kontrak trace eksekusi. Laporan evaluasi memisahkan kelima grup tersebut. Ini bukan klaim akurasi model.

## Struktur Project

```
dibantu-ai/
├── app/
│   ├── agent/          # Agent loop + GLM client + DibantuAgent (memori & trace)
│   ├── api/            # Routes (chat, webhook, approvals, knowledge, memory, observability)
│   ├── approval/       # Approval store persisten + executor allowlist (Step 13/16)
│   ├── db/             # Engine/session, models, repository, seed
│   ├── memory/         # Agent memory per owner: repository, manager, tools (Step 14)
│   ├── models/         # Schema Pydantic
│   ├── observability/  # Tracing run/event: manager, repository (Step 15)
│   ├── rag/            # RAG: chunker, embeddings, vector store, retriever, service, tool, CLI
│   └── tools/          # Business tools + registry + sensitive actions (Step 16)
├── alembic/            # Migration database
├── data/
│   ├── knowledge/      # Dokumen kebijakan Markdown (demo)
│   └── rag/            # Vector store + cache model (generated, di-gitignore)
├── docker/             # entrypoint.sh (migrate + seed + uvicorn)
├── evaluation/         # Dataset + evaluator deterministik
├── frontend/           # Dashboard Next.js
├── tests/              # 350 test pytest (1 skip kondisional)
├── docker-compose.yml
└── Dockerfile
```

## Status & Limitations

✅ **Step 1–16 selesai dan ter-audit** — 349 test lulus (+1 skip kondisional), evaluasi 52/52, lint+build frontend bersih, Docker healthy, migration `0005` applied, alur approval→execute terverifikasi live (termasuk uji race konkurensi sungguhan via `threading.Barrier`).

| Step | Deliverable |
| --- | --- |
| 1–2 | Setup proyek + chat GLM dasar (custom agent loop, tanpa LangChain) |
| 3–5 | Business tools (stok/order/customer/laporan) + tool-calling loop + multi-step workflow |
| 6 | Evaluation suite deterministik (offline, 52 skenario) |
| 7 | WhatsApp webhook simulator (lokal) |
| 8 | Human-in-the-loop approval (intersepsi aksi sensitif) |
| 9–10 | Dockerisasi + dashboard Next.js |
| 11 | PostgreSQL persistence (Alembic, transaksi atomik, seed idempotent) |
| 12 | RAG knowledge base (ChromaDB + embeddings lokal, ber-sitasi) |
| 13 | Approval persisten (keputusan atomik di database) |
| 14 | Agent memory persisten per owner (layar rahasia, isolasi) |
| 15 | Observability end-to-end (`run_id` stabil, metadata aman by construction) |
| 16 | Deferred approval execution (allowlist executor, claim atomik, idempoten, refund penuh/parsial) |

Belum diimplementasikan (backlog yang disengaja): WhatsApp Cloud API sungguhan (Meta auth + verifikasi signature), tunnel ngrok, integrasi Google Sheets, autentikasi, dan deployment. Endpoint inventory/orders/reports khusus belum ada — halaman frontend terkait masih lewat assistant.

Limitasi per fitur (jujur dan lengkap) didokumentasikan di masing-masing section di atas. Limitasi RAG: tidak ada UI manajemen dokumen (tambah/ubah dokumen = edit file + re-ingest); dokumen yang dihapus dari `data/knowledge/` tidak otomatis menghapus chunk lama di store (re-ingest dokumen berubah sudah ditangani); belum ada re-ranking maupun filter similarity threshold (relevansi dinilai GLM dari passage yang kembali); embedding model default berbahasa Inggris — dokumen Indonesia tetap ter-retrieve dengan baik lewat overlap kosakata, tapi model multibahasa (mis. `paraphrase-multilingual-MiniLM`) bisa lebih akurat dan tinggal ganti `RAG_EMBEDDING_MODEL` + hapus `data/rag/` + re-ingest.
