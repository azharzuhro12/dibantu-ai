# DibantuAI

**AI Agent for Business Operations** — asisten AI untuk operasional bisnis kecil (warung, kafe, toko): cek stok, buat pesanan, catat customer, dan laporan penjualan, semuanya lewat percakapan.

Masalah yang dipecahkan: pemilik usaha kecil tidak butuh aplikasi kasir kompleks — mereka butuh asisten yang bisa diajak chat dan langsung mengoperasikan data bisnis mereka dengan aman (stok tidak pernah bocor/oversell, aksi sensitif butuh persetujuan manusia).

## Fitur Utama

- **Chat agent dengan tool calling** — agent memanggil business tools (stok, order, customer, laporan) lewat loop GLM; tidak pernah mengarang data bisnis.
- **RAG knowledge base (Step 12)** — agent menjawab pertanyaan kebijakan/prosedur (refund, stok, order) dari dokumen basis pengetahuan lokal (ChromaDB + embeddings lokal), lengkap dengan sitasi sumber; data bisnis live tetap di PostgreSQL dan keduanya tidak pernah tertukar.
- **PostgreSQL persistence** — data bisnis tersimpan di Postgres (SQLAlchemy + Alembic migration); restart container tidak menghilangkan data.
- **Human-in-the-loop approval** — aksi sensitif (refund, cancel order, bulk stock update) tidak dieksekusi agent; menunggu persetujuan manusia lewat API/frontend.
- **WhatsApp webhook simulator** — endpoint webhook ber-format WhatsApp untuk pengujian lokal (bukan integrasi Meta sungguhan).
- **Dashboard Next.js** — halaman chat, inventory, orders, reports, dan approvals.
- **Evaluation suite deterministik** — 38 skenario offline yang memverifikasi tool selection, task completion, error handling, dan grounding jawaban (34 bisnis + 4 RAG).

## Arsitektur & Alur Agent

```
Pesan user → FastAPI /api/chat → Agent loop
  → GLM (endpoint Anthropic-compatible z.ai, model glm-5.3)
  → tool_use → registry ─┬→ business tools → PostgreSQL   (data live)
  │                       └→ search_knowledge_base → ChromaDB (kebijakan)
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
| **Chat** | `POST /api/chat` — loading, error, dan retry per pesan. |
| **Inventory** | Lewat assistant (`/api/chat`); placeholder untuk endpoint khusus masa depan. |
| **Orders** | Lewat assistant; placeholder endpoint masa depan. |
| **Reports** | Laporan on-demand dari assistant; placeholder API tersimpan. |
| **Approvals** | `GET /api/approvals` + approve/reject, lengkap penanganan 404/409. |

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
| POST | `/api/chat` | Chat dengan agent (tool calling). |
| POST | `/webhook/whatsapp` | Simulator webhook WhatsApp (lokal, tanpa Meta). |
| GET | `/api/approvals` | Daftar approval pending. |
| POST | `/api/approvals/{id}/approve` | Setujui approval. |
| POST | `/api/approvals/{id}/reject` | Tolak approval. |
| GET | `/api/knowledge/search?q=...` | Cari passage di knowledge base (retrieval saja, tanpa LLM). |
| POST | `/api/knowledge/ingest` | (Re)ingest dokumen knowledge base (idempotent; tanpa parameter path — selalu direktori terkonfigurasi). |

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

## Testing & Evaluation

```bash
docker compose up -d postgres   # test database butuh Postgres
source .venv/bin/activate
pytest -q                       # 218 test
python -m evaluation.evaluator  # 38 skenario
```

- Test bisnis/DB berjalan di **database scratch terpisah** (`dibantu_ai_test`, dibuat & di-drop otomatis) — tidak pernah menyentuh data bisnis utama. Tanpa Postgres, test DB di-skip dengan alasan eksplisit.
- Test & evaluasi RAG memakai **embeddings `hashing` deterministik + vector store scratch di tmp** — tanpa download model, tanpa network, dan `data/rag` milik developer tidak pernah disentuh (test model lokal men-skip dirinya jika model tak bisa di-download).
- Evaluation **deterministik dan offline** (client script, tanpa API key): 38 skenario (34 bisnis + 4 RAG), 100% pass rate — memverifikasi tool selection, jumlah tool call, task completion, error handling, dan grounding jawaban. Laporan evaluasi memisahkan angka bisnis vs RAG. Ini bukan klaim akurasi model.

## Struktur Project

```
dibantu-ai/
├── app/
│   ├── agent/          # Agent loop + GLM client
│   ├── api/            # Routes (chat, webhook, approvals, knowledge)
│   ├── approval/       # Human-in-the-loop approval store
│   ├── db/             # Engine/session, models, repository, seed
│   ├── models/         # Schema Pydantic
│   ├── rag/            # RAG: chunker, embeddings, vector store, retriever, service, tool, CLI
│   └── tools/          # Business tools + registry
├── alembic/            # Migration database
├── data/
│   ├── knowledge/      # Dokumen kebijakan Markdown (demo)
│   └── rag/            # Vector store + cache model (generated, di-gitignore)
├── docker/             # entrypoint.sh (migrate + seed + uvicorn)
├── evaluation/         # Dataset + evaluator deterministik
├── frontend/           # Dashboard Next.js
├── tests/              # 218 test pytest
├── docker-compose.yml
└── Dockerfile
```

## Status & Limitations

🚧 **Step 12: RAG / Knowledge Base — selesai.** Seluruh Step 1–11 (agent + tool calling + approval + webhook simulator + PostgreSQL persistence) tetap utuh, ditambah knowledge base lokal: retrieval ChromaDB + embeddings lokal, tool `search_knowledge_base` di agent, endpoint knowledge API, ingestion idempotent, 218 test, dan evaluasi 38/38.

Belum diimplementasikan: WhatsApp Cloud API sungguhan (Meta auth + verifikasi signature), tunnel ngrok, integrasi Google Sheets, autentikasi, dan deployment. Approval store masih in-memory (single-process MVP). Endpoint inventory/orders/reports khusus belum ada — halaman frontend terkait masih lewat assistant.

Limitasi RAG saat ini: tidak ada UI manajemen dokumen (tambah/ubah dokumen = edit file + re-ingest); dokumen yang dihapus dari `data/knowledge/` tidak otomatis menghapus chunk lama di store (re-ingest dokumen berubah sudah ditangani); belum ada re-ranking maupun filter similarity threshold (relevansi dinilai GLM dari passage yang kembali); embedding model default berbahasa Inggris — dokumen Indonesia tetap ter-retrieve dengan baik lewat overlap kosakata, tapi model multibahasa (mis. `paraphrase-multilingual-MiniLM`) bisa lebih akurat dan tinggal ganti `RAG_EMBEDDING_MODEL` + hapus `data/rag/` + re-ingest.
