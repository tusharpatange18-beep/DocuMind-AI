# DocuMind AI (FastAPI + Vanilla Frontend)

## Setup (important)

- Create a `.env` file in `rag-chatbot/` (do not commit it).
- Start from `.env.example` and set `GOOGLE_API_KEY`.

## Backend

### Install

```bash
pip install -r requirements.txt
```

### OCR for scanned PDFs (optional)

If your PDFs are scanned/image-only, enable OCR:

```bash
pip install pytesseract pillow pymupdf
```

Install:
- Tesseract OCR
- Poppler (only required if you use `pdf2image`; with `pymupdf` installed it is usually not needed)

Environment variables (Windows examples):
- `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
- `POPPLER_PATH=C:\poppler\Library\bin`

### Run

From the `rag-chatbot/` folder:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend endpoints:
- `GET /` health check
- `POST /upload-pdf` (multipart form field: `file`)
- `POST /rebuild-knowledge-base`
- `POST /ask` JSON: `{ "question": "..." }`
- `GET /history`
- `DELETE /history`

## Frontend

Serve the frontend with any static server (recommended):

```bash
cd frontend
python -m http.server 5173
```

Then open:
- `http://127.0.0.1:5173/`

The frontend expects the backend at:
- `http://127.0.0.1:8000`

## Hosting (free options)

You can host this as **two services** (recommended for free tiers):

### 1) Host the backend (FastAPI) for free

Good free options:
- **Render (Free Web Service)**: simplest for FastAPI; sleeps when idle.
- **Google Cloud Run**: generous free tier, but requires adding a billing account.

#### Render steps (recommended)

1. Push this repo to GitHub.
2. In Render, you can deploy either:
	- **Blueprint** (recommended): **New → Blueprint** and select this repo (uses `render.yaml`).
	- Or manually: **New → Web Service** → connect your repo.
3. If deploying manually, set **Root Directory** to `rag-chatbot` (so `backend/` and `requirements.txt` are found).
4. Build Command: (leave blank) or `pip install -r requirements.txt`
5. Start Command:

	```bash
	uvicorn backend.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips "*"
	```

6. Add environment variables:
	- `GOOGLE_API_KEY` (required)
	- `CORS_ALLOW_ORIGINS` (recommended): your frontend URL, e.g. `https://your-site.pages.dev`
	- Optional but recommended for persistence (Supabase Storage):
	  - `STORAGE_BACKEND=supabase`
	  - `SUPABASE_URL` (from Supabase project settings)
	  - `SUPABASE_SERVICE_ROLE_KEY` (server secret; keep it private)
	  - `SUPABASE_STORAGE_BUCKET` (bucket name; default `rag-chatbot`)
	  - Optional: `SUPABASE_STORAGE_PREFIX` (folder prefix inside the bucket)
	- Optional: `ASK_TIMEOUT_SECONDS`, `EMBED_MODEL_NAME`

Notes about free hosting:
- Free instances usually have **ephemeral disk**. Uploads and rebuilt indexes may reset when the service sleeps/redeploys.
- With Supabase Storage configured, uploaded PDFs and rebuilt knowledge-base artifacts are persisted automatically.

### 2) Host the frontend (static) for free

Good free options:
- **Cloudflare Pages**
- **GitHub Pages**
- **Netlify**

Deploy the `rag-chatbot/frontend/` folder as a static site.

Then set the backend URL in the frontend by editing:
- [frontend/index.html](frontend/index.html) → set `<meta name="api-base" content="https://YOUR-BACKEND-URL">`

Alternatively, you can keep it empty and open the site with a query param:
- `https://your-frontend.site/?api=https://YOUR-BACKEND-URL`

## Persistent storage (recommended for hosted deployments)

### Supabase Storage (free)

1. Create a Supabase project.
2. Create a Storage bucket (example name: `rag-chatbot`).
3. Add the environment variables listed in the Render section.

What gets persisted when enabled:
- Uploaded PDFs are stored under `pdfs/` in the bucket.
- Knowledge-base artifacts are stored under `kb/` (chunks, metadata, vectors, FAISS index, manifest).
- Chat history is stored as a JSON blob at `history/chat_history.json`.

Security note:
- This backend uses the **service role key** to access Storage server-side. Do not expose it in the frontend.

## Data folders

- `data/raw_data/` uploaded PDFs
- `data/processed_data/` chunks + embeddings metadata
- `data/faiss_index/` FAISS index
