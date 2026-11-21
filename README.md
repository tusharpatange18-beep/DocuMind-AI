# DocuMind AI

Chat with your PDFs. Upload a document, ask questions in plain language, and get answers taken only from that document, with the page numbers they came from.

Built with FastAPI, FAISS, Sentence Transformers and Google Gemini, with a plain HTML/JavaScript frontend.

## Features

- **Answers only from your documents.** If the answer isn't in the PDF, it says so instead of guessing.
- **Sources for every answer**, with the PDF name and page number.
- **Hybrid search.** Meaning-based search (embeddings + FAISS) combined with exact-keyword search, so both related ideas and exact terms are found.
- **Tolerates spelling mistakes.** Misspelled words are corrected to the closest word in your documents ("microwve" becomes "microwave").
- **Chat memory.** Follow-up questions like "what about its textbooks?" work within a chat.
- **Ready right after upload.** Each PDF is indexed during upload. Uploading the same file twice does not create a duplicate.
- **OCR support** for scanned PDFs (optional).
- **Backup models.** If the main Gemini model is out of quota or busy, a backup model answers instead.

## How it works

1. **Upload:** the PDF's text is extracted page by page and cut into overlapping passages of about 2,000 characters.
2. **Index:** each passage is turned into an embedding (a list of numbers that represents its meaning) with `all-MiniLM-L6-v2`, which runs locally, and stored in a FAISS index.
3. **Ask:** your question is searched two ways, by meaning and by keywords, and the two rankings are merged. The best passages, plus the following pages of the same PDF (topics often continue on the next page), are sent to Gemini.
4. **Answer:** Gemini answers using only those passages, and the app shows the answer with its sources.

## Project structure

```
backend/
  main.py              FastAPI app
  requirements.txt
  app/
    api.py             HTTP endpoints
    pdf_ingestion.py   saves uploads (skips duplicates)
    rebuild_pipeline.py  text extraction, chunking, embeddings, index
    retriever.py       hybrid search + spelling correction
    rag_pipeline.py    builds the context, chat memory
    prompt.py          prompt sent to Gemini
    llm.py             Gemini calls with backup models
frontend/              index.html, script.js, styles.css
data/                  uploads and index (created at runtime, not committed)
```

## Run it locally

You need **Python 3.12** and a free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey).

**1. Install**

From the project root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
```

**2. Add your API key**

Copy `.env.example` to `.env` in the project root and set `GOOGLE_API_KEY`. Never commit `.env`.

**3. Start the backend**

From the project root:

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

The first start downloads the embedding model (about 90 MB), so it can take a minute. Check it at http://127.0.0.1:8000/, which should return `{"status": "DocuMind AI backend running"}`.

**4. Start the frontend**

In a second terminal:

```bash
cd frontend
python -m http.server 5173
```

Open http://127.0.0.1:5173/, click **Upload PDF**, wait for "indexed", and ask a question.

## Tips for good answers

- Ask specific questions: "What are the lab experiments in Microwave Communication?" works better than "tell me everything".
- Include the topic's name in your first question. Follow-ups in the same chat can then say "it" or "that course".
- Use **New chat** when you change subject, so the old topic doesn't affect search.
- "I do not know based on the provided documents" usually means the answer isn't in your PDFs.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload-pdf` | Upload a PDF (multipart field `file`). It is indexed before the response. |
| POST | `/rebuild-knowledge-base` | Re-index everything in `data/raw_data/` |
| POST | `/ask` | `{"question": "...", "history": [{"question": "...", "answer": "..."}]}`. `history` is optional (the last 3 turns of the chat). |
| GET | `/history` | Saved questions and answers |
| DELETE | `/history` | Clear saved history |

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL` | `models/gemini-flash-latest` | Main model |
| `GEMINI_FALLBACK_MODELS` | `models/gemini-3-flash-preview,models/gemini-flash-lite-latest` | Backup models, tried in order |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model for search |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | `2000` / `200` | Passage size and overlap |
| `ASK_TIMEOUT_SECONDS` | `120` | Max time for one answer |
| `CORS_ALLOW_ORIGINS` | `*` | Allowed frontend URLs, comma-separated |
| `STORAGE_BACKEND` | (local) | Set to `supabase` for hosted persistence |

### OCR for scanned PDFs (optional)

Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract). If it isn't on your PATH, set `TESSERACT_CMD`, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`.

## Troubleshooting

- **`429 You exceeded your current quota`:** the free Gemini tier allows about 20 requests per day per model. The app switches to the backup models automatically. If all of them are used up, wait for the daily reset or use a paid key.
- **`DLL load failed ... An Application Control policy has blocked this file` (Windows):** Windows Smart App Control can block very new builds of some packages. Install slightly older ones: `pip install grpcio==1.71.0 faiss-cpu==1.9.0.post1`.
- **`ModuleNotFoundError: No module named 'app.api'`:** start the backend with the command above (`uvicorn main:app --app-dir backend ...`) from the project root.
- **The page shows an old version after an update:** hard-refresh the browser with Ctrl+F5.

## Hosting (free)

Host the backend and the frontend as two services.

### Backend on Render

**Blueprint (recommended):** in Render, choose **New → Blueprint** and select this repo. It uses `render.yaml`.

**Or manually:** choose **New → Web Service**, connect the repo, leave **Root Directory** empty, and set:

- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn main:app --app-dir backend --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips "*"`

Environment variables:

- `GOOGLE_API_KEY` (required)
- `CORS_ALLOW_ORIGINS`: your frontend URL, e.g. `https://your-site.pages.dev`
- For persistence with Supabase Storage (recommended): `STORAGE_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`

Free Render instances have temporary disks, so uploads and the index are lost when the service sleeps or redeploys unless Supabase Storage is configured.

### Frontend on Cloudflare Pages, GitHub Pages or Netlify

Deploy the `frontend/` folder as a static site. Then point it at your backend by setting the meta tag in `frontend/index.html`:

```html
<meta name="api-base" content="https://YOUR-BACKEND-URL">
```

Or open the site with a query parameter: `https://your-frontend.site/?api=https://YOUR-BACKEND-URL`.

### Supabase Storage (free persistence)

1. Create a Supabase project and a Storage bucket (for example `rag-chatbot`).
2. Add the Supabase environment variables listed above.

When enabled, uploaded PDFs are stored under `pdfs/`, the index under `kb/`, and chat history at `history/chat_history.json`. The service role key is used only on the server. Never put it in the frontend.
