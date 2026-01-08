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
uvicorn backend.main:app --reload
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

## Data folders

- `data/raw_data/` uploaded PDFs
- `data/processed_data/` chunks + embeddings metadata
- `data/faiss_index/` FAISS index
