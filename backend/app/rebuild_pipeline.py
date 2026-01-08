from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _EMBEDDER


def _current_manifest() -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for pdf_path in sorted(RAW_DATA_DIR.glob("*.pdf")):
        st = pdf_path.stat()
        manifest[pdf_path.name] = {
            "size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        }
    return manifest


def _load_manifest() -> dict[str, dict[str, int]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _outputs_exist() -> bool:
    return (
        CHUNKS_PATH.exists()
        and METADATA_PATH.exists()
        and VECTORS_PATH.exists()
        and FAISS_INDEX_PATH.exists()
    )


def _should_skip_rebuild(current: dict[str, dict[str, int]]) -> bool:
    previous = _load_manifest()
    if not previous:
        return False
    if previous != current:
        return False
    return _outputs_exist()


def _ocr_available() -> bool:
    try:
        import pytesseract  # type: ignore

        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image_png_bytes(png_bytes: bytes) -> str:
    try:
        import io
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        ocr_lang = os.getenv("OCR_LANG", "eng")
        ocr_psm = os.getenv("OCR_PSM", "6")
        ocr_oem = os.getenv("OCR_OEM", "1")
        config = f"--oem {ocr_oem} --psm {ocr_psm}"

        img = Image.open(io.BytesIO(png_bytes))
        text = pytesseract.image_to_string(img, lang=ocr_lang, config=config)
        return (text or "").strip()
    except Exception:
        return ""


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [s for s in out if s]


def _extract_pdf_pages_fast(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text per page using PyMuPDF when available; fallback to pypdf."""
    # PyMuPDF path (fast + enables rendering for OCR without Poppler)
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        results: list[tuple[int, str]] = []
        for page_no in range(1, doc.page_count + 1):
            page = doc.load_page(page_no - 1)
            text = (page.get_text("text") or "").strip()
            results.append((page_no, text))
        return results
    except Exception:
        pass

    # Fallback (digital text only)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        results = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            results.append((page_no, text))
        return results
    except Exception:
        return []


def _render_pages_for_ocr(pdf_path: Path) -> list[tuple[int, bytes]]:
    """Render pages to PNG bytes for OCR.

    Uses PyMuPDF when available; falls back to pdf2image (requires Poppler).
    """
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        try:
            zoom = float(os.getenv("OCR_ZOOM", "1.3"))
        except Exception:
            zoom = 1.3
        mat = fitz.Matrix(zoom, zoom)

        out: list[tuple[int, bytes]] = []
        for page_no in range(1, doc.page_count + 1):
            page = doc.load_page(page_no - 1)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out.append((page_no, pix.tobytes("png")))
        return out
    except Exception:
        pass

    try:
        from pdf2image import convert_from_path  # type: ignore

        poppler_path = os.getenv("POPPLER_PATH")
        try:
            dpi = int(os.getenv("OCR_DPI", "200"))
        except Exception:
            dpi = 200
        images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
        out2: list[tuple[int, bytes]] = []
        for i, img in enumerate(images, start=1):
            import io

            bio = io.BytesIO()
            img.save(bio, format="PNG")
            out2.append((i, bio.getvalue()))
        return out2
    except Exception:
        return []


def rebuild_knowledge_base() -> None:
    """Rebuild chunks, embeddings, and FAISS index from PDFs in data/raw_data/.

    Speed optimizations:
    - Skips rebuild when PDFs unchanged (manifest)
    - Uses PyMuPDF for fast extraction when available
    - OCRs only pages that have no extractable text
    - Parallel OCR on rendered images (no repeated PDF opens)
    - Batches embeddings
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("[rebuild] starting knowledge base rebuild...", flush=True)

    current = _current_manifest()
    if _should_skip_rebuild(current):
        print("[rebuild] skipped (no PDF changes detected).", flush=True)
        return

    try:
        embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    except Exception:
        embed_batch_size = 64
    embed_batch_size = max(1, min(256, embed_batch_size))

    try:
        ocr_workers = int(os.getenv("OCR_WORKERS", "2"))
    except Exception:
        ocr_workers = 2
    ocr_workers = max(1, min(8, ocr_workers))

    try:
        chunk_size = int(os.getenv("CHUNK_SIZE_CHARS", "2000"))
    except Exception:
        chunk_size = 2000
    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
    except Exception:
        chunk_overlap = 200

    ocr_is_available = _ocr_available()
    ocr_missing_hits = 0
    ocr_attempt_hits = 0

    chunks: list[dict] = []
    metadata: list[dict] = []

    pdfs = sorted(RAW_DATA_DIR.glob("*.pdf"))
    print(f"[rebuild] PDFs found: {len(pdfs)}", flush=True)

    for pdf_path in pdfs:
        print(f"[rebuild] extracting: {pdf_path.name}", flush=True)
        pages = _extract_pdf_pages_fast(pdf_path)
        if not pages:
            print(f"[rebuild] no pages/text extracted: {pdf_path.name}", flush=True)
            continue

        print(f"[rebuild] pages: {len(pages)}", flush=True)

        # If many pages have no extractable text, treat as scanned.
        probe = pages[: min(3, len(pages))]
        likely_scanned = all((t or "").strip() == "" for _pn, t in probe)

        if likely_scanned:
            print(f"[rebuild] detected scanned PDF: {pdf_path.name}", flush=True)
            if not ocr_is_available:
                ocr_missing_hits += len(pages)
                print("[rebuild] OCR unavailable; skipping scanned pages.", flush=True)
                continue

            rendered = _render_pages_for_ocr(pdf_path)
            if not rendered:
                # Could not render for OCR.
                ocr_attempt_hits += len(pages)
                print("[rebuild] failed to render pages for OCR; skipping.", flush=True)
                continue

            ocr_attempt_hits += len(rendered)
            ocr_text_by_page: dict[int, str] = {}
            print(f"[rebuild] OCR pages: {len(rendered)} (workers={ocr_workers})", flush=True)
            with ThreadPoolExecutor(max_workers=ocr_workers) as ex:
                fut_map = {ex.submit(_ocr_image_png_bytes, b): pn for pn, b in rendered}
                for fut in as_completed(fut_map):
                    pn = fut_map[fut]
                    try:
                        ocr_text_by_page[pn] = (fut.result() or "").strip()
                    except Exception:
                        ocr_text_by_page[pn] = ""

            for page_no, _ in pages:
                text = (ocr_text_by_page.get(page_no, "") or "").strip()
                if not text:
                    continue
                for part in _split_text(text, chunk_size, chunk_overlap):
                    chunks.append({"chunk_text": part, "source": pdf_path.name, "page": page_no})
                    metadata.append({"index": len(metadata), "source": pdf_path.name, "page": page_no})

            continue

        # Digital PDFs: embed extracted text; OCR only pages that are empty.
        empty_pages: list[int] = [pn for pn, t in pages if not (t or "").strip()]
        ocr_text_by_page: dict[int, str] = {}
        if empty_pages:
            print(f"[rebuild] empty pages requiring OCR: {len(empty_pages)}", flush=True)
            if ocr_is_available:
                rendered = _render_pages_for_ocr(pdf_path)
                rendered_map = {pn: b for pn, b in rendered if pn in set(empty_pages)}
                if rendered_map:
                    ocr_attempt_hits += len(rendered_map)
                    print(
                        f"[rebuild] OCR empty pages: {len(rendered_map)} (workers={min(ocr_workers, len(rendered_map))})",
                        flush=True,
                    )
                    with ThreadPoolExecutor(max_workers=min(ocr_workers, len(rendered_map))) as ex:
                        fut_map = {ex.submit(_ocr_image_png_bytes, b): pn for pn, b in rendered_map.items()}
                        for fut in as_completed(fut_map):
                            pn = fut_map[fut]
                            try:
                                ocr_text_by_page[pn] = (fut.result() or "").strip()
                            except Exception:
                                ocr_text_by_page[pn] = ""
            else:
                ocr_missing_hits += len(empty_pages)
                print("[rebuild] OCR unavailable; leaving empty pages blank.", flush=True)

        for page_no, text in pages:
            text = (text or "").strip()
            if not text:
                text = (ocr_text_by_page.get(page_no, "") or "").strip()
            if not text:
                continue
            for part in _split_text(text, chunk_size, chunk_overlap):
                chunks.append({"chunk_text": part, "source": pdf_path.name, "page": page_no})
                metadata.append({"index": len(metadata), "source": pdf_path.name, "page": page_no})

    if not chunks:
        if ocr_missing_hits > 0 and not ocr_is_available:
            raise ValueError(
                "No extractable text found. This often happens with scanned PDFs. "
                "Install Tesseract OCR and set TESSERACT_CMD, and ensure pytesseract + pillow are installed."
            )
        if ocr_attempt_hits > 0 and ocr_is_available:
            raise ValueError(
                "No extractable text found. OCR was attempted but produced no text. "
                "Verify Tesseract is installed, TESSERACT_CMD is correct, and try increasing OCR_ZOOM."
            )
        raise ValueError("No valid text found in uploaded PDFs.")

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[rebuild] chunks: {len(chunks)}", flush=True)

    texts = [c["chunk_text"] for c in chunks]
    model = _get_embedder()

    print(f"[rebuild] embedding (batch_size={embed_batch_size})...", flush=True)
    emb_batches: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, embed_batch_size):
        batch = texts[start : start + embed_batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            batch_size=embed_batch_size,
            show_progress_bar=False,
        )
        emb_batches.append(emb)
        done = min(start + len(batch), total)
        print(f"[rebuild] embedded {done}/{total}", flush=True)

    embeddings = np.vstack(emb_batches)

    np.save(str(VECTORS_PATH), embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    MANIFEST_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - started
    print(f"[rebuild] done in {elapsed:.1f}s", flush=True)
