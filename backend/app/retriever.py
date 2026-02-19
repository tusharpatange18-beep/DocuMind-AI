from __future__ import annotations

import json
import os
from pathlib import Path
from functools import lru_cache

from app.faiss_store import load_faiss_index
from app.persistence import ensure_kb_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data" / "processed_data" / "metadata.json"

TOP_K = 5
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_embedding_model():
    # 🔥 Lazy import (important for Render startup)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def _get_metadata() -> list[dict]:
    ensure_kb_artifacts()
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "metadata.json not found. Upload a PDF and rebuild the knowledge base first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _get_index():
    return load_faiss_index()


def retrieve_relevant_chunks(query: str):
    metadata = _get_metadata()
    index = _get_index()
    model = _get_embedding_model()

    query_embedding = model.encode([query], convert_to_numpy=True)
    _distances, indices = index.search(query_embedding, TOP_K)

    results = []
    for idx in indices[0]:
        if 0 <= int(idx) < len(metadata):
            results.append(metadata[int(idx)])

    return results


def clear_retriever_caches() -> None:
    try:
        _get_metadata.cache_clear()
    except Exception:
        pass
    try:
        _get_index.cache_clear()
    except Exception:
        pass
