from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from app.persistence import ensure_kb_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTORS_PATH = PROJECT_ROOT / "data" / "processed_data" / "vectors.npy"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index" / "index.faiss"


def build_faiss_index() -> None:
    vectors = np.load(str(VECTORS_PATH))
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def load_faiss_index():
    ensure_kb_artifacts()
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Rebuild the knowledge base first.")
    return faiss.read_index(str(FAISS_INDEX_PATH))
