from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from backend.app.faiss_store import load_faiss_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data" / "processed_data" / "metadata.json"

TOP_K = 5


def retrieve_relevant_chunks(query: str):
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "metadata.json not found. Upload a PDF and rebuild the knowledge base first."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    index = load_faiss_index()

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    query_embedding = model.encode([query], convert_to_numpy=True)

    _distances, indices = index.search(query_embedding, TOP_K)

    results = []
    for idx in indices[0]:
        if 0 <= int(idx) < len(metadata):
            results.append(metadata[int(idx)])

    return results
