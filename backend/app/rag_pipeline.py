from __future__ import annotations

import json
from pathlib import Path

from app.llm import generate_answer
from app.prompt import build_prompt
from app.retriever import retrieve_relevant_chunks
from app.persistence import ensure_kb_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_data" / "chunks.json"


def run_rag(question: str):
    ensure_kb_artifacts()
    retrieved = retrieve_relevant_chunks(question)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. Rebuild the knowledge base first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    context_parts: list[str] = []
    for item in retrieved:
        idx = int(item.get("index", -1))
        if 0 <= idx < len(chunks):
            context_parts.append(chunks[idx]["chunk_text"])

    context = "\n\n".join(context_parts)
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    return answer, retrieved
