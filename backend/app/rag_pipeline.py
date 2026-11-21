from __future__ import annotations

import json
from pathlib import Path

from app.llm import generate_answer
from app.prompt import build_prompt
from app.retriever import retrieve_relevant_chunks
from app.persistence import ensure_kb_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_data" / "chunks.json"

# After each passage the search finds, also add the passages from this
# many following pages of the same PDF (a course here spans about 3 pages).
NEXT_PAGES = 2

# Most passages kept when combining the two searches (with chat memory).
MAX_PASSAGES = 6


def run_rag(question: str, history: list[dict] | None = None):
    ensure_kb_artifacts()
    history = history or []
    retrieved = find_passages(question, history)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. Rebuild the knowledge base first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    context_indexes = pick_context_indexes(retrieved, chunks)

    context_parts: list[str] = []
    for idx in context_indexes:
        chunk = chunks[idx]
        # Label each passage with its page so the model knows where it came from.
        context_parts.append(f"[Page {chunk['page']}]\n{chunk['chunk_text']}")

    context = "\n\n".join(context_parts)
    prompt = build_prompt(context, question, history)
    answer = generate_answer(prompt)

    return answer, retrieved


def find_passages(question: str, history: list[dict]) -> list[dict]:
    # Search 1: the new question on its own.
    alone = retrieve_relevant_chunks(question)
    if not history:
        return alone

    # Search 2: the new question plus the earlier questions of this chat,
    # so "what about its textbooks?" also searches for the course named before.
    earlier_questions = " ".join(turn.get("question", "") for turn in history)
    with_memory = retrieve_relevant_chunks(earlier_questions + " " + question)

    # Take results from both searches in turn, skipping repeats.
    merged: list[dict] = []
    seen: set[int] = set()
    for i in range(max(len(alone), len(with_memory))):
        for results in (with_memory, alone):
            if i < len(results) and results[i]["index"] not in seen:
                seen.add(results[i]["index"])
                merged.append(results[i])
    return merged[:MAX_PASSAGES]


def pick_context_indexes(retrieved: list[dict], chunks: list[dict]) -> list[int]:
    # For each passage found, also give the model the next few passages
    # from the same PDF. A topic often continues on the next page
    # (e.g. a course name on page 31, its module list on page 32).
    context_indexes: list[int] = []
    for item in retrieved:
        idx = int(item.get("index", -1))
        if idx < 0 or idx >= len(chunks):
            continue
        source = chunks[idx]["source"]
        last_page = int(chunks[idx]["page"]) + NEXT_PAGES
        next_idx = idx
        while next_idx < len(chunks):
            if chunks[next_idx]["source"] != source:
                break  # do not run into a different PDF
            if int(chunks[next_idx]["page"]) > last_page:
                break  # far enough
            if next_idx not in context_indexes:
                context_indexes.append(next_idx)
            next_idx += 1

    return context_indexes
