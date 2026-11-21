from __future__ import annotations

import difflib
import json
import math
import os
import re
from pathlib import Path
from functools import lru_cache

from app.faiss_store import load_faiss_index
from app.persistence import ensure_kb_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data" / "processed_data" / "metadata.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_data" / "chunks.json"

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


@lru_cache(maxsize=1)
def _get_chunk_texts() -> list[str]:
    # Text of every chunk, in the same order as metadata.json.
    if not CHUNKS_PATH.exists():
        return []
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [c.get("chunk_text", "") for c in chunks]


# Common words that say nothing about the topic; ignored by keyword search.
STOP_WORDS = {
    "the", "and", "for", "are", "was", "what", "which", "who", "how", "why",
    "when", "where", "with", "from", "this", "that", "there", "these", "those",
    "is", "of", "to", "in", "on", "a", "an", "it", "its", "be", "do", "does",
    "can", "you", "your", "about", "tell", "give", "list", "explain", "any",
    "all", "has", "have", "will", "would", "should", "into", "than", "then",
}

# Used when merging the two rankings (standard "reciprocal rank fusion" value).
RANK_FUSION_K = 60


def _words(text: str) -> list[str]:
    # Lowercase words, with a simple plural cut so "rounds" matches "round"
    # and "technologies" matches "technology".
    words = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return words


@lru_cache(maxsize=1)
def _get_chunk_word_sets() -> list[set[str]]:
    # The set of words in every chunk, so keyword search is fast.
    return [set(_words(text)) for text in _get_chunk_texts()]


def _keyword_ranking(query: str) -> list[int]:
    # Rank chunks by which question words they contain exactly.
    # Rare words (like "stipend") count much more than common ones.
    word_sets = _get_chunk_word_sets()
    total = len(word_sets)
    scores = [0.0] * total

    query_words = set()
    for word in _words(query):
        if len(word) > 2 and word not in STOP_WORDS:
            query_words.add(word)

    for word in query_words:
        chunks_with_word = sum(1 for s in word_sets if word in s)
        if chunks_with_word == 0:
            continue
        rarity = math.log((total + 1) / chunks_with_word)
        for i in range(total):
            if word in word_sets[i]:
                scores[i] += rarity

    # Only chunks that matched at least one word, best first.
    matched = [i for i in range(total) if scores[i] > 0]
    matched.sort(key=lambda i: scores[i], reverse=True)
    return matched


def _meaning_ranking(query: str) -> list[int]:
    # Rank chunks by meaning, using the embedding model and FAISS.
    index = _get_index()
    model = _get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    # The index is small (a few hundred chunks), so rank every chunk.
    _distances, indices = index.search(query_embedding, index.ntotal)
    return [int(i) for i in indices[0] if int(i) >= 0]


@lru_cache(maxsize=1)
def _get_vocabulary() -> list[str]:
    # Every distinct word (4+ letters) that appears in the PDFs.
    vocabulary = set()
    for text in _get_chunk_texts():
        for word in re.findall(r"[a-z]+", text.lower()):
            if len(word) >= 4:
                vocabulary.add(word)
    return sorted(vocabulary)


def _fix_spelling(query: str) -> str:
    # Replace misspelled words with the closest word found in the PDFs,
    # e.g. "microwve" -> "microwave", "commincation" -> "communication".
    vocabulary = _get_vocabulary()
    known = set(vocabulary)
    fixed_words = []
    for word in re.findall(r"[A-Za-z0-9]+", query):
        lower = word.lower()
        if len(lower) < 5 or lower in known or lower in STOP_WORDS:
            fixed_words.append(word)  # short, known or common: keep as is
            continue
        matches = difflib.get_close_matches(lower, vocabulary, n=1, cutoff=0.8)
        fixed_words.append(matches[0] if matches else word)
    return " ".join(fixed_words)


def retrieve_relevant_chunks(query: str):
    # Hybrid search: combine meaning search and keyword search.
    # Meaning search finds related ideas even with different words;
    # keyword search makes sure exact terms like "stipend" are not missed.
    metadata = _get_metadata()
    texts = _get_chunk_texts()
    query = _fix_spelling(query)

    # Each chunk gets points from its position in both rankings.
    fused_scores: dict[int, float] = {}
    for ranking in (_meaning_ranking(query), _keyword_ranking(query)):
        for position, idx in enumerate(ranking):
            points = 1.0 / (RANK_FUSION_K + position)
            fused_scores[idx] = fused_scores.get(idx, 0.0) + points

    best_first = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)

    results = []
    seen_texts = set()
    for idx in best_first:
        if idx < 0 or idx >= len(metadata):
            continue

        # Skip a chunk whose text we already picked (duplicate upload).
        text = texts[idx].strip() if idx < len(texts) else ""
        if text and text in seen_texts:
            continue
        seen_texts.add(text)

        results.append(metadata[idx])
        if len(results) == TOP_K:
            break

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
    try:
        _get_chunk_texts.cache_clear()
    except Exception:
        pass
    try:
        _get_chunk_word_sets.cache_clear()
    except Exception:
        pass
    try:
        _get_vocabulary.cache_clear()
    except Exception:
        pass
