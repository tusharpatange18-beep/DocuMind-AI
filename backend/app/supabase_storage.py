from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def supabase_enabled() -> bool:
    return (os.getenv("STORAGE_BACKEND", "local") or "local").strip().lower() == "supabase"


def _required_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"{name} is not set")
    return val


@lru_cache(maxsize=1)
def _client():
    if not supabase_enabled():
        raise RuntimeError("Supabase storage is not enabled")

    # Lazy import so local-only deployments don't need the dependency.
    from supabase import create_client  # type: ignore

    url = _required_env("SUPABASE_URL")
    key = _required_env("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def _bucket() -> str:
    return (os.getenv("SUPABASE_STORAGE_BUCKET") or "").strip() or "rag-chatbot"


def _prefix() -> str:
    # Optional namespace within the bucket
    p = (os.getenv("SUPABASE_STORAGE_PREFIX") or "").strip().strip("/")
    return p


def _join(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(cleaned)


def upload_bytes(*, object_key: str, data: bytes, content_type: str) -> None:
    client = _client()
    bucket = _bucket()

    # supabase-py expects file content as bytes
    try:
        client.storage.from_(bucket).upload(
            path=object_key,
            file=data,
            file_options={"content-type": content_type, "upsert": True},
        )
    except Exception as e:
        raise RuntimeError(f"Supabase upload failed for {object_key}: {e}")


def download_bytes(*, object_key: str) -> bytes:
    client = _client()
    bucket = _bucket()

    try:
        data = client.storage.from_(bucket).download(object_key)
        if isinstance(data, bytes):
            return data
        # Some versions may return a bytearray-like object
        return bytes(data)
    except Exception as e:
        raise FileNotFoundError(f"Supabase download failed for {object_key}: {e}")


def list_folder(*, folder: str) -> list[dict[str, Any]]:
    client = _client()
    bucket = _bucket()

    folder = folder.strip("/")
    try:
        items = client.storage.from_(bucket).list(path=folder)
        if not isinstance(items, list):
            return []
        # Items are dicts with at least a 'name'
        return [i for i in items if isinstance(i, dict)]
    except Exception:
        return []


def pdfs_folder() -> str:
    return _join(_prefix(), os.getenv("SUPABASE_PDFS_FOLDER", "pdfs"))


def kb_folder() -> str:
    return _join(_prefix(), os.getenv("SUPABASE_KB_FOLDER", "kb"))


def history_key() -> str:
    # Single global history file by default (you can change later to per-user if you add auth).
    return _join(_prefix(), os.getenv("SUPABASE_HISTORY_KEY", "history/chat_history.json"))
