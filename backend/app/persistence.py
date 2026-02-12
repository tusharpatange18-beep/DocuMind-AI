from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.supabase_storage import (
    download_bytes,
    history_key,
    kb_folder,
    list_folder,
    pdfs_folder,
    supabase_enabled,
    upload_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"

HISTORY_FILE = PROJECT_ROOT / "chat_history.json"

# Core artifacts required for /ask
_KB_ARTIFACTS: list[tuple[Path, str, str]] = [
    (PROCESSED_DIR / "chunks.json", "chunks.json", "application/json"),
    (PROCESSED_DIR / "metadata.json", "metadata.json", "application/json"),
    (PROCESSED_DIR / "vectors.npy", "vectors.npy", "application/octet-stream"),
    (FAISS_DIR / "index.faiss", "index.faiss", "application/octet-stream"),
    (PROCESSED_DIR / "manifest.json", "manifest.json", "application/json"),
]


def sync_remote_pdfs_to_local() -> int:
    """Download PDFs from Supabase Storage into data/raw_data/.

    No-op for local deployments.
    Returns the number of files downloaded.
    """
    if not supabase_enabled():
        return 0

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    folder = pdfs_folder()
    # list_folder wants the folder path without the object name
    items = list_folder(folder=folder)
    downloaded = 0

    for item in items:
        name = (item.get("name") or "").strip()
        if not name.lower().endswith(".pdf"):
            continue

        object_key = f"{folder}/{name}" if folder else name
        dest = RAW_DATA_DIR / name
        if dest.exists():
            continue

        try:
            data = download_bytes(object_key=object_key)
            dest.write_bytes(data)
            downloaded += 1
        except Exception:
            # Ignore missing/failed objects; rebuild will surface errors if nothing is available.
            continue

    return downloaded


def upload_local_pdf_to_remote(*, local_path: Path, object_name: str) -> None:
    if not supabase_enabled():
        return

    folder = pdfs_folder()
    object_key = f"{folder}/{object_name}" if folder else object_name
    upload_bytes(object_key=object_key, data=local_path.read_bytes(), content_type="application/pdf")


def ensure_kb_artifacts() -> None:
    """Ensure KB artifacts exist locally; if missing, try downloading from Supabase."""
    if not supabase_enabled():
        return

    folder = kb_folder()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    for local_path, remote_name, _content_type in _KB_ARTIFACTS:
        if local_path.exists():
            continue
        object_key = f"{folder}/{remote_name}" if folder else remote_name
        try:
            data = download_bytes(object_key=object_key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        except Exception:
            # If it doesn't exist remotely either, the existing code will raise helpful errors.
            continue


def upload_kb_artifacts() -> None:
    """Upload KB artifacts to Supabase after a rebuild (no-op if disabled)."""
    if not supabase_enabled():
        return

    folder = kb_folder()
    for local_path, remote_name, content_type in _KB_ARTIFACTS:
        if not local_path.exists():
            continue
        object_key = f"{folder}/{remote_name}" if folder else remote_name
        upload_bytes(object_key=object_key, data=local_path.read_bytes(), content_type=content_type)


def load_history() -> list[dict[str, Any]]:
    """Load chat history.

    - Local mode: reads chat_history.json from disk
    - Supabase mode: reads a JSON blob from Storage (history/chat_history.json by default)
    """
    if supabase_enabled():
        try:
            raw = download_bytes(object_key=history_key())
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items: list[dict[str, Any]]) -> None:
    if supabase_enabled():
        payload = json.dumps(items, indent=2, ensure_ascii=False).encode("utf-8")
        upload_bytes(object_key=history_key(), data=payload, content_type="application/json")
        return

    HISTORY_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
