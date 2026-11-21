from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.persistence import upload_local_pdf_to_remote
from app.supabase_storage import supabase_enabled

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"


def save_uploaded_pdf(file: UploadFile) -> str:
    """Save an uploaded PDF into data/raw_data/ and return the saved path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    # Avoid overwriting if multiple uploads share the same name.
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix or ".pdf"
    unique_name = f"{stem}-{uuid4().hex[:8]}{suffix}"
    dest = RAW_DATA_DIR / unique_name

    # UploadFile.file is a SpooledTemporaryFile (sync file-like object)
    with dest.open("wb") as f:
        try:
            file.file.seek(0)
        except Exception:
            pass
        # Use a larger buffer to reduce Python overhead on big PDFs.
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    # If exactly the same file was uploaded before, keep the old copy
    # and delete the new one, so the same PDF is never stored twice.
    new_hash = _file_hash(dest)
    for existing in RAW_DATA_DIR.glob("*.pdf"):
        if existing == dest:
            continue
        if existing.stat().st_size == dest.stat().st_size and _file_hash(existing) == new_hash:
            dest.unlink()
            return str(existing)

    # Persist to Supabase Storage when enabled.
    if supabase_enabled():
        upload_local_pdf_to_remote(local_path=dest, object_name=dest.name)

    return str(dest)


def _file_hash(path: Path) -> str:
    # A fingerprint of the file's contents; equal fingerprints = identical files.
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
