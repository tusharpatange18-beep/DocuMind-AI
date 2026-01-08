from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"


def save_uploaded_pdf(file: UploadFile) -> str:
    """Save an uploaded PDF into data/raw_data/ and return the saved path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    dest = RAW_DATA_DIR / safe_name

    # UploadFile.file is a SpooledTemporaryFile (sync file-like object)
    with dest.open("wb") as f:
        try:
            file.file.seek(0)
        except Exception:
            pass
        # Use a larger buffer to reduce Python overhead on big PDFs.
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    return str(dest)
