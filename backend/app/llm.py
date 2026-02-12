from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import google.generativeai as genai
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Note: available model IDs can vary by API key/project.
# Defaulting to a commonly available "flash" model; override via GEMINI_MODEL.
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))


@lru_cache(maxsize=1)
def _get_model() -> genai.GenerativeModel:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def generate_answer(prompt: str) -> str:
    try:
        model = _get_model()
        response = model.generate_content(
            (prompt or "").strip(),
            request_options={"timeout": DEFAULT_TIMEOUT_SECONDS},
        )

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            parts: list[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        parts.append(part_text.strip())
            if parts:
                return "\n".join(parts).strip()

        return "Model returned no text output."
    except Exception as e:
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return "GOOGLE_API_KEY is invalid."
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return f"LLM request timed out after {DEFAULT_TIMEOUT_SECONDS}s."
        return f"LLM request failed: {msg}"
