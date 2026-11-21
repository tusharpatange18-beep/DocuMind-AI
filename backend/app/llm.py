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

# Backup models, tried in order when the main one is out of quota or busy.
# Each Gemini model has its own free daily limit, so a backup usually works.
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "models/gemini-3-flash-preview,models/gemini-flash-lite-latest",
    ).split(",")
    if m.strip()
]


@lru_cache(maxsize=None)
def _get_model(model_name: str = MODEL_NAME) -> genai.GenerativeModel:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _is_quota_or_busy_error(message: str) -> bool:
    # 429 = out of quota, 503 = model overloaded, 404 = model not available.
    text = message.lower()
    return "429" in text or "503" in text or "404" in text or "quota" in text


def generate_answer(prompt: str) -> str:
    # Try the main model first, then each backup model.
    models_to_try = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    answer = ""
    for model_name in models_to_try:
        answer = _generate_with_model(model_name, prompt)
        if not answer.startswith("RETRY_NEXT_MODEL:"):
            return answer
    # Every model failed; show the last error.
    return answer.replace("RETRY_NEXT_MODEL:", "LLM request failed (all models):", 1)


def _generate_with_model(model_name: str, prompt: str) -> str:
    try:
        model = _get_model(model_name)
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

        # An empty reply also means "try the next model".
        return f"RETRY_NEXT_MODEL: {model_name} returned no text output."
    except Exception as e:
        msg = str(e)
        if _is_quota_or_busy_error(msg):
            return f"RETRY_NEXT_MODEL: {model_name}: {msg[:200]}"
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return "GOOGLE_API_KEY is invalid."
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return f"LLM request timed out after {DEFAULT_TIMEOUT_SECONDS}s."
        return f"LLM request failed: {msg}"
