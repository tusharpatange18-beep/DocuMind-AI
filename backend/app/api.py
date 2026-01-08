from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.pdf_ingestion import save_uploaded_pdf
from backend.app.rag_pipeline import run_rag
from backend.app.rebuild_pipeline import rebuild_knowledge_base

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_FILE = PROJECT_ROOT / "chat_history.json"


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(items: list[dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/")
async def health_check():
    return {"status": "DocuMind AI backend running"}


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        saved_path = await run_in_threadpool(save_uploaded_pdf, file)
        return {"success": True, "filename": Path(saved_path).name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(e)}")


@router.post("/rebuild-knowledge-base")
async def rebuild_kb():
    try:
        await run_in_threadpool(rebuild_knowledge_base)
        return {"success": True, "status": "Knowledge base rebuilt"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, sources = await run_in_threadpool(run_rag, question)

        history = _load_history()
        history.append({"question": question, "answer": answer, "sources": sources})
        _save_history(history)

        return {"answer": answer, "sources": sources}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")


@router.get("/history")
async def get_history():
    return _load_history()


@router.delete("/history")
async def clear_history():
    try:
        _save_history([])
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
