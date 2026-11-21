from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.persistence import load_history as _load_history
from app.persistence import save_history as _save_history
from app.persistence import sync_remote_pdfs_to_local
from app.persistence import upload_kb_artifacts
from app.pdf_ingestion import save_uploaded_pdf
from app.rag_pipeline import run_rag
from app.rebuild_pipeline import rebuild_knowledge_base
from app.retriever import clear_retriever_caches

router = APIRouter()

ASK_TIMEOUT_SECONDS = float(os.getenv("ASK_TIMEOUT_SECONDS", "120"))


class HistoryTurn(BaseModel):
    # One earlier question and answer from the same chat.
    question: str = ""
    answer: str = ""


class AskRequest(BaseModel):
    question: str
    # Recent turns of this chat (chat memory); empty for a new chat.
    history: list[HistoryTurn] = []


class SourceItem(BaseModel):
    source: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


@router.get("/")
async def health_check():
    return {"status": "DocuMind AI backend running"}


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        saved_path = await run_in_threadpool(save_uploaded_pdf, file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(e)}")

    # Index the PDF right away, so it can be searched as soon as
    # the upload finishes (no need to press "Rebuild Index").
    try:
        await run_in_threadpool(rebuild_knowledge_base)
        clear_retriever_caches()
        await run_in_threadpool(upload_kb_artifacts)
    except ValueError as e:
        # e.g. a scanned PDF with no readable text
        raise HTTPException(status_code=400, detail=f"PDF saved but could not be read: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF saved but indexing failed: {str(e)}")

    return {"success": True, "filename": Path(saved_path).name}


@router.post("/rebuild-knowledge-base")
async def rebuild_kb():
    try:
        # In hosted mode with remote storage, make sure all uploaded PDFs are present locally.
        await run_in_threadpool(sync_remote_pdfs_to_local)
        await run_in_threadpool(rebuild_knowledge_base)
        # Ensure the next /ask sees the updated index + metadata.
        clear_retriever_caches()
        # Persist rebuilt artifacts so /ask works after restarts.
        await run_in_threadpool(upload_kb_artifacts)
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
        try:
            with anyio.fail_after(ASK_TIMEOUT_SECONDS):
                # Keep only the last 3 turns, as plain dicts.
                recent_turns = [t.model_dump() for t in req.history[-3:]]
                answer, sources = await run_in_threadpool(run_rag, question, recent_turns)
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timed out after {int(ASK_TIMEOUT_SECONDS)}s while generating an answer.",
            )

        history = _load_history()
        history.append({"question": question, "answer": answer, "sources": sources})
        _save_history(history)

        return {"answer": answer, "sources": sources}
    except FileNotFoundError:
        # No PDFs have been uploaded and indexed yet.
        raise HTTPException(
            status_code=400,
            detail="No documents yet. Click \"Upload PDF\" to add a document, then ask your question.",
        )
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
