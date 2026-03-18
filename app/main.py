from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, ReindexResponse
from app.rag import load_rag

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Mini RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_rag():
    return load_rag()


@app.post("/api/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    rag = get_rag()
    try:
        indexed_chunks, documents_processed = rag.build_index()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReindexResponse(
        indexed_chunks=indexed_chunks,
        documents_processed=documents_processed,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag = get_rag()
    try:
        answer, contexts, model_used = rag.answer(payload.question.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        question=payload.question.strip(),
        answer=answer,
        retrieved_context=contexts,
        model_used=model_used,
    )


@app.get("/")
def root() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
