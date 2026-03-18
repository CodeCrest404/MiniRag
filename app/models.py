from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_name: str
    source_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_name: str
    source_path: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[RetrievedChunk]
    model_used: str


class ReindexResponse(BaseModel):
    indexed_chunks: int
    documents_processed: int
