from __future__ import annotations

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.document_loader import chunk_documents, load_documents, load_saved_chunks, save_chunks
from app.models import Chunk, RetrievedChunk
from app.prompts import SYSTEM_PROMPT, build_user_prompt


class MiniRAG:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def ensure_index(self) -> None:
        if self.index is not None and self.chunks:
            return
        if self.settings.index_path.exists():
            self.index = faiss.read_index(str(self.settings.index_path))
            self.chunks = load_saved_chunks()
            if self.chunks:
                return
        self.build_index()

    def build_index(self) -> tuple[int, int]:
        documents = load_documents()
        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError(
                "No source documents found in data/raw. Add the assessment files first."
            )

        embeddings = self._embed_texts([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.index_path))
        save_chunks(chunks)

        self.index = index
        self.chunks = chunks
        return len(chunks), len(documents)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.ensure_index()
        assert self.index is not None

        query_embedding = self._embed_texts([question])
        limit = top_k or self.settings.top_k
        scores, indices = self.index.search(query_embedding, limit)

        retrieved: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    source_path=chunk.source_path,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
        return retrieved

    def answer(self, question: str) -> tuple[str, list[RetrievedChunk], str]:
        contexts = self.retrieve(question)
        model_used = self.settings.openrouter_model

        if not contexts:
            return (
                "The answer is not available in the provided documents.",
                [],
                model_used,
            )

        answer = self._generate_answer(question, contexts)
        return answer, contexts, model_used

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        prompt = build_user_prompt(
            question,
            [
                {"document_name": item.document_name, "text": item.text}
                for item in contexts
            ],
        )

        if self.settings.openrouter_api_key:
            return self._call_openrouter(prompt)
        return (
            "No LLM is configured. Retrieved context is available, but you must set "
            "OPENROUTER_API_KEY in .env to generate a final answer."
        )

    def _call_openrouter(self, prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.openrouter_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        if not response.ok:
            detail = response.text.strip()
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message") or payload.get("message") or detail
            except ValueError:
                pass
            raise RuntimeError(
                f"OpenRouter request failed ({response.status_code}) for model "
                f"'{self.settings.openrouter_model}': {detail}"
            )
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


def load_rag() -> MiniRAG:
    return MiniRAG()
