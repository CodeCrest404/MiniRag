from __future__ import annotations

import requests
from collections import Counter
from math import sqrt

from app.config import get_settings
from app.document_loader import chunk_documents, load_documents, load_saved_chunks, save_chunks
from app.models import Chunk, RetrievedChunk
from app.prompts import SYSTEM_PROMPT, build_user_prompt


class MiniRAG:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunks: list[Chunk] = []

    def ensure_index(self) -> None:
        if self.chunks:
            return
        if self.settings.chunk_store_path.exists():
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

        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        save_chunks(chunks)

        self.chunks = chunks
        return len(chunks), len(documents)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.ensure_index()
        limit = top_k or self.settings.top_k
        query_terms = self._tokenize(question)
        if not query_terms:
            return []

        scored_chunks: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            score = self._score_chunk(query_terms, chunk.text)
            if score <= 0:
                continue
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        retrieved: list[RetrievedChunk] = []
        for score, chunk in scored_chunks[:limit]:
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

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.strip(".,!?;:\"'()[]{}").lower()
            for token in text.split()
            if token.strip(".,!?;:\"'()[]{}")
        ]

    def _score_chunk(self, query_terms: list[str], chunk_text: str) -> float:
        chunk_terms = self._tokenize(chunk_text)
        if not chunk_terms:
            return 0.0

        query_counts = Counter(query_terms)
        chunk_counts = Counter(chunk_terms)
        overlap = 0.0
        for term, count in query_counts.items():
            overlap += min(count, chunk_counts.get(term, 0))

        if overlap == 0:
            return 0.0

        return overlap / sqrt(len(query_terms) * len(chunk_terms))

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
