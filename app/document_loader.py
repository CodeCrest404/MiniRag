from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.config import get_settings
from app.models import Chunk


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class LoadedDocument:
    name: str
    source_path: Path
    text: str


def load_documents() -> list[LoadedDocument]:
    settings = get_settings()
    documents: list[LoadedDocument] = []

    for path in sorted(settings.raw_data_dir.glob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue

        text = _read_text(path).strip()
        if not text:
            continue
        documents.append(LoadedDocument(name=path.name, source_path=path, text=text))

    return documents


def chunk_documents(documents: list[LoadedDocument]) -> list[Chunk]:
    settings = get_settings()
    chunks: list[Chunk] = []

    for document in documents:
        pieces = _split_text(
            document.text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_path.stem}-{index}",
                    document_name=document.name,
                    source_path=str(document.source_path),
                    text=piece,
                    metadata={"chunk_index": index},
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    settings = get_settings()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.chunk_store_path.write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], indent=2),
        encoding="utf-8",
    )


def load_saved_chunks() -> list[Chunk]:
    settings = get_settings()
    if not settings.chunk_store_path.exists():
        return []
    raw = json.loads(settings.chunk_store_path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in raw]


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
    return chunks
