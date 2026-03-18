from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="meta-llama/llama-3.2-3b-instruct:free",
        alias="OPENROUTER_MODEL",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    top_k: int = Field(default=4, alias="TOP_K")
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    index_path: Path = BASE_DIR / "data" / "processed" / "chunks.index"
    chunk_store_path: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    frontend_dir: Path = BASE_DIR / "frontend"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
