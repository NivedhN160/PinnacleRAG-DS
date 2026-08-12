"""
PinnacleRAG-DS Settings Module.

Single source of truth for all tunable parameters.
Loads configuration from environment variables / .env file.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunable parameters for PinnacleRAG-DS."""

    # ── Groq LLM ──────────────────────────────────────────────────────
    groq_api_key: str = Field(
        ...,
        description="Groq API key (only required secret)",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for generation",
    )
    temperature: float = Field(
        default=0.1,
        description="LLM temperature (low for factual RAG)",
    )
    max_tokens: int = Field(
        default=2048,
        description="Max tokens for LLM generation",
    )

    # ── Embedding Model ───────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings (free, local)",
    )

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlap between consecutive chunks",
    )

    # ── Retrieval ─────────────────────────────────────────────────────
    top_k_retrieve: int = Field(
        default=20,
        description="Number of documents to retrieve before reranking",
    )
    top_k_rerank: int = Field(
        default=5,
        description="Number of documents to keep after reranking",
    )
    hybrid_alpha: float = Field(
        default=0.5,
        description="Weight for dense retrieval (1-alpha goes to BM25)",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking (free, local)",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    raw_data_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "raw"),
        description="Path to raw documents",
    )
    processed_data_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "processed"),
        description="Path to processed/chunked data",
    )
    vectorstore_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "vectorstore"),
        description="Path to persistent vector store",
    )
    golden_set_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "golden" / "golden_set.json"),
        description="Path to golden Q&A set for evaluation",
    )

    # ── Evaluation Thresholds ─────────────────────────────────────────
    min_faithfulness: float = Field(
        default=0.85,
        description="Minimum acceptable faithfulness score",
    )
    min_answer_relevancy: float = Field(
        default=0.80,
        description="Minimum acceptable answer relevancy score",
    )
    min_context_precision: float = Field(
        default=0.75,
        description="Minimum acceptable context precision score",
    )
    min_context_recall: float = Field(
        default=0.75,
        description="Minimum acceptable context recall score",
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and return validated settings singleton."""
    return Settings()
