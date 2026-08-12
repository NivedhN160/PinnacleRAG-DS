"""
Common utility functions for PinnacleRAG-DS.

Token counting, file I/O, path helpers, and document formatting.
"""

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: Input text to tokenize.
        model: Tokenizer model name (tiktoken-compatible).

    Returns:
        Number of tokens.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def ensure_dir(path: str) -> Path:
    """
    Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to ensure.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_json(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """
    Save data as a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
    """
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def format_sources(docs: list[Document]) -> list[dict]:
    """
    Format a list of LangChain Documents into a clean source list.

    Args:
        docs: List of retrieved documents.

    Returns:
        List of dicts with source metadata.
    """
    sources = []
    for i, doc in enumerate(docs):
        source_info = {
            "rank": i + 1,
            "content_preview": doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        }
        # Include additional metadata if available
        for key in ("page", "chunk_id", "section", "word_count"):
            if key in doc.metadata:
                source_info[key] = doc.metadata[key]
        sources.append(source_info)
    return sources
