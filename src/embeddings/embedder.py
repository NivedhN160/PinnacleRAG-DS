"""
Local free embedding model for PinnacleRAG-DS.

Uses sentence-transformers for consistent embeddings across indexing and querying.
No API key required — runs entirely locally.
"""

from typing import Optional

from sentence_transformers import SentenceTransformer

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Free local embedding model using sentence-transformers."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the embedding model.

        Downloads the model on first use (one-time cost).

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._model_name = settings.embedding_model_name

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded: {self._model_name} "
            f"(dimension={self._dimension})"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} document(s)")
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            batch_size=64,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
