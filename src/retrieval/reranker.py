"""
Cross-encoder reranker for PinnacleRAG-DS.

Critical precision component: reorders initial retrieval candidates using a
free cross-encoder model for dramatically improved accuracy.
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Free cross-encoder reranker using sentence-transformers."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the cross-encoder reranker.

        Downloads the model on first use (one-time cost).

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._model_name = settings.reranker_model

        logger.info(f"Loading reranker model: {self._model_name}")
        self._model = CrossEncoder(self._model_name)
        logger.info(f"Reranker model loaded: {self._model_name}")

    def rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: User query string.
            documents: Candidate documents from initial retrieval.
            top_k: Number of top documents to return.

        Returns:
            Reranked list of top-k Documents.
        """
        results = self.rerank_with_scores(query, documents, top_k)
        return [doc for doc, _ in results]

    def rerank_with_scores(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents with relevance scores.

        Args:
            query: User query string.
            documents: Candidate documents from initial retrieval.
            top_k: Number of top documents to return.

        Returns:
            List of (Document, score) tuples sorted by relevance descending.
        """
        if not documents:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Pair documents with scores and sort
        doc_scores = list(zip(documents, scores.tolist()))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Add rerank score to metadata
        reranked = []
        for rank, (doc, score) in enumerate(doc_scores[:top_k]):
            enriched_metadata = {**doc.metadata, "rerank_score": round(score, 4), "rerank_position": rank + 1}
            enriched_doc = Document(page_content=doc.page_content, metadata=enriched_metadata)
            reranked.append((enriched_doc, score))

        logger.debug(
            f"Reranked {len(documents)} → {len(reranked)} documents "
            f"(top score: {reranked[0][1]:.4f})" if reranked else "No documents to rerank"
        )
        return reranked
