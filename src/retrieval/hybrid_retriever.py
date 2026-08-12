"""
Hybrid retriever for PinnacleRAG-DS.

Combines dense vector search (ChromaDB) with sparse BM25 search for
superior recall — the single largest accuracy improvement over pure vector search.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import Settings
from src.embeddings.embedder import EmbeddingModel
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class HybridRetriever:
    """Dense (vector) + Sparse (BM25) hybrid retrieval with score fusion."""

    COLLECTION_NAME = "pinnacle_rag"

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        vectorstore_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            settings: Application settings.
            embedder: Embedding model for dense retrieval.
            vectorstore_path: Override path for ChromaDB persistence.
        """
        self.settings = settings
        self.embedder = embedder
        self.alpha = settings.hybrid_alpha  # Weight for dense scores
        self.top_k = settings.top_k_retrieve

        # ChromaDB setup
        vs_path = vectorstore_path or settings.vectorstore_path
        ensure_dir(vs_path)
        self._chroma_client = chromadb.PersistentClient(path=vs_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (in-memory, persisted separately)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []
        self._bm25_path = os.path.join(vs_path, "bm25_index.pkl")

        # Try loading existing BM25 index
        self._load_bm25()

    def build_index(self, chunks: list[Document]) -> None:
        """
        Build/rebuild both dense and sparse indices from chunks.

        Args:
            chunks: List of chunked Document objects.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        logger.info(f"Building index from {len(chunks)} chunks")

        # ── Dense Index (ChromaDB) ─────────────────────────────────
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()}
            for c in chunks
        ]

        # Clear existing and add new
        try:
            self._collection.delete(ids=self._collection.get()["ids"])
        except Exception:
            pass

        # Add in batches (ChromaDB has limits)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self._collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"Dense index built: {len(chunks)} vectors in ChromaDB")

        # ── Sparse Index (BM25) ───────────────────────────────────
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = list(chunks)
        self._save_bm25()

        logger.info(f"Sparse index built: BM25 with {len(chunks)} documents")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        Retrieve documents using hybrid (dense + BM25) fusion.

        Args:
            query: User query string.
            top_k: Number of documents to return (overrides settings).

        Returns:
            List of top-k Documents sorted by fused score.
        """
        results = self.retrieve_with_scores(query, top_k)
        return [doc for doc, _ in results]

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with hybrid fusion scores.

        Args:
            query: User query string.
            top_k: Number of documents to return.

        Returns:
            List of (Document, score) tuples sorted by fused score descending.
        """
        k = top_k or self.top_k

        # ── Dense Retrieval ────────────────────────────────────────
        dense_results = self._dense_retrieve(query, k)

        # ── Sparse Retrieval ───────────────────────────────────────
        sparse_results = self._sparse_retrieve(query, k)

        # ── Reciprocal Rank Fusion ─────────────────────────────────
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k)

        logger.debug(f"Hybrid retrieval: {len(fused)} results for query: {query[:50]}...")
        return fused

    def save_index(self, path: str) -> None:
        """Save indices to disk (ChromaDB auto-persists; this saves BM25)."""
        self._save_bm25(path)

    def load_index(self, path: str) -> None:
        """Load BM25 index from disk."""
        self._load_bm25(path)

    def _dense_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Retrieve from ChromaDB dense index."""
        if self._collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        actual_k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_with_scores = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            # ChromaDB returns distances; convert to similarity
            score = 1.0 - results["distances"][0][i]
            docs_with_scores.append((doc, score))

        return docs_with_scores

    def _sparse_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Retrieve from BM25 sparse index."""
        if self._bm25 is None or not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        max_score = max(scores) if max(scores) > 0 else 1.0
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = scores[idx] / max_score
                results.append((self._bm25_docs[idx], normalized_score))

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).

        RRF is more robust than simple score interpolation because it
        normalizes across different score distributions.
        """
        doc_scores: dict[str, tuple[Document, float]] = {}

        # Score from dense results
        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:100]  # Use content prefix as key
            rrf_score = self.alpha / (rrf_k + rank + 1)
            if key in doc_scores:
                existing_doc, existing_score = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        # Score from sparse results
        for rank, (doc, _) in enumerate(sparse_results):
            key = doc.page_content[:100]
            rrf_score = (1.0 - self.alpha) / (rrf_k + rank + 1)
            if key in doc_scores:
                existing_doc, existing_score = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        # Sort by fused score
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _save_bm25(self, path: Optional[str] = None) -> None:
        """Persist BM25 index to disk."""
        save_path = path or self._bm25_path
        if self._bm25 is not None:
            data = {"bm25": self._bm25, "docs": self._bm25_docs}
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"BM25 index saved to {save_path}")

    def _load_bm25(self, path: Optional[str] = None) -> None:
        """Load BM25 index from disk."""
        load_path = path or self._bm25_path
        if os.path.exists(load_path):
            try:
                with open(load_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._bm25_docs = data["docs"]
                logger.info(f"BM25 index loaded: {len(self._bm25_docs)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
