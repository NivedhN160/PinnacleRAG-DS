"""
Sparse retriever for PinnacleRAG-DS.
Uses BM25 for keyword search with domain-scoped filtering.
"""

import os
import pickle
from typing import Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SparseRetriever:
    def __init__(self, settings: Settings, vectorstore_path: Optional[str] = None):
        self.settings = settings
        vs_path = vectorstore_path or settings.vectorstore_path
        self._bm25_path = os.path.join(vs_path, "bm25_index.pkl")
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []
        self._load_bm25()

    def build_index(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        
        texts = [c.page_content for c in chunks]
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = list(chunks)
        self.save_index()
        logger.info(f"Sparse index built: BM25 with {len(chunks)} documents")

    def retrieve(self, query: str, top_k: int, domain: Optional[str] = None) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        
        # Filter by domain before ranking
        valid_indices = list(range(len(scores)))
        if domain and domain.lower() not in ("general", "all", ""):
            valid_indices = [
                i for i in valid_indices
                if self._bm25_docs[i].metadata.get("domain", "general") == domain.lower()
            ]
        
        top_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        max_score = max((scores[i] for i in valid_indices), default=1.0)
        if max_score <= 0:
            max_score = 1.0
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = scores[idx] / max_score
                results.append((self._bm25_docs[idx], normalized_score))

        return results

    def save_index(self) -> None:
        if self._bm25 is not None:
            data = {"bm25": self._bm25, "docs": self._bm25_docs}
            with open(self._bm25_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"BM25 index saved to {self._bm25_path}")

    def _load_bm25(self) -> None:
        if os.path.exists(self._bm25_path):
            try:
                with open(self._bm25_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._bm25_docs = data["docs"]
                logger.info(f"BM25 index loaded: {len(self._bm25_docs)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
