"""
Hybrid retriever for PinnacleRAG-DS.
Combines Dense and Sparse retrievers using Reciprocal Rank Fusion.
Domain filtering is threaded through to both sub-retrievers.
"""

from typing import Optional
from langchain_core.documents import Document

from config.settings import Settings
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, settings: Settings, dense: DenseRetriever, sparse: SparseRetriever):
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.alpha = settings.hybrid_alpha
        self.top_k = settings.top_k_retrieve

    def build_index(self, chunks: list[Document]) -> None:
        self.dense.build_index(chunks)
        self.sparse.build_index(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None, domain: Optional[str] = None) -> list[Document]:
        results = self.retrieve_with_scores(query, top_k, domain=domain)
        return [doc for doc, _ in results]

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None, domain: Optional[str] = None) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        dense_results = self.dense.retrieve(query, k, domain=domain)
        sparse_results = self.sparse.retrieve(query, k, domain=domain)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k)
        logger.debug(f"Hybrid retrieval: {len(fused)} results for query: {query[:50]}... (domain={domain})")
        return fused

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        doc_scores: dict[str, tuple[Document, float]] = {}

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:100]
            rrf_score = self.alpha / (rrf_k + rank + 1)
            if key in doc_scores:
                existing_doc, existing_score = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        for rank, (doc, _) in enumerate(sparse_results):
            key = doc.page_content[:100]
            rrf_score = (1.0 - self.alpha) / (rrf_k + rank + 1)
            if key in doc_scores:
                existing_doc, existing_score = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
