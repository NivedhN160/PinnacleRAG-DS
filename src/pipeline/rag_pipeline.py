"""
PinnacleRAG Pipeline — end-to-end RAG orchestration.

Wires together all components: ingestion, embedding, hybrid retrieval,
cross-encoder reranking, and grounded generation. Supports a simple
query path and an optional self-correction loop.
"""

import time
from typing import Any, Optional

from langchain_core.documents import Document

from config.settings import Settings
from src.embeddings.embedder import EmbeddingModel
from src.generation.generator import GroqGenerator
from src.ingestion.chunker import DocumentChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.helpers import ensure_dir, format_sources
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class PinnacleRAGPipeline:
    """
    Main pipeline class that orchestrates the full RAG workflow.

    Ingest: Load → Clean → Chunk → Embed → Index
    Query:  Retrieve (Hybrid) → Rerank (Cross-Encoder) → Generate (Groq)
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize all pipeline components.

        Args:
            settings: Application settings.
        """
        setup_logging(settings.log_level)
        self.settings = settings

        logger.info("Initializing PinnacleRAG pipeline...")

        # Ingestion components
        self._loader = DocumentLoader(settings)
        self._cleaner = TextCleaner(settings)
        self._chunker = DocumentChunker(settings)

        # Embedding
        self._embedder = EmbeddingModel(settings)

        # Retrieval
        self._retriever = HybridRetriever(settings, self._embedder)
        self._reranker = CrossEncoderReranker(settings)

        # Generation
        self._generator = GroqGenerator(settings)

        logger.info("PinnacleRAG pipeline initialized successfully")

    def ingest(self, data_path: Optional[str] = None) -> dict:
        """
        Run the full ingestion pipeline: Load → Clean → Chunk → Embed → Index.

        Args:
            data_path: Path to raw documents (defaults to settings.raw_data_path).

        Returns:
            Dict with ingestion statistics.
        """
        path = data_path or self.settings.raw_data_path
        start_time = time.time()

        logger.info(f"Starting ingestion from: {path}")

        # Step 1: Load
        raw_docs = self._loader.load_directory(path)
        if not raw_docs:
            logger.warning("No documents found to ingest")
            return {"status": "no_documents", "documents_loaded": 0}

        # Step 2: Clean
        cleaned_docs = self._cleaner.clean_batch(raw_docs)

        # Step 3: Chunk
        chunks = self._chunker.chunk_documents(cleaned_docs)

        # Step 4: Build Index (embed + index)
        self._retriever.build_index(chunks)

        elapsed = time.time() - start_time
        stats = {
            "status": "success",
            "documents_loaded": len(raw_docs),
            "documents_after_cleaning": len(cleaned_docs),
            "total_chunks": len(chunks),
            "chunk_stats": self._chunker.get_chunk_stats(chunks),
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            f"Ingestion complete: {len(raw_docs)} docs → {len(cleaned_docs)} cleaned → "
            f"{len(chunks)} chunks in {elapsed:.1f}s"
        )
        return stats

    def query(self, question: str, return_sources: bool = True) -> dict:
        """
        Run the full query pipeline: Retrieve → Rerank → Generate.

        Args:
            question: User question.
            return_sources: Whether to include formatted sources in response.

        Returns:
            Dict with answer, sources, context_docs, and metadata.
        """
        start_time = time.time()

        logger.info(f"Query: {question[:80]}...")

        # Step 1: Hybrid Retrieval
        retrieved_docs = self._retriever.retrieve(question, self.settings.top_k_retrieve)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": [],
                "context_docs": [],
                "retrieval_count": 0,
            }

        # Step 2: Rerank
        reranked_docs = self._reranker.rerank(
            question, retrieved_docs, self.settings.top_k_rerank
        )

        # Step 3: Generate
        result = self._generator.generate(question, reranked_docs)

        elapsed = time.time() - start_time

        response = {
            "answer": result["answer"],
            "sources": result["sources"] if return_sources else [],
            "context_docs": reranked_docs,
            "retrieval_count": len(retrieved_docs),
            "reranked_count": len(reranked_docs),
            "model": result["model"],
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(f"Query answered in {elapsed:.1f}s (retrieved={len(retrieved_docs)}, reranked={len(reranked_docs)})")
        return response

    def query_with_correction(self, question: str, max_retries: int = 2) -> dict:
        """
        Query with optional agentic self-correction loop.

        If the initial answer has low confidence (e.g., "I cannot answer"),
        reformulates the query and tries again with a broader retrieval.

        Args:
            question: User question.
            max_retries: Maximum correction attempts.

        Returns:
            Dict with answer, sources, and correction metadata.
        """
        result = self.query(question)

        # Check if the answer indicates low confidence
        low_confidence_indicators = [
            "i cannot answer",
            "not enough information",
            "no relevant",
            "insufficient context",
        ]

        answer_lower = result["answer"].lower()
        needs_correction = any(ind in answer_lower for ind in low_confidence_indicators)

        if not needs_correction or max_retries <= 0:
            result["correction_applied"] = False
            return result

        logger.info("Low confidence detected — attempting self-correction")

        # Try with expanded retrieval (2x top_k)
        expanded_docs = self._retriever.retrieve(question, self.settings.top_k_retrieve * 2)

        if expanded_docs:
            expanded_reranked = self._reranker.rerank(
                question, expanded_docs, self.settings.top_k_rerank * 2
            )
            retry_result = self._generator.generate(question, expanded_reranked)

            result["answer"] = retry_result["answer"]
            result["sources"] = retry_result["sources"]
            result["context_docs"] = expanded_reranked
            result["correction_applied"] = True
            result["correction_strategy"] = "expanded_retrieval"
            logger.info("Self-correction applied: expanded retrieval")

        return result

    def get_retriever(self) -> HybridRetriever:
        """Return the hybrid retriever component."""
        return self._retriever

    def get_generator(self) -> GroqGenerator:
        """Return the Groq generator component."""
        return self._generator

    def health_check(self) -> dict:
        """
        Check pipeline health and component status.

        Returns:
            Dict with component statuses and configuration info.
        """
        return {
            "status": "healthy",
            "components": {
                "loader": "ready",
                "cleaner": "ready",
                "chunker": "ready",
                "embedder": {
                    "status": "ready",
                    "model": self._embedder.model_name(),
                    "dimension": self._embedder.get_dimension(),
                },
                "retriever": {
                    "status": "ready",
                    "hybrid_alpha": self.settings.hybrid_alpha,
                },
                "reranker": {
                    "status": "ready",
                    "model": self.settings.reranker_model,
                },
                "generator": {
                    "status": "ready",
                    "model": self.settings.groq_model,
                },
            },
            "config": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "top_k_retrieve": self.settings.top_k_retrieve,
                "top_k_rerank": self.settings.top_k_rerank,
            },
        }
