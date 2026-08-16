"""
Tests for the retrieval module — hybrid retriever and reranker.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock Settings with a temp vectorstore path."""
    settings = MagicMock(spec=Settings)
    settings.hybrid_alpha = 0.5
    settings.top_k_retrieve = 5
    settings.top_k_rerank = 3
    settings.embedding_model_name = "all-MiniLM-L6-v2"
    settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    settings.vectorstore_path = str(tmp_path / "vectorstore")
    return settings


@pytest.fixture
def sample_chunks():
    """Sample chunks for retrieval testing."""
    return [
        Document(
            page_content="Python is a high-level programming language known for readability.",
            metadata={"source": "python.txt", "chunk_id": "a1"},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml.txt", "chunk_id": "b2"},
        ),
        Document(
            page_content="RAG combines retrieval with generation for factual answers.",
            metadata={"source": "rag.txt", "chunk_id": "c3"},
        ),
        Document(
            page_content="Vector databases store embeddings for similarity search.",
            metadata={"source": "vectors.txt", "chunk_id": "d4"},
        ),
    ]


class TestHybridRetriever:
    """Tests for HybridRetriever (uses mocked embedder to avoid model download)."""

    def test_build_index_empty(self, mock_settings):
        from src.retrieval.hybrid import HybridRetriever

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = []

        retriever = HybridRetriever(mock_settings, mock_embedder)
        retriever.build_index([])

        # Should not crash on empty input

    def test_retrieve_empty_index(self, mock_settings):
        from src.retrieval.hybrid import HybridRetriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384

        retriever = HybridRetriever(mock_settings, mock_embedder)
        results = retriever.retrieve("test query")

        assert results == []

    def test_build_and_retrieve(self, mock_settings, sample_chunks):
        from src.retrieval.hybrid import HybridRetriever

        mock_embedder = MagicMock()
        # Return mock embeddings
        mock_embedder.embed_documents.return_value = [
            [float(i)] * 384 for i in range(len(sample_chunks))
        ]
        mock_embedder.embed_query.return_value = [1.0] * 384

        retriever = HybridRetriever(mock_settings, mock_embedder)
        retriever.build_index(sample_chunks)

        results = retriever.retrieve("What is Python?", top_k=3)
        assert len(results) <= 3

    def test_retrieve_with_scores_returns_tuples(self, mock_settings, sample_chunks):
        from src.retrieval.hybrid import HybridRetriever

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [
            [float(i)] * 384 for i in range(len(sample_chunks))
        ]
        mock_embedder.embed_query.return_value = [1.0] * 384

        retriever = HybridRetriever(mock_settings, mock_embedder)
        retriever.build_index(sample_chunks)

        results = retriever.retrieve_with_scores("test", top_k=2)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker (mocked to avoid model download)."""

    def test_rerank_empty(self):
        from src.retrieval.reranker import CrossEncoderReranker

        mock_settings = MagicMock()
        mock_settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        with patch.object(CrossEncoderReranker, "__init__", lambda self, s: None):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker.settings = mock_settings

            result = reranker.rerank("query", [], top_k=3)
            assert result == []

    def test_rerank_preserves_content(self):
        """Verify reranking doesn't alter document content."""
        import numpy as np
        from src.retrieval.reranker import CrossEncoderReranker

        mock_settings = MagicMock()

        with patch.object(CrossEncoderReranker, "__init__", lambda self, s: None):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker.settings = mock_settings
            reranker._model = MagicMock()
            reranker._model.predict.return_value = np.array([0.9, 0.3, 0.7])

            docs = [
                Document(page_content="Doc A", metadata={"id": "a"}),
                Document(page_content="Doc B", metadata={"id": "b"}),
                Document(page_content="Doc C", metadata={"id": "c"}),
            ]

            reranked = reranker.rerank("query", docs, top_k=2)

            assert len(reranked) == 2
            # Highest score doc should be first
            assert reranked[0].page_content == "Doc A"  # score 0.9
