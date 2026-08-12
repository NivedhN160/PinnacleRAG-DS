"""
End-to-end pipeline tests for PinnacleRAG-DS.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock Settings with temp paths."""
    settings = MagicMock(spec=Settings)
    settings.groq_api_key = "test_key"
    settings.groq_model = "llama-3.3-70b-versatile"
    settings.temperature = 0.1
    settings.max_tokens = 512
    settings.embedding_model_name = "all-MiniLM-L6-v2"
    settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    settings.chunk_size = 200
    settings.chunk_overlap = 30
    settings.top_k_retrieve = 5
    settings.top_k_rerank = 3
    settings.hybrid_alpha = 0.5
    settings.raw_data_path = str(tmp_path / "raw")
    settings.processed_data_path = str(tmp_path / "processed")
    settings.vectorstore_path = str(tmp_path / "vectorstore")
    settings.golden_set_path = str(tmp_path / "golden_set.json")
    settings.log_level = "WARNING"
    settings.min_faithfulness = 0.85
    settings.min_answer_relevancy = 0.80
    settings.min_context_precision = 0.75
    settings.min_context_recall = 0.75
    return settings


class TestPipelineStructure:
    """Test that pipeline components are properly wired."""

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_pipeline_initializes(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        pipeline = PinnacleRAGPipeline(mock_settings)

        assert pipeline.settings == mock_settings
        mock_embed.assert_called_once_with(mock_settings)
        mock_generator.assert_called_once_with(mock_settings)

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_health_check(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        mock_embed_instance = mock_embed.return_value
        mock_embed_instance.model_name.return_value = "all-MiniLM-L6-v2"
        mock_embed_instance.get_dimension.return_value = 384

        pipeline = PinnacleRAGPipeline(mock_settings)
        health = pipeline.health_check()

        assert health["status"] == "healthy"
        assert "components" in health
        assert "config" in health

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_ingest_no_documents(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        pipeline = PinnacleRAGPipeline(mock_settings)
        result = pipeline.ingest()

        assert result["status"] == "no_documents"
        assert result["documents_loaded"] == 0

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_query_no_results(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        pipeline = PinnacleRAGPipeline(mock_settings)
        pipeline._retriever.retrieve.return_value = []

        result = pipeline.query("What is RAG?")

        assert "no relevant" in result["answer"].lower() or "ingest" in result["answer"].lower()
        assert result["retrieval_count"] == 0

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_query_returns_correct_structure(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from langchain_core.documents import Document
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        pipeline = PinnacleRAGPipeline(mock_settings)

        # Mock retrieval results
        mock_docs = [
            Document(page_content="Test content", metadata={"source": "test.txt"}),
        ]
        pipeline._retriever.retrieve.return_value = mock_docs
        pipeline._reranker.rerank.return_value = mock_docs
        pipeline._generator.generate.return_value = {
            "answer": "Test answer",
            "sources": [{"source": "test.txt"}],
            "raw_response": "Test answer",
            "model": "llama-3.3-70b-versatile",
            "context_docs_used": 1,
        }

        result = pipeline.query("Test question")

        assert "answer" in result
        assert "sources" in result
        assert "context_docs" in result
        assert "elapsed_seconds" in result

    @patch("src.pipeline.rag_pipeline.GroqGenerator")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.HybridRetriever")
    @patch("src.pipeline.rag_pipeline.EmbeddingModel")
    def test_get_retriever_and_generator(
        self, mock_embed, mock_retriever, mock_reranker, mock_generator, mock_settings
    ):
        from src.pipeline.rag_pipeline import PinnacleRAGPipeline

        pipeline = PinnacleRAGPipeline(mock_settings)

        assert pipeline.get_retriever() is not None
        assert pipeline.get_generator() is not None
