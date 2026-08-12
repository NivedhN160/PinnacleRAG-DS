"""
Tests for the ingestion module — loader, cleaner, and chunker.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """Create a mock Settings object for testing."""
    settings = MagicMock(spec=Settings)
    settings.chunk_size = 200
    settings.chunk_overlap = 30
    settings.raw_data_path = str(Path(__file__).parent / "fixtures")
    return settings


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first test document. It contains multiple sentences. "
            "The purpose is to test the ingestion pipeline.",
            metadata={"source": "test1.txt", "filename": "test1.txt"},
        ),
        Document(
            page_content="# Heading One\n\nParagraph under heading one.\n\n"
            "## Heading Two\n\nParagraph under heading two with more content.",
            metadata={"source": "test2.md", "filename": "test2.md"},
        ),
        Document(
            page_content="   \n\n  \t  \n  ",  # Empty/whitespace document
            metadata={"source": "empty.txt", "filename": "empty.txt"},
        ),
    ]


# ── DocumentLoader Tests ─────────────────────────────────────────────

class TestDocumentLoader:
    def test_supported_extensions(self, mock_settings):
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader(mock_settings)
        exts = loader.supported_extensions()
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".md" in exts
        assert ".docx" in exts

    def test_load_directory_missing(self, mock_settings):
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader(mock_settings)
        docs = loader.load_directory("/nonexistent/path")
        assert docs == []

    def test_load_file_unsupported(self, mock_settings):
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader(mock_settings)
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_file("test.xyz")


# ── TextCleaner Tests ────────────────────────────────────────────────

class TestTextCleaner:
    def test_clean_document_normalizes_whitespace(self, mock_settings):
        from src.ingestion.cleaner import TextCleaner
        cleaner = TextCleaner(mock_settings)

        doc = Document(
            page_content="Hello   world\t\ttest\n\n\n\n\n\nend",
            metadata={"source": "test.txt"},
        )
        cleaned = cleaner.clean_document(doc)

        assert "\t" not in cleaned.page_content
        assert "\n\n\n\n\n\n" not in cleaned.page_content

    def test_clean_batch_removes_empty(self, mock_settings, sample_documents):
        from src.ingestion.cleaner import TextCleaner
        cleaner = TextCleaner(mock_settings)

        cleaned = cleaner.clean_batch(sample_documents)
        # Should remove the empty document
        assert len(cleaned) == 2

    def test_extract_metadata(self, mock_settings):
        from src.ingestion.cleaner import TextCleaner
        cleaner = TextCleaner(mock_settings)

        doc = Document(
            page_content="Hello world. This is a test. It has three sentences.",
            metadata={"source": "test.txt"},
        )
        metadata = cleaner.extract_metadata(doc)

        assert "word_count" in metadata
        assert "char_count" in metadata
        assert "sentence_count" in metadata
        assert metadata["word_count"] > 0

    def test_encoding_fix(self, mock_settings):
        from src.ingestion.cleaner import TextCleaner
        cleaner = TextCleaner(mock_settings)

        doc = Document(
            page_content="Smart \u2018quotes\u2019 and \u2013 dashes",
            metadata={"source": "test.txt"},
        )
        cleaned = cleaner.clean_document(doc)
        assert "\u2018" not in cleaned.page_content
        assert "\u2019" not in cleaned.page_content


# ── DocumentChunker Tests ────────────────────────────────────────────

class TestDocumentChunker:
    def test_chunk_single_small_doc(self, mock_settings):
        from src.ingestion.chunker import DocumentChunker
        chunker = DocumentChunker(mock_settings)

        doc = Document(page_content="Short text.", metadata={"source": "test.txt"})
        chunks = chunker.chunk_single(doc)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_chunk_single_large_doc(self, mock_settings):
        from src.ingestion.chunker import DocumentChunker
        mock_settings.chunk_size = 50
        mock_settings.chunk_overlap = 10
        chunker = DocumentChunker(mock_settings)

        long_text = "This is a sentence. " * 20
        doc = Document(page_content=long_text, metadata={"source": "long.txt"})
        chunks = chunker.chunk_single(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_documents_batch(self, mock_settings, sample_documents):
        from src.ingestion.chunker import DocumentChunker
        chunker = DocumentChunker(mock_settings)

        non_empty = [d for d in sample_documents if d.page_content.strip()]
        chunks = chunker.chunk_documents(non_empty)

        assert len(chunks) >= len(non_empty)

    def test_chunk_stats(self, mock_settings):
        from src.ingestion.chunker import DocumentChunker
        chunker = DocumentChunker(mock_settings)

        chunks = [
            Document(page_content="Hello world", metadata={}),
            Document(page_content="Foo bar baz", metadata={}),
        ]
        stats = chunker.get_chunk_stats(chunks)

        assert stats["count"] == 2
        assert stats["avg_size"] > 0
        assert stats["min_size"] > 0
        assert stats["max_size"] > 0

    def test_chunk_empty_returns_empty(self, mock_settings):
        from src.ingestion.chunker import DocumentChunker
        chunker = DocumentChunker(mock_settings)

        doc = Document(page_content="", metadata={"source": "empty.txt"})
        chunks = chunker.chunk_single(doc)
        assert chunks == []
