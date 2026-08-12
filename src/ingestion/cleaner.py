"""
Text cleaning and normalization for PinnacleRAG-DS.

Classic DS data-cleaning stage: normalize, remove noise, enrich metadata.
"""

import re
from typing import Optional

from langchain_core.documents import Document

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Clean and normalize document text for optimal chunking and retrieval."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the text cleaner.

        Args:
            settings: Application settings.
        """
        self.settings = settings

    def clean_document(self, doc: Document) -> Document:
        """
        Clean a single document's text content.

        Applies:
        - Whitespace normalization
        - Header/footer removal
        - Encoding fix
        - Metadata enrichment

        Args:
            doc: Input document.

        Returns:
            Cleaned document with enriched metadata.
        """
        text = doc.page_content

        # Normalize unicode
        text = self._fix_encoding(text)

        # Remove common noise patterns
        text = self._remove_noise(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Enrich metadata
        metadata = {**doc.metadata, **self.extract_metadata_from_text(text)}

        return Document(page_content=text, metadata=metadata)

    def clean_batch(self, docs: list[Document]) -> list[Document]:
        """
        Clean a batch of documents.

        Args:
            docs: List of documents to clean.

        Returns:
            List of cleaned documents (empty docs removed).
        """
        cleaned = []
        for doc in docs:
            try:
                result = self.clean_document(doc)
                if result.page_content.strip():
                    cleaned.append(result)
            except Exception as e:
                logger.error(f"Failed to clean document from {doc.metadata.get('source', 'unknown')}: {e}")

        removed_count = len(docs) - len(cleaned)
        if removed_count > 0:
            logger.info(f"Cleaned {len(cleaned)} documents ({removed_count} empty docs removed)")
        else:
            logger.info(f"Cleaned {len(cleaned)} documents")

        return cleaned

    def extract_metadata(self, doc: Document) -> dict:
        """
        Extract enrichment metadata from a document.

        Args:
            doc: Document to analyze.

        Returns:
            Dict with extracted metadata fields.
        """
        return self.extract_metadata_from_text(doc.page_content)

    def extract_metadata_from_text(self, text: str) -> dict:
        """
        Extract metadata fields from text content.

        Args:
            text: Text to analyze.

        Returns:
            Dict with word_count, char_count, sentence_count, has_code, has_tables.
        """
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "has_code": bool(re.search(r"```|def |class |import |function ", text)),
            "has_tables": bool(re.search(r"\|.*\|.*\|", text)),
        }

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        replacements = {
            "\u2018": "'", "\u2019": "'",   # Smart quotes
            "\u201c": '"', "\u201d": '"',   # Smart double quotes
            "\u2013": "-", "\u2014": "--",  # En/em dashes
            "\u2026": "...",                 # Ellipsis
            "\u00a0": " ",                   # Non-breaking space
            "\ufffd": "",                    # Replacement character
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_noise(self, text: str) -> str:
        """Remove common noise patterns (headers, footers, page numbers)."""
        # Remove page numbers (standalone numbers on a line)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Remove repeated separator patterns
        text = re.sub(r"[-=_]{10,}", "", text)

        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")

        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Collapse multiple spaces within lines (but not leading spaces)
        text = re.sub(r"([^\n ]) {2,}", r"\1 ", text)

        return text.strip()
