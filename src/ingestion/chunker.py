"""
Structure-aware document chunking for PinnacleRAG-DS.

High-leverage accuracy lever: recursive + heading-aware splitting with overlap.
"""

import re
import uuid
from typing import Optional

from langchain_core.documents import Document

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Chunk documents using structure-aware recursive splitting."""

    # Separators ordered from most to least significant
    SEPARATORS = [
        "\n\n\n",   # Section breaks
        "\n\n",      # Paragraph breaks
        "\n",         # Line breaks
        ". ",         # Sentence boundaries
        " ",          # Word boundaries
    ]

    # Heading patterns for structure-aware splitting
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the chunker.

        Args:
            settings: Application settings (chunk_size, chunk_overlap).
        """
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        """
        Chunk a list of documents.

        Args:
            docs: Documents to chunk.

        Returns:
            List of chunked Document objects with metadata.
        """
        all_chunks: list[Document] = []
        for doc in docs:
            chunks = self.chunk_single(doc)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(docs)} document(s) into {len(all_chunks)} chunk(s)")
        stats = self.get_chunk_stats(all_chunks)
        logger.info(
            f"Chunk stats — avg: {stats['avg_size']:.0f} chars, "
            f"min: {stats['min_size']}, max: {stats['max_size']}, "
            f"total tokens: ~{stats['total_estimated_tokens']}"
        )
        return all_chunks

    def chunk_single(self, doc: Document) -> list[Document]:
        """
        Chunk a single document using structure-aware recursive splitting.

        Args:
            doc: Document to chunk.

        Returns:
            List of chunk Documents with inherited + chunk-specific metadata.
        """
        text = doc.page_content
        if not text.strip():
            return []

        # Try structure-aware splitting first (by headings)
        sections = self._split_by_headings(text)

        chunks: list[Document] = []
        for section_text, section_title in sections:
            # Recursively split each section if too large
            sub_chunks = self._recursive_split(section_text, self.SEPARATORS)

            # Apply overlap
            sub_chunks = self._apply_overlap(sub_chunks)

            for i, chunk_text in enumerate(sub_chunks):
                if not chunk_text.strip():
                    continue

                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": str(uuid.uuid4())[:8],
                    "chunk_index": len(chunks),
                    "chunk_size": len(chunk_text),
                    "section": section_title or "untitled",
                }
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunks

    def get_chunk_stats(self, chunks: list[Document]) -> dict:
        """
        Compute statistics about a set of chunks.

        Args:
            chunks: List of chunk Documents.

        Returns:
            Dict with avg_size, min_size, max_size, total_chars, total_estimated_tokens, count.
        """
        if not chunks:
            return {
                "avg_size": 0, "min_size": 0, "max_size": 0,
                "total_chars": 0, "total_estimated_tokens": 0, "count": 0,
            }

        sizes = [len(c.page_content) for c in chunks]
        total = sum(sizes)

        return {
            "avg_size": total / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": total,
            "total_estimated_tokens": total // 4,  # Rough estimate
            "count": len(sizes),
        }

    def _split_by_headings(self, text: str) -> list[tuple[str, Optional[str]]]:
        """
        Split text by markdown headings for structure-aware chunking.

        Returns list of (section_text, heading_title) tuples.
        """
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            return [(text, None)]

        sections: list[tuple[str, Optional[str]]] = []

        # Content before first heading
        if matches[0].start() > 0:
            pre_text = text[: matches[0].start()].strip()
            if pre_text:
                sections.append((pre_text, None))

        # Each heading section
        for i, match in enumerate(matches):
            heading_title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_text, heading_title))

        return sections

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using a hierarchy of separators.

        Tries the most significant separator first; falls back to less
        significant ones for chunks that are still too large.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: hard split by chunk_size
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If this single part is too large, split recursively
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            overlapped.append(overlap_text + chunks[i])

        return overlapped
