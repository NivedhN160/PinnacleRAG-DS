"""
Dense retriever for PinnacleRAG-DS.
Uses ChromaDB for vector search with domain-scoped filtering.
"""

from typing import Optional
import chromadb
from langchain_core.documents import Document

from config.settings import Settings
from src.embeddings.embedder import EmbeddingModel
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

class DenseRetriever:
    COLLECTION_NAME = "pinnacle_rag"

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        vectorstore_path: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        
        vs_path = vectorstore_path or settings.vectorstore_path
        ensure_dir(vs_path)
        self._chroma_client = chromadb.PersistentClient(path=vs_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, chunks: list[Document]) -> None:
        if not chunks:
            return
            
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [{k: str(v) for k, v in c.metadata.items()} for c in chunks]

        try:
            self._collection.delete(ids=self._collection.get()["ids"])
        except Exception:
            pass

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

    def retrieve(self, query: str, top_k: int, domain: Optional[str] = None) -> list[tuple[Document, float]]:
        if self._collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        actual_k = min(top_k, self._collection.count())

        # Build where filter for domain-scoped retrieval
        where_filter = None
        if domain and domain.lower() not in ("general", "all", ""):
            where_filter = {"domain": domain.lower()}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        docs_with_scores = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            score = 1.0 - results["distances"][0][i]
            docs_with_scores.append((doc, score))

        return docs_with_scores
