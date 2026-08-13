"""
Ingestion pipeline for PinnacleRAG-DS.
"""
from config.settings import Settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingModel
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

class IngestPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = DocumentLoader(settings)
        self.cleaner = TextCleaner(settings)
        self.chunker = DocumentChunker(settings)
        self.embedder = EmbeddingModel(settings)
        
        dense = DenseRetriever(settings, self.embedder)
        sparse = SparseRetriever(settings)
        self.retriever = HybridRetriever(settings, dense, sparse)

    def run(self, data_path: str = None) -> dict:
        path = data_path or self.settings.raw_data_path
        raw_docs = self.loader.load_directory(path)
        if not raw_docs:
            return {"status": "no_documents", "documents_loaded": 0}
            
        cleaned = self.cleaner.clean_batch(raw_docs)
        chunks = self.chunker.chunk_documents(cleaned)
        
        self.retriever.build_index(chunks)
        
        return {
            "status": "success",
            "documents_loaded": len(raw_docs),
            "chunks_created": len(chunks),
        }
