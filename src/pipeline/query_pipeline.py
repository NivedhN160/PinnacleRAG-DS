"""
Query pipeline for PinnacleRAG-DS.
budget -> hybrid -> rerank -> generate -> citations -> usage
"""
from config.settings import Settings
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.llm import GroqLLM
from src.embeddings.embedder import EmbeddingModel
from src.budget.guard import BudgetGuard

class QueryPipeline:
    def __init__(self, settings: Settings, budget_guard: BudgetGuard):
        self.settings = settings
        self.budget_guard = budget_guard
        
        self.embedder = EmbeddingModel(settings)
        dense = DenseRetriever(settings, self.embedder)
        sparse = SparseRetriever(settings)
        self.retriever = HybridRetriever(settings, dense, sparse)
        
        self.reranker = CrossEncoderReranker(settings)
        self.llm = GroqLLM(settings)

    def run(self, query: str) -> dict:
        self.budget_guard.check_budget()
        
        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(query, top_k=self.settings.top_k_retrieve)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have any relevant information to answer this question. Please upload some documents.",
                "citations": [],
                "usage": {"llm_calls": 0, "tokens": 0, "budget_remaining_calls": self.budget_guard.get_remaining()},
                "mode": "simple"
            }
            
        # 2. Rerank
        reranked = self.reranker.rerank(query, retrieved_docs, top_k=self.settings.top_k_rerank)
        
        # 3. Generate
        gen_result = self.llm.generate(query, reranked)
        self.budget_guard.record_call()
        
        # Add budget remaining to usage
        usage = gen_result["usage"]
        usage["budget_remaining_calls"] = self.budget_guard.get_remaining()
        
        return {
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "usage": usage,
            "mode": "simple"
        }
