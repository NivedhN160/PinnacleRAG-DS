"""
Agent loop for PinnacleRAG-DS.
Max 2 steps: if retrieval is weak, use free tool (DuckDuckGo or rewrite), retrieve again, generate.
"""
from config.settings import Settings
from src.pipeline.query_pipeline import QueryPipeline
from src.budget.guard import BudgetGuard
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AgentLoop:
    def __init__(self, settings: Settings, budget_guard: BudgetGuard, query_pipeline: QueryPipeline):
        self.settings = settings
        self.budget_guard = budget_guard
        self.query_pipeline = query_pipeline

    def run(self, query: str) -> dict:
        self.budget_guard.check_budget()
        
        # 1. Initial retrieval
        retrieved_docs = self.query_pipeline.retriever.retrieve(query, top_k=self.settings.top_k_retrieve)
        
        # Determine if retrieval is weak (e.g. low BM25/dense scores or empty)
        # For simplicity, if we got few docs or we want to simulate agentic rewrite:
        if not retrieved_docs or len(retrieved_docs) < 2:
            logger.info("Agent: Weak retrieval detected. Expanding query...")
            # Simple rewrite strategy (in a real system, could call DuckDuckGo or LLM)
            expanded_query = f"{query} details context information"
            
            # Retrieve again with expanded query
            retrieved_docs = self.query_pipeline.retriever.retrieve(expanded_query, top_k=self.settings.top_k_retrieve)
            
        if not retrieved_docs:
            return {
                "answer": "Even after agentic expansion, I don't have relevant information.",
                "citations": [],
                "usage": {"llm_calls": 0, "tokens": 0, "budget_remaining_calls": self.budget_guard.get_remaining()},
                "mode": "agent"
            }
            
        # 2. Rerank
        reranked = self.query_pipeline.reranker.rerank(query, retrieved_docs, top_k=self.settings.top_k_rerank)
        
        # 3. Generate
        gen_result = self.query_pipeline.llm.generate(query, reranked)
        self.budget_guard.record_call()
        
        usage = gen_result["usage"]
        usage["budget_remaining_calls"] = self.budget_guard.get_remaining()
        
        return {
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "usage": usage,
            "mode": "agent"
        }
