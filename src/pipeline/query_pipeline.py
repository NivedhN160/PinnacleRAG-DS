"""
Query pipeline for PinnacleRAG-DS.
budget -> hybrid(domain) -> rerank -> generate -> citations -> usage
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

    def run(self, query: str, domain: str = "general", rewrite: bool = False, check_faithfulness: bool = False) -> dict:
        self.budget_guard.check_budget()
        
        domain = (domain or "general").lower().strip()
        
        from src.domains import get_domain_adapter
        adapter = get_domain_adapter(domain)
        
        if rewrite or getattr(self.settings, "enable_query_rewrite", False):
            query = self.llm.rewrite_query(query)
            self.budget_guard.record_call()
        
        # 1. Retrieve with domain filter
        retrieved_docs = self.retriever.retrieve(
            query, top_k=self.settings.top_k_retrieve, domain=domain
        )
        
        if not retrieved_docs:
            return {
                "answer": f"I don't have any relevant information for the '{domain}' domain to answer this question. Please upload some documents.",
                "citations": [],
                "usage": {"llm_calls": 0, "tokens": 0, "budget_remaining_calls": self.budget_guard.get_remaining()},
                "mode": "simple",
                "domain": domain,
            }
            
        # 2. Rerank
        reranked = self.reranker.rerank(query, retrieved_docs, top_k=self.settings.top_k_rerank)
        
        # 3. Generate
        gen_result = self.llm.generate(query, reranked, system_prompt=adapter.get_system_prompt())
        self.budget_guard.record_call()
        
        answer = adapter.post_process_answer(gen_result["answer"], reranked)
        gen_result["answer"] = answer
        
        if check_faithfulness or getattr(self.settings, "enable_faithfulness_check", False):
            is_faithful = self.llm.check_faithfulness(answer, reranked)
            self.budget_guard.record_call()
            if not is_faithful:
                gen_result["answer"] += "\n\n[Warning: The faithfulness check determined that part of this answer might not be fully supported by the provided context.]"
                gen_result["faithfulness_warning"] = True
        
        # Add budget remaining to usage
        usage = gen_result["usage"]
        usage["budget_remaining_calls"] = self.budget_guard.get_remaining()
        
        response = {
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "usage": usage,
            "mode": "simple",
            "domain": domain,
        }
        
        if rewrite or getattr(self.settings, "enable_query_rewrite", False):
            response["rewritten_query"] = query
            
        return response
