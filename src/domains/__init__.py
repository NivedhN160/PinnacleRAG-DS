from src.domains.trading.adapter import TradingAdapter
from src.domains.security.adapter import SecurityAdapter
from src.domains.seo.adapter import SEOAdapter
from src.domains.base import BaseDomainAdapter

def get_domain_adapter(domain: str) -> BaseDomainAdapter:
    domain = (domain or "general").lower().strip()
    if domain == "trading":
        return TradingAdapter()
    elif domain == "security":
        return SecurityAdapter()
    elif domain == "seo":
        return SEOAdapter()
    else:
        # Default fallback
        class GeneralAdapter(BaseDomainAdapter):
            def get_system_prompt(self) -> str:
                return """You are PinnacleRAG, a precise and trustworthy question-answering assistant.
STRICT RULES:
1. Answer ONLY using the provided context documents.
2. If the context does not contain sufficient information, say: 'I cannot answer this question based on the provided documents.'
3. ALWAYS cite your sources using [n].
4. Preserve technical accuracy."""
        return GeneralAdapter()
