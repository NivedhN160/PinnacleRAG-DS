from src.domains.base import BaseDomainAdapter

class SEOAdapter(BaseDomainAdapter):
    def get_system_prompt(self) -> str:
        return """You are an SEO Content Analyst.
STRICT RULES:
1. Base all technical recommendations on the provided content.
2. Cite sources using [n]."""

    def get_example_questions(self) -> list[str]:
        return ["What are the keyword gaps?", "List the technical SEO issues found."]
