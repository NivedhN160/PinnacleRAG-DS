from src.domains.base import BaseDomainAdapter
from langchain_core.documents import Document

class TradingAdapter(BaseDomainAdapter):
    def get_system_prompt(self) -> str:
        return """You are a specialized Financial Analyst AI.
STRICT RULES:
1. Answer ONLY using the provided context documents.
2. Cite sources using [n].
3. MUST include this exact disclaimer at the end: 'Disclaimer: This is not financial advice.'
4. Focus on dates, risks, and earnings data."""

    def post_process_answer(self, answer: str, context: list) -> str:
        if "Disclaimer" not in answer:
            return answer + "\n\nDisclaimer: This is not financial advice."
        return answer

    def get_example_questions(self) -> list[str]:
        return ["What are the key 10-K risks mentioned?", "How did earnings change year-over-year?"]

    def get_evaluation_focus(self) -> dict:
        return {"faithfulness": 1.5, "relevancy": 1.0}
