from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseDomainAdapter(ABC):
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    def enrich_metadata(self, doc: Document) -> Document:
        return doc

    def post_process_answer(self, answer: str, context: list) -> str:
        return answer

    def get_example_questions(self) -> list[str]:
        return []

    def get_evaluation_focus(self) -> dict:
        return {"faithfulness": 1.0, "relevancy": 1.0}
