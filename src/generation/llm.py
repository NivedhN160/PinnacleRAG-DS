"""
Groq LLM integration for PinnacleRAG-DS.
Includes mocking support and budget guard integration.
"""

from typing import Optional

from groq import Groq
from langchain_core.documents import Document

from config.settings import Settings
from src.generation.prompts import build_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)

class GroqLLM:
    """Groq LLM client with strict grounded generation and citation enforcement."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = settings.groq_model
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        
        # Determine if we should mock (no key or mock explicitly requested)
        self.is_mock = False
        if not settings.groq_api_key or settings.groq_api_key == "your_key_here":
            if settings.mock_if_no_key:
                logger.warning("No valid Groq API key found. Running in MOCK mode.")
                self.is_mock = True
            else:
                raise ValueError("Groq API key is required and mock_if_no_key is False.")
        else:
            self._client = Groq(api_key=settings.groq_api_key)
            logger.info(f"Groq LLM initialized: model={self._model}")

    def generate(self, query: str, context_docs: list[Document]) -> dict:
        """
        Generate a grounded answer.
        """
        messages = build_prompt(query, context_docs)
        
        if self.is_mock:
            raw_response, tokens_used = self._mock_call(messages, context_docs)
        else:
            raw_response, tokens_used = self._call_groq(messages)

        # Extract source references from context
        sources = []
        for i, doc in enumerate(context_docs, start=1):
            sources.append({
                "id": i,
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:150],
                "score": doc.metadata.get("score", 0.0),
            })

        return {
            "answer": raw_response,
            "citations": sources,
            "raw_response": raw_response,
            "model": "mock" if self.is_mock else self._model,
            "usage": {
                "llm_calls": 1,
                "tokens": tokens_used
            }
        }

    def _call_groq(self, messages: list[dict]) -> tuple[str, int]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            answer = response.choices[0].message.content
            usage = response.usage
            
            return answer, usage.total_tokens

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
            
    def _mock_call(self, messages: list[dict], context_docs: list[Document]) -> tuple[str, int]:
        """Generate a mock response using snippets from the context."""
        if not context_docs:
            return "I cannot answer this question based on the provided documents.", 20
            
        mock_answer = "This is a mock answer generated without an API key.\n\n"
        for i, doc in enumerate(context_docs[:2], start=1):
            mock_answer += f"Based on the context, here is a snippet [1]: '{doc.page_content[:100]}...'\n\n"
            
        return mock_answer, 150
