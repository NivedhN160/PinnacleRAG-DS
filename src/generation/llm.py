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
        
        if not settings.groq_api_key or settings.groq_api_key == "your_key_here":
            raise ValueError("Valid Groq API key is required. Mocking has been disabled.")
        
        self._client = Groq(api_key=settings.groq_api_key)
        logger.info(f"Groq LLM initialized: model={self._model}")

    def generate(self, query: str, context_docs: list[Document], system_prompt: Optional[str] = None) -> dict:
        """
        Generate a grounded answer.
        """
        messages = build_prompt(query, context_docs, system_prompt=system_prompt)
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
            "model": self._model,
            "usage": {
                "llm_calls": 1,
                "tokens": tokens_used
            }
        }

    def _call_groq(self, messages: list[dict], max_retries: int = 3) -> tuple[str, int]:
        import time
        for attempt in range(max_retries):
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
                logger.warning(f"Groq API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Groq API call failed after max retries.")
                    raise
                time.sleep((2 ** attempt) + 1)
            

