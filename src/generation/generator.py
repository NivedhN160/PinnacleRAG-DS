"""
Groq LLM generator for PinnacleRAG-DS.

Strict grounded generation: the LLM must answer ONLY from provided context
and cite sources. This is the only component that talks to the Groq API.
"""

from typing import Optional, Generator

from groq import Groq
from langchain_core.documents import Document

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PinnacleRAG, a precise and trustworthy question-answering assistant.

STRICT RULES:
1. Answer ONLY using the provided context documents. Do NOT use any prior knowledge.
2. If the context does not contain sufficient information to answer the question, say: "I cannot answer this question based on the provided documents."
3. ALWAYS cite your sources using [Source: filename] notation after each claim.
4. Be concise but thorough. Provide complete answers without unnecessary filler.
5. If multiple sources support a claim, cite all of them.
6. Preserve technical accuracy — do not paraphrase in a way that changes meaning.
7. Structure your response with clear paragraphs or bullet points when appropriate.

Remember: Faithfulness to the source material is your #1 priority."""


class GroqGenerator:
    """Groq LLM client with strict grounded generation and citation enforcement."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the Groq generator.

        Args:
            settings: Application settings (must include groq_api_key).
        """
        self.settings = settings
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens

        logger.info(f"Groq generator initialized: model={self._model}")

    def generate(self, query: str, context_docs: list[Document]) -> dict:
        """
        Generate a grounded answer from query and context documents.

        Args:
            query: User question.
            context_docs: Retrieved and reranked context documents.

        Returns:
            Dict with keys: answer, sources, raw_response, model, tokens_used.
        """
        messages = self._build_prompt(query, context_docs)
        raw_response = self._call_groq(messages)

        # Extract source references from context
        sources = []
        for doc in context_docs:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "chunk_id": doc.metadata.get("chunk_id", None),
                "content_preview": doc.page_content[:150],
            })

        return {
            "answer": raw_response,
            "sources": sources,
            "raw_response": raw_response,
            "model": self._model,
            "context_docs_used": len(context_docs),
        }

    def stream_generate(
        self, query: str, context_docs: list[Document]
    ) -> Generator[str, None, None]:
        """
        Stream a grounded answer token by token.

        Args:
            query: User question.
            context_docs: Retrieved context documents.

        Yields:
            Answer tokens as they arrive.
        """
        messages = self._build_prompt(query, context_docs)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_prompt(
        self, query: str, context_docs: list[Document]
    ) -> list[dict]:
        """
        Build the chat messages with strict grounding instructions.

        Args:
            query: User question.
            context_docs: Context documents to ground the answer in.

        Returns:
            List of message dicts for the Groq API.
        """
        # Format context documents
        context_parts = []
        for i, doc in enumerate(context_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page else ""
            context_parts.append(
                f"[Document {i} — {source}{page_str}]\n{doc.page_content}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        user_message = f"""CONTEXT DOCUMENTS:
{context_text}

---

QUESTION: {query}

Answer the question using ONLY the context documents above. Cite sources using [Source: filename] notation."""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _call_groq(self, messages: list[dict]) -> str:
        """
        Make a single call to the Groq API.

        This is the ONLY method that communicates with Groq.

        Args:
            messages: Chat messages for the API.

        Returns:
            Generated text response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            answer = response.choices[0].message.content
            usage = response.usage

            logger.info(
                f"Groq response: {usage.prompt_tokens} prompt + "
                f"{usage.completion_tokens} completion = {usage.total_tokens} total tokens"
            )

            return answer

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
