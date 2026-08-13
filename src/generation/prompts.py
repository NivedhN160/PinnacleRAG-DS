"""
Prompts for PinnacleRAG-DS.
Enforces strict grounding and citation format [n].
"""

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are PinnacleRAG, a precise and trustworthy question-answering assistant.

STRICT RULES:
1. Answer ONLY using the provided context documents. Do NOT use any prior knowledge.
2. If the context does not contain sufficient information to answer the question, say: "I cannot answer this question based on the provided documents."
3. ALWAYS cite your sources using [n] notation corresponding to the Document number. Example: "This is a fact [1]."
4. Be concise but thorough. Provide complete answers without unnecessary filler.
5. If multiple sources support a claim, cite all of them like [1][2].
6. Preserve technical accuracy — do not paraphrase in a way that changes meaning.
7. Structure your response with clear paragraphs or bullet points when appropriate.

Remember: Faithfulness to the source material is your #1 priority."""

def build_prompt(query: str, context_docs: list[Document], system_prompt: str = None) -> list[dict]:
    """
    Build the chat messages with strict grounding instructions.
    
    Args:
        query: User question.
        context_docs: Context documents to ground the answer in.
        system_prompt: Optional domain-specific system prompt.
        
    Returns:
        List of message dicts for the Groq API.
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT
    context_parts = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Document {i} — {source}]\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""CONTEXT DOCUMENTS:
{context_text}

---

QUESTION: {query}

Answer the question using ONLY the context documents above. Cite sources using [n] notation."""

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_message},
    ]
