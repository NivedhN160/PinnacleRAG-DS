"""
RAG evaluation metrics for PinnacleRAG-DS.

Custom implementations of RAGAS-style metrics that work without paid APIs.
Uses Groq (free-tier) as LLM judge for claim extraction and support checking.
"""

import re
from typing import Optional

from groq import Groq

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_groq_client(settings: Settings) -> Groq:
    """Get a Groq client for LLM-as-judge evaluations."""
    return Groq(api_key=settings.groq_api_key)


def _llm_judge(client: Groq, model: str, prompt: str) -> str:
    """Call Groq as LLM judge for evaluation."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def compute_faithfulness(
    answer: str,
    contexts: list[str],
    question: str,
    settings: Optional[Settings] = None,
) -> float:
    """
    Compute faithfulness: is the answer grounded in the provided contexts?

    Extracts claims from the answer and checks if each is supported
    by at least one context passage.

    Args:
        answer: Generated answer.
        contexts: List of context passage texts.
        question: Original question.
        settings: Settings for LLM judge (uses Groq).

    Returns:
        Faithfulness score between 0.0 and 1.0.
    """
    if not answer.strip() or not contexts:
        return 0.0

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    client = _get_groq_client(settings)
    model = settings.groq_model

    # Step 1: Extract claims from the answer
    extract_prompt = f"""Extract all factual claims from the following answer. List each claim on a separate line.
Only extract factual statements, not opinions or filler text.

Answer: {answer}

Claims (one per line):"""

    claims_text = _llm_judge(client, model, extract_prompt)
    claims = [c.strip().lstrip("- ").lstrip("• ") for c in claims_text.strip().split("\n") if c.strip()]

    if not claims:
        return 1.0  # No claims to verify

    # Step 2: Check each claim against contexts
    context_text = "\n\n---\n\n".join(contexts)
    supported = 0

    for claim in claims:
        check_prompt = f"""Given the following context, determine if the claim is supported.
Answer with exactly "SUPPORTED" or "NOT SUPPORTED".

Context:
{context_text}

Claim: {claim}

Verdict:"""

        verdict = _llm_judge(client, model, check_prompt)
        if "SUPPORTED" in verdict.upper() and "NOT SUPPORTED" not in verdict.upper():
            supported += 1

    score = supported / len(claims) if claims else 1.0
    logger.debug(f"Faithfulness: {supported}/{len(claims)} claims supported = {score:.3f}")
    return round(score, 4)


def compute_answer_relevancy(
    answer: str,
    question: str,
    settings: Optional[Settings] = None,
) -> float:
    """
    Compute answer relevancy: does the answer address the question?

    Uses LLM judge to rate relevancy on a 0-1 scale.

    Args:
        answer: Generated answer.
        question: Original question.
        settings: Settings for LLM judge.

    Returns:
        Relevancy score between 0.0 and 1.0.
    """
    if not answer.strip():
        return 0.0

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    client = _get_groq_client(settings)
    model = settings.groq_model

    prompt = f"""Rate how well the following answer addresses the question.
Give a score from 0.0 to 1.0 where:
- 1.0 = perfectly addresses the question
- 0.5 = partially addresses the question
- 0.0 = completely irrelevant

Question: {question}
Answer: {answer}

Score (just the number, nothing else):"""

    result = _llm_judge(client, model, prompt)

    try:
        score = float(re.search(r"(\d+\.?\d*)", result).group(1))
        return round(min(max(score, 0.0), 1.0), 4)
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse relevancy score from: {result}")
        return 0.5


def compute_context_precision(
    question: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> float:
    """
    Compute context precision: are the retrieved contexts relevant to the question?

    For each context, checks if it contains information useful for answering.

    Args:
        question: Original question.
        contexts: List of retrieved context texts.
        ground_truth: Optional expected answer for reference.
        settings: Settings for LLM judge.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if not contexts:
        return 0.0

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    client = _get_groq_client(settings)
    model = settings.groq_model

    relevant_count = 0
    gt_ref = f"\nExpected Answer: {ground_truth}" if ground_truth else ""

    for ctx in contexts:
        prompt = f"""Is the following context passage relevant and useful for answering the question?
Answer with exactly "RELEVANT" or "NOT RELEVANT".

Question: {question}{gt_ref}

Context passage:
{ctx[:1000]}

Verdict:"""

        verdict = _llm_judge(client, model, prompt)
        if "RELEVANT" in verdict.upper() and "NOT RELEVANT" not in verdict.upper():
            relevant_count += 1

    score = relevant_count / len(contexts)
    logger.debug(f"Context precision: {relevant_count}/{len(contexts)} relevant = {score:.3f}")
    return round(score, 4)


def compute_context_recall(
    question: str,
    contexts: list[str],
    ground_truth: str,
    settings: Optional[Settings] = None,
) -> float:
    """
    Compute context recall: do the contexts contain all information needed
    for the ground truth answer?

    Extracts claims from ground truth and checks if contexts support them.

    Args:
        question: Original question.
        contexts: List of retrieved context texts.
        ground_truth: Expected correct answer.
        settings: Settings for LLM judge.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not ground_truth.strip() or not contexts:
        return 0.0

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    client = _get_groq_client(settings)
    model = settings.groq_model

    # Extract claims from ground truth
    extract_prompt = f"""Extract all factual claims from the following answer. List each claim on a separate line.

Answer: {ground_truth}

Claims (one per line):"""

    claims_text = _llm_judge(client, model, extract_prompt)
    claims = [c.strip().lstrip("- ").lstrip("• ") for c in claims_text.strip().split("\n") if c.strip()]

    if not claims:
        return 1.0

    # Check each claim against contexts
    context_text = "\n\n---\n\n".join(contexts)
    supported = 0

    for claim in claims:
        check_prompt = f"""Is the following claim supported by the context?
Answer with exactly "SUPPORTED" or "NOT SUPPORTED".

Context:
{context_text}

Claim: {claim}

Verdict:"""

        verdict = _llm_judge(client, model, check_prompt)
        if "SUPPORTED" in verdict.upper() and "NOT SUPPORTED" not in verdict.upper():
            supported += 1

    score = supported / len(claims)
    logger.debug(f"Context recall: {supported}/{len(claims)} GT claims supported = {score:.3f}")
    return round(score, 4)
