"""
Pydantic schemas for PinnacleRAG-DS API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Query ---
class QueryRequest(BaseModel):
    question: str
    mode: str = Field(default="simple", description="'simple' or 'agent'")
    domain: str = Field(default="general", description="Domain logic (general, trading, security, seo)")
    rewrite: bool = Field(default=False, description="Rewrite query before retrieval")
    check_faithfulness: bool = Field(default=False, description="Verify answer against context")

class Citation(BaseModel):
    id: int
    source: str
    snippet: str
    score: float

class Usage(BaseModel):
    llm_calls: int
    tokens: int
    budget_remaining_calls: int

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    usage: Usage
    mode: str
    domain: Optional[str] = None
    rewritten_query: Optional[str] = None

# --- Eval ---
class EvalResult(BaseModel):
    question: str
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: float

class EvalResponse(BaseModel):
    results: List[EvalResult]
    averages: Dict[str, float]

# --- Health ---
class HealthParseResponse(BaseModel):
    structured: Dict[str, Any]
    explanation: str
    indexed: bool
