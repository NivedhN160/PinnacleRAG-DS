"""
FastAPI application for PinnacleRAG-DS.
Provides a REST API for the RAG pipeline.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings, Settings
from src.pipeline.rag_pipeline import PinnacleRAGPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PinnacleRAG-DS API",
    description="Professional-grade RAG pipeline API",
    version="1.0.0",
)

# Global pipeline instance
pipeline: Optional[PinnacleRAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on application startup."""
    global pipeline
    logger.info("Initializing PinnacleRAG-DS pipeline...")
    try:
        settings = get_settings()
        pipeline = PinnacleRAGPipeline(settings)
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # We don't raise here to allow the app to start, but health check will fail


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system", example="What is RAG?")
    return_sources: bool = Field(default=True, description="Whether to return source documents")
    use_correction: bool = Field(default=False, description="Whether to use the agentic self-correction loop")


class SourceDocument(BaseModel):
    source: str
    content_snippet: str
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None
    context_docs_used: int
    elapsed_seconds: float
    model_used: str


@app.get("/health")
async def health_check():
    """Check the health of the API and the underlying pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    health = pipeline.health_check()
    if health["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health)
    
    return health


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the RAG pipeline.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        if request.use_correction:
            # Using the self-correction loop
            result = pipeline.query_with_correction(request.question)
        else:
            # Standard query
            result = pipeline.query(request.question, return_sources=request.return_sources)
            
        # Format sources for response
        sources = []
        if request.return_sources and "sources" in result:
            for s in result["sources"]:
                sources.append(
                    SourceDocument(
                        source=s.get("source", "Unknown"),
                        content_snippet=s.get("snippet", ""),
                        chunk_id=s.get("chunk_id"),
                        page=s.get("page"),
                        section=s.get("section")
                    )
                )
                
        return QueryResponse(
            answer=result["answer"],
            sources=sources if request.return_sources else None,
            context_docs_used=result.get("context_docs_used", 0),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
            model_used=result.get("model", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest():
    """
    Trigger the ingestion pipeline to process raw data and update the vectorstore.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info("Triggering background ingestion pipeline")
        result = pipeline.ingest()
        return result
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
