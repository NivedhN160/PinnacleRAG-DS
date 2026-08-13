"""
FastAPI application for PinnacleRAG-DS.
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.routes import query, eval, health, ingest

app = FastAPI(
    title="PinnacleRAG-DS API",
    description="Professional-grade RAG pipeline API",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler to prevent stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."}
    )

# Include routers
app.include_router(query.router, prefix="/api/query")
app.include_router(eval.router, prefix="/api/eval")
app.include_router(health.router, prefix="/api/health")
app.include_router(ingest.router, prefix="/api/ingest")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
