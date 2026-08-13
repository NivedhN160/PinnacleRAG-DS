from fastapi import APIRouter, HTTPException
from config.settings import get_settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.packs.docs import DocsPack

router = APIRouter()

settings = get_settings()
ingest_pipeline = IngestPipeline(settings)
docs_pack = DocsPack(settings, ingest_pipeline)

@router.post("/")
async def ingest_endpoint():
    try:
        result = docs_pack.process_directory()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
