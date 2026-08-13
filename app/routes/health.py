import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from config.settings import get_settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.generation.llm import GroqLLM
from src.packs.health import HealthPack
from app.schemas import HealthParseResponse
import tempfile

router = APIRouter()

settings = get_settings()
ingest_pipeline = IngestPipeline(settings)
llm = GroqLLM(settings)
health_pack = HealthPack(settings, ingest_pipeline, llm)

@router.post("/parse", response_model=HealthParseResponse)
async def parse_health_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        result = health_pack.process_lab_report(tmp_path)
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
