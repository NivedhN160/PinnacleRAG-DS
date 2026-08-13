from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import shutil

from config.settings import get_settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.packs.docs import DocsPack

router = APIRouter()
settings = get_settings()

@router.post("/")
async def ingest_endpoint():
    """Rebuild index from all of data/raw/"""
    try:
        ingest_pipeline = IngestPipeline(settings)
        docs_pack = DocsPack(settings, ingest_pipeline)
        return docs_pack.process_directory()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    domain: str = Form(...), 
    rebuild: bool = Form(False)
):
    """Upload multipart files (pdf/txt/md/docx)"""
    try:
        domain_dir = os.path.join(settings.raw_data_path, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        file_path = os.path.join(domain_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = {"status": "success", "message": f"File {file.filename} saved to {domain} domain."}
        
        if rebuild:
            ingest_pipeline = IngestPipeline(settings)
            docs_pack = DocsPack(settings, ingest_pipeline)
            result["ingest_stats"] = docs_pack.process_directory()
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextUploadRequest(BaseModel):
    text: str
    filename: str
    domain: str
    rebuild: bool = False

@router.post("/text")
async def upload_text(request: TextUploadRequest):
    """Paste-text path for notes without files"""
    try:
        domain_dir = os.path.join(settings.raw_data_path, request.domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        # Ensure it has .txt extension
        filename = request.filename if request.filename.endswith(".txt") else f"{request.filename}.txt"
        file_path = os.path.join(domain_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(request.text)
            
        result = {"status": "success", "message": f"Text saved as {filename} in {request.domain} domain."}
        
        if request.rebuild:
            ingest_pipeline = IngestPipeline(settings)
            docs_pack = DocsPack(settings, ingest_pipeline)
            result["ingest_stats"] = docs_pack.process_directory()
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
