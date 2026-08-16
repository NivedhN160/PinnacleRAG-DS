import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from config.settings import get_settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.packs.docs import DocsPack

router = APIRouter()
settings = get_settings()

VALID_DOMAINS = {"general", "trading", "security", "seo"}
VALID_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

def sanitize_filename(filename: str) -> str:
    if not filename:
        filename = "unnamed"
    basename = os.path.basename(filename)
    safe_uuid = uuid.uuid4().hex[:8]
    return f"{safe_uuid}_{basename}"

def validate_domain(domain: str) -> str:
    domain_lower = domain.lower().strip()
    if domain_lower not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Must be one of: {VALID_DOMAINS}")
    return domain_lower

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
async def upload_files(
    files: List[UploadFile] = File(...), 
    domain: str = Form(...), 
    rebuild: bool = Form(False)
):
    """Upload multipart files (pdf/txt/md/docx)"""
    try:
        valid_domain = validate_domain(domain)
        domain_dir = os.path.join(settings.raw_data_path, valid_domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        saved_paths = []
        for file in files:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in VALID_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Invalid extension {ext}. Allowed: {VALID_EXTENSIONS}")
            
            safe_name = sanitize_filename(file.filename)
            file_path = os.path.join(domain_dir, safe_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
            
        result = {
            "status": "success", 
            "domain": valid_domain,
            "saved_paths": saved_paths,
            "message": f"Successfully uploaded {len(files)} files to {valid_domain} domain."
        }
        
        if rebuild:
            ingest_pipeline = IngestPipeline(settings)
            docs_pack = DocsPack(settings, ingest_pipeline)
            result["ingest_stats"] = docs_pack.process_directory()
            
        return result
    except HTTPException:
        raise
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
        valid_domain = validate_domain(request.domain)
        domain_dir = os.path.join(settings.raw_data_path, valid_domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        filename = request.filename if request.filename.endswith(".txt") else f"{request.filename}.txt"
        safe_name = sanitize_filename(filename)
        file_path = os.path.join(domain_dir, safe_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(request.text)
            
        result = {
            "status": "success", 
            "domain": valid_domain,
            "saved_paths": [file_path],
            "message": f"Text saved as {safe_name} in {valid_domain} domain."
        }
        
        if request.rebuild:
            ingest_pipeline = IngestPipeline(settings)
            docs_pack = DocsPack(settings, ingest_pipeline)
            result["ingest_stats"] = docs_pack.process_directory()
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
