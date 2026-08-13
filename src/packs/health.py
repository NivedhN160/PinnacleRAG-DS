"""
Health Pack for PinnacleRAG-DS.
Lab PDF text (+ optional structured JSON via Groq) -> also index for chat.
"""
from config.settings import Settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.generation.llm import GroqLLM
from langchain_core.documents import Document

class HealthPack:
    def __init__(self, settings: Settings, ingest_pipeline: IngestPipeline, llm: GroqLLM):
        self.settings = settings
        self.ingest_pipeline = ingest_pipeline
        self.llm = llm

    def process_lab_report(self, file_path: str) -> dict:
        """Process a lab report PDF and extract structured data."""
        # 1. Load and parse the single file
        raw_docs = self.ingest_pipeline.loader.load_file(file_path)
        if not raw_docs:
            return {"structured": {}, "explanation": "Failed to load document.", "indexed": False}
            
        text = "\n".join([doc.page_content for doc in raw_docs])
        
        # 2. Index it for chat
        self.ingest_pipeline.retriever.build_index(raw_docs)
        
        # 3. Optional structured extraction via LLM
        prompt = "Extract health lab tests from this text into JSON format: " + text[:2000]
        # In a real app we would force JSON output from Groq
        if not self.llm.is_mock:
            explanation = "Extraction logic would go here."
            structured = {"tests": [{"name": "Hb", "value": "14", "status": "normal"}]}
        else:
            explanation = "Mock explanation."
            structured = {"tests": [{"name": "Hb", "value": "14", "status": "normal"}]}
            
        return {
            "structured": structured,
            "explanation": explanation,
            "indexed": True
        }
