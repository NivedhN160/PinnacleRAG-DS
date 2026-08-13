"""
Standard Docs Pack for PinnacleRAG-DS.
"""
from config.settings import Settings
from src.pipeline.ingest_pipeline import IngestPipeline

class DocsPack:
    def __init__(self, settings: Settings, ingest_pipeline: IngestPipeline):
        self.settings = settings
        self.ingest_pipeline = ingest_pipeline

    def process_directory(self, path: str = None) -> dict:
        """Process standard documents."""
        return self.ingest_pipeline.run(path)
