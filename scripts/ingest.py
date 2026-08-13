"""
Ingest documents into the PinnacleRAG-DS vector store.

Usage:
    python scripts/ingest.py [--data-path PATH]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.pipeline.ingest_pipeline import IngestPipeline
from src.packs.docs import DocsPack
from src.utils.logging import setup_logging

def main() -> None:
    """Run the full data ingestion pipeline."""
    parser = argparse.ArgumentParser(description="PinnacleRAG-DS Data Ingestion")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw documents directory (defaults to config setting)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.data_path:
        settings.raw_data_path = args.data_path
        
    setup_logging(settings.log_level)

    print("\n PinnacleRAG-DS - Data Ingestion Pipeline")
    print("=" * 50)

    ingest_pipeline = IngestPipeline(settings)
    docs_pack = DocsPack(settings, ingest_pipeline)
    
    stats = docs_pack.process_directory()

    print("\n Ingestion Summary:")
    print(f"  Documents loaded:      {stats.get('documents_loaded', 0)}")
    print(f"  Total chunks:          {stats.get('total_chunks', 0)}")
    print(f"  Time elapsed:          {stats.get('elapsed_seconds', 0):.1f}s")
    print(f"  Status:                {stats.get('status', 'unknown')}")
    print()

    chunk_stats = stats.get("chunk_stats", {})
    if chunk_stats:
        print(f"  Avg chunk size:        {chunk_stats.get('avg_size', 0):.0f} chars")
        print(f"  Total est. tokens:     ~{chunk_stats.get('total_estimated_tokens', 0)}")

    print(f"  Time elapsed:          {stats.get('elapsed_seconds', 0):.1f}s")
    print(f"  Status:                {stats.get('status', 'unknown')}")
    print()


if __name__ == "__main__":
    main()
