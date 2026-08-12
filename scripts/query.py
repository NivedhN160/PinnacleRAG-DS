"""
Interactive query CLI for PinnacleRAG-DS.

Usage:
    python scripts/query.py                              # Interactive mode
    python scripts/query.py --question "What is X?"      # Single question
    python scripts/query.py --correction                 # Enable self-correction
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.pipeline.rag_pipeline import PinnacleRAGPipeline
from src.utils.logging import setup_logging


def print_result(result: dict) -> None:
    """Pretty-print a query result."""
    print("\n" + "─" * 60)
    print("📝 Answer:")
    print(result["answer"])

    sources = result.get("sources", [])
    if sources:
        print("\n📚 Sources:")
        for src in sources:
            source_name = src.get("source", "unknown")
            page = src.get("page", "")
            page_str = f" (p.{page})" if page else ""
            print(f"  • {source_name}{page_str}")

    print(f"\n⏱  {result.get('elapsed_seconds', 0):.1f}s | "
          f"Retrieved: {result.get('retrieval_count', 0)} | "
          f"Reranked: {result.get('reranked_count', 0)} | "
          f"Model: {result.get('model', 'unknown')}")
    print("─" * 60 + "\n")


def main() -> None:
    """Run the query CLI."""
    parser = argparse.ArgumentParser(description="PinnacleRAG-DS Query CLI")
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Single question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--correction",
        action="store_true",
        help="Enable self-correction loop for low-confidence answers",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    print("\n🔍 PinnacleRAG-DS — Query Interface")
    print("=" * 50)

    pipeline = PinnacleRAGPipeline(settings)

    # Single question mode
    if args.question:
        query_fn = pipeline.query_with_correction if args.correction else pipeline.query
        result = query_fn(args.question)
        print_result(result)
        return

    # Interactive mode
    print("Type your questions below. Enter 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        query_fn = pipeline.query_with_correction if args.correction else pipeline.query
        result = query_fn(question)
        print_result(result)


if __name__ == "__main__":
    main()
