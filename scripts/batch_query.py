"""
Batch query runner for PinnacleRAG-DS.

Usage:
    python scripts/batch_query.py --input questions.txt --output results.json
    python scripts/batch_query.py --input questions.txt --format csv
    cat questions.txt | python scripts/batch_query.py --output results.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.pipeline.rag_pipeline import PinnacleRAGPipeline
from src.utils.helpers import save_json
from src.utils.logging import setup_logging


def read_questions(input_path: str | None) -> list[str]:
    """Read questions from file or stdin."""
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        print("Reading questions from stdin (one per line, Ctrl+D to finish):")
        lines = sys.stdin.readlines()

    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main() -> None:
    """Run batch queries and output results."""
    parser = argparse.ArgumentParser(description="PinnacleRAG-DS Batch Query")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to file with questions (one per line). Reads stdin if omitted.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="batch_results.json",
        help="Output file path (default: batch_results.json)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--correction",
        action="store_true",
        help="Enable self-correction loop",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    print("\n📋 PinnacleRAG-DS — Batch Query Runner")
    print("=" * 50)

    questions = read_questions(args.input)
    print(f"Loaded {len(questions)} questions")

    if not questions:
        print("No questions to process.")
        return

    pipeline = PinnacleRAGPipeline(settings)
    query_fn = pipeline.query_with_correction if args.correction else pipeline.query

    results = []
    start_time = time.time()

    for i, question in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {question[:60]}...")
        try:
            result = query_fn(question)
            results.append({
                "question": question,
                "answer": result["answer"],
                "sources": [s.get("source", "") for s in result.get("sources", [])],
                "model": result.get("model", ""),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "sources": [],
                "error": str(e),
            })

    elapsed = time.time() - start_time

    # Save results
    if args.format == "json":
        save_json(results, args.output)
    elif args.format == "csv":
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer", "sources", "elapsed_seconds"])
            writer.writeheader()
            for r in results:
                r["sources"] = "; ".join(r.get("sources", []))
                writer.writerow({k: r.get(k, "") for k in ["question", "answer", "sources", "elapsed_seconds"]})

    print(f"\n✅ Processed {len(questions)} questions in {elapsed:.1f}s")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()
