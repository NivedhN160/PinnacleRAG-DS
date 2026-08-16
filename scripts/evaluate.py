"""
Run evaluation on the PinnacleRAG-DS golden set.

Usage:
    python scripts/evaluate.py [--golden-path PATH] [--save]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.evaluation.evaluator import RAGEvaluator
from src.pipeline.query_pipeline import QueryPipeline
from src.pipeline.agent_loop import AgentLoop
from src.budget.guard import BudgetGuard
from src.utils.logging import setup_logging


def main() -> None:
    """Run RAG evaluation and print results."""
    parser = argparse.ArgumentParser(description="PinnacleRAG-DS Evaluation Runner")
    parser.add_argument(
        "--golden-path",
        type=str,
        default=None,
        help="Path to golden set JSON file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for results",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    print("\n📊 PinnacleRAG-DS — Evaluation Runner")
    print("=" * 50)

    # Initialize pipeline and evaluator
    budget_guard = BudgetGuard(settings)
    query_pipeline = QueryPipeline(settings, budget_guard)
    pipeline = AgentLoop(settings, budget_guard, query_pipeline)
    evaluator = RAGEvaluator(settings, pipeline)

    # Load golden set
    golden_set = evaluator.load_golden_set(args.golden_path)
    if not golden_set:
        print("❌ No golden set found. Create one at:", settings.golden_set_path)
        return

    # Run evaluation
    results = evaluator.evaluate(golden_set)

    # Print summary
    evaluator.print_summary(results)

    # Save if requested
    if args.save:
        evaluator.save_results(results, args.output)
        print(f"✅ Results saved to {args.output or 'data/golden/eval_results.json'}")


if __name__ == "__main__":
    main()
