"""
RAG evaluation runner for PinnacleRAG-DS.

Runs golden-set evaluation with standard RAG metrics and produces
aggregate + per-question reports for iteration.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings
from src.evaluation.metrics import (
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
)
from src.utils.helpers import load_json, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """Professional evaluation runner for RAG pipeline quality assessment."""

    def __init__(self, settings: Settings, pipeline: Any) -> None:
        """
        Initialize the evaluator.

        Args:
            settings: Application settings.
            pipeline: PinnacleRAGPipeline instance to evaluate.
        """
        self.settings = settings
        self.pipeline = pipeline

    def load_golden_set(self, path: Optional[str] = None) -> list[dict]:
        """
        Load the golden Q&A evaluation set.

        Expected format: list of dicts with keys:
        - "question": str (required)
        - "ground_truth": str (required for recall metrics)
        - "metadata": dict (optional)

        Args:
            path: Path to golden set JSON file.

        Returns:
            List of evaluation items.
        """
        golden_path = path or self.settings.golden_set_path

        if not Path(golden_path).exists():
            logger.warning(f"Golden set not found: {golden_path}")
            return []

        golden_set = load_json(golden_path)
        logger.info(f"Loaded golden set: {len(golden_set)} items from {golden_path}")

        # Validate
        for i, item in enumerate(golden_set):
            if "question" not in item:
                raise ValueError(f"Golden set item {i} missing 'question' field")

        return golden_set

    def evaluate(self, golden_set: Optional[list[dict]] = None) -> dict:
        """
        Run full evaluation on the golden set.

        Args:
            golden_set: Optional pre-loaded golden set. If None, loads from settings path.

        Returns:
            Dict with aggregate scores and per-question details.
        """
        if golden_set is None:
            golden_set = self.load_golden_set()

        if not golden_set:
            logger.warning("No golden set items to evaluate")
            return {"error": "Empty golden set"}

        logger.info(f"Starting evaluation on {len(golden_set)} questions")
        start_time = time.time()

        per_question_results = []
        all_faithfulness = []
        all_relevancy = []
        all_precision = []
        all_recall = []

        for i, item in enumerate(golden_set):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            logger.info(f"Evaluating [{i + 1}/{len(golden_set)}]: {question[:60]}...")

            try:
                # Run the pipeline
                domain = item.get("domain", "general")
                result = self.pipeline.run(question, domain=domain)
                answer = result["answer"]
                context_texts = [c["snippet"] for c in result.get("citations", [])]

                # Compute metrics
                faithfulness = compute_faithfulness(answer, context_texts, question, self.settings)
                relevancy = compute_answer_relevancy(answer, question, self.settings)
                precision = compute_context_precision(question, context_texts, ground_truth, self.settings)
                recall = (
                    compute_context_recall(question, context_texts, ground_truth, self.settings)
                    if ground_truth
                    else None
                )

                all_faithfulness.append(faithfulness)
                all_relevancy.append(relevancy)
                all_precision.append(precision)
                if recall is not None:
                    all_recall.append(recall)

                per_question_results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "metrics": {
                        "faithfulness": faithfulness,
                        "answer_relevancy": relevancy,
                        "context_precision": precision,
                        "context_recall": recall,
                    },
                    "sources_used": len(context_texts),
                })

            except Exception as e:
                logger.error(f"Evaluation failed for question {i + 1}: {e}")
                per_question_results.append({
                    "question": question,
                    "error": str(e),
                })
                
            # Add delay between questions to avoid rate limits
            if i < len(golden_set) - 1:
                delay = getattr(self.settings, 'eval_delay_seconds', 3)
                logger.debug(f"Sleeping for {delay} seconds before next evaluation...")
                time.sleep(delay)

        elapsed = time.time() - start_time

        # Aggregate scores
        aggregate = {
            "faithfulness": self._safe_mean(all_faithfulness),
            "answer_relevancy": self._safe_mean(all_relevancy),
            "context_precision": self._safe_mean(all_precision),
            "context_recall": self._safe_mean(all_recall),
        }

        # Check against thresholds
        thresholds_met = {
            "faithfulness": aggregate["faithfulness"] >= self.settings.min_faithfulness,
            "answer_relevancy": aggregate["answer_relevancy"] >= self.settings.min_answer_relevancy,
            "context_precision": aggregate["context_precision"] >= self.settings.min_context_precision,
            "context_recall": (
                aggregate["context_recall"] >= self.settings.min_context_recall
                if aggregate["context_recall"] is not None
                else None
            ),
        }

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(golden_set),
            "successful_evaluations": len(all_faithfulness),
            "elapsed_seconds": round(elapsed, 2),
            "aggregate_scores": aggregate,
            "thresholds_met": thresholds_met,
            "per_question": per_question_results,
        }

        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        return results

    def save_results(self, results: dict, path: Optional[str] = None) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dict.
            path: Output path (defaults to data/golden/eval_results.json).
        """
        if path is None:
            path = str(Path(self.settings.golden_set_path).parent / "eval_results.json")

        save_json(results, path)
        logger.info(f"Evaluation results saved to {path}")

    def print_summary(self, results: dict) -> None:
        """
        Print a formatted evaluation summary to console.

        Args:
            results: Evaluation results dict.
        """
        print("\n" + "=" * 60)
        print("  PinnacleRAG-DS — Evaluation Report")
        print("=" * 60)

        agg = results.get("aggregate_scores", {})
        thresholds = results.get("thresholds_met", {})

        print(f"\n  Total Questions:  {results.get('total_questions', 0)}")
        print(f"  Successful:       {results.get('successful_evaluations', 0)}")
        print(f"  Time Elapsed:     {results.get('elapsed_seconds', 0):.1f}s")

        print("\n  METRIC                  SCORE     THRESHOLD   STATUS")
        print("  " + "-" * 56)

        metrics_config = [
            ("Faithfulness", "faithfulness", self.settings.min_faithfulness),
            ("Answer Relevancy", "answer_relevancy", self.settings.min_answer_relevancy),
            ("Context Precision", "context_precision", self.settings.min_context_precision),
            ("Context Recall", "context_recall", self.settings.min_context_recall),
        ]

        for label, key, threshold in metrics_config:
            score = agg.get(key)
            met = thresholds.get(key)

            if score is not None:
                status = "✓ PASS" if met else "✗ FAIL"
                print(f"  {label:<22} {score:.4f}    ≥ {threshold:.2f}      {status}")
            else:
                print(f"  {label:<22} N/A")

        print("\n" + "=" * 60 + "\n")

    @staticmethod
    def _safe_mean(values: list[float]) -> Optional[float]:
        """Compute mean, returning None for empty lists."""
        if not values:
            return None
        return round(sum(values) / len(values), 4)
