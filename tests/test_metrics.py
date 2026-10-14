from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.analysis import _summary_rows
from alignmenttax.metrics import (
    calibration_bins,
    metric_summary,
    paired_bootstrap_deltas,
    score_from_label_logprobs,
    two_way_softmax,
)


class MetricsTest(unittest.TestCase):
    def test_two_way_softmax_is_stable(self) -> None:
        p_a, p_b = two_way_softmax(-1000.0, -1001.0)
        self.assertAlmostEqual(p_a + p_b, 1.0)
        self.assertGreater(p_a, p_b)

    def test_score_from_logprobs(self) -> None:
        score = score_from_label_logprobs(
            logprob_a=math.log(0.8),
            logprob_b=math.log(0.2),
            correct_label="A",
        )
        self.assertEqual(score["predicted_label"], "A")
        self.assertTrue(score["correct"])
        self.assertAlmostEqual(score["p_correct"], 0.8)
        self.assertAlmostEqual(score["confidence"], 0.8)

    def test_metric_summary_and_ece(self) -> None:
        rows = [
            {"correct": True, "confidence": 0.9, "p_correct": 0.9},
            {"correct": False, "confidence": 0.8, "p_correct": 0.2},
            {"correct": True, "confidence": 0.6, "p_correct": 0.6},
            {"correct": True, "confidence": 0.7, "p_correct": 0.7},
        ]
        summary = metric_summary(rows, ece_bins=2)
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["accuracy"], 0.75)
        self.assertGreater(summary["brier"], 0.0)
        self.assertGreater(summary["nll"], 0.0)
        bins = calibration_bins(rows, bins=2)
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(sum(row["ece_contribution"] for row in bins), summary["ece"])

    def test_paired_bootstrap_is_deterministic(self) -> None:
        base_rows = [
            {"question_id": "q1", "correct": True, "confidence": 0.6, "p_correct": 0.6},
            {"question_id": "q2", "correct": False, "confidence": 0.7, "p_correct": 0.3},
            {"question_id": "q3", "correct": True, "confidence": 0.8, "p_correct": 0.8},
        ]
        instruct_rows = [
            {"question_id": "q1", "correct": True, "confidence": 0.8, "p_correct": 0.8},
            {"question_id": "q2", "correct": True, "confidence": 0.6, "p_correct": 0.6},
            {"question_id": "q3", "correct": False, "confidence": 0.9, "p_correct": 0.1},
        ]
        first = paired_bootstrap_deltas(
            base_rows=base_rows,
            instruct_rows=instruct_rows,
            iterations=25,
            confidence_level=0.95,
            seed=123,
        )
        second = paired_bootstrap_deltas(
            base_rows=base_rows,
            instruct_rows=instruct_rows,
            iterations=25,
            confidence_level=0.95,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["n_pairs"], 3)
        self.assertIn("accuracy", first["metrics"])

    def test_summary_delta_uses_the_bootstrap_question_pairs(self) -> None:
        # A resumed/interrupted run can hold base rows for questions instruct has not scored.
        base_rows = [
            {"question_id": "q1", "correct": True, "confidence": 0.6, "p_correct": 0.6},
            {"question_id": "q2", "correct": False, "confidence": 0.7, "p_correct": 0.3},
        ]
        instruct_rows = [{"question_id": "q1", "correct": True, "confidence": 0.8, "p_correct": 0.8}]
        grouped = {("shared", "base"): base_rows, ("shared", "instruct"): instruct_rows}
        rows = {row["metric"]: row for row in _summary_rows(grouped, ece_bins=10, binning="equal_frequency")}
        paired = paired_bootstrap_deltas(
            base_rows=base_rows, instruct_rows=instruct_rows, iterations=5, confidence_level=0.95, seed=1,
        )
        self.assertEqual((rows["n"]["base"], rows["n"]["instruct"]), (1, 1))
        for metric, delta in paired["metrics"].items():
            self.assertAlmostEqual(rows[metric]["delta_instruct_minus_base"], delta["point"])
        only_base = _summary_rows({("shared", "base"): base_rows}, ece_bins=10, binning="equal_frequency")
        self.assertEqual(only_base[0]["base"], 2)


if __name__ == "__main__":
    unittest.main()

