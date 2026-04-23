from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.calibration_stage import _scaled_row, _stable_split, fit_temperature


class CalibrationStageTest(unittest.TestCase):
    def test_stable_split_is_deterministic(self) -> None:
        first = _stable_split("question-1", seed=123, calibration_fraction=0.5)
        second = _stable_split("question-1", seed=123, calibration_fraction=0.5)
        self.assertEqual(first, second)
        self.assertIn(first, {"calibration", "test"})

    def test_temperature_scaling_preserves_prediction_and_softens_confidence(self) -> None:
        row = {
            "raw_logprob_A": math.log(0.99),
            "raw_logprob_B": math.log(0.01),
            "correct_label": "B",
        }
        raw = _scaled_row(row, temperature=1.0)
        softened = _scaled_row(row, temperature=10.0)
        self.assertEqual(raw["predicted_label"], softened["predicted_label"])
        self.assertGreater(raw["confidence"], softened["confidence"])
        self.assertGreater(softened["p_correct"], raw["p_correct"])

    def test_fit_temperature_finds_softening_for_overconfident_rows(self) -> None:
        rows = [
            {
                "raw_logprob_A": math.log(0.99),
                "raw_logprob_B": math.log(0.01),
                "correct_label": "B",
            },
            {
                "raw_logprob_A": math.log(0.98),
                "raw_logprob_B": math.log(0.02),
                "correct_label": "B",
            },
            {
                "raw_logprob_A": math.log(0.97),
                "raw_logprob_B": math.log(0.03),
                "correct_label": "B",
            },
        ]
        self.assertGreater(fit_temperature(rows), 1.0)


if __name__ == "__main__":
    unittest.main()

