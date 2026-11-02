from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ece_changes() -> dict[str, dict[str, tuple[float, float]]]:
    """Instruct-minus-base ECE interval per pair, for each scoring rule."""
    changes: dict[str, dict[str, tuple[float, float]]] = {"labels": {}, "strings": {}}
    with (ROOT / "results/reanalysis/candidate_control.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] != "shared_plain_ab_label" or row["metric"] != "ece":
                continue
            for name in changes:
                if row["condition"] == f"matched_{name}/instruct_minus_base":
                    changes[name][row["pair_id"]] = (float(row["ci_low"]), float(row["ci_high"]))
    return changes


class HeadlineCountsTest(unittest.TestCase):
    def test_label_scoring_worsens_ece_and_answer_scoring_improves_it(self) -> None:
        changes = ece_changes()
        self.assertEqual(len(changes["labels"]), 9)
        self.assertEqual(len(changes["strings"]), 9)
        worse = sum(low > 0 for low, _ in changes["labels"].values())
        better = sum(high < 0 for _, high in changes["labels"].values())
        self.assertEqual((worse, better), (8, 0))
        worse = sum(low > 0 for low, _ in changes["strings"].values())
        better = sum(high < 0 for _, high in changes["strings"].values())
        self.assertEqual((worse, better), (0, 6))


if __name__ == "__main__":
    unittest.main()
