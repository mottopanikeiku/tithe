from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| \[[^\]]+\]\((?:\.\./)?results/day_scale/(\w+)\) \|(.+)\|$")
CELL = re.compile(r"([+-]\d+\.\d\d) \[([+-]\d+\.\d\d), ([+-]\d+\.\d\d)\]")


class ReadmeResultsTest(unittest.TestCase):
    def test_standard_table_matches_committed_summary(self) -> None:
        with (ROOT / "results/day_scale/analysis/summary.csv").open(newline="", encoding="utf-8") as handle:
            summary = {
                (row["pair_id"], row["metric"]): row
                for row in csv.DictReader(handle)
                if row["benchmark"] == "standard" and row["protocol"] == "shared_plain_ab_label"
            }
        pairs = set()
        for line in (ROOT / "docs/method.md").read_text(encoding="utf-8").splitlines():
            match = ROW.match(line)
            if not match:
                continue
            pair_id = match.group(1)
            pairs.add(pair_id)
            cells = CELL.findall(match.group(2))
            self.assertEqual(len(cells), 3, line)
            for metric, cell in zip(("mc1_accuracy", "mc2", "mc1_ece"), cells):
                row = summary[pair_id, metric]
                expected = tuple(f"{100 * float(row[key]):+.2f}" for key in ("delta", "ci_low", "ci_high"))
                self.assertEqual(cell, expected, f"{pair_id} {metric}")
        self.assertEqual(pairs, {pair_id for pair_id, _metric in summary})

    def test_candidate_control_table_matches_committed_reanalysis(self) -> None:
        with (ROOT / "results/reanalysis/candidate_control.csv").open(newline="", encoding="utf-8") as handle:
            control = {
                (row["pair_id"], row["condition"]): row
                for row in csv.DictReader(handle)
                if row["protocol"] == "shared_plain_ab_label" and row["metric"] == "ece"
            }
        lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        conditions = ("matched_labels/instruct_minus_base", "matched_strings/instruct_minus_base")
        pairs = set()
        for line in lines:
            match = ROW.match(line)
            if not match:
                continue
            pair_id = match.group(1)
            pairs.add(pair_id)
            cells = [cell.strip() for cell in match.group(2).split("|")]
            self.assertEqual(len(cells), 2, line)
            for condition, cell in zip(conditions, cells):
                row = control[pair_id, condition]
                marker = "*" if float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0 else ""
                expected = f"{100 * float(row['point']):+.1f}{marker}"
                self.assertEqual(cell, expected, f"{pair_id} {condition}")
        self.assertEqual(pairs, {pair_id for pair_id, _condition in control})


if __name__ == "__main__":
    unittest.main()
