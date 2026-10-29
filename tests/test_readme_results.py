from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| \[[^\]]+\]\(results/day_scale/(\w+)\) \|(.+)\|$")
CELL = re.compile(r"([+-]\d+\.\d\d) \[([+-]\d+\.\d\d), ([+-]\d+\.\d\d)\]")
NAMED_ROW = re.compile(r"^\| \[([^\]]+)\]\(results/day_scale/(\w+)\) \|")
PLAIN_ROW = re.compile(r"^\| ([^|\[]+?) \|(.+)\|$")


class ReadmeResultsTest(unittest.TestCase):
    def test_standard_table_matches_committed_summary(self) -> None:
        with (ROOT / "results/day_scale/analysis/summary.csv").open(newline="", encoding="utf-8") as handle:
            summary = {
                (row["pair_id"], row["metric"]): row
                for row in csv.DictReader(handle)
                if row["benchmark"] == "standard" and row["protocol"] == "shared_plain_ab_label"
            }
        pairs = set()
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
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
        names = {match.group(1): match.group(2) for match in map(NAMED_ROW.match, lines) if match}
        conditions = ("matched_labels/instruct_minus_base", "matched_strings/instruct_minus_base", "labels_minus_strings_delta")
        pairs = set()
        for line in lines:
            match = PLAIN_ROW.match(line)
            if not match or match.group(1) not in names:
                continue
            pair_id = names[match.group(1)]
            pairs.add(pair_id)
            cells = CELL.findall(match.group(2))
            self.assertEqual(len(cells), 3, line)
            for condition, cell in zip(conditions, cells):
                row = control[pair_id, condition]
                expected = tuple(f"{100 * float(row[key]):+.2f}" for key in ("point", "ci_low", "ci_high"))
                self.assertEqual(cell, expected, f"{pair_id} {condition}")
        self.assertEqual(pairs, {pair_id for pair_id, _condition in control})

    def test_candidate_control_counts_state_their_basis(self) -> None:
        with (ROOT / "results/reanalysis/candidate_control.csv").open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["metric"] == "ece"]
        n = len({row["pair_id"] for row in rows})

        def counts(protocol: str, condition: str) -> tuple[dict[str, int], dict[str, int]]:
            selected = [row for row in rows if row["protocol"] == protocol and row["condition"] == condition]
            self.assertEqual(len(selected), n)
            point = {sign: sum((float(row["point"]) > 0) - (float(row["point"]) < 0) == value for row in selected)
                     for sign, value in (("+", 1), ("-", -1))}
            interval = {"+": sum(float(row["ci_low"]) > 0 for row in selected),
                        "-": sum(float(row["ci_high"]) < 0 for row in selected)}
            return point, interval

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        shared, native = "shared_plain_ab_label", "native_prompt_ab_label"
        labels, strings, paired = ("matched_labels/instruct_minus_base", "matched_strings/instruct_minus_base",
                                   "labels_minus_strings_delta")
        label_point, label_interval = counts(shared, labels)
        string_point, string_interval = counts(shared, strings)
        paired_point, paired_interval = counts(shared, paired)
        none = lambda count: "none" if count == 0 else str(count)
        fragments = [
            f"the 95% interval for label ΔECE lies above zero in {label_interval['+']} of {n} base/instruct pairs",
            f"For full-string ΔECE it lies below zero in {string_interval['-']}, above zero in {none(string_interval['+'])} "
            f"and includes zero in {n - string_interval['-'] - string_interval['+']}",
            f"by point estimate, {string_point['-']} are negative and {string_point['+']} is positive",
            f"The interval for their paired difference lies above zero in {paired_interval['+']} of {n}",
            f"{string_point['-']} of {n} point estimates are negative and {string_interval['-']} of {n} intervals lie below zero",
        ]
        # The lead's "ninth pair" sentence holds only while both 8-of-9 counts agree on point and interval bases.
        self.assertEqual((label_point["+"], paired_point["+"]), (label_interval["+"], paired_interval["+"]))
        native_label = counts(native, labels)[1]["+"]
        native_point, native_interval = counts(native, strings)
        self.assertEqual(native_point["-"], n)
        fragments.append(f"the label ΔECE interval lies above zero in {native_label} of {n} pairs, and the two-string ΔECE "
                         f"interval lies below zero in {native_interval['-']} of {n} (all {n} point estimates are negative)")
        for fragment in fragments:
            self.assertIn(fragment, readme)


if __name__ == "__main__":
    unittest.main()
