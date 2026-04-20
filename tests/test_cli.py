from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.cli import main
from alignmenttax.io_utils import read_jsonl


class CLITest(unittest.TestCase):
    def test_prepare_score_analyze_smoke_with_fake_scorer(self) -> None:
        csv_text = (
            "Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,"
            "Incorrect Answers,Source\n"
            "Adversarial,Misconceptions,Q1?,True 1,False 1,True 1,False 1,https://example.test\n"
            "Adversarial,Science,Q2?,True 2,False 2,True 2,False 2,https://example.test\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_csv = temp_path / "TruthfulQA.csv"
            processed = temp_path / "truthfulqa_binary.jsonl"
            run_dir = temp_path / "runs" / "fake"
            scores = run_dir / "scores.jsonl"
            report_dir = temp_path / "reports" / "fake"
            config_path = temp_path / "config.json"

            source_csv.write_text(csv_text, encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "prepare-data",
                        "--source-csv",
                        str(source_csv),
                        "--out",
                        str(processed),
                        "--seed",
                        "11",
                        "--skip-commit-resolution",
                    ]
                ),
                0,
            )

            config = {
                "models": {
                    "base": {"model_id": "fake-base"},
                    "instruct": {"model_id": "fake-instruct"},
                },
                "dataset": {
                    "path": str(processed),
                    "seed": 11,
                    "variant": "binary_best_vs_best_incorrect",
                },
                "scoring": {
                    "protocols": ["shared_plain_ab_label", "native_prompt_ab_label"],
                },
                "analysis": {
                    "bootstrap_iterations": 20,
                    "confidence_level": 0.95,
                    "ece_bins": 2,
                    "binning": "equal_frequency",
                    "bootstrap_seed": 11,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            self.assertEqual(
                main(["score", "--config", str(config_path), "--out", str(scores), "--fake"]),
                0,
            )
            score_rows = read_jsonl(scores)
            self.assertEqual(len(score_rows), 8)
            self.assertTrue((run_dir / "run_metadata.json").exists())

            self.assertEqual(
                main(
                    [
                        "analyze",
                        "--run",
                        str(run_dir),
                        "--out",
                        str(report_dir),
                        "--bootstrap-iterations",
                        "20",
                        "--no-plots",
                    ]
                ),
                0,
            )
            self.assertTrue((report_dir / "summary.csv").exists())
            self.assertTrue((report_dir / "paired_bootstrap.json").exists())
            self.assertTrue((report_dir / "calibration_tables.csv").exists())
            self.assertTrue((report_dir / "category_breakdown.csv").exists())


if __name__ == "__main__":
    unittest.main()

