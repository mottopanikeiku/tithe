from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.data import build_binary_dataset, parse_truthfulqa_csv_text, prepare_data
from alignmenttax.io_utils import read_jsonl


class TruthfulQATest(unittest.TestCase):
    def test_parse_csv_with_commas_and_newlines(self) -> None:
        csv_text = (
            "Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,"
            "Incorrect Answers,Source\n"
            'Adversarial,Misconceptions,"Question, with comma?","Line one\n'
            'line two","Wrong, answer",Correct,Incorrect,https://example.test\n'
        )
        rows = parse_truthfulqa_csv_text(csv_text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Question"], "Question, with comma?")
        self.assertEqual(rows[0]["Best Answer"], "Line one\nline two")
        self.assertEqual(rows[0]["Best Incorrect Answer"], "Wrong, answer")

    def test_binary_randomization_is_deterministic_and_preserves_correct_label(self) -> None:
        rows = [
            {
                "Type": "Adversarial",
                "Category": "Misconceptions",
                "Question": "Q1?",
                "Best Answer": "True one",
                "Best Incorrect Answer": "False one",
                "Source": "source",
            },
            {
                "Type": "Adversarial",
                "Category": "Science",
                "Question": "Q2?",
                "Best Answer": "True two",
                "Best Incorrect Answer": "False two",
                "Source": "source",
            },
        ]
        first = build_binary_dataset(rows, seed=7)
        second = build_binary_dataset(rows, seed=7)
        third = build_binary_dataset(rows, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(
            [row["correct_label"] for row in first],
            [row["correct_label"] for row in third],
        )
        for row in first:
            self.assertEqual(row["choices"][row["correct_label"]], row["correct_answer"])

    def test_prepare_data_writes_jsonl_from_local_csv(self) -> None:
        csv_text = (
            "Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,"
            "Incorrect Answers,Source\n"
            "Adversarial,Misconceptions,Q?,True,False,True,False,https://example.test\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "TruthfulQA.csv"
            out_path = temp_path / "truthfulqa_binary.jsonl"
            csv_path.write_text(csv_text, encoding="utf-8")

            count = prepare_data(
                out=out_path,
                source_csv=csv_path,
                seed=20260420,
                resolve_commit=False,
            )
            self.assertEqual(count, 1)
            records = read_jsonl(out_path)
            self.assertEqual(records[0]["source_metadata"]["variant"], "binary_best_vs_best_incorrect")
            self.assertEqual(records[0]["source_metadata"]["source_path"], str(csv_path))


if __name__ == "__main__":
    unittest.main()

