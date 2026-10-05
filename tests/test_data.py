from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEMP_ROOT = ROOT / ".tmp-tests"

from alignmenttax.data import (
    TRUTHFULQA_COMMIT,
    TRUTHFULQA_CSV_SHA256,
    TRUTHFULQA_CSV_URL,
    build_binary_dataset,
    load_or_download_csv,
    parse_truthfulqa_csv_text,
    prepare_data,
)
from alignmenttax.io_utils import read_jsonl


class TruthfulQATest(unittest.TestCase):
    CSV_BYTES = (
        b"Type,Category,Question,Best Answer,Best Incorrect Answer,Source\n"
        b"Adversarial,Misconceptions,Q?,True,False,https://example.test\n"
    )

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
        TEMP_ROOT.mkdir(exist_ok=True)
        temp_path = TEMP_ROOT / f"data_{uuid.uuid4().hex}"
        temp_path.mkdir()
        csv_path = temp_path / "TruthfulQA.csv"
        out_path = temp_path / "truthfulqa_binary.jsonl"
        csv_path.write_text(csv_text, encoding="utf-8")

        count = prepare_data(
            out=out_path,
            source_csv=csv_path,
            seed=20260420,
        )
        self.assertEqual(count, 1)
        records = read_jsonl(out_path)
        self.assertEqual(records[0]["source_metadata"]["variant"], "binary_best_vs_best_incorrect")
        self.assertEqual(records[0]["source_metadata"]["source_path"], str(csv_path))
        metadata = records[0]["source_metadata"]
        self.assertIsNone(metadata["source_commit"])
        self.assertIsNone(metadata["upstream_url"])
        self.assertEqual(metadata["csv_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())

    def test_default_url_uses_the_pinned_revision(self) -> None:
        self.assertEqual(TRUTHFULQA_COMMIT, "d71c110897f5d31c5d7f309e7bc316c152f6f031")
        self.assertEqual(
            TRUTHFULQA_CSV_URL,
            f"https://raw.githubusercontent.com/sylinrl/TruthfulQA/{TRUTHFULQA_COMMIT}/TruthfulQA.csv",
        )
        self.assertEqual(
            TRUTHFULQA_CSV_SHA256,
            "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c",
        )

    def test_pinned_download_and_cache_have_identical_provenance(self) -> None:
        digest = hashlib.sha256(self.CSV_BYTES).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "raw" / "TruthfulQA.csv"
            downloaded_out = Path(directory) / "downloaded.jsonl"
            cached_out = Path(directory) / "cached.jsonl"
            with (
                patch("alignmenttax.data.TRUTHFULQA_CSV_SHA256", digest),
                patch("alignmenttax.data._request_bytes", return_value=self.CSV_BYTES) as request,
            ):
                prepare_data(out=downloaded_out, cache_path=cache)
                request.assert_called_once_with(TRUTHFULQA_CSV_URL)
                request.reset_mock()
                prepare_data(out=cached_out, cache_path=cache)
                request.assert_not_called()
            self.assertEqual(cache.read_bytes(), self.CSV_BYTES)
            downloaded = read_jsonl(downloaded_out)
            cached = read_jsonl(cached_out)
            self.assertEqual(downloaded, cached)
            metadata = cached[0]["source_metadata"]
            self.assertEqual(metadata["source_commit"], TRUTHFULQA_COMMIT)
            self.assertEqual(metadata["upstream_url"], TRUTHFULQA_CSV_URL)
            self.assertEqual(metadata["csv_sha256"], digest)

    def test_existing_unverified_cache_does_not_claim_pinned_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "TruthfulQA.csv"
            out = Path(directory) / "processed.jsonl"
            cache.write_bytes(self.CSV_BYTES)
            with patch("alignmenttax.data._request_bytes") as request:
                prepare_data(out=out, cache_path=cache)
                request.assert_not_called()
            metadata = read_jsonl(out)[0]["source_metadata"]
            self.assertIsNone(metadata["source_commit"])
            self.assertIsNone(metadata["upstream_url"])
            self.assertEqual(metadata["csv_sha256"], hashlib.sha256(self.CSV_BYTES).hexdigest())

    def test_modified_pinned_cache_loses_commit_attribution(self) -> None:
        digest = hashlib.sha256(self.CSV_BYTES).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "TruthfulQA.csv"
            with (
                patch("alignmenttax.data.TRUTHFULQA_CSV_SHA256", digest),
                patch("alignmenttax.data._request_bytes", return_value=self.CSV_BYTES) as request,
            ):
                load_or_download_csv(cache_path=cache)
                changed_bytes = self.CSV_BYTES.replace(b"Q?", b"Changed question?")
                cache.write_bytes(changed_bytes)
                request.reset_mock()
                _, metadata = load_or_download_csv(cache_path=cache)
                request.assert_not_called()
            self.assertIsNone(metadata["source_commit"])
            self.assertIsNone(metadata["upstream_url"])
            self.assertEqual(metadata["csv_sha256"], hashlib.sha256(changed_bytes).hexdigest())

    def test_wrong_pinned_download_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "TruthfulQA.csv"
            with patch("alignmenttax.data._request_bytes", return_value=self.CSV_BYTES):
                with self.assertRaisesRegex(ValueError, "pinned SHA256"):
                    load_or_download_csv(cache_path=cache)
            self.assertFalse(cache.exists())

    def test_custom_download_does_not_claim_a_truthfulqa_commit(self) -> None:
        url = "https://example.test/custom.csv"
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "TruthfulQA.csv"
            with patch("alignmenttax.data._request_bytes", return_value=self.CSV_BYTES) as request:
                _, metadata = load_or_download_csv(cache_path=cache, url=url)
                request.assert_called_once_with(url)
                self.assertIsNone(metadata["source_commit"])
                self.assertEqual(metadata["upstream_url"], url)
                self.assertEqual(metadata["csv_sha256"], hashlib.sha256(self.CSV_BYTES).hexdigest())
                request.reset_mock()
                _, cached_metadata = load_or_download_csv(cache_path=cache, url=TRUTHFULQA_CSV_URL)
                request.assert_not_called()
            self.assertIsNone(cached_metadata["source_commit"])
            self.assertIsNone(cached_metadata["upstream_url"])
            self.assertEqual(cached_metadata["csv_sha256"], metadata["csv_sha256"])

    def test_local_csv_is_offline_and_hashes_unmodified_bytes(self) -> None:
        csv_bytes = b"\xef\xbb\xbf" + self.CSV_BYTES.replace(b"\n", b"\r\n")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "TruthfulQA.csv"
            out = Path(directory) / "processed.jsonl"
            source.write_bytes(csv_bytes)
            with patch("alignmenttax.data.urllib.request.urlopen") as request:
                prepare_data(out=out, source_csv=source)
                request.assert_not_called()
            record = read_jsonl(out)[0]
            self.assertEqual(record["question"], "Q?")
            self.assertEqual(record["source_metadata"]["csv_sha256"], hashlib.sha256(csv_bytes).hexdigest())
            self.assertIsNone(record["source_metadata"]["source_commit"])

    def test_local_csv_matching_pin_has_verified_provenance_offline(self) -> None:
        digest = hashlib.sha256(self.CSV_BYTES).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "TruthfulQA.csv"
            source.write_bytes(self.CSV_BYTES)
            with (
                patch("alignmenttax.data.TRUTHFULQA_CSV_SHA256", digest),
                patch("alignmenttax.data.urllib.request.urlopen") as request,
            ):
                _, metadata = load_or_download_csv(source_csv=source)
                request.assert_not_called()
            self.assertEqual(metadata["source_commit"], TRUTHFULQA_COMMIT)
            self.assertEqual(metadata["upstream_url"], TRUTHFULQA_CSV_URL)
            self.assertEqual(metadata["csv_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
