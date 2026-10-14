from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.cross_family import (
    MODELS,
    PROTOCOLS,
    _bootstrap_deltas,
    _classification,
    _metric_arrays,
    _resampled_metrics,
    analyze_cross_family,
    main,
)
from alignmenttax.io_utils import write_json, write_jsonl
from alignmenttax.metrics import METRIC_NAMES, metric_summary, percentile
from alignmenttax.scoring import NATIVE_PROMPT_PROTOCOL, _score_row

FIXTURE_COUNT = 13


def fixture(pair_id: str = "fixture_a") -> tuple[dict, list[dict]]:
    """Synthetic unit data with the same real-run fields; never published results."""
    config = {
        "experiment": {"pair_id": pair_id, "family": "Fixture & family", "parameters_billion": 0.5},
        "models": {
            model: {"model_id": f"fixture/{pair_id}-{model}", "revision": ("a" if model == "base" else "b") * 40}
            for model in MODELS
        },
        "scoring": {"protocols": list(PROTOCOLS)},
    }
    records = []
    for protocol in PROTOCOLS:
        for model in MODELS:
            for index in range(FIXTURE_COUNT):
                p_a = (0.5, 0.6, 0.6, 0.9)[index % 4]
                if model == "instruct":
                    p_a = (0.8, 0.4, 0.4, 0.7)[index % 4]
                if protocol == NATIVE_PROMPT_PROTOCOL and model == "instruct":
                    p_a = 1.0 - p_a
                item = {
                    "id": f"q{index:03d}", "source_index": index, "question": f"Fixture question {index}?",
                    "correct_label": "A" if index % 3 else "B",
                    "choices": {"A": "Fixture answer A", "B": "Fixture answer B"},
                    "category": "Fixture category", "type": "Adversarial",
                }
                records.append(_score_row(
                    item=item, model_key=model, model_id=config["models"][model]["model_id"],
                    protocol=protocol, logprob_a=math.log(p_a) - 2.0, logprob_b=math.log(1.0 - p_a) - 2.0,
                    device="cuda:0", dtype="torch.bfloat16", elapsed_seconds=0.01,
                    label_token_counts={"A": 1, "B": 1},
                    prompt_format="chat_template" if protocol == NATIVE_PROMPT_PROTOCOL and model == "instruct" else "plain",
                    model_revision=config["models"][model]["revision"],
                ))
    metadata = {"fake": False, "config": config, "question_rows": FIXTURE_COUNT, "score_rows_total": len(records)}
    return metadata, records


def write_pair(root: Path, metadata: dict, records: list[dict], *, compressed: bool = False) -> Path:
    directory = root / metadata["config"]["experiment"]["pair_id"]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(metadata, directory / "run_metadata.json")
    if compressed:
        with gzip.open(directory / "scores.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    else:
        write_jsonl(records, directory / "scores.jsonl")
    return directory


class CrossFamilyTest(unittest.TestCase):
    def analyze(self, root: Path, out: Path, *, iterations: int = 19) -> dict:
        return analyze_cross_family(root, out, iterations=iterations, seed=20260420, _expected_question_count=FIXTURE_COUNT)

    def test_outputs_preserve_all_metrics_protocols_and_question_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
            metadata, records = fixture()
            write_pair(root, metadata, list(reversed(records)))
            second_metadata, second_records = fixture("fixture_b")
            second_metadata["config"]["experiment"]["parameters_billion"] = 1.5
            write_pair(root, second_metadata, second_records, compressed=True)
            summary = self.analyze(root, out)
            self.assertEqual(summary["pair_count"], 2)
            self.assertEqual(summary["question_count"], FIXTURE_COUNT)
            self.assertEqual(summary["question_ids"], [f"q{index:03d}" for index in range(FIXTURE_COUNT)])
            self.assertEqual(summary["seed"], 20260420)
            self.assertTrue(summary["validation"]["fixture_question_count_override"])
            self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
            for pair in summary["pairs"]:
                self.assertEqual(set(pair["protocols"]), set(PROTOCOLS))
                for protocol, analysis in pair["protocols"].items():
                    self.assertEqual(analysis["role"], "sensitivity" if protocol == NATIVE_PROMPT_PROTOCOL else "primary")
                    self.assertEqual(set(analysis["deltas"]["metrics"]), set(METRIC_NAMES))
                    self.assertEqual(set(analysis["classification"]["signs"]), set(METRIC_NAMES))
                    for model in MODELS:
                        rows = [row for row in records if row["model_key"] == model and row["prompt_protocol"] == protocol]
                        expected = metric_summary(rows)
                        for metric in METRIC_NAMES:
                            self.assertAlmostEqual(analysis["models"][model][metric], expected[metric])
                    for metric, delta in analysis["deltas"]["metrics"].items():
                        self.assertAlmostEqual(delta["point"], analysis["models"]["instruct"][metric] - analysis["models"]["base"][metric])
                        self.assertLessEqual(delta["ci_low"], delta["ci_high"])
            with (out / "summary.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2 * 2 * len(METRIC_NAMES))
            self.assertEqual({row["role"] for row in csv_rows}, {"primary", "sensitivity"})
            self.assertNotIn(b"\r", (out / "summary.csv").read_bytes())
            svg = ElementTree.parse(out / "cross_family.svg")
            text = " ".join(svg.getroot().itertext())
            self.assertIn("Δaccuracy (percentage points)", text)
            self.assertIn("ΔECE (percentage points)", text)
            self.assertIn("Shared plain · primary", text)
            self.assertIn("Native prompt · sensitivity", text)
            self.assertIn("Fixture & family", text)
            self.assertFalse(any("href" in key for element in svg.iter() for key in element.attrib))

    def test_vectorized_metrics_match_reference_for_ties_remainders_and_clipping(self) -> None:
        metadata, records = fixture()
        rows = [row for row in records if row["model_key"] == "base" and row["prompt_protocol"] == PROTOCOLS[0]]
        # Include underflowed p_correct to exercise the same EPSILON NLL rule.
        rows[0] = {**rows[0], "correct": False, "confidence": 1.0, "p_correct": 0.0}
        indices = np.asarray([
            list(range(FIXTURE_COUNT)),
            [1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9, 10, 0],
            [0] * FIXTURE_COUNT,
            list(reversed(range(FIXTURE_COUNT))),
        ])
        for bins in (1, 2, 10, 20):
            with self.subTest(bins=bins):
                actual = _resampled_metrics(_metric_arrays(rows), indices, ece_bins=bins)
                expected = np.asarray([
                    [metric_summary([rows[int(index)] for index in draw], ece_bins=bins)[metric] for metric in METRIC_NAMES]
                    for draw in indices
                ])
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        # Equal-confidence rows with different correctness must retain sample order.
        tied_rows = [{"correct": index >= 3, "confidence": 0.7, "p_correct": 0.7 if index >= 3 else 0.3} for index in range(6)]
        tie_indices = np.asarray([[0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5]])
        values = _resampled_metrics(_metric_arrays(tied_rows), tie_indices, ece_bins=2)
        self.assertNotAlmostEqual(values[0, 5], values[1, 5])
        for index, draw in enumerate(tie_indices):
            self.assertAlmostEqual(values[index, 5], metric_summary([tied_rows[int(i)] for i in draw], ece_bins=2)["ece"])

    def test_bootstrap_matches_slow_reference_on_identical_draws(self) -> None:
        _metadata, records = fixture()
        base = [row for row in records if row["model_key"] == "base" and row["prompt_protocol"] == PROTOCOLS[0]]
        instruct = [row for row in records if row["model_key"] == "instruct" and row["prompt_protocol"] == PROTOCOLS[0]]
        iterations, seed = 37, 81
        draws = np.random.default_rng(seed).integers(0, len(base), size=(iterations, len(base)))
        samples = {metric: [] for metric in METRIC_NAMES}
        for draw in draws:
            summaries = [metric_summary([rows[int(index)] for index in draw]) for rows in (base, instruct)]
            for metric in METRIC_NAMES:
                samples[metric].append(summaries[1][metric] - summaries[0][metric])
        actual = _bootstrap_deltas(base, instruct, iterations=iterations, seed=seed, chunk_size=7)
        other_chunk = _bootstrap_deltas(base, instruct, iterations=iterations, seed=seed, chunk_size=2)
        for metric, values in actual["metrics"].items():
            self.assertAlmostEqual(values["ci_low"], percentile(samples[metric], 0.025))
            self.assertAlmostEqual(values["ci_high"], percentile(samples[metric], 0.975))
            self.assertAlmostEqual(values["point"], metric_summary(instruct)[metric] - metric_summary(base)[metric])
            for field in ("point", "ci_low", "ci_high"):
                self.assertAlmostEqual(values[field], other_chunk["metrics"][metric][field])

    def test_count_override_does_not_weaken_real_run_validation(self) -> None:
        mutations = {
            "fake_metadata": lambda metadata, rows: metadata.update(fake=True),
            "missing_fake_flag": lambda metadata, rows: metadata.pop("fake"),
            "fake_device": lambda metadata, rows: rows[0].update(device="fake"),
            "fake_dtype": lambda metadata, rows: rows[0].update(dtype="fake"),
            "synthetic_format": lambda metadata, rows: rows[0].update(prompt_format="synthetic"),
            "fake_row_flag": lambda metadata, rows: rows[0].update(fake=True),
            "missing_tokens": lambda metadata, rows: rows[0].pop("label_token_counts"),
            "bool_tokens": lambda metadata, rows: rows[0].update(label_token_counts={"A": True, "B": 1}),
            "nan_logprob": lambda metadata, rows: rows[0].update(raw_logprob_A=float("nan")),
            "infinite_logprob": lambda metadata, rows: rows[0].update(raw_logprob_B=float("-inf")),
            "boolean_logprob": lambda metadata, rows: rows[0].update(raw_logprob_A=True),
            "positive_logprob": lambda metadata, rows: rows[0].update(raw_logprob_A=0.1),
            "tampered_confidence": lambda metadata, rows: rows[0].update(confidence=0.99),
            "tampered_correctness": lambda metadata, rows: rows[0].update(correct=not rows[0]["correct"]),
            "unpinned_config": lambda metadata, rows: metadata["config"]["models"]["base"].update(revision="main"),
            "row_revision_mismatch": lambda metadata, rows: rows[0].update(model_revision="c" * 40),
            "row_model_mismatch": lambda metadata, rows: rows[0].update(model_id="fixture/wrong"),
            "missing_record": lambda metadata, rows: rows.pop(),
            "duplicate_record": lambda metadata, rows: rows.append(copy.deepcopy(rows[0])),
            "different_group_ids": lambda metadata, rows: rows[0].update(question_id="different"),
            "changed_content": lambda metadata, rows: rows[0].update(question="Changed fixture question"),
            "wrong_label": lambda metadata, rows: rows[0].update(correct_label="C"),
            "wrong_model_key": lambda metadata, rows: rows[0].update(model_key="other"),
            "wrong_protocol": lambda metadata, rows: rows[0].update(prompt_protocol="other"),
            "missing_protocol_config": lambda metadata, rows: metadata["config"]["scoring"].update(protocols=[PROTOCOLS[0]]),
            "missing_family": lambda metadata, rows: metadata["config"]["experiment"].pop("family"),
            "invalid_size": lambda metadata, rows: metadata["config"]["experiment"].update(parameters_billion=0),
            "bad_metadata_count": lambda metadata, rows: metadata.update(question_rows=12),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
                metadata, records = fixture()
                mutate(metadata, records)
                write_pair(root, metadata, records)
                with self.assertRaises(ValueError):
                    self.analyze(root, out)
                self.assertFalse(out.exists())

    def test_cross_pair_id_and_content_mismatch_and_duplicate_pair_rejected(self) -> None:
        for issue in ("ids", "content", "duplicate"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as temporary:
                root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
                first_metadata, first_rows = fixture()
                write_pair(root, first_metadata, first_rows)
                metadata, rows = fixture("fixture_b")
                if issue == "ids":
                    for row in rows:
                        row["question_id"] = "other_" + row["question_id"]
                elif issue == "content":
                    for row in rows:
                        row["question"] += " changed"
                else:
                    metadata["config"]["models"] = copy.deepcopy(first_metadata["config"]["models"])
                    for row in rows:
                        row["model_id"] = metadata["config"]["models"][row["model_key"]]["model_id"]
                write_pair(root, metadata, rows)
                with self.assertRaises(ValueError):
                    self.analyze(root, out)
                self.assertFalse(out.exists())

    def test_ambiguous_compressed_and_plain_input_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
            metadata, records = fixture()
            write_pair(root, metadata, records)
            write_pair(root, metadata, records, compressed=True)
            with self.assertRaisesRegex(ValueError, "both plain and compressed"):
                self.analyze(root, out)

    def test_production_count_cannot_be_relaxed_by_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
            metadata, records = fixture()
            write_pair(root, metadata, records)
            with self.assertRaisesRegex(ValueError, "exactly 790"):
                analyze_cross_family(root, out, iterations=1)

    def test_missing_scores_or_metadata_rejected(self) -> None:
        for missing in ("scores", "metadata"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root, out = Path(temporary) / "scores", Path(temporary) / "analysis"
                metadata, records = fixture()
                directory = root / "fixture_a"
                directory.mkdir(parents=True)
                if missing == "scores":
                    write_json(metadata, directory / "run_metadata.json")
                else:
                    write_jsonl(records, directory / "scores.jsonl")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.analyze(root, out)
                self.assertFalse(out.exists())

    def test_classification_reports_all_signs_without_hiding_accuracy_losses(self) -> None:
        def deltas(accuracy: tuple, ece: tuple) -> dict:
            result = {metric: {"point": 0.0, "ci_low": -0.1, "ci_high": 0.1} for metric in METRIC_NAMES}
            for metric, values in (("accuracy", accuracy), ("ece", ece)):
                result[metric] = dict(zip(("point", "ci_low", "ci_high"), values))
            return result

        tradeoff = _classification(deltas((0.1, 0.01, 0.2), (0.1, 0.01, 0.2)))
        self.assertEqual(tradeoff["outcome"], "supports_tradeoff")
        improvement = _classification(deltas((-0.1, -0.2, -0.01), (-0.1, -0.2, -0.01)))
        self.assertEqual(improvement["outcome"], "supports_calibration_improvement")
        self.assertEqual(improvement["signs"]["accuracy"]["ci"], "negative")
        self.assertIn("does not imply", improvement["explanation"])
        worsened = _classification(deltas((-0.1, -0.2, -0.01), (0.1, 0.01, 0.2)))
        self.assertEqual(worsened["outcome"], "accuracy_and_calibration_worsened")
        for accuracy, ece in (((0.1, 0.01, 0.2), (0.0, -0.1, 0.1)), ((0.1, 0.0, 0.2), (0.1, 0.01, 0.2))):
            self.assertEqual(_classification(deltas(accuracy, ece))["outcome"], "uncertain")
        self.assertEqual(set(tradeoff["signs"]), set(METRIC_NAMES))

    def test_cli_defaults_and_no_question_count_override(self) -> None:
        with patch("alignmenttax.cross_family.analyze_cross_family") as analyze:
            self.assertEqual(main(["--root", "scores", "--out", "analysis"]), 0)
            analyze.assert_called_once_with(Path("scores"), Path("analysis"), 10000, 20260420)
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--root", "scores", "--out", "analysis", "--expected-question-count", "13"])

    def test_invalid_bootstrap_arguments_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in ({"iterations": 0}, {"iterations": True}, {"seed": -1}, {"_expected_question_count": 0}):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    analyze_cross_family(Path(temporary), Path(temporary) / "out", **arguments)


if __name__ == "__main__":
    unittest.main()
