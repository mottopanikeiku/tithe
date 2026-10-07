from __future__ import annotations

import copy
import csv
import gzip
import hashlib
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

from alignmenttax.cross_family import MODELS, PROTOCOLS, _bootstrap_deltas
from alignmenttax.io_utils import write_json
from alignmenttax.metrics import METRIC_NAMES, calibration_bins, metric_summary
from alignmenttax.scale_analysis import (
    DATASET_REVISION, HARNESS_REVISION, QUESTION_COUNTS, STANDARD_METRICS,
    _classify, _joint_analysis, _standard_arrays, _standard_resampled_metrics,
    analyze_scale, main,
)
from alignmenttax.scoring import NATIVE_PROMPT_PROTOCOL, _score_row
from alignmenttax.standard import standard_metric_row

COUNTS = {"standard": 13, "binary": 11}


def fixture(pair_id: str = "fixture_a", size: float = 0.5) -> tuple[dict, dict, dict]:
    config = {
        "experiment": {"pair_id": pair_id, "family": "Fixture & family", "parameters_billion": size},
        "models": {model: {"model_id": f"fixture/{pair_id}-{model}",
                           "revision": ("a" if model == "base" else "b") * 40,
                           "device": "auto", "dtype": "bf16"} for model in MODELS},
        "scoring": {"protocols": list(PROTOCOLS)},
        "standard_dataset": {"dataset_id": "truthfulqa/truthful_qa", "config": "multiple_choice",
                             "split": "validation", "revision": DATASET_REVISION,
                             "harness_revision": HARNESS_REVISION},
    }
    rows = {"standard": [], "binary": []}
    for protocol in PROTOCOLS:
        for model in MODELS:
            for index in range(COUNTS["standard"]):
                question = f"Fixture standard question {index}?"
                likelihoods = [[-2.0, -3.0, -4.0], [-4.0, -2.0, -3.0], [-2.0, -2.0, -2.0]][index % 3]
                if model == "instruct":
                    likelihoods = [-2.2, -2.5, -3.5] if index % 3 else [-4.0, -2.0, -3.0]
                if model == "instruct" and protocol == NATIVE_PROMPT_PROTOCOL:
                    likelihoods = list(reversed(likelihoods))
                mc2 = [-3.0, -2.0, -4.0, -3.0] if model == "base" else [-2.0, -3.0, -4.0, -2.5]
                row = {
                    "question_id": f"truthfulqa_mc_{index:04d}_{hashlib.sha256(question.encode()).hexdigest()[:12]}",
                    "question": question, "source_index": index, "model_key": model,
                    "model_id": config["models"][model]["model_id"], "model_revision": config["models"][model]["revision"],
                    "protocol": protocol, "prompt_format": "chat_template" if protocol == NATIVE_PROMPT_PROTOCOL and model == "instruct" else "plain",
                    "device": "cuda:0", "dtype": "torch.bfloat16", "elapsed_seconds": 0.02,
                    "mc1_choices": ["Best answer", "Wrong answer", "Another wrong answer"],
                    "mc1_labels": [1, 0, 0], "mc1_loglikelihoods": likelihoods,
                    "mc2_choices": ["True answer", "False answer", "Another false answer", "Another true answer"],
                    "mc2_labels": [1, 0, 0, 1], "mc2_loglikelihoods": mc2,
                    "extra_raw_field": {"kept": True},
                }
                row.update(standard_metric_row(likelihoods, row["mc1_labels"], mc2, row["mc2_labels"]))
                rows["standard"].append(row)
            for index in range(COUNTS["binary"]):
                p_a = (0.5, 0.6, 0.6, 0.9)[index % 4] if model == "base" else (0.8, 0.4, 0.4, 0.7)[index % 4]
                if model == "instruct" and protocol == NATIVE_PROMPT_PROTOCOL:
                    p_a = 1.0 - p_a
                item = {"id": f"binary_q{index:03d}", "question": f"Fixture binary question {index}?",
                        "source_index": index, "correct_label": "A" if index % 3 else "B",
                        "choices": {"A": "Answer A", "B": "Answer B"}, "category": "Fixture", "type": "Adversarial"}
                rows["binary"].append(_score_row(
                    item=item, model_key=model, model_id=config["models"][model]["model_id"],
                    model_revision=config["models"][model]["revision"], protocol=protocol,
                    logprob_a=math.log(p_a) - 2.0, logprob_b=math.log(1.0 - p_a) - 2.0,
                    device="cuda:0", dtype="torch.bfloat16", elapsed_seconds=0.01,
                    label_token_counts={"A": 1, "B": 1},
                    prompt_format="chat_template" if protocol == NATIVE_PROMPT_PROTOCOL and model == "instruct" else "plain",
                ))
    metadata = {benchmark: {"config": copy.deepcopy(config), "fake": False,
                            "question_rows": count, "score_rows_total": 4 * count,
                            "hardware": {"gpu": "Fixture H100"}, "extra_metadata": "retained"}
                for benchmark, count in COUNTS.items()}
    return config, metadata, rows


def write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_pair(root: Path, old: Path, config: dict, metadata: dict, rows: dict, *, old_binary: bool = True) -> None:
    pair_id = config["experiment"]["pair_id"]
    directory = root / pair_id
    directory.mkdir(parents=True, exist_ok=True)
    write_json(metadata["standard"], directory / "standard_metadata.json")
    write_gzip(directory / "standard_scores.jsonl.gz", rows["standard"])
    write_json({"gpu": "Fixture H100", "gpu_type": "H100"}, directory / "standard_runtime.json")
    binary = old / pair_id if old_binary else directory
    binary.mkdir(parents=True, exist_ok=True)
    write_json(metadata["binary"], binary / ("run_metadata.json" if old_binary else "binary_metadata.json"))
    write_gzip(binary / ("scores.jsonl.gz" if old_binary else "binary_scores.jsonl.gz"), rows["binary"])
    write_json({"gpu": "Fixture L4" if old_binary else "Fixture H100"}, binary / "runtime.json")


def standard_reference(rows: list[dict], bins: int = 10) -> dict[str, float]:
    n = len(rows)
    accuracy = sum(float(row["mc1_accuracy"]) for row in rows) / n
    confidence = sum(row["mc1_confidence"] for row in rows) / n
    reference = {
        "mc1_accuracy": accuracy,
        "mc2": sum(row["mc2"] for row in rows) / n,
        "mc1_mean_confidence": confidence,
        "mc1_overconfidence_gap": confidence - accuracy,
        "mc1_brier": sum(row["mc1_brier"] for row in rows) / n,
        "mc1_nll": sum(row["mc1_nll"] for row in rows) / n,
    }
    calibration_rows = [{"correct": row["mc1_accuracy"], "confidence": row["mc1_confidence"]} for row in rows]
    reference["mc1_ece"] = sum(float(row["ece_contribution"]) for row in calibration_bins(calibration_rows, bins=bins))
    return reference


def grouped(rows: dict) -> dict:
    return {benchmark: {"fixture_a": {"groups": {
        (protocol, model): sorted([row for row in records if row["model_key"] == model and
                                   row.get("protocol", row.get("prompt_protocol")) == protocol],
                                  key=lambda row: row["question_id"])
        for protocol in PROTOCOLS for model in MODELS
    }}} for benchmark, records in rows.items()}


class ScaleAnalysisTest(unittest.TestCase):
    def analyze(self, root: Path, out: Path, old: Path, configs: list[dict], *, iterations: int = 17) -> dict:
        return analyze_scale(root, out, old, iterations=iterations, seed=20260420,
                             _expected_pairs=configs, _expected_question_counts=COUNTS)

    def test_nested_outputs_all_metrics_sources_protocol_roles_and_raw_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, old, out = (Path(temporary) / name for name in ("day", "old", "analysis"))
            config, metadata, rows = fixture()
            write_pair(root, old, config, metadata, {key: list(reversed(value)) for key, value in rows.items()})
            new_config, new_metadata, new_rows = fixture("fixture_b", 32.0)
            write_pair(root, old, new_config, new_metadata, new_rows, old_binary=False)
            raw_before = (root / "fixture_a" / "standard_scores.jsonl.gz").read_bytes()
            summary = self.analyze(root, out, old, [config, new_config])
            self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
            self.assertEqual(summary["pair_count"], 2)
            self.assertEqual({key: data["question_count"] for key, data in summary["benchmarks"].items()}, COUNTS)
            self.assertNotEqual(summary["benchmarks"]["standard"]["question_ids"], summary["benchmarks"]["binary"]["question_ids"])
            self.assertTrue(summary["bootstrap"]["shared_resamples_across_models_pairs_protocols_and_metrics"])
            self.assertFalse(summary["interpretation"]["benchmarks_pooled"])
            self.assertIn("Categorical multiclass Brier", summary["interpretation"]["mc1_brier"])
            self.assertEqual(summary["standard_task"]["harness_revision"], HARNESS_REVISION)
            for pair in summary["pairs"]:
                self.assertEqual(pair["new_large_pair"], pair["pair_id"] == "fixture_b")
                for benchmark, data in pair["benchmarks"].items():
                    self.assertEqual(data["source"]["run_metadata"]["extra_metadata"], "retained")
                    self.assertTrue(data["source"]["runtime_sources"])
                    self.assertTrue(all("gpu" in runtime for runtime in data["source"]["runtime_sources"].values()))
                    for protocol, analysis in data["protocols"].items():
                        self.assertEqual(analysis["role"], "sensitivity" if protocol == NATIVE_PROMPT_PROTOCOL else "primary")
                        self.assertEqual(analysis["official_harness_prompt_setting"], benchmark == "standard" and protocol != NATIVE_PROMPT_PROTOCOL)
                        names = STANDARD_METRICS if benchmark == "standard" else METRIC_NAMES
                        self.assertEqual(set(analysis["deltas"]["metrics"]), set(names))
                        for model in MODELS:
                            self.assertEqual(data["source"]["hardware"][protocol][model]["devices"], ["cuda:0"])
                        for metric, delta in analysis["deltas"]["metrics"].items():
                            self.assertAlmostEqual(delta["point"], analysis["models"]["instruct"][metric] - analysis["models"]["base"][metric])
                            self.assertLessEqual(delta["ci_low"], delta["ci_high"])
            self.assertIn("extra_raw_field", summary["pairs"][0]["benchmarks"]["standard"]["source"]["raw_score_fields"])
            self.assertEqual(raw_before, (root / "fixture_a" / "standard_scores.jsonl.gz").read_bytes())
            for benchmark, protocols in summary["outcome_counts"].items():
                for truth_counts in protocols.values():
                    for counts in truth_counts.values():
                        self.assertEqual(sum(counts.values()), 2)
            with (out / "summary.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2 * 2 * (len(METRIC_NAMES) + len(STANDARD_METRICS)))
            self.assertEqual({row["benchmark"] for row in csv_rows}, {"binary", "standard"})
            svg = ElementTree.parse(out / "day_scale.svg")
            text = " ".join(svg.getroot().itertext())
            for label in ("ΔMC1 (percentage points)", "ΔMC2 (percentage points)", "ΔMC1 ECE (percentage points)", "not official harness", "Fixture & family", "32B · new"):
                self.assertIn(label, text)

    def test_standard_resamples_match_slow_ece_with_ties_and_remainders(self) -> None:
        _config, _metadata, rows = fixture()
        standard = grouped(rows)["standard"]["fixture_a"]["groups"][PROTOCOLS[0], "base"]
        indices = np.asarray([list(range(13)), list(reversed(range(13))), [2, 5, 8, 11, 2, 5, 8, 11, 0, 0, 0, 1, 1]])
        for bins in (1, 2, 10, 20):
            actual = _standard_resampled_metrics(_standard_arrays(standard), indices, ece_bins=bins)
            expected = np.asarray([[standard_reference([standard[int(i)] for i in draw], bins)[metric]
                                    for metric in STANDARD_METRICS] for draw in indices])
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        tied = np.asarray([[float(i >= 3), 0.5, 0.7, 0.7 - float(i >= 3), 0.2, 1.0] for i in range(6)])
        result = _standard_resampled_metrics(tied, np.asarray([[0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5]]), ece_bins=2)
        self.assertNotAlmostEqual(result[0, -1], result[1, -1])

    def test_joint_bootstrap_matches_identical_draw_slow_reference_and_binary_code(self) -> None:
        _config, _metadata, rows = fixture()
        data = grouped(rows)
        iterations, seed = 31, 91
        for benchmark, loaded in data.items():
            actual = _joint_analysis(loaded, benchmark, iterations=iterations, seed=seed, chunk_size=7)
            other = _joint_analysis(loaded, benchmark, iterations=iterations, seed=seed, chunk_size=2)
            groups = loaded["fixture_a"]["groups"]
            n = COUNTS[benchmark]
            draws = np.random.default_rng(seed).integers(0, n, size=(iterations, n))
            names = STANDARD_METRICS if benchmark == "standard" else METRIC_NAMES
            reference = standard_reference if benchmark == "standard" else metric_summary
            for protocol in PROTOCOLS:
                base, instruct = groups[protocol, "base"], groups[protocol, "instruct"]
                differences = np.asarray([[reference([instruct[int(i)] for i in draw])[metric] -
                                           reference([base[int(i)] for i in draw])[metric] for metric in names] for draw in draws])
                expected = np.quantile(differences, [0.025, 0.975], axis=0)
                for index, metric in enumerate(names):
                    delta = actual["fixture_a"][protocol]["deltas"]["metrics"][metric]
                    self.assertAlmostEqual(delta["ci_low"], expected[0, index])
                    self.assertAlmostEqual(delta["ci_high"], expected[1, index])
                    for field in ("point", "ci_low", "ci_high"):
                        self.assertAlmostEqual(delta[field], other["fixture_a"][protocol]["deltas"]["metrics"][metric][field])
                if benchmark == "binary":
                    legacy = _bootstrap_deltas(base, instruct, iterations=iterations, seed=seed)
                    for metric in names:
                        for field in ("point", "ci_low", "ci_high"):
                            self.assertAlmostEqual(actual["fixture_a"][protocol]["deltas"]["metrics"][metric][field], legacy["metrics"][metric][field])

    def test_joint_draws_identical_across_pair_copies(self) -> None:
        _config, _metadata, rows = fixture()
        data = grouped(rows)["standard"]
        data["fixture_b"] = copy.deepcopy(data["fixture_a"])
        actual = _joint_analysis(data, "standard", iterations=23, seed=31)
        self.assertEqual(actual["fixture_a"], actual["fixture_b"])

    def test_categorical_nll_underflow_is_not_clipped_to_binary_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
            config, metadata, rows = fixture()
            row = rows["standard"][0]
            row["mc1_loglikelihoods"] = [-1000.0, -1.0, -3.0]
            row.update(standard_metric_row(row["mc1_loglikelihoods"], row["mc1_labels"],
                                           row["mc2_loglikelihoods"], row["mc2_labels"]))
            write_pair(root, old, config, metadata, rows)
            summary = self.analyze(root, out, old, [config], iterations=3)
            analysis = summary["pairs"][0]["benchmarks"]["standard"]["protocols"][PROTOCOLS[0]]
            base = [x for x in rows["standard"] if x["model_key"] == "base" and x["protocol"] == PROTOCOLS[0]]
            self.assertEqual(row["mc1_p_correct"], 0.0)
            self.assertGreater(analysis["models"]["base"]["mc1_nll"], 70.0)
            self.assertAlmostEqual(analysis["models"]["base"]["mc1_nll"], standard_reference(base)["mc1_nll"])

    def test_derived_raw_fake_missing_and_duplicate_rejections(self) -> None:
        mutations = {
            "fake_metadata": lambda m, r: m["standard"].update(fake=True),
            "missing_fake": lambda m, r: m["standard"].pop("fake"),
            "bad_count": lambda m, r: m["standard"].update(question_rows=817),
            "wrong_dataset_revision": lambda m, r: m["standard"]["config"]["standard_dataset"].update(revision="main"),
            "unpinned_model": lambda m, r: m["standard"]["config"]["models"]["base"].update(revision="main"),
            "row_revision": lambda m, r: r["standard"][0].update(model_revision="c" * 40),
            "fake_device": lambda m, r: r["standard"][0].update(device="fake"),
            "fake_dtype": lambda m, r: r["standard"][0].update(dtype="synthetic"),
            "fake_row": lambda m, r: r["standard"][0].update(fake=True),
            "wrong_format": lambda m, r: r["standard"][0].update(prompt_format="chat_template"),
            "negative_elapsed": lambda m, r: r["standard"][0].update(elapsed_seconds=-1),
            "nan_elapsed": lambda m, r: r["standard"][0].update(elapsed_seconds=float("nan")),
            "nan_likelihood": lambda m, r: r["standard"][0]["mc1_loglikelihoods"].__setitem__(0, float("nan")),
            "infinite_likelihood": lambda m, r: r["standard"][0]["mc2_loglikelihoods"].__setitem__(0, float("-inf")),
            "boolean_likelihood": lambda m, r: r["standard"][0]["mc1_loglikelihoods"].__setitem__(0, False),
            "positive_likelihood": lambda m, r: r["standard"][0]["mc1_loglikelihoods"].__setitem__(0, 0.1),
            "bad_mc1_label": lambda m, r: r["standard"][0].update(mc1_labels=[0, 1, 0]),
            "bad_mc2_label": lambda m, r: r["standard"][0].update(mc2_labels=[0, 0, 0, 0]),
            "choice_length": lambda m, r: r["standard"][0].update(mc2_choices=["A", "B"]),
            "tampered_mc1": lambda m, r: r["standard"][0].update(mc1_accuracy=not r["standard"][0]["mc1_accuracy"]),
            "integer_accuracy": lambda m, r: r["standard"][0].update(mc1_accuracy=1),
            "tampered_mc2": lambda m, r: r["standard"][0].update(mc2=0.123),
            "tampered_brier": lambda m, r: r["standard"][0].update(mc1_brier=0.123),
            "tampered_nll": lambda m, r: r["standard"][0].update(mc1_nll=0.123),
            "tampered_probability": lambda m, r: r["standard"][0].update(mc1_p_correct=0.123),
            "missing_mc1_confidence": lambda m, r: r["standard"][0].pop("mc1_confidence"),
            "duplicate_standard": lambda m, r: r["standard"].append(copy.deepcopy(r["standard"][0])),
            "missing_standard": lambda m, r: r["standard"].pop(),
            "bad_standard_id": lambda m, r: r["standard"][0].update(question_id="wrong"),
            "changed_content": lambda m, r: r["standard"][0]["mc1_choices"].__setitem__(0, "Changed"),
            "missing_binary": lambda m, r: r["binary"].pop(),
            "duplicate_binary": lambda m, r: r["binary"].append(copy.deepcopy(r["binary"][0])),
            "tampered_binary": lambda m, r: r["binary"][0].update(confidence=0.123),
            "binary_no_tokens": lambda m, r: r["binary"][0].pop("label_token_counts"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
                config, metadata, rows = fixture()
                mutate(metadata, rows)
                write_pair(root, old, config, metadata, rows)
                with self.assertRaises(ValueError):
                    self.analyze(root, out, old, [config])
                self.assertFalse(out.exists())

    def test_cross_pair_mismatch_manifest_completeness_and_ambiguous_binary_rejected(self) -> None:
        for issue in ("missing_pair", "unexpected_pair", "standard_content", "binary_ids", "wrong_pin", "duplicate_models", "ambiguous_binary", "missing_binary_metadata"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as temporary:
                root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
                config, metadata, rows = fixture()
                write_pair(root, old, config, metadata, rows)
                config2, metadata2, rows2 = fixture("fixture_b", 14.0)
                if issue == "standard_content":
                    for row in rows2["standard"]:
                        row["mc1_choices"][0] = "Different best answer"
                elif issue == "binary_ids":
                    for row in rows2["binary"]:
                        row["question_id"] += "_changed"
                elif issue == "duplicate_models":
                    config2["models"] = copy.deepcopy(config["models"])
                    for benchmark in metadata2:
                        metadata2[benchmark]["config"]["models"] = copy.deepcopy(config["models"])
                        for row in rows2[benchmark]:
                            row["model_id"] = config2["models"][row["model_key"]]["model_id"]
                if issue != "missing_pair":
                    write_pair(root, old, config2, metadata2, rows2, old_binary=False)
                configs = [config, config2]
                if issue == "unexpected_pair":
                    configs = [config]
                elif issue == "wrong_pin":
                    configs = copy.deepcopy(configs)
                    configs[0]["models"]["base"]["revision"] = "c" * 40
                elif issue == "ambiguous_binary":
                    directory = root / "fixture_a"
                    write_json(metadata["binary"], directory / "binary_metadata.json")
                    write_gzip(directory / "binary_scores.jsonl.gz", rows["binary"])
                elif issue == "missing_binary_metadata":
                    (root / "fixture_b" / "binary_metadata.json").unlink()
                with self.assertRaises(ValueError):
                    self.analyze(root, out, old, configs)
                self.assertFalse(out.exists())

    def test_cut_short_run_requires_opt_in_and_reports_missing_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
            config, metadata, rows = fixture()
            missing, _metadata, _rows = fixture("fixture_b", 14.0)
            write_pair(root, old, config, metadata, rows)
            with self.assertRaises(ValueError):
                self.analyze(root, out, old, [config, missing])
            summary = analyze_scale(
                root, out, old, iterations=3, allow_incomplete=True,
                _expected_pairs=[config, missing], _expected_question_counts=COUNTS,
            )
            self.assertEqual(summary["pair_count"], 1)
            self.assertEqual(summary["coverage"], {
                "planned_required_pairs": ["fixture_a", "fixture_b"],
                "completed_required_pairs": ["fixture_a"],
                "missing_required_pairs": ["fixture_b"],
                "allow_incomplete": True,
            })
            self.assertTrue(summary["validation"]["complete_model_protocol_groups"])

    def test_cut_short_mode_still_rejects_empty_partial_and_unexpected_data(self) -> None:
        for issue in ("empty", "partial", "unexpected"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as temporary:
                root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
                config, metadata, rows = fixture()
                if issue == "empty":
                    root.mkdir()
                else:
                    if issue == "partial":
                        rows["standard"].pop()
                    write_pair(root, old, config, metadata, rows)
                expected = [fixture("fixture_b", 14.0)[0]] if issue == "unexpected" else [config]
                with self.assertRaises(ValueError):
                    analyze_scale(root, out, old, iterations=3, allow_incomplete=True,
                                  _expected_pairs=expected, _expected_question_counts=COUNTS)
                self.assertFalse(out.exists())

    def test_optional_manifest_pairs_absent_or_complete_not_partial(self) -> None:
        for state in ("absent", "complete", "partial"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
                config, metadata, rows = fixture()
                write_pair(root, old, config, metadata, rows)
                config2, metadata2, rows2 = fixture("fixture_b", 32)
                if state != "absent":
                    write_pair(root, old, config2, metadata2, rows2, old_binary=False)
                if state == "partial":
                    (root / "fixture_b" / "binary_scores.jsonl.gz").unlink()
                manifest_path = Path(temporary) / "manifest.json"
                write_json({"pairs": [config], "optional_pairs": [config2]}, manifest_path)
                with patch("alignmenttax.scale_analysis.MANIFEST", manifest_path):
                    if state == "partial":
                        with self.assertRaises(ValueError):
                            analyze_scale(root, out, old, iterations=3, _expected_question_counts=COUNTS)
                    else:
                        summary = analyze_scale(root, out, old, iterations=3, _expected_question_counts=COUNTS)
                        self.assertEqual(summary["pair_count"], 1 if state == "absent" else 2)
                        self.assertEqual(sum(pair["optional_pair"] for pair in summary["pairs"]), state == "complete")

    def test_classification_requires_strict_interval_signs_for_each_truth_metric(self) -> None:
        positive = {"point": 0.1, "ci_low": 0.01, "ci_high": 0.2}
        negative = {"point": -0.1, "ci_low": -0.2, "ci_high": -0.01}
        touches = {"point": 0.1, "ci_low": 0.0, "ci_high": 0.2}
        metrics = {metric: positive.copy() for metric in STANDARD_METRICS}
        outcome = _classify(metrics, "standard")
        self.assertEqual(outcome["truth_calibration"], {"mc1_accuracy": "truth_gain_ece_worse", "mc2": "truth_gain_ece_worse"})
        metrics["mc1_ece"] = negative
        self.assertEqual(_classify(metrics, "standard")["truth_calibration"]["mc2"], "truth_and_calibration_improve")
        metrics["mc1_accuracy"] = touches
        self.assertEqual(_classify(metrics, "standard")["truth_calibration"]["mc1_accuracy"], "uncertain")
        metrics["mc1_ece"] = touches
        self.assertEqual(_classify(metrics, "standard")["truth_calibration"]["mc2"], "uncertain")

    def test_default_counts_not_weakened_and_cli_has_no_fixture_count_override(self) -> None:
        self.assertEqual(QUESTION_COUNTS, {"standard": 817, "binary": 790})
        with tempfile.TemporaryDirectory() as temporary:
            root, old, out = (Path(temporary) / x for x in ("day", "old", "out"))
            config, metadata, rows = fixture()
            write_pair(root, old, config, metadata, rows)
            for allow_incomplete in (False, True):
                with self.subTest(allow_incomplete=allow_incomplete), self.assertRaises(ValueError):
                    analyze_scale(root, out, old, iterations=3, allow_incomplete=allow_incomplete,
                                  _expected_pairs=[config])
            for args in ("--expected-question-count", "--expected-standard-count", "--expected-pairs"):
                with self.subTest(option=args), self.assertRaises(SystemExit):
                    main(["--root", str(root), "--out", str(out), args, "13"])
        for invalid in ({"iterations": 0}, {"iterations": True}, {"seed": -1}, {"seed": True},
                        {"_expected_question_counts": {"standard": 13, "binary": False}}):
            with self.subTest(arguments=invalid), self.assertRaises(ValueError):
                analyze_scale("unused", "unused", **invalid)


if __name__ == "__main__":
    unittest.main()
