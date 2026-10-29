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

from alignmenttax.cross_family import MODELS, PROTOCOLS
from alignmenttax.io_utils import write_json
from alignmenttax.metrics import METRIC_NAMES, metric_summary, score_from_label_logprobs, two_way_softmax
from alignmenttax.reanalysis import (
    analyze_reanalysis, answer_key, bootstrap_mass, held_out_analysis, main, match_candidates,
    matched_arrays, mc2_mass, score_arrays, split_manifest,
)
from alignmenttax.scale_analysis import DATASET_REVISION, HARNESS_REVISION, analyze_scale
from alignmenttax.scoring import NATIVE_PROMPT_PROTOCOL, _score_row
from alignmenttax.standard import standard_metric_row
from alignmenttax.uncertainty import (
    TEMPERATURE_BOUNDS, baseline_logits, bootstrap_metrics, categorical_values,
    fit_categorical_temperature, label_diagnostics, probability_rows, risk_coverage,
)

SEED = 20260420
COUNTS = {"standard": 13, "binary": 11}


def standard_identity(index: int, question: str, choices: list[str], labels: list[int] | None = None) -> dict:
    labels = labels or [1] + [0] * (len(choices) - 1)
    return {"question": question, "source_index": index, "mc1_choices": choices, "mc1_labels": labels,
            "mc2_choices": choices, "mc2_labels": labels}


def binary_identity(index: int, question: str, a: str, b: str, correct: str = "A") -> dict:
    return {"question": question, "source_index": index, "choices": {"A": a, "B": b}, "correct_label": correct,
            "category": "Fixture", "type": "Adversarial"}


def matching_fixture() -> tuple[dict, dict]:
    standard = {
        "s0": standard_identity(0, "Exact?", ["Yes", "No", "Maybe"]),
        "s1": standard_identity(1, "Period?", ["Paris", "London.", "Rome"]),
        "s2": standard_identity(2, "Duplicate?", ["Yes", "No"]),
        "s3": standard_identity(3, "Duplicate?", ["Yes", "No"]),
        "s4": standard_identity(4, "Missing answer?", ["Yes", "No"]),
        "s5": standard_identity(5, "Ambiguous answer?", ["Yes", "No", "No."]),
        "s6": standard_identity(6, "Disagreeing label?", ["Yes", "No"]),
        "s7": standard_identity(7, "Case?", ["Yes", "No"]),
        "s8": standard_identity(8, "Unused standard question?", ["Yes", "No"]),
    }
    binary = {
        "b0": binary_identity(0, "Exact?", "No", "Yes", correct="B"),
        "b1": binary_identity(1, "Period?", "Paris.", "London"),
        "b2": binary_identity(2, "Not in the standard pool?", "Yes", "No"),
        "b3": binary_identity(3, "Duplicate?", "Yes", "No"),
        "b4": binary_identity(4, "Missing answer?", "Yes", "Unlisted"),
        "b5": binary_identity(5, "Ambiguous answer?", "Yes", "No"),
        "b6": binary_identity(6, "Disagreeing label?", "No", "Yes"),
        "b7": binary_identity(7, "Case?", "yes", "No"),
    }
    return binary, standard


def random_logits(rng: np.random.Generator, n: int, width: int, *, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    logits = rng.normal(0.0, 2.0, size=(n, width))
    probabilities = probability_rows(logits, temperature)[0]
    targets = np.asarray([rng.choice(width, p=row) for row in probabilities])
    return logits, targets


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    return float(categorical_values(logits, targets, temperature)[:, 4].mean())


class MatchingTest(unittest.TestCase):
    def test_answer_key_only_strips_outer_whitespace_and_terminal_periods(self) -> None:
        self.assertEqual(answer_key("  Paris. "), "Paris")
        self.assertEqual(answer_key("etc.."), "etc")
        self.assertEqual(answer_key("U.S. law"), "U.S. law")
        self.assertNotEqual(answer_key("Yes"), answer_key("yes"))
        self.assertNotEqual(answer_key("No,"), answer_key("No"))

    def test_matches_exclusions_and_manifest(self) -> None:
        binary, standard = matching_fixture()
        before = copy.deepcopy((binary, standard))
        manifest = match_candidates(binary, standard)
        self.assertEqual((binary, standard), before)
        self.assertEqual((manifest["binary_n"], manifest["standard_n"], manifest["matched_n"]), (8, 9, 2))
        matches = {row["binary_id"]: row for row in manifest["matches"]}
        self.assertEqual(set(matches), {"b0", "b1"})
        self.assertEqual(matches["b0"]["candidate_indices"], {"A": 1, "B": 0})
        self.assertFalse(matches["b0"]["normalization_required"])
        self.assertEqual(matches["b1"]["candidate_indices"], {"A": 0, "B": 1})
        self.assertTrue(matches["b1"]["normalization_required"])
        # The manifest records the original scored strings, never the normalized keys.
        self.assertEqual(matches["b1"]["binary_choices"], {"A": "Paris.", "B": "London"})
        self.assertEqual(matches["b1"]["standard_choices"], ["Paris", "London.", "Rome"])
        reasons = {row["binary_id"]: row["reason"] for row in manifest["binary_exclusions"]}
        self.assertEqual(reasons, {"b2": "question_missing", "b3": "question_ambiguous", "b4": "answer_missing",
                                   "b5": "answer_ambiguous", "b6": "answer_label_disagreement", "b7": "answer_missing"})
        self.assertEqual(next(row for row in manifest["binary_exclusions"] if row["binary_id"] == "b3")["standard_ids"], ["s2", "s3"])
        self.assertEqual({row["standard_id"] for row in manifest["standard_exclusions"]}, set(standard) - {"s0", "s1"})

    def test_duplicate_or_empty_matches_raise(self) -> None:
        binary, standard = matching_fixture()
        binary["b8"] = binary_identity(8, "Exact?", "Yes", "No")
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            match_candidates(binary, standard)
        with self.assertRaisesRegex(ValueError, "No unambiguous"):
            match_candidates({"b2": matching_fixture()[0]["b2"]}, standard)


class RepresentationTest(unittest.TestCase):
    def rows(self) -> tuple[list[dict], list[dict], dict]:
        binary, standard = matching_fixture()
        manifest = match_candidates(binary, standard)
        binary_rows = [{"question_id": key, "raw_logprob_A": -1.0 - i, "raw_logprob_B": -2.5 + 0.5 * i,
                        "correct_label": row["correct_label"]} for i, (key, row) in enumerate(sorted(binary.items()))]
        standard_rows = [{"question_id": key, "mc1_loglikelihoods": [-3.0 - j - 0.25 * i for j in range(len(row["mc1_choices"]))],
                          "mc1_labels": row["mc1_labels"]} for i, (key, row) in enumerate(sorted(standard.items()))]
        return binary_rows, standard_rows, manifest

    def test_two_answer_strings_renormalize_over_only_the_matched_candidates(self) -> None:
        binary_rows, standard_rows, manifest = self.rows()
        arrays = matched_arrays(binary_rows, standard_rows, manifest)
        logits, targets = arrays["matched_strings"]
        np.testing.assert_array_equal(targets, [1, 0])
        np.testing.assert_array_equal(logits, [[-4.0, -3.0], [-3.25, -4.25]])
        np.testing.assert_array_equal(arrays["matched_labels"][1], targets)
        values = categorical_values(logits, targets)
        for row, (a, b) in enumerate(logits):
            p_a, p_b = two_way_softmax(a, b)
            self.assertAlmostEqual(values[row, 1], max(p_a, p_b), places=15)
        # A third, unmatched candidate changes full-MC1 probabilities but not the two-string control.
        changed = copy.deepcopy(standard_rows)
        changed[0]["mc1_loglikelihoods"][2] = 0.0
        changed_arrays = matched_arrays(binary_rows, changed, manifest)
        np.testing.assert_array_equal(categorical_values(*changed_arrays["matched_strings"]), values)
        self.assertFalse(np.array_equal(categorical_values(*changed_arrays["matched_mc1_all_candidates"]),
                                        categorical_values(*arrays["matched_mc1_all_candidates"])))

    def test_binary_values_match_legacy_metrics_with_documented_brier_and_nll_conventions(self) -> None:
        rng = np.random.default_rng(1)
        logits = rng.normal(-3.0, 2.0, size=(41, 2))
        logits[0] = [-1.0, -40.0]  # p(B) < 1e-12: legacy NLL clips, the new NLL does not.
        targets = (np.arange(41) % 3 == 0).astype(int)
        legacy = [score_from_label_logprobs(logprob_a=a, logprob_b=b, correct_label="AB"[t]) for (a, b), t in zip(logits, targets)]
        values = categorical_values(logits, targets)
        np.testing.assert_array_equal(values[:, 0], [float(row["correct"]) for row in legacy])
        np.testing.assert_allclose(values[:, 1], [row["confidence"] for row in legacy], rtol=0, atol=1e-14)
        np.testing.assert_allclose(values[:, 3], [2 * (1 - row["p_correct"]) ** 2 for row in legacy], rtol=0, atol=1e-14)
        np.testing.assert_allclose(values[1:, 4], [-math.log(row["p_correct"]) for row in legacy[1:]], rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(values[0, 4], 39.0, places=9)
        summary = metric_summary(legacy)
        self.assertAlmostEqual(values[:, 4].mean() - summary["nll"], (values[0, 4] + math.log(1e-12)) / 41, places=9)
        points = bootstrap_metrics({"x": values}, iterations=1, seed=0)["x"]
        for metric in ("accuracy", "mean_confidence", "overconfidence_gap", "ece"):
            self.assertAlmostEqual(points[metric]["point"], summary[metric], places=14)
        self.assertAlmostEqual(points["brier"]["point"], 2 * summary["brier"], places=14)

    def test_padded_mc1_values_match_the_published_standard_row(self) -> None:
        rows = [{"mc1_loglikelihoods": [-2.0, -1.0, -4.0], "mc1_labels": [1, 0, 0]},
                {"mc1_loglikelihoods": [-1.0, -3.0], "mc1_labels": [1, 0]},
                {"mc1_loglikelihoods": [-2.0, -2.0, -2.0, -9.0], "mc1_labels": [1, 0, 0, 0]}]
        logits, targets = score_arrays(rows, "mc1")
        self.assertEqual(logits.shape, (3, 4))
        self.assertTrue(np.isneginf(logits[1, 2:]).all())
        values = categorical_values(logits, targets)
        for row, value in zip(rows, values):
            expected = standard_metric_row(row["mc1_loglikelihoods"], row["mc1_labels"], [-1.0, -2.0], [1, 0])
            self.assertEqual(value[0], float(expected["mc1_accuracy"]))
            self.assertAlmostEqual(value[1], expected["mc1_confidence"], places=14)
            self.assertAlmostEqual(value[3], expected["mc1_brier"], places=14)
            self.assertAlmostEqual(value[4], expected["mc1_nll"], places=14)

    def test_label_diagnostics_expose_position_bias(self) -> None:
        logits = np.asarray([[0.0, -1.0]] * 9 + [[-1.0, 0.0]])
        targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        labels = label_diagnostics(logits, targets)
        self.assertEqual((labels["A"]["prediction_n"], labels["A"]["prediction_frequency"]), (9, 0.9))
        self.assertEqual((labels["A"]["target_n"], labels["A"]["accuracy_given_target"]), (4, 1.0))
        self.assertAlmostEqual(labels["B"]["accuracy_given_target"], 1 / 6)
        self.assertAlmostEqual(labels["A"]["accuracy_given_prediction"], 4 / 9)
        self.assertEqual(labels["B"]["accuracy_given_prediction"], 1.0)


class BootstrapTest(unittest.TestCase):
    def test_contrasts_use_the_same_question_draws(self) -> None:
        rng = np.random.default_rng(2)
        values = categorical_values(*random_logits(rng, 37, 3))
        shifted = values.copy()
        shifted[:, 4] += 0.1
        result = bootstrap_metrics({"a": values, "b": shifted, "noise": categorical_values(*random_logits(rng, 37, 3))},
                                   iterations=300, seed=SEED, contrasts={"b_minus_a": {"b": 1, "a": -1}})
        for metric in METRIC_NAMES:
            expected = 0.1 if metric == "nll" else 0.0
            for field in ("point", "ci_low", "ci_high"):
                self.assertAlmostEqual(result["b_minus_a"][metric][field], expected, places=12)
        # An unpaired comparison of the same two arrays would not collapse to a point.
        self.assertGreater(result["a"]["nll"]["ci_high"] - result["a"]["nll"]["ci_low"], 0.1)

    def test_contrast_interval_comes_from_per_draw_differences(self) -> None:
        rng = np.random.default_rng(3)
        first, second = (categorical_values(*random_logits(rng, 29, 2)) for _ in range(2))
        result = bootstrap_metrics({"first": first, "second": second}, iterations=300, seed=7,
                                   contrasts={"difference": {"second": 1, "first": -1}})
        draws = np.random.default_rng(7)
        indices = np.concatenate([draws.integers(0, 29, size=(256, 29)), draws.integers(0, 29, size=(44, 29))])
        differences = second[indices, 3].mean(axis=1) - first[indices, 3].mean(axis=1)
        low, high = np.quantile(differences, [0.025, 0.975])
        self.assertAlmostEqual(result["difference"]["brier"]["ci_low"], low, places=12)
        self.assertAlmostEqual(result["difference"]["brier"]["ci_high"], high, places=12)
        self.assertNotAlmostEqual(result["difference"]["brier"]["ci_low"],
                                  result["second"]["brier"]["ci_low"] - result["first"]["brier"]["ci_high"])

    def test_mc2_mass_bootstrap_shares_metric_draws(self) -> None:
        rng = np.random.default_rng(4)
        values = {key: categorical_values(*random_logits(rng, 31, 3)) for key in ("base/raw", "instruct/raw")}
        metrics = bootstrap_metrics(values, iterations=300, seed=SEED)
        masses = bootstrap_mass({key: value[:, 0] for key, value in values.items()}, iterations=300, seed=SEED,
                                contrasts={"difference": {"instruct/raw": 1, "base/raw": -1}})
        for key in values:
            for field in ("point", "ci_low", "ci_high"):
                self.assertAlmostEqual(masses[key][field], metrics[key]["accuracy"][field], places=12)
        self.assertAlmostEqual(masses["difference"]["point"], metrics["instruct/raw"]["accuracy"]["point"] - metrics["base/raw"]["accuracy"]["point"])

    def test_invalid_bootstrap_inputs_raise(self) -> None:
        values = categorical_values(np.zeros((3, 2)), np.zeros(3, dtype=int))
        for arguments in ({"values": {"x": values}, "iterations": 0}, {"values": {}, "iterations": 3},
                          {"values": {"x": values, "y": values[:2]}, "iterations": 3}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                bootstrap_metrics(seed=0, **arguments)


class TemperatureTest(unittest.TestCase):
    def test_interior_fit_matches_dense_grid_and_ignores_padding(self) -> None:
        logits, targets = random_logits(np.random.default_rng(5), 400, 4, temperature=2.0)
        fit = fit_categorical_temperature(logits, targets)
        self.assertIsNone(fit["boundary"])
        grid = np.exp(np.linspace(np.log(TEMPERATURE_BOUNDS[0]), np.log(TEMPERATURE_BOUNDS[1]), 4001))
        losses = np.asarray([nll(logits, targets, temperature) for temperature in grid])
        best = int(losses.argmin())
        self.assertLessEqual(fit["calibration_nll"], losses[best] + 1e-12)
        self.assertTrue(grid[best - 1] <= fit["temperature"] <= grid[best + 1])
        self.assertAlmostEqual(fit["calibration_nll"], nll(logits, targets, fit["temperature"]), places=14)
        padded = np.column_stack((logits, np.full(len(logits), -np.inf)))
        self.assertAlmostEqual(fit_categorical_temperature(padded, targets)["temperature"], fit["temperature"], places=10)

    def test_boundaries_are_reported_not_silently_clipped(self) -> None:
        logits = np.asarray([[0.0, -5.0, -6.0], [-7.0, 0.0, -4.0]] * 10)
        right, wrong = np.asarray([0, 1] * 10), np.asarray([2, 2] * 10)
        sharp = fit_categorical_temperature(logits, right)
        self.assertEqual((sharp["temperature"], sharp["boundary"]), (TEMPERATURE_BOUNDS[0], "lower"))
        flat = fit_categorical_temperature(logits, wrong)
        self.assertEqual((flat["temperature"], flat["boundary"]), (TEMPERATURE_BOUNDS[1], "upper"))
        uninformative = fit_categorical_temperature(np.zeros((4, 2)), np.asarray([0, 1, 0, 1]))
        self.assertEqual(uninformative["boundary"], "upper")
        with self.assertRaises(ValueError):
            fit_categorical_temperature(logits[:0], right[:0])
        for temperature in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                probability_rows(logits, temperature)


class HeldOutTest(unittest.TestCase):
    def fixture(self) -> tuple[list[str], dict]:
        rng = np.random.default_rng(6)
        questions = [f"Held-out question {i}?" for i in range(120)]
        logits, targets = random_logits(rng, 120, 2)
        # Base is overconfident and instruct underconfident about the same targets.
        return questions, {"base": (2.0 * logits, targets), "instruct": (0.5 * logits, targets)}

    def test_split_depends_only_on_question_text(self) -> None:
        questions = [f"Split question {i}?" for i in range(50)]
        calibration, rows = split_manifest(questions, seed=SEED)
        reverse, reverse_rows = split_manifest(list(reversed(questions)), seed=SEED)
        self.assertEqual({row["question"]: row["split"] for row in rows}, {row["question"]: row["split"] for row in reverse_rows})
        np.testing.assert_array_equal(calibration, reverse[::-1])
        self.assertTrue(all(row["question_sha256"] == hashlib.sha256(row["question"].encode()).hexdigest() for row in rows))
        self.assertTrue(0 < calibration.sum() < len(questions))
        with self.assertRaisesRegex(ValueError, "nonempty calibration and test"):
            split_manifest(questions[:1], seed=SEED)

    def test_test_questions_never_influence_fits_or_constant_priors(self) -> None:
        questions, arrays = self.fixture()
        calibration, _ = split_manifest(questions, seed=SEED)
        result, details = held_out_analysis(arrays, questions, task="binary", iterations=20, seed=SEED)
        self.assertEqual((result["calibration_n"], result["test_n"]), (int(calibration.sum()), int((~calibration).sum())))
        self.assertEqual(set(details), {"uniform", "constant", "base/raw", "base/temperature", "instruct/raw", "instruct/temperature"})
        changed = {}
        rng = np.random.default_rng(8)
        targets = arrays["base"][1].copy()
        targets[~calibration] = 1 - targets[~calibration]
        for model, (logits, _) in arrays.items():
            logits = logits.copy()
            logits[~calibration] = rng.normal(size=((~calibration).sum(), 2))
            changed[model] = (logits, targets)
        again, _ = held_out_analysis(changed, questions, task="binary", iterations=20, seed=SEED)
        self.assertEqual(again["fits"], result["fits"])
        self.assertEqual(again["constant_priors"], result["constant_priors"])
        self.assertNotEqual(again["estimates"]["base/raw"], result["estimates"]["base/raw"])
        changed["base"][0][calibration] *= 3
        self.assertNotEqual(held_out_analysis(changed, questions, task="binary", iterations=20, seed=SEED)[0]["fits"]["base"], result["fits"]["base"])

    def test_test_estimates_use_fitted_temperatures_on_test_rows_only(self) -> None:
        questions, arrays = self.fixture()
        calibration, _ = split_manifest(questions, seed=SEED)
        result, _ = held_out_analysis(arrays, questions, task="binary", iterations=20, seed=SEED)
        for model, (logits, targets) in arrays.items():
            temperature = result["fits"][model]["temperature"]
            self.assertAlmostEqual(result["estimates"][f"{model}/temperature"]["nll"]["point"],
                                   nll(logits[~calibration], targets[~calibration], temperature), places=13)
            self.assertAlmostEqual(result["fits"][model]["calibration_nll"],
                                   nll(logits[calibration], targets[calibration], temperature), places=13)
            # Temperature never changes two-choice predictions.
            self.assertEqual(result["estimates"][f"{model}/temperature"]["accuracy"], result["estimates"][f"{model}/raw"]["accuracy"])
        self.assertGreater(result["fits"]["base"]["temperature"], result["fits"]["instruct"]["temperature"])

    def test_mc2_truth_mass_is_paired_with_baselines_on_the_same_test_questions(self) -> None:
        rng = np.random.default_rng(10)
        questions = [f"MC2 question {i}?" for i in range(90)]
        calibration, _ = split_manifest(questions, seed=SEED)
        logits = rng.normal(size=(90, 3))
        arrays = {model: (scale * logits, np.zeros(90, dtype=int)) for model, scale in (("base", 1.0), ("instruct", 2.0))}
        # Flat MC2 likelihoods give every condition the uniform per-question truth mass.
        rows = [{"question": question, "mc2_loglikelihoods": [-1.0] * (2 + i % 3),
                 "mc2_labels": [1] * (1 + i % 2) + [0] * (1 + i % 3 - i % 2)} for i, question in enumerate(questions)]
        result, _ = held_out_analysis(arrays, questions, task="mc1", iterations=50, seed=SEED,
                                      standard_rows={model: rows for model in MODELS})
        masses = result["mc2_truth_mass"]
        expected = np.mean([sum(row["mc2_labels"]) / len(row["mc2_labels"]) for row, selected in zip(rows, ~calibration) if selected])
        self.assertAlmostEqual(masses["base/raw"]["point"], expected, places=14)
        for key in ("base/temperature_minus_uniform", "instruct/temperature_minus_uniform", "instruct_minus_base/raw"):
            for field in ("point", "ci_low", "ci_high"):
                self.assertAlmostEqual(masses[key][field], 0.0, places=12)
        self.assertGreater(masses["uniform"]["ci_high"] - masses["uniform"]["ci_low"], 0.01)


class BaselineTest(unittest.TestCase):
    def test_baselines_are_blind_to_model_scores(self) -> None:
        rng = np.random.default_rng(9)
        questions = [f"Baseline question {i}?" for i in range(80)]
        calibration, _ = split_manifest(questions, seed=SEED)
        logits = rng.normal(size=(80, 5))
        logits[::3, 4] = -np.inf
        targets = np.zeros(80, dtype=int)
        first = baseline_logits(logits, targets, questions, calibration, seed=SEED, permute_positions=True)
        other = np.where(np.isfinite(logits), rng.normal(size=logits.shape), -np.inf)
        second = baseline_logits(other, targets, questions, calibration, seed=SEED, permute_positions=True)
        for key in ("uniform", "constant"):
            np.testing.assert_array_equal(first[0][key], second[0][key])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(set(first[2]), {"4", "5"})

    def test_truth_first_mc1_order_is_not_an_oracle(self) -> None:
        questions = [f"MC1 question {i}?" for i in range(400)]
        calibration, _ = split_manifest(questions, seed=SEED)
        logits, targets = np.zeros((400, 4)), np.zeros(400, dtype=int)
        baselines, blinded, priors = baseline_logits(logits, targets, questions, calibration, seed=SEED, permute_positions=True)
        self.assertEqual(sorted(set(blinded.tolist())), [0, 1, 2, 3])
        accuracy = categorical_values(baselines["uniform"], blinded)[:, 0].mean()
        self.assertLess(abs(accuracy - 0.25), 0.07)
        self.assertEqual(categorical_values(baselines["uniform"], targets)[:, 0].mean(), 1.0)
        self.assertLess(max(priors["4"]["probabilities"]), 0.4)
        self.assertEqual(priors["4"]["calibration_n"], int(calibration.sum()))

    def test_constant_prior_is_smoothed_calibration_frequency(self) -> None:
        questions = [f"Prior question {i}?" for i in range(60)]
        calibration, _ = split_manifest(questions, seed=SEED)
        targets = (np.arange(60) % 4 == 0).astype(int)
        logits = np.zeros((60, 2))
        baselines, blinded, priors = baseline_logits(logits, targets, questions, calibration, seed=SEED, permute_positions=False)
        np.testing.assert_array_equal(blinded, targets)
        counts = np.bincount(targets[calibration], minlength=2) + 1.0
        np.testing.assert_allclose(priors["2"]["probabilities"], counts / counts.sum(), rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.exp(baselines["constant"][0]), counts / counts.sum(), rtol=1e-14)
        changed = targets.copy()
        changed[~calibration] = 1 - changed[~calibration]
        self.assertEqual(baseline_logits(logits, changed, questions, calibration, seed=SEED, permute_positions=False)[2], priors)
        # A candidate count absent from calibration falls back to uniform.
        padded = np.column_stack((logits, np.full(60, -np.inf)))
        padded[~calibration, 2] = 0.0
        unseen = baseline_logits(padded, targets, questions, calibration, seed=SEED, permute_positions=False)[2]
        self.assertEqual(unseen["3"], {"calibration_n": 0, "probabilities": [1 / 3] * 3})

    def test_mc2_truth_mass_conventions(self) -> None:
        rows = [{"question": "Q one?", "mc2_loglikelihoods": [-1.0, -2.0, -0.5], "mc2_labels": [1, 0, 1]},
                {"question": "Q two?", "mc2_loglikelihoods": [-3.0, -1.0], "mc2_labels": [0, 1]}]
        for row, mass in zip(rows, mc2_mass(rows)):
            self.assertAlmostEqual(mass, standard_metric_row([0.0, -1.0], [1, 0], row["mc2_loglikelihoods"], row["mc2_labels"])["mc2"], places=14)
        np.testing.assert_allclose(mc2_mass(rows, baseline="uniform"), [2 / 3, 1 / 2])
        np.testing.assert_allclose(mc2_mass(rows, temperature=1e6), [2 / 3, 1 / 2], atol=1e-5)
        priors = {"2": {"probabilities": [0.9, 0.1]}}
        constant = mc2_mass(rows, baseline="constant", priors=priors, seed=SEED)
        self.assertAlmostEqual(constant[0], 2 / 3)
        self.assertIn(round(constant[1], 12), (0.9, 0.1))


class RiskCoverageTest(unittest.TestCase):
    def test_ties_use_expected_risk_and_are_order_invariant(self) -> None:
        values = np.asarray([[1.0, 0.9], [0.0, 0.9], [1.0, 0.5]])
        values = np.column_stack((values, np.zeros((3, 3))))
        result = risk_coverage(values)
        self.assertEqual([row["n"] for row in result["curve"]], [1, 2, 3])
        np.testing.assert_allclose([row["risk"] for row in result["curve"]], [0.5, 0.5, 1 / 3])
        self.assertAlmostEqual(result["aurc"], 4 / 9)
        self.assertAlmostEqual(result["random_ranking_risk"], 1 / 3)
        self.assertEqual(risk_coverage(values[[1, 0, 2]]), result)
        self.assertEqual(risk_coverage(values[[2, 1, 0]]), result)

    def test_all_tied_confidence_has_no_selective_value(self) -> None:
        correct = (np.arange(40) % 5 != 0).astype(float)
        values = np.column_stack((correct, np.full(40, 0.7), np.zeros((40, 3))))
        result = risk_coverage(values)
        self.assertAlmostEqual(result["aurc"], 0.2)
        self.assertTrue(all(abs(row["risk"] - 0.2) < 1e-15 for row in result["curve"]))
        self.assertEqual(result["curve"][-1]["coverage"], 1.0)


def pair_fixture(pair_id: str, size: float, shift: float) -> tuple[dict, dict, dict]:
    config = {
        "experiment": {"pair_id": pair_id, "family": "Fixture", "parameters_billion": size},
        "models": {model: {"model_id": f"fixture/{pair_id}-{model}", "revision": ("a" if model == "base" else "b") * 40,
                           "device": "auto", "dtype": "bf16"} for model in MODELS},
        "scoring": {"protocols": list(PROTOCOLS)},
        "standard_dataset": {"dataset_id": "truthfulqa/truthful_qa", "config": "multiple_choice", "split": "validation",
                             "revision": DATASET_REVISION, "harness_revision": HARNESS_REVISION},
    }
    rows = {"standard": [], "binary": []}
    for p, protocol in enumerate(PROTOCOLS):
        for m, model in enumerate(MODELS):
            prompt_format = "chat_template" if protocol == NATIVE_PROMPT_PROTOCOL and model == "instruct" else "plain"
            for index in range(COUNTS["standard"]):
                question = f"Fixture question {index}?"
                choices = [f"Best {index}", f"Wrong {index}", f"Other {index}"][: 2 + index % 2]
                likelihoods = [-1.0 - ((index * 7 + j * 3 + m * 5 + p) % 11) / 3 - shift * j for j in range(len(choices))]
                mc2_choices, mc2_labels = [f"True {index}", f"False {index}", f"Also true {index}"], [1, 0, 1]
                mc2 = [-2.0 - ((index + j + m) % 4) / 2 for j in range(3)]
                row = {"question_id": f"truthfulqa_mc_{index:04d}_{hashlib.sha256(question.encode()).hexdigest()[:12]}",
                       "question": question, "source_index": index, "model_key": model,
                       "model_id": config["models"][model]["model_id"], "model_revision": config["models"][model]["revision"],
                       "protocol": protocol, "prompt_format": prompt_format, "device": "cuda:0", "dtype": "torch.bfloat16",
                       "elapsed_seconds": 0.02, "mc1_choices": choices, "mc1_labels": [1] + [0] * (len(choices) - 1),
                       "mc1_loglikelihoods": likelihoods, "mc2_choices": mc2_choices, "mc2_labels": mc2_labels,
                       "mc2_loglikelihoods": mc2}
                row.update(standard_metric_row(likelihoods, row["mc1_labels"], mc2, mc2_labels))
                rows["standard"].append(row)
            for index in range(COUNTS["binary"]):
                correct = "A" if index % 3 else "B"
                best, wrong = f"Best {index}" + ("." if index == 4 else ""), "Not a candidate" if index == 9 else f"Wrong {index}"
                question = "A binary-only question?" if index == 10 else f"Fixture question {index}?"
                item = {"id": f"binary_q{index:03d}", "question": question, "source_index": 100 + index, "correct_label": correct,
                        "choices": {correct: best, "B" if correct == "A" else "A": wrong}, "category": "Fixture", "type": "Adversarial"}
                p_a = 0.2 + 0.6 * (((index * 5 + m * 3 + p) % 7) / 6)
                rows["binary"].append(_score_row(
                    item=item, model_key=model, model_id=config["models"][model]["model_id"],
                    model_revision=config["models"][model]["revision"], protocol=protocol,
                    logprob_a=math.log(p_a) - 2.0, logprob_b=math.log(1.0 - p_a) - 2.0, device="cuda:0",
                    dtype="torch.bfloat16", elapsed_seconds=0.01, label_token_counts={"A": 1, "B": 1}, prompt_format=prompt_format))
    metadata = {benchmark: {"config": copy.deepcopy(config), "fake": False, "question_rows": count, "score_rows_total": 4 * count}
                for benchmark, count in COUNTS.items()}
    return config, metadata, rows


def write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_pair(root: Path, old: Path, config: dict, metadata: dict, rows: dict, *, old_binary: bool) -> None:
    directory = root / config["experiment"]["pair_id"]
    directory.mkdir(parents=True)
    write_json(metadata["standard"], directory / "standard_metadata.json")
    write_gzip(directory / "standard_scores.jsonl.gz", rows["standard"])
    binary = old / config["experiment"]["pair_id"] if old_binary else directory
    binary.mkdir(parents=True, exist_ok=True)
    write_json(metadata["binary"], binary / ("run_metadata.json" if old_binary else "binary_metadata.json"))
    write_gzip(binary / ("scores.jsonl.gz" if old_binary else "binary_scores.jsonl.gz"), rows["binary"])


class ReanalysisEndToEndTest(unittest.TestCase):
    def run_fixture(self, temporary: Path) -> tuple[dict, Path, Path, Path]:
        root, old, out, full = (temporary / name for name in ("day", "old", "reanalysis", "full"))
        configs = []
        for pair_id, size, shift, old_binary in (("fixture_a", 0.5, 0.2, True), ("fixture_b", 7.0, 0.6, False)):
            config, metadata, rows = pair_fixture(pair_id, size, shift)
            write_pair(root, old, config, metadata, rows, old_binary=old_binary)
            configs.append(config)
        write_json({"pairs": configs, "optional_pairs": []}, temporary / "manifest.json")
        with patch("alignmenttax.scale_analysis.MANIFEST", temporary / "manifest.json"), \
                patch.dict("alignmenttax.scale_analysis.QUESTION_COUNTS", COUNTS):
            summary = analyze_reanalysis(root, old, out, full_out=full, iterations=9, seed=SEED)
            reference = analyze_scale(root, temporary / "reference", old, iterations=9, seed=SEED)
        self.assertEqual(json.loads((full / "summary.json").read_text()), reference)
        return summary, root, out, full

    def test_outputs_splits_and_byte_identical_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            summary, root, out, full = self.run_fixture(temporary)
            raw = {path: path.read_bytes() for path in sorted(root.rglob("*.gz"))}
            outputs = {path: path.read_bytes() for directory in (out, full) for path in sorted(directory.iterdir())}
            for name in ("summary.json", "diagnostics.json", "matching.json", "splits.json", "candidate_control.csv", "held_out.csv",
                         "temperatures.csv", "label_diagnostics.csv", "ece_sensitivity.csv", "candidate_control_shared.svg",
                         "risk_coverage_mc1_native.svg", "reliability_matched_strings_shared.svg"):
                self.assertIn(out / name, outputs)
            for path in outputs:
                if path.suffix == ".svg":
                    ElementTree.fromstring(outputs[path])
            self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
            self.assertEqual(summary["matching"]["matched_n"], 9)
            self.assertEqual(summary["matching"]["binary_exclusion_counts"], {"answer_missing": 1, "question_missing": 1})
            self.assertEqual(summary["matching"]["normalization_required_n"], 1)
            splits = json.loads((out / "splits.json").read_text())
            by_task = {task: {row["question"]: row["split"] for row in rows} for task, rows in splits.items()}
            for question, split in by_task["matched"].items():
                self.assertEqual((by_task["binary"][question], by_task["mc1"][question]), (split, split))
            for pair in summary["pairs"].values():
                for protocol in PROTOCOLS:
                    control = pair["candidate_control"][protocol]["estimates"]
                    for metric in METRIC_NAMES:
                        expected = (control["matched_labels/instruct_minus_base"][metric]["point"]
                                    - control["matched_strings/instruct_minus_base"][metric]["point"])
                        self.assertAlmostEqual(control["labels_minus_strings_delta"][metric]["point"], expected, places=14)
                    for task, count in (("binary", 11), ("mc1", 13), ("matched_labels", 9), ("matched_strings", 9)):
                        held_out = pair["held_out"][task][protocol]
                        self.assertEqual(held_out["calibration_n"] + held_out["test_n"], count)
                    self.assertIn("mc2_truth_mass", pair["held_out"]["mc1"][protocol])
                    self.assertNotIn("mc2_truth_mass", pair["held_out"]["binary"][protocol])
                    self.assertEqual(set(pair["label_diagnostics"]), {"binary", "matched_labels", "matched_strings"})
            with (out / "held_out.csv").open(newline="") as handle:
                tasks = {row["task"] for row in csv.DictReader(handle)}
            self.assertEqual(tasks, {"binary", "mc1", "mc2", "matched_labels", "matched_strings"})
            # Rerunning the same command rewrites identical bytes and never touches saved scores.
            self.run_fixture_again(temporary, root, out, full)
            self.assertEqual({path: path.read_bytes() for directory in (out, full) for path in sorted(directory.iterdir())}, outputs)
            self.assertEqual({path: path.read_bytes() for path in sorted(root.rglob("*.gz"))}, raw)

    def run_fixture_again(self, temporary: Path, root: Path, out: Path, full: Path) -> None:
        with patch("alignmenttax.scale_analysis.MANIFEST", temporary / "manifest.json"), \
                patch.dict("alignmenttax.scale_analysis.QUESTION_COUNTS", COUNTS):
            analyze_reanalysis(root, temporary / "old", out, full_out=full, iterations=9, seed=SEED)

    def test_cli_defaults_regenerate_committed_locations(self) -> None:
        with patch("alignmenttax.reanalysis.analyze_reanalysis",
                   return_value={"matching": {"matched_n": 779}, "pairs": {}}) as analyze, patch("builtins.print"):
            self.assertEqual(main([]), 0)
        analyze.assert_called_once_with(Path("results/day_scale"), Path("results/cross_family"), Path("results/reanalysis"),
                                        full_out=Path("results/day_scale/analysis"), iterations=10000, seed=SEED)


if __name__ == "__main__":
    unittest.main()
