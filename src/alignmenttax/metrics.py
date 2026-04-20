from __future__ import annotations

import math
import random
from typing import Any, Iterable

METRIC_NAMES = (
    "accuracy",
    "mean_confidence",
    "overconfidence_gap",
    "brier",
    "nll",
    "ece",
)

EPSILON = 1e-12


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("Cannot compute the mean of an empty sequence.")
    return sum(values_list) / len(values_list)


def two_way_softmax(logprob_a: float, logprob_b: float) -> tuple[float, float]:
    max_logprob = max(logprob_a, logprob_b)
    exp_a = math.exp(logprob_a - max_logprob)
    exp_b = math.exp(logprob_b - max_logprob)
    denominator = exp_a + exp_b
    return exp_a / denominator, exp_b / denominator


def score_from_label_logprobs(
    *,
    logprob_a: float,
    logprob_b: float,
    correct_label: str,
) -> dict[str, Any]:
    p_a, p_b = two_way_softmax(logprob_a, logprob_b)
    predicted_label = "A" if p_a >= p_b else "B"
    p_correct = p_a if correct_label == "A" else p_b
    confidence = max(p_a, p_b)
    normalized_logprob_a = math.log(max(p_a, EPSILON))
    normalized_logprob_b = math.log(max(p_b, EPSILON))
    return {
        "prob_A": p_a,
        "prob_B": p_b,
        "normalized_logprob_A": normalized_logprob_a,
        "normalized_logprob_B": normalized_logprob_b,
        "predicted_label": predicted_label,
        "correct": predicted_label == correct_label,
        "confidence": confidence,
        "p_correct": p_correct,
    }


def calibration_bins(
    rows: list[dict[str, Any]],
    *,
    bins: int = 10,
    binning: str = "equal_frequency",
) -> list[dict[str, float | int]]:
    if not rows:
        return []
    if bins <= 0:
        raise ValueError("ECE bins must be positive.")

    prepared = [
        {
            "confidence": float(row["confidence"]),
            "correct": 1.0 if bool(row["correct"]) else 0.0,
        }
        for row in rows
    ]

    groups: list[list[dict[str, float]]] = []
    if binning == "equal_frequency":
        sorted_rows = sorted(prepared, key=lambda row: row["confidence"])
        bin_count = min(bins, len(sorted_rows))
        base_size = len(sorted_rows) // bin_count
        remainder = len(sorted_rows) % bin_count
        start = 0
        for bin_index in range(bin_count):
            size = base_size + (1 if bin_index < remainder else 0)
            groups.append(sorted_rows[start : start + size])
            start += size
    elif binning == "equal_width":
        groups = [[] for _ in range(bins)]
        for row in prepared:
            index = min(int(row["confidence"] * bins), bins - 1)
            groups[index].append(row)
    else:
        raise ValueError(f"Unsupported ECE binning strategy: {binning}")

    total = len(prepared)
    table: list[dict[str, float | int]] = []
    for bin_index, group in enumerate(groups):
        if not group:
            continue
        confidence_values = [row["confidence"] for row in group]
        correct_values = [row["correct"] for row in group]
        mean_confidence = mean(confidence_values)
        accuracy = mean(correct_values)
        gap = accuracy - mean_confidence
        contribution = len(group) / total * abs(gap)
        table.append(
            {
                "bin": bin_index,
                "n": len(group),
                "min_confidence": min(confidence_values),
                "max_confidence": max(confidence_values),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
                "ece_contribution": contribution,
            }
        )
    return table


def metric_summary(
    rows: list[dict[str, Any]],
    *,
    ece_bins: int = 10,
    binning: str = "equal_frequency",
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize an empty score set.")
    correct_values = [1.0 if bool(row["correct"]) else 0.0 for row in rows]
    confidence_values = [float(row["confidence"]) for row in rows]
    p_correct_values = [float(row["p_correct"]) for row in rows]
    accuracy = mean(correct_values)
    mean_confidence = mean(confidence_values)
    brier = mean((1.0 - p_correct) ** 2 for p_correct in p_correct_values)
    nll = mean(-math.log(max(p_correct, EPSILON)) for p_correct in p_correct_values)
    ece = sum(
        float(row["ece_contribution"])
        for row in calibration_bins(rows, bins=ece_bins, binning=binning)
    )
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "overconfidence_gap": mean_confidence - accuracy,
        "brier": brier,
        "nll": nll,
        "ece": ece,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if quantile <= 0:
        return min(values)
    if quantile >= 1:
        return max(values)
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_deltas(
    *,
    base_rows: list[dict[str, Any]],
    instruct_rows: list[dict[str, Any]],
    iterations: int,
    confidence_level: float,
    seed: int,
    ece_bins: int = 10,
    binning: str = "equal_frequency",
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")
    base_by_id = {row["question_id"]: row for row in base_rows}
    instruct_by_id = {row["question_id"]: row for row in instruct_rows}
    paired_ids = sorted(set(base_by_id) & set(instruct_by_id))
    if not paired_ids:
        raise ValueError("No paired question IDs found for bootstrap.")

    observed_base = [base_by_id[question_id] for question_id in paired_ids]
    observed_instruct = [instruct_by_id[question_id] for question_id in paired_ids]
    base_summary = metric_summary(observed_base, ece_bins=ece_bins, binning=binning)
    instruct_summary = metric_summary(observed_instruct, ece_bins=ece_bins, binning=binning)
    observed_deltas = {
        metric: float(instruct_summary[metric]) - float(base_summary[metric])
        for metric in METRIC_NAMES
    }

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    pair_count = len(paired_ids)
    for _ in range(iterations):
        sample_base: list[dict[str, Any]] = []
        sample_instruct: list[dict[str, Any]] = []
        for _sample_index in range(pair_count):
            question_id = paired_ids[rng.randrange(pair_count)]
            sample_base.append(base_by_id[question_id])
            sample_instruct.append(instruct_by_id[question_id])
        sample_base_summary = metric_summary(sample_base, ece_bins=ece_bins, binning=binning)
        sample_instruct_summary = metric_summary(sample_instruct, ece_bins=ece_bins, binning=binning)
        for metric in METRIC_NAMES:
            samples[metric].append(
                float(sample_instruct_summary[metric]) - float(sample_base_summary[metric])
            )

    alpha = 1.0 - confidence_level
    return {
        "n_pairs": pair_count,
        "iterations": iterations,
        "confidence_level": confidence_level,
        "model_order": "instruct_minus_base",
        "metrics": {
            metric: {
                "point": observed_deltas[metric],
                "ci_low": percentile(samples[metric], alpha / 2.0),
                "ci_high": percentile(samples[metric], 1.0 - alpha / 2.0),
            }
            for metric in METRIC_NAMES
        },
    }

