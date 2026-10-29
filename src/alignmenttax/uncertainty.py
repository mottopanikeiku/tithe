"""Saved-likelihood diagnostics; no model loading or inference."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .cross_family import BOOTSTRAP_CHUNK_SIZE, _resampled_metrics
from .metrics import METRIC_NAMES, calibration_bins

TEMPERATURE_BOUNDS = (0.01, 200.0)


def probability_rows(logits: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    shifted = (logits - logits.max(axis=1, keepdims=True)) / temperature
    log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probabilities = shifted - log_normalizer
    return np.exp(log_probabilities), log_probabilities


def categorical_values(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    probabilities, log_probabilities = probability_rows(logits, temperature)
    indices = np.arange(len(targets))
    correct = (logits.argmax(axis=1) == targets).astype(float)
    confidence = probabilities.max(axis=1)
    brier = (probabilities ** 2).sum(axis=1) - 2 * probabilities[indices, targets] + 1
    return np.column_stack((correct, confidence, confidence - correct, brier, -log_probabilities[indices, targets]))


def fit_categorical_temperature(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Minimize calibration NLL in inverse temperature, a convex one-dimensional problem."""
    if not len(targets):
        raise ValueError("Cannot fit a temperature without calibration questions.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    finite = np.where(np.isfinite(shifted), shifted, 0.0)
    target_logits = shifted[np.arange(len(targets)), targets]

    def derivative(beta: float) -> float:
        weights = np.exp(shifted * beta)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        return float(((probabilities * finite).sum(axis=1) - target_logits).mean())

    low, high = 1 / TEMPERATURE_BOUNDS[1], 1 / TEMPERATURE_BOUNDS[0]
    if derivative(low) >= 0:
        temperature, boundary = TEMPERATURE_BOUNDS[1], "upper"
    elif derivative(high) <= 0:
        temperature, boundary = TEMPERATURE_BOUNDS[0], "lower"
    else:
        for _ in range(64):
            middle = (low + high) / 2
            if derivative(middle) > 0:
                high = middle
            else:
                low = middle
        temperature, boundary = 1 / ((low + high) / 2), None
    nll = categorical_values(logits, targets, temperature)[:, 4].mean()
    return {"temperature": temperature, "boundary": boundary, "calibration_nll": float(nll)}


def baseline_logits(
    logits: np.ndarray, targets: np.ndarray, questions: list[str], calibration: np.ndarray, *,
    seed: int, permute_positions: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Question-independent position priors, separately for each candidate count.

    MC1 stores the best answer first. Hash-permute its coordinates before estimating
    a baseline, or a constant index-zero oracle would leak the benchmark key.
    This does not change a model's prompt, scores, or tie-breaking convention.
    """
    counts = np.isfinite(logits).sum(axis=1)
    baseline_targets = targets.copy()
    if permute_positions:
        for i, (question, count) in enumerate(zip(questions, counts)):
            order = sorted(range(int(count)), key=lambda j: hashlib.sha256(f"{seed}|{question}|position|{j}".encode()).digest())
            baseline_targets[i] = order.index(int(targets[i]))
    uniform = np.where(np.isfinite(logits), 0.0, -np.inf)
    constant = uniform.copy()
    priors = {}
    for count in sorted(set(counts.tolist())):
        selected = calibration & (counts == count)
        frequencies = np.bincount(baseline_targets[selected], minlength=count) + 1.0
        probabilities = frequencies / frequencies.sum()
        constant[counts == count, :count] = np.log(probabilities)
        priors[str(count)] = {"calibration_n": int(selected.sum()), "probabilities": probabilities.tolist()}
    return {"uniform": uniform, "constant": constant}, baseline_targets, priors


def risk_coverage(values: np.ndarray) -> dict[str, Any]:
    """Expected prefix risk under random ordering inside exact confidence ties."""
    confidence, errors = values[:, 1], 1 - values[:, 0]
    order = np.argsort(-confidence, kind="stable")
    confidence, errors = confidence[order], errors[order]
    starts = np.r_[0, np.flatnonzero(confidence[1:] != confidence[:-1]) + 1]
    ends = np.r_[starts[1:], len(errors)]
    expected_errors = np.empty(len(errors))
    prior_errors = 0.0
    for start, end in zip(starts, ends):
        rate = errors[start:end].mean()
        expected_errors[start:end] = prior_errors + np.arange(1, end - start + 1) * rate
        prior_errors += errors[start:end].sum()
    risks = expected_errors / np.arange(1, len(errors) + 1)
    retained = np.unique(np.ceil(np.linspace(0.05, 1.0, 20) * len(errors)).astype(int))
    return {"aurc": float(risks.mean()), "random_ranking_risk": float(errors.mean()),
            "tie_policy": "Expected risk over random ordering within exact confidence ties",
            "curve": [{"n": int(k), "coverage": float(k / len(errors)), "risk": float(risks[k - 1])} for k in retained]}


def diagnostics(values: np.ndarray) -> dict[str, Any]:
    rows = [{"correct": bool(correct), "confidence": float(confidence)} for correct, confidence in values[:, :2]]
    reliability = {strategy: calibration_bins(rows, bins=10, binning=strategy)
                   for strategy in ("equal_frequency", "equal_width")}
    sensitivity = {strategy: {str(bins): sum(float(row["ece_contribution"]) for row in calibration_bins(rows, bins=bins, binning=strategy))
                              for bins in (5, 10, 15, 20, 50)} for strategy in reliability}
    return {"metrics": dict(zip(METRIC_NAMES, _resampled_metrics(values, np.arange(len(values))[None, :])[0].tolist())),
            "reliability": reliability, "ece_sensitivity": sensitivity, "risk_coverage": risk_coverage(values)}


def label_diagnostics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    return {label: {"target_n": int((targets == index).sum()),
                    "accuracy_given_target": float((predictions[targets == index] == index).mean()) if (targets == index).any() else None,
                    "prediction_n": int((predictions == index).sum()),
                    "prediction_frequency": float((predictions == index).mean()),
                    "accuracy_given_prediction": float((targets[predictions == index] == index).mean()) if (predictions == index).any() else None}
            for index, label in enumerate(("A", "B"))}


def bootstrap_metrics(
    values: dict[str, np.ndarray], *, iterations: int, seed: int,
    contrasts: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Shared question draws for every supplied condition and linear contrast."""
    if iterations <= 0 or not values:
        raise ValueError("Positive bootstrap iterations and nonempty conditions required.")
    n = len(next(iter(values.values())))
    if n == 0 or any(value.shape != (n, 5) for value in values.values()):
        raise ValueError("Bootstrap conditions must be aligned nonempty five-column arrays.")
    points = {key: _resampled_metrics(value, np.arange(n)[None, :])[0] for key, value in values.items()}
    draws = {key: np.empty((iterations, len(METRIC_NAMES))) for key in values}
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        for key, value in values.items():
            draws[key][start:stop] = _resampled_metrics(value, indices)
    for key, terms in (contrasts or {}).items():
        points[key] = sum(points[term] * weight for term, weight in terms.items())
        draws[key] = sum(draws[term] * weight for term, weight in terms.items())
    result = {}
    for key, point in points.items():
        bounds = np.quantile(draws[key], [0.025, 0.975], axis=0, method="linear")
        result[key] = {metric: {"point": float(point[j]), "ci_low": float(bounds[0, j]), "ci_high": float(bounds[1, j])}
                       for j, metric in enumerate(METRIC_NAMES)}
    return result
