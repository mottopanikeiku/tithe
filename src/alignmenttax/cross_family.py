from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_jsonl, write_json
from .metrics import EPSILON, METRIC_NAMES, metric_summary, score_from_label_logprobs
from .scoring import NATIVE_PROMPT_PROTOCOL, SHARED_PLAIN_PROTOCOL, configured_protocols

PROTOCOLS = (SHARED_PLAIN_PROTOCOL, NATIVE_PROMPT_PROTOCOL)
MODELS = ("base", "instruct")
EXPECTED_QUESTION_COUNT = 790
ECE_BINS = 10
BOOTSTRAP_CHUNK_SIZE = 256


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite real number.")
    return float(value)


def _pinned_revision(value: Any, context: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError(f"{context} must be a pinned 40-hex model revision.")
    return value


def _read_scores(directory: Path) -> list[dict[str, Any]]:
    plain, compressed = directory / "scores.jsonl", directory / "scores.jsonl.gz"
    if plain.exists() and compressed.exists():
        raise ValueError(f"{directory.name}: both plain and compressed scores exist; input is ambiguous.")
    if plain.exists():
        return read_jsonl(plain)
    if not compressed.exists():
        raise ValueError(f"{directory.name}: missing scores.jsonl or scores.jsonl.gz.")
    records = []
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{directory.name}: invalid compressed JSONL at line {line_number}.") from exc
    return records


def _load_pair(
    directory: Path, expected_question_count: int
) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    metadata_path = directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("fake") is not False:
        raise ValueError(f"{directory.name}: run metadata must explicitly declare fake=false.")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{directory.name}: missing metadata config.")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"{directory.name}: missing experiment metadata.")
    pair_id, family = experiment.get("pair_id"), experiment.get("family")
    if pair_id != directory.name or not isinstance(family, str) or not family.strip():
        raise ValueError(f"{directory.name}: experiment pair_id must match its directory and family must be set.")
    size = _finite_number(experiment.get("parameters_billion"), f"{pair_id} parameters_billion")
    if size <= 0:
        raise ValueError(f"{pair_id}: parameters_billion must be positive.")
    models = config.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError(f"{pair_id}: exactly base and instruct model configs are required.")
    for model_key, model in models.items():
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str) or not model["model_id"]:
            raise ValueError(f"{pair_id}: missing {model_key} model_id.")
        _pinned_revision(model.get("revision"), f"{pair_id} {model_key}")
    if models["base"]["model_id"] == models["instruct"]["model_id"]:
        raise ValueError(f"{pair_id}: base and instruct must be distinct model IDs.")
    configured = configured_protocols(config)
    if len(configured) != len(PROTOCOLS) or set(configured) != set(PROTOCOLS):
        raise ValueError(f"{pair_id}: both shared primary and native sensitivity protocols are required.")

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        (protocol, model): {} for protocol in PROTOCOLS for model in MODELS
    }
    identities: dict[str, Any] = {}
    records = _read_scores(directory)
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{pair_id}: every score record must be an object.")
        question_id = row.get("question_id")
        group_key = (row.get("prompt_protocol"), row.get("model_key"))
        if (not isinstance(question_id, str) or not question_id or
                not all(isinstance(value, str) for value in group_key) or group_key not in grouped):
            raise ValueError(f"{pair_id}: invalid question ID, model key or prompt protocol.")
        context = f"{pair_id}/{group_key[0]}/{group_key[1]}/{question_id}"
        if question_id in grouped[group_key]:
            raise ValueError(f"{context}: duplicate score key.")
        model = models[group_key[1]]
        if row.get("model_id") != model["model_id"] or row.get("model_revision") != model["revision"]:
            raise ValueError(f"{context}: model ID/revision does not match pinned config.")
        for field in ("device", "dtype"):
            value = row.get(field)
            if not isinstance(value, str) or not value or any(
                marker in value.lower() for marker in ("fake", "synthetic", "mock")
            ):
                raise ValueError(f"{context}: fake or missing {field} is not accepted.")
        expected_format = "chat_template" if group_key == (NATIVE_PROMPT_PROTOCOL, "instruct") else "plain"
        if row.get("prompt_format") != expected_format or row.get("fake", False) is not False:
            raise ValueError(f"{context}: fake or incorrect prompt format is not accepted.")
        counts = row.get("label_token_counts")
        if not isinstance(counts, dict) or any(
            type(counts.get(label)) is not int or counts[label] <= 0 for label in ("A", "B")
        ):
            raise ValueError(f"{context}: real label token counts are required.")
        correct_label = row.get("correct_label")
        choices = row.get("choices")
        if correct_label not in ("A", "B") or not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError(f"{context}: missing question or correct label.")
        if not isinstance(choices, dict) or set(choices) != {"A", "B"} or any(
            not isinstance(value, str) or not value for value in choices.values()
        ):
            raise ValueError(f"{context}: both answer choices are required.")
        identity = {field: row.get(field) for field in ("question", "choices", "correct_label", "source_index", "category", "type")}
        if question_id in identities and identities[question_id] != identity:
            raise ValueError(f"{context}: question content differs across model/protocol records.")
        identities[question_id] = identity
        logprobs = [_finite_number(row.get(f"raw_logprob_{label}"), f"{context} raw_logprob_{label}") for label in ("A", "B")]
        if any(value > 0 for value in logprobs):
            raise ValueError(f"{context}: label loglikelihoods cannot be positive.")
        derived = score_from_label_logprobs(logprob_a=logprobs[0], logprob_b=logprobs[1], correct_label=correct_label)
        for field, expected in derived.items():
            actual = row.get(field)
            if isinstance(expected, (float, int)) and not isinstance(expected, bool):
                actual_number = _finite_number(actual, f"{context} {field}")
                matches = math.isclose(actual_number, expected, rel_tol=1e-10, abs_tol=1e-12)
            else:
                matches = type(actual) is type(expected) and actual == expected
            if not matches:
                raise ValueError(f"{context}: {field} disagrees with raw label loglikelihoods.")
        grouped[group_key][question_id] = row

    canonical_ids: set[str] | None = None
    ordered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, by_id in grouped.items():
        if len(by_id) != expected_question_count:
            raise ValueError(f"{pair_id}/{key}: expected exactly {expected_question_count} question IDs, found {len(by_id)}.")
        ids = set(by_id)
        if canonical_ids is not None and ids != canonical_ids:
            raise ValueError(f"{pair_id}: question ID sets differ across models/protocols.")
        canonical_ids = ids
        ordered[key] = [by_id[question_id] for question_id in sorted(ids)]
    for field, expected in (("question_rows", expected_question_count), ("score_rows_total", 4 * expected_question_count)):
        if field in metadata and metadata[field] != expected:
            raise ValueError(f"{pair_id}: metadata {field} disagrees with complete scores.")
    return {"pair_id": pair_id, "family": family, "parameters_billion": size, "models": models}, ordered, identities


def _metric_arrays(rows: list[dict[str, Any]]) -> np.ndarray:
    correct = np.asarray([float(row["correct"]) for row in rows])
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    p_correct = np.asarray([row["p_correct"] for row in rows], dtype=np.float64)
    return np.column_stack((correct, confidence, confidence - correct, (1.0 - p_correct) ** 2, -np.log(np.maximum(p_correct, EPSILON))))


def _resampled_metrics(values: np.ndarray, indices: np.ndarray, *, ece_bins: int = ECE_BINS) -> np.ndarray:
    """Match metric_summary, including stable sample-order ties in ECE bins."""
    sampled = values[indices]
    result = np.empty((len(indices), len(METRIC_NAMES)), dtype=np.float64)
    result[:, :5] = sampled.mean(axis=1)
    # Equal-frequency bins are rebuilt for each draw, not reused from observed data.
    order = np.argsort(sampled[:, :, 1], axis=1, kind="stable")
    residuals = np.take_along_axis(sampled[:, :, 2], order, axis=1)
    n = indices.shape[1]
    bin_count = min(ece_bins, n)
    sizes = np.full(bin_count, n // bin_count, dtype=np.int64)
    sizes[: n % bin_count] += 1
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    result[:, 5] = np.abs(np.add.reduceat(residuals, starts, axis=1)).sum(axis=1) / n
    return result


def _bootstrap_deltas(
    base_rows: list[dict[str, Any]], instruct_rows: list[dict[str, Any]], *, iterations: int, seed: int,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> dict[str, Any]:
    base_summary, instruct_summary = metric_summary(base_rows), metric_summary(instruct_rows)
    base_values, instruct_values = _metric_arrays(base_rows), _metric_arrays(instruct_rows)
    samples = np.empty((iterations, len(METRIC_NAMES)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(base_rows)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = _resampled_metrics(instruct_values, indices) - _resampled_metrics(base_values, indices)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0, method="linear")
    return {
        "n_pairs": n, "iterations": iterations, "confidence_level": 0.95,
        "model_order": "instruct_minus_base",
        "metrics": {
            metric: {"point": float(instruct_summary[metric]) - float(base_summary[metric]),
                     "ci_low": float(bounds[0, index]), "ci_high": float(bounds[1, index])}
            for index, metric in enumerate(METRIC_NAMES)
        },
    }


def _classification(deltas: dict[str, dict[str, float]]) -> dict[str, Any]:
    signs = {
        metric: {"point": "positive" if values["point"] > 0 else "negative" if values["point"] < 0 else "zero",
                 "ci": "positive" if values["ci_low"] > 0 else "negative" if values["ci_high"] < 0 else "crosses_or_touches_zero"}
        for metric, values in deltas.items()
    }
    accuracy, ece = signs["accuracy"]["ci"], signs["ece"]["ci"]
    if accuracy == "positive" and ece == "positive":
        outcome = "supports_tradeoff"
        explanation = "Accuracy increased and ECE worsened; both intervals exclude zero."
    elif ece == "negative":
        outcome = "supports_calibration_improvement"
        explanation = "ECE improved; this does not imply that accuracy improved."
    elif ece == "positive" and accuracy == "negative":
        outcome = "accuracy_and_calibration_worsened"
        explanation = "Accuracy fell and ECE worsened; both intervals exclude zero."
    else:
        outcome = "uncertain"
        explanation = "At least one interval relevant to the tradeoff crosses or touches zero."
    return {"outcome": outcome, "explanation": explanation, "signs": signs}


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = ["pair_id", "family", "parameters_billion", "prompt_protocol", "role", "metric", "base", "instruct", "delta", "ci_low", "ci_high", "point_sign", "ci_sign", "classification"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in summary["pairs"]:
            for protocol, analysis in pair["protocols"].items():
                for metric, delta in analysis["deltas"]["metrics"].items():
                    writer.writerow({
                        **{key: pair[key] for key in ("pair_id", "family", "parameters_billion")},
                        "prompt_protocol": protocol, "role": analysis["role"], "metric": metric,
                        "base": analysis["models"]["base"][metric], "instruct": analysis["models"]["instruct"][metric],
                        "delta": delta["point"], "ci_low": delta["ci_low"], "ci_high": delta["ci_high"],
                        "point_sign": analysis["classification"]["signs"][metric]["point"],
                        "ci_sign": analysis["classification"]["signs"][metric]["ci"],
                        "classification": analysis["classification"]["outcome"],
                    })


def _write_svg(summary: dict[str, Any], path: Path) -> None:
    pairs = summary["pairs"]
    width, height = 1120, 160 + 62 * len(pairs)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
              '<title id="title">Matched base/instruct deltas with 95% paired-bootstrap intervals</title>',
              '<desc id="desc">Shared plain prompts are the primary comparison. Native prompts are sensitivity results. Positive accuracy means higher accuracy; positive ECE means worse calibration.</desc>',
              '<rect width="100%" height="100%" fill="white"/>',
              '<g font-family="sans-serif" font-size="12" fill="#202020">',
              '<text x="20" y="25" font-size="17">Instruct minus base · 95% paired-bootstrap intervals</text>',
              '<circle cx="28" cy="49" r="4" fill="#17649b"/><text x="40" y="53">Shared plain · primary</text>',
              '<path d="M 285 44 l 5 5 -5 5 -5 -5 Z" fill="#b75b16"/><text x="297" y="53">Native prompt · sensitivity</text>']
    for panel, (metric, label) in enumerate((("accuracy", "Δaccuracy (percentage points)"), ("ece", "ΔECE (percentage points)"))):
        left, right = 350 + panel * 390, 705 + panel * 390
        all_bounds = [100 * values[key] for pair in pairs for protocol in PROTOCOLS
                      for values in [pair["protocols"][protocol]["deltas"]["metrics"][metric]] for key in ("point", "ci_low", "ci_high")]
        lower, upper = min(0.0, min(all_bounds)), max(0.0, max(all_bounds))
        padding = max((upper - lower) * 0.1, 1.0)
        lower, upper = lower - padding, upper + padding

        def x(value: float) -> float:
            return left + (value - lower) / (upper - lower) * (right - left)

        pieces.append(f'<text x="{(left + right) / 2}" y="83" text-anchor="middle" font-size="14">{label}</text>')
        pieces.append(f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="95" y2="{height - 57}" stroke="#999" stroke-dasharray="4 3"/>')
        for tick in np.linspace(lower, upper, 5):
            pieces.append(f'<text x="{x(float(tick)):.2f}" y="{height - 37}" text-anchor="middle">{tick:.1f}</text>')
        for row_index, pair in enumerate(pairs):
            center = 116 + row_index * 62
            if panel == 0:
                pieces.append(f'<text x="20" y="{center + 4}">{escape(pair["models"]["base"]["model_id"].split("/")[-1])}</text>')
                pieces.append(f'<text x="20" y="{center + 20}" fill="#666">{escape(pair["family"])} · {pair["parameters_billion"]:g}B</text>')
            for protocol_index, protocol in enumerate(PROTOCOLS):
                values = pair["protocols"][protocol]["deltas"]["metrics"][metric]
                y = center - 7 + protocol_index * 17
                color = "#17649b" if protocol_index == 0 else "#b75b16"
                low, high, point = (x(100 * values[key]) for key in ("ci_low", "ci_high", "point"))
                pieces.append(f'<path d="M {low:.2f} {y} H {high:.2f} M {low:.2f} {y - 4} V {y + 4} M {high:.2f} {y - 4} V {y + 4}" stroke="{color}" fill="none" stroke-width="1.7"/>')
                if protocol_index == 0:
                    pieces.append(f'<circle cx="{point:.2f}" cy="{y}" r="4" fill="{color}"/>')
                else:
                    pieces.append(f'<path d="M {point:.2f} {y - 5} l 5 5 -5 5 -5 -5 Z" fill="{color}"/>')
    pieces.extend([f'<text x="20" y="{height - 12}" fill="#555">Accuracy is a truthfulness proxy; lower ECE is better. These matched comparisons do not establish causality or a family-population effect.</text>', '</g></svg>'])
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def analyze_cross_family(
    root: str | Path, out: str | Path, iterations: int = 10000, seed: int = 20260420,
    *, _expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> dict[str, Any]:
    """Analyze complete real score pairs; the private count override is for unit fixtures only."""
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("Bootstrap iterations must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer.")
    if type(_expected_question_count) is not int or _expected_question_count <= 0:
        raise ValueError("Expected question count must be a positive integer.")
    root_path, out_path = Path(root), Path(out)
    directories = sorted(path for path in root_path.iterdir() if path.is_dir() and
                         any((path / name).exists() for name in ("scores.jsonl", "scores.jsonl.gz", "run_metadata.json")))
    if not directories:
        raise ValueError("No pair score directories found.")
    loaded = [_load_pair(directory, _expected_question_count) for directory in directories]
    reference_identities = loaded[0][2]
    model_pairs: set[tuple[tuple[str, str], ...]] = set()
    for pair, _grouped, identities in loaded:
        if identities.keys() != reference_identities.keys():
            raise ValueError(f"{pair['pair_id']}: question ID set differs across pairs.")
        if identities != reference_identities:
            raise ValueError(f"{pair['pair_id']}: question content differs across pairs.")
        model_pair = tuple((pair["models"][model]["model_id"], pair["models"][model]["revision"]) for model in MODELS)
        if model_pair in model_pairs:
            raise ValueError(f"{pair['pair_id']}: duplicate pinned model pair.")
        model_pairs.add(model_pair)
    question_ids = sorted(reference_identities)
    summary: dict[str, Any] = {
        "question_count": len(question_ids), "question_ids": question_ids,
        "question_ids_sha256": hashlib.sha256(json.dumps(question_ids, separators=(",", ":")).encode()).hexdigest(),
        "pair_count": len(loaded), "iterations": iterations, "seed": seed, "confidence_level": 0.95,
        "bootstrap": {"unit": "question_id", "interval": "percentile", "quantile_method": "linear",
                      "rng": "numpy.default_rng/PCG64", "shared_resamples_across_pairs_and_protocols": True,
                      "chunk_size": BOOTSTRAP_CHUNK_SIZE},
        "ece": {"bins": ECE_BINS, "binning": "equal_frequency", "tie_order": "stable sampled order", "recomputed_per_resample": True},
        "protocol_roles": {SHARED_PLAIN_PROTOCOL: "primary", NATIVE_PROMPT_PROTOCOL: "sensitivity"},
        "interpretation": {
            "delta_order": "instruct_minus_base", "accuracy": "Truthfulness proxy on this binary task, not general truthfulness.",
            "primary_calibration_metric": "ece", "proper_score_checks": ["brier", "nll"],
            "classification_rule": "Accuracy CI strictly above zero and ECE CI strictly above zero supports a tradeoff. ECE CI strictly below zero supports calibration improvement only. Crossing or touching zero is uncertain for that metric.",
            "scope": "Descriptive matched comparisons on the same questions; no causal, family-population or population meta-inference. Intervals are per-comparison and not multiplicity-adjusted.",
        },
        "validation": {"fake_records_accepted": False, "unique_score_keys": True, "pinned_model_revisions": True,
                       "complete_model_protocol_groups": True, "same_question_ids_and_content": True,
                       "derived_metrics_checked_against_raw_loglikelihoods": True,
                       "fixture_question_count_override": _expected_question_count != EXPECTED_QUESTION_COUNT},
        "pairs": [],
    }
    for pair, grouped, _identities in loaded:
        pair["protocols"] = {}
        for protocol in PROTOCOLS:
            base_rows, instruct_rows = grouped[(protocol, "base")], grouped[(protocol, "instruct")]
            deltas = _bootstrap_deltas(base_rows, instruct_rows, iterations=iterations, seed=seed)
            pair["protocols"][protocol] = {
                "role": summary["protocol_roles"][protocol],
                "models": {"base": metric_summary(base_rows), "instruct": metric_summary(instruct_rows)},
                "deltas": deltas, "classification": _classification(deltas["metrics"]),
            }
        summary["pairs"].append(pair)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(summary, out_path / "summary.json")
    _write_csv(summary, out_path / "summary.csv")
    _write_svg(summary, out_path / "cross_family.svg")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare complete real base/instruct scores across matched model pairs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args(argv)
    analyze_cross_family(args.root, args.out, args.iterations, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
