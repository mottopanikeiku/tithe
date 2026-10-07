from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .cross_family import (
    BOOTSTRAP_CHUNK_SIZE, ECE_BINS, MODELS, PROTOCOLS,
    _finite_number, _metric_arrays, _pinned_revision, _resampled_metrics,
)
from .io_utils import write_json
from .metrics import METRIC_NAMES, score_from_label_logprobs
from .scoring import NATIVE_PROMPT_PROTOCOL, SHARED_PLAIN_PROTOCOL
from .standard import standard_metric_row

STANDARD_METRICS = (
    "mc1_accuracy", "mc2", "mc1_mean_confidence", "mc1_overconfidence_gap",
    "mc1_brier", "mc1_nll", "mc1_ece",
)
QUESTION_COUNTS = {"standard": 817, "binary": 790}
DATASET_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "day_scale.json"


def _source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be an object: {path}")
    return value


def _pair_config(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    experiment, models = config.get("experiment"), config.get("models")
    if not isinstance(experiment, dict) or experiment.get("pair_id") != pair_id:
        raise ValueError(f"{pair_id}: experiment pair_id must match directory.")
    family = experiment.get("family")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"{pair_id}: family is required.")
    size = _finite_number(experiment.get("parameters_billion"), f"{pair_id} size")
    if size <= 0:
        raise ValueError(f"{pair_id}: parameters_billion must be positive.")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError(f"{pair_id}: exactly base and instruct models required.")
    for model_key, model in models.items():
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str) or not model["model_id"]:
            raise ValueError(f"{pair_id}: {model_key} model_id required.")
        _pinned_revision(model.get("revision"), f"{pair_id}/{model_key}")
    if models["base"]["model_id"] == models["instruct"]["model_id"]:
        raise ValueError(f"{pair_id}: base and instruct must be distinct.")
    return {"pair_id": pair_id, "family": family, "parameters_billion": size, "models": models}


def _same_pair(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual[key] == expected[key] for key in ("pair_id", "family", "parameters_billion")) and all(
        all(actual["models"][model][field] == expected["models"][model][field]
            for field in ("model_id", "revision")) for model in MODELS
    )


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing required scores: {path}")
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: score must be an object.")
                    rows.append(value)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid compressed JSONL: {path}") from exc
    return rows


def _check_derived(row: dict[str, Any], derived: dict[str, Any], context: str) -> None:
    for field, expected in derived.items():
        actual = row.get(field)
        if isinstance(expected, bool):
            matches = type(actual) is bool and actual == expected
        elif isinstance(expected, (float, int)):
            actual = _finite_number(actual, f"{context}/{field}")
            matches = math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise ValueError(f"{context}: {field} disagrees with raw loglikelihoods.")


def _standard_identity(row: dict[str, Any], context: str) -> dict[str, Any]:
    for task in ("mc1", "mc2"):
        choices, labels, likelihoods = (row.get(f"{task}_{field}") for field in ("choices", "labels", "loglikelihoods"))
        if not isinstance(choices, list) or len(choices) < 2 or any(not isinstance(x, str) or not x for x in choices):
            raise ValueError(f"{context}: {task} answer strings required.")
        if not isinstance(labels, list) or len(labels) != len(choices) or any(type(x) is not int or x not in (0, 1) for x in labels):
            raise ValueError(f"{context}: {task} labels must be aligned zeros/ones.")
        if not any(labels) or (task == "mc1" and (sum(labels) != 1 or labels[0] != 1)):
            raise ValueError(f"{context}: MC1 best answer must be index zero; MC2 needs true answers.")
        if not isinstance(likelihoods, list) or len(likelihoods) != len(choices):
            raise ValueError(f"{context}: {task} likelihoods must align with choices.")
        if any(_finite_number(x, f"{context}/{task}") > 0 for x in likelihoods):
            raise ValueError(f"{context}: summed loglikelihoods cannot be positive.")
    index, question = row["source_index"], row["question"]
    expected_id = f"truthfulqa_mc_{index:04d}_{hashlib.sha256(question.encode()).hexdigest()[:12]}"
    if row["question_id"] != expected_id:
        raise ValueError(f"{context}: standard question ID disagrees with source index/question.")
    return {field: row[field] for field in (
        "question", "source_index", "mc1_choices", "mc1_labels", "mc2_choices", "mc2_labels",
    )}


def _binary_identity(row: dict[str, Any], context: str) -> dict[str, Any]:
    choices, label = row.get("choices"), row.get("correct_label")
    if label not in ("A", "B") or not isinstance(choices, dict) or set(choices) != {"A", "B"} or any(
        not isinstance(value, str) or not value for value in choices.values()
    ):
        raise ValueError(f"{context}: binary choices and correct label required.")
    counts = row.get("label_token_counts")
    if not isinstance(counts, dict) or any(type(counts.get(x)) is not int or counts[x] <= 0 for x in ("A", "B")):
        raise ValueError(f"{context}: real label token counts required.")
    likelihoods = [_finite_number(row.get(f"raw_logprob_{x}"), context) for x in ("A", "B")]
    if any(x > 0 for x in likelihoods):
        raise ValueError(f"{context}: label likelihoods cannot be positive.")
    _check_derived(row, score_from_label_logprobs(logprob_a=likelihoods[0], logprob_b=likelihoods[1], correct_label=label), context)
    return {field: row.get(field) for field in ("question", "source_index", "choices", "correct_label", "category", "type")}


def _load_benchmark(
    directory: Path, benchmark: str, expected: dict[str, Any], count: int, *, old_binary: bool = False,
) -> dict[str, Any]:
    score_name = "scores.jsonl.gz" if old_binary else f"{benchmark}_scores.jsonl.gz"
    metadata_name = "run_metadata.json" if old_binary else f"{benchmark}_metadata.json"
    score_path, metadata_path = directory / score_name, directory / metadata_name
    if score_path.with_suffix("").exists():
        raise ValueError(f"{directory.name}: uncompressed score input is not accepted alongside day-wave gzip sources.")
    metadata = _read_json(metadata_path)
    if metadata.get("fake") is not False:
        raise ValueError(f"{directory.name}: metadata must explicitly declare fake=false.")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{directory.name}: metadata config required.")
    pair = _pair_config(config, directory.name)
    if not _same_pair(pair, expected):
        raise ValueError(f"{directory.name}: models/family/size differ from fixed manifest.")
    if benchmark == "standard":
        dataset = config.get("standard_dataset")
        required_dataset = {"dataset_id": "truthfulqa/truthful_qa", "config": "multiple_choice",
                            "split": "validation", "revision": DATASET_REVISION,
                            "harness_revision": HARNESS_REVISION}
        if not isinstance(dataset, dict) or any(dataset.get(key) != value for key, value in required_dataset.items()):
            raise ValueError(f"{directory.name}: standard dataset and harness must match pinned task contract.")
    for field, value in (("question_rows", count), ("score_rows_total", 4 * count)):
        if type(metadata.get(field)) is not int or metadata[field] != value:
            raise ValueError(f"{directory.name}: metadata {field} must equal {value}.")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        (protocol, model): {} for protocol in PROTOCOLS for model in MODELS
    }
    identities: dict[str, Any] = {}
    hardware: dict[str, Any] = {}
    rows = _raw_rows(score_path)
    for row in rows:
        protocol = row.get("protocol") if benchmark == "standard" else row.get("prompt_protocol")
        key, question_id = (protocol, row.get("model_key")), row.get("question_id")
        if not all(isinstance(x, str) for x in key) or key not in grouped or not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{directory.name}: invalid model, protocol or question ID.")
        context = f"{directory.name}/{benchmark}/{protocol}/{key[1]}/{question_id}"
        if question_id in grouped[key]:
            raise ValueError(f"{context}: duplicate score key.")
        model = pair["models"][key[1]]
        if row.get("model_id") != model["model_id"] or row.get("model_revision") != model["revision"]:
            raise ValueError(f"{context}: row model/revision differs from config.")
        for field in ("device", "dtype"):
            value = row.get(field)
            if not isinstance(value, str) or not value or any(x in value.lower() for x in ("fake", "synthetic", "mock")):
                raise ValueError(f"{context}: fake/missing {field}.")
        expected_format = "chat_template" if key == (NATIVE_PROMPT_PROTOCOL, "instruct") else "plain"
        if row.get("fake", False) is not False or row.get("prompt_format") != expected_format:
            raise ValueError(f"{context}: fake/incorrect prompt format.")
        if not isinstance(row.get("question"), str) or not row["question"] or type(row.get("source_index")) is not int or row["source_index"] < 0:
            raise ValueError(f"{context}: question and nonnegative source_index required.")
        if _finite_number(row.get("elapsed_seconds"), context) < 0:
            raise ValueError(f"{context}: elapsed_seconds cannot be negative.")
        if benchmark == "standard":
            identity = _standard_identity(row, context)
            derived = standard_metric_row(row["mc1_loglikelihoods"], row["mc1_labels"], row["mc2_loglikelihoods"], row["mc2_labels"])
            _check_derived(row, derived, context)
        else:
            identity = _binary_identity(row, context)
        if question_id in identities and identities[question_id] != identity:
            raise ValueError(f"{context}: content differs across models/protocols.")
        identities[question_id] = identity
        grouped[key][question_id] = row
        observed = hardware.setdefault(protocol, {}).setdefault(key[1], {"devices": set(), "dtypes": set()})
        observed["devices"].add(row["device"])
        observed["dtypes"].add(row["dtype"])
    ids = sorted(identities)
    for key, group in grouped.items():
        if len(group) != count or set(group) != set(ids):
            raise ValueError(f"{directory.name}/{benchmark}/{key}: expected {count} identical question IDs in all four groups.")
    source_indices = [identities[x]["source_index"] for x in ids]
    if len(set(source_indices)) != count or (benchmark == "standard" and set(source_indices) != set(range(count))):
        raise ValueError(f"{directory.name}/{benchmark}: duplicate/missing source indices.")
    for protocol in hardware.values():
        for observed in protocol.values():
            for field in observed:
                observed[field] = sorted(observed[field])
    runtime_sources = {
        _source_path(path): _read_json(path)
        for path in (directory / "runtime.json", directory / f"{benchmark}_runtime.json")
        if path.is_file()
    }
    return {
        "groups": {key: [group[x] for x in ids] for key, group in grouped.items()}, "identities": identities,
        "source": {"scores": _source_path(score_path), "metadata": _source_path(metadata_path),
                   "raw_score_fields": sorted({field for row in rows for field in row}),
                   "raw_fields_retained": True, "run_metadata": metadata, "hardware": hardware,
                   "runtime_sources": runtime_sources},
    }


def _standard_arrays(rows: list[dict[str, Any]]) -> np.ndarray:
    correct = np.asarray([row["mc1_accuracy"] for row in rows], dtype=np.float64)
    confidence = np.asarray([row["mc1_confidence"] for row in rows], dtype=np.float64)
    return np.column_stack((correct, [row["mc2"] for row in rows], confidence, confidence - correct,
                            [row["mc1_brier"] for row in rows], [row["mc1_nll"] for row in rows]))


def _standard_resampled_metrics(values: np.ndarray, indices: np.ndarray, *, ece_bins: int = ECE_BINS) -> np.ndarray:
    sampled = values[indices]
    result = np.empty((len(indices), len(STANDARD_METRICS)), dtype=np.float64)
    result[:, :6] = sampled.mean(axis=1)
    order = np.argsort(sampled[:, :, 2], axis=1, kind="stable")
    residuals = np.take_along_axis(sampled[:, :, 3], order, axis=1)
    n = indices.shape[1]
    bins = min(ece_bins, n)
    sizes = np.full(bins, n // bins, dtype=np.int64)
    sizes[:n % bins] += 1
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    result[:, 6] = np.abs(np.add.reduceat(residuals, starts, axis=1)).sum(axis=1) / n
    return result


def _joint_analysis(
    loaded: dict[str, dict[str, Any]], benchmark: str, *, iterations: int, seed: int,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> dict[str, Any]:
    """One set of question draws for every model, protocol, pair and metric within a benchmark."""
    names = STANDARD_METRICS if benchmark == "standard" else METRIC_NAMES
    arrays_fn = _standard_arrays if benchmark == "standard" else _metric_arrays
    resample = _standard_resampled_metrics if benchmark == "standard" else _resampled_metrics
    arrays = {pair: {key: arrays_fn(rows) for key, rows in data["groups"].items()} for pair, data in loaded.items()}
    n = len(next(iter(next(iter(arrays.values())).values())))
    points = {pair: {key: resample(values, np.arange(n)[None, :])[0] for key, values in groups.items()}
              for pair, groups in arrays.items()}
    samples = {(pair, protocol): np.empty((iterations, len(names)), dtype=np.float64)
               for pair in arrays for protocol in PROTOCOLS}
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        for pair, groups in arrays.items():
            for protocol in PROTOCOLS:
                samples[pair, protocol][start:stop] = (
                    resample(groups[protocol, "instruct"], indices) - resample(groups[protocol, "base"], indices)
                )
    result = {}
    for pair in arrays:
        result[pair] = {}
        for protocol in PROTOCOLS:
            bounds = np.quantile(samples[pair, protocol], [0.025, 0.975], axis=0, method="linear")
            delta = points[pair][protocol, "instruct"] - points[pair][protocol, "base"]
            metrics = {metric: {"point": float(delta[i]), "ci_low": float(bounds[0, i]), "ci_high": float(bounds[1, i])}
                       for i, metric in enumerate(names)}
            result[pair][protocol] = {
                "role": "primary" if protocol == SHARED_PLAIN_PROTOCOL else "sensitivity",
                "official_harness_prompt_setting": benchmark == "standard" and protocol == SHARED_PLAIN_PROTOCOL,
                "models": {model: {"n": n, **{metric: float(points[pair][protocol, model][i])
                                               for i, metric in enumerate(names)}} for model in MODELS},
                "deltas": {"n_pairs": n, "iterations": iterations, "confidence_level": 0.95,
                           "model_order": "instruct_minus_base", "metrics": metrics},
                "classification": _classify(metrics, benchmark),
            }
    return result


def _classify(metrics: dict[str, Any], benchmark: str) -> dict[str, Any]:
    signs = {metric: {"point": "positive" if value["point"] > 0 else "negative" if value["point"] < 0 else "zero",
                      "ci": "positive" if value["ci_low"] > 0 else "negative" if value["ci_high"] < 0 else "crosses_or_touches_zero"}
             for metric, value in metrics.items()}
    ece = "mc1_ece" if benchmark == "standard" else "ece"
    truth_metrics = ("mc1_accuracy", "mc2") if benchmark == "standard" else ("accuracy",)
    outcomes = {}
    for truth in truth_metrics:
        truth_sign, ece_sign = signs[truth]["ci"], signs[ece]["ci"]
        if "crosses_or_touches_zero" in (truth_sign, ece_sign):
            outcome = "uncertain"
        elif truth_sign == "positive":
            outcome = "truth_gain_ece_worse" if ece_sign == "positive" else "truth_and_calibration_improve"
        else:
            outcome = "truth_and_calibration_worsen" if ece_sign == "positive" else "truth_worse_calibration_improves"
        outcomes[truth] = outcome
    return {"signs": signs, "truth_calibration": outcomes,
            "rule": "Both marginal 95% intervals must strictly exclude zero; touching zero is uncertain. Counts are descriptive, not a joint or multiplicity-adjusted test."}


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = ["pair_id", "family", "parameters_billion", "benchmark", "question_count", "protocol", "role",
              "metric", "base", "instruct", "delta", "ci_low", "ci_high", "point_sign", "ci_sign"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in summary["pairs"]:
            for benchmark, data in pair["benchmarks"].items():
                for protocol, analysis in data["protocols"].items():
                    for metric, delta in analysis["deltas"]["metrics"].items():
                        writer.writerow({
                            **{key: pair[key] for key in ("pair_id", "family", "parameters_billion")},
                            "benchmark": benchmark, "question_count": data["question_count"], "protocol": protocol,
                            "role": analysis["role"], "metric": metric, "base": analysis["models"]["base"][metric],
                            "instruct": analysis["models"]["instruct"][metric], "delta": delta["point"],
                            "ci_low": delta["ci_low"], "ci_high": delta["ci_high"],
                            "point_sign": analysis["classification"]["signs"][metric]["point"],
                            "ci_sign": analysis["classification"]["signs"][metric]["ci"],
                        })


def _write_svg(summary: dict[str, Any], path: Path) -> None:
    pairs = summary["pairs"]
    width, height = 1240, 175 + 54 * len(pairs)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
              '<title id="title">Standard TruthfulQA MC1/MC2: instruct minus base</title>',
              '<desc id="desc">Three panels show MC1 accuracy, MC2 true-answer probability mass, and MC1 categorical ECE changes with 95% paired-question percentile intervals. Native prompts are sensitivity, not official harness scores.</desc>',
              '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#202020">',
              '<text x="20" y="26" font-size="17">TruthfulQA · instruct minus base · 95% paired-question intervals</text>',
              '<circle cx="27" cy="49" r="4" fill="#17649b"/><text x="39" y="53">Shared plain · primary (harness prompt)</text>',
              '<path d="M 390 44 l 5 5 -5 5 -5 -5 Z" fill="#b75b16"/><text x="402" y="53">Native prompt · sensitivity (not official harness)</text>']
    for panel, (metric, title) in enumerate((("mc1_accuracy", "ΔMC1 (percentage points)"), ("mc2", "ΔMC2 (percentage points)"), ("mc1_ece", "ΔMC1 ECE (percentage points)"))):
        left, right = 300 + 310 * panel, 580 + 310 * panel
        values = [100 * delta[field] for pair in pairs for protocol in PROTOCOLS
                  for delta in [pair["benchmarks"]["standard"]["protocols"][protocol]["deltas"]["metrics"][metric]]
                  for field in ("point", "ci_low", "ci_high")]
        lower, upper = min(0.0, min(values)), max(0.0, max(values))
        padding = max(1.0, (upper - lower) * 0.1)
        lower, upper = lower - padding, upper + padding

        def x(value: float) -> float:
            return left + (value - lower) * (right - left) / (upper - lower)

        pieces.append(f'<text x="{(left + right) / 2}" y="84" text-anchor="middle" font-size="13">{title}</text>')
        pieces.append(f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="96" y2="{height - 64}" stroke="#999" stroke-dasharray="4 3"/>')
        for tick in np.linspace(lower, upper, 5):
            pieces.append(f'<text x="{x(float(tick)):.2f}" y="{height - 43}" text-anchor="middle">{tick:.1f}</text>')
        for index, pair in enumerate(pairs):
            center = 117 + 54 * index
            if panel == 0:
                label = pair["models"]["base"]["model_id"].split("/")[-1]
                pieces.append(f'<text x="20" y="{center + 1}">{escape(label)}</text>')
                wave = "new" if pair["new_large_pair"] else "existing"
                pieces.append(f'<text x="20" y="{center + 18}" fill="#666">{escape(pair["family"])} · {pair["parameters_billion"]:g}B · {wave}</text>')
            for pindex, protocol in enumerate(PROTOCOLS):
                delta = pair["benchmarks"]["standard"]["protocols"][protocol]["deltas"]["metrics"][metric]
                y = center - 6 + 16 * pindex
                low, high, point = (x(100 * delta[field]) for field in ("ci_low", "ci_high", "point"))
                color = "#17649b" if pindex == 0 else "#b75b16"
                pieces.append(f'<path d="M {low:.2f} {y} H {high:.2f} M {low:.2f} {y - 4} V {y + 4} M {high:.2f} {y - 4} V {y + 4}" stroke="{color}" fill="none" stroke-width="1.7"/>')
                if pindex == 0:
                    pieces.append(f'<circle cx="{point:.2f}" cy="{y}" r="4" fill="{color}"/>')
                else:
                    pieces.append(f'<path d="M {point:.2f} {y - 5} l 5 5 -5 5 -5 -5 Z" fill="{color}"/>')
    pieces.extend([f'<text x="20" y="{height - 19}" fill="#555">Higher MC1/MC2 is better; lower MC1 ECE is better. Benchmarks are not pooled. Intervals are descriptive and not multiplicity-adjusted.</text>', '</g></svg>'])
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def analyze_scale(
    root: str | Path, out: str | Path, old_binary_root: str | Path = "results/cross_family",
    iterations: int = 10000, seed: int = 20260420, *,
    _expected_pairs: list[dict[str, Any]] | None = None,
    _expected_question_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate fixed matched pairs and analyze benchmarks separately; private overrides are unit fixtures only."""
    if type(iterations) is not int or iterations <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Iterations must be a positive integer and seed a nonnegative integer.")
    counts = QUESTION_COUNTS if _expected_question_counts is None else _expected_question_counts
    if set(counts) != set(QUESTION_COUNTS) or any(type(x) is not int or x <= 0 for x in counts.values()):
        raise ValueError("Expected counts require positive standard and binary counts.")
    root_path, old_path, out_path = Path(root), Path(old_binary_root), Path(out)
    manifest = _read_json(MANIFEST) if _expected_pairs is None else {"pairs": _expected_pairs}
    required, optional = manifest.get("pairs"), manifest.get("optional_pairs", [])
    if not isinstance(required, list) or not required or not isinstance(optional, list):
        raise ValueError("Manifest requires nonempty pairs and optional_pairs lists.")
    expected = {}
    for config in required + optional:
        if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
            raise ValueError("Invalid manifest pair config.")
        pair_id = config["experiment"].get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in expected:
            raise ValueError("Manifest pair IDs must be unique nonempty strings.")
        expected[pair_id] = _pair_config(config, pair_id)
    if not root_path.is_dir():
        raise ValueError("No day-scale score root found.")
    present = {directory.name for directory in root_path.iterdir() if directory.is_dir() and any(
        (directory / name).exists() for name in ("standard_scores.jsonl.gz", "standard_metadata.json", "binary_scores.jsonl.gz", "binary_metadata.json")
    )}
    mandatory = {config["experiment"]["pair_id"] for config in required}
    if not mandatory <= present or not present <= set(expected):
        raise ValueError(f"Day-scale pair set differs from manifest: missing={sorted(mandatory - present)}, unexpected={sorted(present - set(expected))}")
    loaded: dict[str, dict[str, Any]] = {benchmark: {} for benchmark in counts}
    pairs = []
    seen_models = set()
    for pair_id in sorted(present, key=lambda x: (expected[x]["family"], expected[x]["parameters_billion"], x)):
        pair = expected[pair_id]
        signature = tuple((pair["models"][model]["model_id"], pair["models"][model]["revision"]) for model in MODELS)
        if signature in seen_models:
            raise ValueError(f"{pair_id}: duplicate pinned model pair.")
        seen_models.add(signature)
        directory = root_path / pair_id
        loaded["standard"][pair_id] = _load_benchmark(directory, "standard", pair, counts["standard"])
        day_binary = any((directory / name).exists() for name in ("binary_scores.jsonl.gz", "binary_metadata.json"))
        old_binary = any((old_path / pair_id / name).exists() for name in ("scores.jsonl.gz", "run_metadata.json"))
        if day_binary and old_binary:
            raise ValueError(f"{pair_id}: ambiguous old and day binary sources.")
        loaded["binary"][pair_id] = _load_benchmark(directory if day_binary else old_path / pair_id, "binary", pair, counts["binary"], old_binary=not day_binary)
        pairs.append({**pair, "new_large_pair": not old_binary, "optional_pair": pair_id not in mandatory, "benchmarks": {}})
    benchmarks = {}
    analyses = {}
    for benchmark, data in loaded.items():
        reference = next(iter(data.values()))["identities"]
        for pair_id, pair_data in data.items():
            if pair_data["identities"] != reference:
                raise ValueError(f"{pair_id}/{benchmark}: question IDs/content differ across pairs.")
        question_ids = sorted(reference)
        benchmarks[benchmark] = {"question_count": counts[benchmark], "question_ids": question_ids,
                                 "question_ids_sha256": hashlib.sha256(json.dumps(question_ids, separators=(",", ":")).encode()).hexdigest()}
        analyses[benchmark] = _joint_analysis(data, benchmark, iterations=iterations, seed=seed)
    outcome_counts = {}
    for benchmark in counts:
        truth_metrics = ("mc1_accuracy", "mc2") if benchmark == "standard" else ("accuracy",)
        outcome_counts[benchmark] = {}
        for protocol in PROTOCOLS:
            outcome_counts[benchmark][protocol] = {}
            for truth in truth_metrics:
                outcomes = [analyses[benchmark][pair["pair_id"]][protocol]["classification"]["truth_calibration"][truth] for pair in pairs]
                outcome_counts[benchmark][protocol][truth] = {outcome: outcomes.count(outcome) for outcome in (
                    "truth_gain_ece_worse", "truth_and_calibration_improve", "truth_and_calibration_worsen",
                    "truth_worse_calibration_improves", "uncertain",
                )}
    for pair in pairs:
        for benchmark in counts:
            pair["benchmarks"][benchmark] = {
                "question_count": counts[benchmark], "source": loaded[benchmark][pair["pair_id"]]["source"],
                "protocols": analyses[benchmark][pair["pair_id"]],
            }
    summary = {
        "pair_count": len(pairs), "iterations": iterations, "seed": seed, "confidence_level": 0.95,
        "manifest_source": "configs/day_scale.json" if _expected_pairs is None else "private unit fixture override",
        "benchmarks": benchmarks, "pairs": pairs, "outcome_counts": outcome_counts,
        "protocol_roles": {SHARED_PLAIN_PROTOCOL: "primary", NATIVE_PROMPT_PROTOCOL: "sensitivity"},
        "bootstrap": {"unit": "question_id", "interval": "percentile", "quantile_method": "linear",
                      "rng": "numpy.default_rng/PCG64", "chunk_size": BOOTSTRAP_CHUNK_SIZE,
                      "shared_resamples_across_models_pairs_protocols_and_metrics": True,
                      "separate_question_resamples_per_benchmark": True},
        "ece": {"bins": ECE_BINS, "binning": "equal_frequency", "tie_order": "stable sampled order", "recomputed_per_resample": True},
        "standard_task": {"dataset_id": "truthfulqa/truthful_qa", "config": "multiple_choice", "split": "validation",
                          "dataset_revision": DATASET_REVISION, "harness_revision": HARNESS_REVISION,
                          "harness_sources": ["lm_eval/tasks/truthfulqa/truthfulqa_mc1.yaml", "lm_eval/tasks/truthfulqa/truthfulqa_mc2.yaml", "lm_eval/tasks/truthfulqa/utils.py"],
                          "scoring": "Raw summed continuation loglikelihoods; leading space plus full answer; no length normalization.",
                          "primary_prompt": "Exact six-example harness primer followed by Q: question and A:.",
                          "native_prompt": "Base uses identical plain prompt; instruct wraps the entire same prompt in user chat template plus assistant generation prompt. Sensitivity only, not official harness prompt setting."},
        "interpretation": {"delta_order": "instruct_minus_base", "benchmarks_pooled": False,
                           "mc1_accuracy": "Argmax over MC1 answer choices; best-answer target is index zero.",
                           "mc2": "Softmax probability mass on true-labeled MC2 answer choices; not binary accuracy.",
                           "mc1_brier": "Categorical multiclass Brier: sum over MC1 choices of (probability - one-hot best-answer target)^2; not the binary Brier definition.",
                           "mc1_nll": "Unclipped categorical negative log probability of best answer, computed as logsumexp(raw summed loglikelihoods) minus best-answer loglikelihood.",
                           "binary_brier": "Legacy binary definition: (1 - probability of correct label)^2, without a factor of two.",
                           "binary_nll": "Legacy binary definition: -log(max(probability of correct label, 1e-12)).",
                           "overconfidence_gap": "Mean confidence minus accuracy within the relevant benchmark; positive is overconfidence.",
                           "outcome_counts": "Counts use strictly positive/negative marginal CI bounds, separately for MC1 gain versus MC1 ECE and MC2 gain versus MC1 ECE; crossing or touching zero is uncertain.",
                           "standard_calibration": "Confidence, overconfidence, Brier, NLL and ECE concern MC1 categorical best-answer probabilities, not MC2 truth-mass classification.",
                           "scope": "Descriptive matched comparisons on fixed questions/models, not causal or family-population effects. Marginal 95% intervals are not multiplicity-adjusted. Outcome counts do not constitute a joint significance test."},
        "validation": {"fake_records_accepted": False, "unique_score_keys": True, "pinned_model_revisions": True,
                       "complete_model_protocol_groups": True, "same_question_ids_and_content_within_each_benchmark": True,
                       "derived_metrics_checked_against_raw_loglikelihoods": True,
                       "fixture_pair_override": _expected_pairs is not None, "fixture_question_count_override": _expected_question_counts is not None},
    }
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(summary, out_path / "summary.json")
    _write_csv(summary, out_path / "summary.csv")
    _write_svg(summary, out_path / "day_scale.svg")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze fixed day-scale base/instruct pairs on separate TruthfulQA benchmarks.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--old-binary-root", type=Path, default=Path("results/cross_family"))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args(argv)
    analyze_scale(args.root, args.out, args.old_binary_root, args.iterations, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
