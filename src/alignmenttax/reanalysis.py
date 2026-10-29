"""Regenerate candidate controls, diagnostics and held-out calibration from saved scores."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .calibration_stage import _stable_split, _write_csv
from .cross_family import BOOTSTRAP_CHUNK_SIZE, MODELS, PROTOCOLS
from .io_utils import write_json
from .reanalysis_report import write_figures
from .scale_analysis import QUESTION_COUNTS, _load_benchmark, analyze_scale
from .uncertainty import (
    TEMPERATURE_BOUNDS, baseline_logits, bootstrap_metrics, categorical_values,
    diagnostics, fit_categorical_temperature, label_diagnostics, probability_rows,
)

TWO_CHOICE_TASKS = ("binary", "matched_labels", "matched_strings")


def answer_key(text: str) -> str:
    # Only a matching key: never edit a scored continuation or its likelihood.
    return text.strip().rstrip(".")


def match_candidates(binary: dict[str, Any], standard: dict[str, Any]) -> dict[str, Any]:
    by_question: dict[str, list[str]] = {}
    for question_id, row in standard.items():
        by_question.setdefault(row["question"], []).append(question_id)
    matches, exclusions = [], []
    for binary_id, row in sorted(binary.items()):
        entry = {"binary_id": binary_id, "question": row["question"], "binary_choices": row["choices"],
                 "correct_label": row["correct_label"], "binary_source_index": row["source_index"]}
        candidates = by_question.get(row["question"], [])
        if len(candidates) != 1:
            exclusions.append({**entry, "reason": "question_missing" if not candidates else "question_ambiguous",
                               "standard_ids": candidates})
            continue
        standard_id = candidates[0]
        other = standard[standard_id]
        indices = {label: [i for i, text in enumerate(other["mc1_choices"]) if answer_key(text) == answer_key(answer)]
                   for label, answer in row["choices"].items()}
        entry.update(standard_id=standard_id, standard_source_index=other["source_index"],
                     standard_choices=other["mc1_choices"], candidate_indices=indices)
        if any(len(value) != 1 for value in indices.values()):
            reason = "answer_missing" if any(not value for value in indices.values()) else "answer_ambiguous"
            exclusions.append({**entry, "reason": reason})
            continue
        selected = {label: value[0] for label, value in indices.items()}
        if selected["A"] == selected["B"] or any(other["mc1_labels"][index] != int(label == row["correct_label"])
                                                for label, index in selected.items()):
            exclusions.append({**entry, "reason": "answer_label_disagreement"})
            continue
        entry["candidate_indices"] = selected
        entry["normalization_required"] = any(row["choices"][label] != other["mc1_choices"][index] for label, index in selected.items())
        matches.append(entry)
    matched_standard = {row["standard_id"] for row in matches}
    if len(matched_standard) != len(matches):
        raise ValueError("Matching must be one-to-one; duplicate binary questions found.")
    if not matches:
        raise ValueError("No unambiguous question-and-answer matches.")
    return {"rule": "Exact question text; answer keys strip surrounding whitespace and terminal periods only; unique candidates with agreeing truth labels required. Scored strings and loglikelihoods are unchanged.",
            "binary_n": len(binary), "standard_n": len(standard), "matched_n": len(matches),
            "matches": matches, "binary_exclusions": exclusions,
            "standard_exclusions": [{"standard_id": key, "question": row["question"],
                                      "reason": "not_in_candidate_matched_pool"} for key, row in sorted(standard.items()) if key not in matched_standard]}


def score_arrays(rows: list[dict[str, Any]], task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "binary":
        return (np.asarray([[row["raw_logprob_A"], row["raw_logprob_B"]] for row in rows]),
                np.asarray([int(row["correct_label"] == "B") for row in rows]))
    width = max(len(row["mc1_loglikelihoods"]) for row in rows)
    logits = np.full((len(rows), width), -np.inf)
    for i, row in enumerate(rows):
        logits[i, :len(row["mc1_loglikelihoods"])] = row["mc1_loglikelihoods"]
    return logits, np.asarray([row["mc1_labels"].index(1) for row in rows])


def matched_arrays(binary_rows: list[dict[str, Any]], standard_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    binary = {row["question_id"]: row for row in binary_rows}
    standard = {row["question_id"]: row for row in standard_rows}
    selected_binary = [binary[row["binary_id"]] for row in manifest["matches"]]
    selected_standard = [standard[row["standard_id"]] for row in manifest["matches"]]
    labels = score_arrays(selected_binary, "binary")
    strings = np.asarray([[standard[row["standard_id"]]["mc1_loglikelihoods"][row["candidate_indices"][label]]
                           for label in ("A", "B")] for row in manifest["matches"]])
    return {"matched_labels": labels, "matched_strings": (strings, labels[1]),
            "matched_mc1_all_candidates": score_arrays(selected_standard, "mc1")}


def split_manifest(questions: list[str], *, seed: int) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows = [{"question": question, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
             "split": _stable_split(question, seed=seed, calibration_fraction=0.5)} for question in questions]
    calibration = np.asarray([row["split"] == "calibration" for row in rows])
    if not calibration.any() or calibration.all():
        raise ValueError("Shared question-hash split requires nonempty calibration and test sets.")
    return calibration, rows


def mc2_mass(rows: list[dict[str, Any]], *, temperature: float = 1.0,
             baseline: str | None = None, priors: dict[str, Any] | None = None, seed: int = 0) -> np.ndarray:
    masses = []
    for row in rows:
        labels = np.asarray(row["mc2_labels"])
        if baseline is None:
            probabilities = probability_rows(np.asarray([row["mc2_loglikelihoods"]]), temperature)[0][0]
        else:
            count = len(labels)
            order = sorted(range(count), key=lambda j: hashlib.sha256(f"{seed}|{row['question']}|position|{j}".encode()).digest())
            labels = labels[order]
            probabilities = (np.asarray(priors[str(count)]["probabilities"]) if baseline == "constant" and str(count) in (priors or {})
                             else np.full(count, 1 / count))
        masses.append(float(probabilities @ labels))
    return np.asarray(masses)


def bootstrap_mass(values: dict[str, np.ndarray], *, iterations: int, seed: int,
                   contrasts: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Mean truth mass with the same question draws as bootstrap_metrics for an equal-length test set."""
    n = len(next(iter(values.values())))
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(iterations) for key in values}
    for start in range(0, iterations, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        for key, value in values.items():
            draws[key][start:stop] = value[indices].mean(axis=1)
    points = {key: float(value.mean()) for key, value in values.items()}
    for key, terms in contrasts.items():
        points[key] = sum(points[term] * weight for term, weight in terms.items())
        draws[key] = sum(draws[term] * weight for term, weight in terms.items())
    return {key: {"point": point, "ci_low": float(np.quantile(draws[key], 0.025, method="linear")),
                  "ci_high": float(np.quantile(draws[key], 0.975, method="linear"))} for key, point in points.items()}


def held_out_analysis(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]], questions: list[str], *, task: str,
    iterations: int, seed: int, standard_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on the calibration half only; evaluate every condition on the same test questions."""
    calibration, _ = split_manifest(questions, seed=seed)
    test = ~calibration
    logits, targets = arrays["base"]
    baselines, baseline_targets, priors = baseline_logits(logits, targets, questions, calibration,
                                                        seed=seed, permute_positions=task == "mc1")
    values = {method: categorical_values(scores[test], baseline_targets[test]) for method, scores in baselines.items()}
    test_rows = None if standard_rows is None else {
        model: [row for row, selected in zip(rows, test) if selected] for model, rows in standard_rows.items()}
    fits, masses = {}, {}
    for model, (scores, labels) in arrays.items():
        fits[model] = fit_categorical_temperature(scores[calibration], labels[calibration])
        for method, temperature in (("raw", 1.0), ("temperature", fits[model]["temperature"])):
            values[f"{model}/{method}"] = categorical_values(scores[test], labels[test], temperature)
            if test_rows is not None:
                masses[f"{model}/{method}"] = mc2_mass(test_rows[model], temperature=temperature)
    contrasts = {f"{model}/temperature_minus_raw": {f"{model}/temperature": 1, f"{model}/raw": -1} for model in MODELS}
    for model in MODELS:
        for baseline in baselines:
            contrasts[f"{model}/temperature_minus_{baseline}"] = {f"{model}/temperature": 1, baseline: -1}
    for method in ("raw", "temperature"):
        contrasts[f"instruct_minus_base/{method}"] = {f"instruct/{method}": 1, f"base/{method}": -1}
    details = {key: diagnostics(value) for key, value in values.items()}
    result = {"calibration_n": int(calibration.sum()), "test_n": int(test.sum()), "fits": fits,
              "constant_priors": priors, "estimates": bootstrap_metrics(values, iterations=iterations, seed=seed, contrasts=contrasts),
              "risk_coverage": {key: {field: value["risk_coverage"][field] for field in ("aurc", "random_ranking_risk")}
                                for key, value in details.items()}}
    if test_rows is not None:
        # Candidate strings and truth labels are identical across models; only likelihoods differ.
        for baseline in baselines:
            masses[baseline] = mc2_mass(test_rows["base"], baseline=baseline, priors=priors, seed=seed)
        result["mc2_truth_mass"] = bootstrap_mass(masses, iterations=iterations, seed=seed, contrasts=contrasts)
    return result, details


def analyze_reanalysis(
    root: Path, old_binary_root: Path, out: Path, *, full_out: Path,
    iterations: int = 10000, seed: int = 20260420,
) -> dict[str, Any]:
    # Reuse the strict loaders and regenerate the published full-task analysis
    # beside the new estimands. Nothing here changes saved likelihoods.
    out = Path(out)
    full = analyze_scale(root, full_out, old_binary_root, iterations=iterations, seed=seed)
    summary: dict[str, Any] = {
        "iterations": iterations, "seed": seed, "confidence_level": 0.95,
        "metric_definitions": {"brier": "Categorical sum over offered choices, including two-choice tasks: twice the legacy binary Brier (1 - p_correct)^2.",
                               "nll": "Unclipped logsumexp minus target loglikelihood; nats. The legacy binary NLL clips p_correct at 1e-12.",
                               "ece": "10 equal-frequency bins; stable sample-order ties, rebuilt in every bootstrap draw.",
                               "risk_coverage": "Most-confident-first error rate, with expected risk for random ordering within exact confidence ties. AURC averages the risk of every prefix k = 1..n; random_ranking_risk is the same model's error rate, the AURC of a random ordering. Curves and AURC are descriptive, without intervals."},
        "tasks": {"binary": "790-question A/B-label derivative: normalized ' A'/' B' likelihoods.",
                  "mc1": "817-question standard MC1: all candidate answer strings.",
                  "matched_labels": "Matched questions, binary A/B-label likelihoods.",
                  "matched_strings": "Matched questions, standard-prompt likelihoods of the same two answer strings, renormalized over those two.",
                  "matched_mc1_all_candidates": "Matched questions, standard MC1 over all of its candidates."},
        "temperature_scaling": {"bounds": list(TEMPERATURE_BOUNDS), "objective": "Calibration-only categorical NLL; convex inverse-temperature bisection.",
                                "split": "SHA256(seed|exact question text), first eight bytes / 2^64 < 0.5; shared across models, protocols and tasks.",
                                "bootstrap": "Paired test-question percentile intervals, conditional on fixed split and fitted temperatures/priors. No refitting; no training-seed, fitting or numerical uncertainty.",
                                "constant": "Calibration-only add-one-smoothed position prior stratified by candidate count; unseen counts use uniform. MC1 positions are hash-permuted first to prevent a truth-first index-zero oracle. Binary and matched tasks keep A/B coordinates.",
                                "uniform": "Equal probability over offered candidates; deterministic argmax. MC1 baselines use the same blinded coordinates as constant.",
                                "mc2": "Held-out truth mass after transferring the MC1-fitted temperature and MC1 constant position priors (uniform for unseen counts) to MC2. MC2 has multiple true answers: no one-hot MC2 target, ECE or calibration-repair claim is defined."},
        "candidate_bootstrap": "Same matched-question draws across representations, models, protocols and pairs; difference-of-deltas is paired, not subtraction of interval endpoints. Pointwise intervals, no multiplicity correction.",
        "full_task_analysis": "Regenerated by this command (--full-out) with the same seed and iterations; its estimands are unchanged.",
        "pairs": {},
    }
    details: dict[str, Any] = {"note": "Reliability bins, ECE binning sensitivity and risk-coverage curves behind summary.json.", "pairs": {}}
    matches = None
    reference_identities = None
    csv_candidates, csv_calibration, csv_temperatures, csv_labels, csv_sensitivity = [], [], [], [], []
    for pair in full["pairs"]:
        pair_id = pair["pair_id"]
        standard = _load_benchmark(root / pair_id, "standard", pair, QUESTION_COUNTS["standard"])
        binary = _load_benchmark(root / pair_id if pair["new_large_pair"] else old_binary_root / pair_id,
                                 "binary", pair, QUESTION_COUNTS["binary"], old_binary=not pair["new_large_pair"])
        identities = {"binary": binary["identities"], "standard": standard["identities"]}
        if matches is None:
            matches = match_candidates(identities["binary"], identities["standard"])
            reference_identities = identities
            write_json(matches, out / "matching.json")
            questions = {"binary": [row["question"] for _, row in sorted(identities["binary"].items())],
                         "mc1": [row["question"] for _, row in sorted(identities["standard"].items())],
                         "matched": [row["question"] for row in matches["matches"]]}
            write_json({task: split_manifest(items, seed=seed)[1] for task, items in questions.items()}, out / "splits.json")
            summary["matching"] = {key: matches[key] for key in ("rule", "binary_n", "standard_n", "matched_n")}
            summary["matching"]["binary_exclusion_counts"] = {reason: sum(row["reason"] == reason for row in matches["binary_exclusions"])
                                                                for reason in sorted({row["reason"] for row in matches["binary_exclusions"]})}
            summary["matching"]["normalization_required_n"] = sum(row["normalization_required"] for row in matches["matches"])
        elif reference_identities != identities:
            raise ValueError("Question content must agree across every model pair.")
        pair_result: dict[str, Any] = {
            "label": pair["models"]["base"]["model_id"].split("/")[-1], "family": pair["family"],
            "parameters_billion": pair["parameters_billion"], "candidate_control": {}, "label_diagnostics": {}, "held_out": {}}
        pair_details: dict[str, Any] = {"full_pool": {}, "held_out": {}}
        for protocol in PROTOCOLS:
            task_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {task: {} for task in summary["tasks"]}
            for model in MODELS:
                binary_rows, standard_rows = binary["groups"][protocol, model], standard["groups"][protocol, model]
                task_arrays["binary"][model] = score_arrays(binary_rows, "binary")
                task_arrays["mc1"][model] = score_arrays(standard_rows, "mc1")
                for task, value in matched_arrays(binary_rows, standard_rows, matches).items():
                    task_arrays[task][model] = value
            candidate_values = {f"{task}/{model}": categorical_values(*arrays)
                                for task, models in task_arrays.items() if task.startswith("matched_") for model, arrays in models.items()}
            contrasts = {f"{task}/instruct_minus_base": {f"{task}/instruct": 1, f"{task}/base": -1}
                         for task in task_arrays if task.startswith("matched_")}
            contrasts["labels_minus_strings_delta"] = {"matched_labels/instruct": 1, "matched_labels/base": -1,
                                                        "matched_strings/instruct": -1, "matched_strings/base": 1}
            estimates = bootstrap_metrics(candidate_values, iterations=iterations, seed=seed, contrasts=contrasts)
            pair_result["candidate_control"][protocol] = {"n": matches["matched_n"], "estimates": estimates}
            for condition, metrics in estimates.items():
                for metric, estimate in metrics.items():
                    csv_candidates.append({"pair_id": pair_id, "protocol": protocol, "condition": condition, "metric": metric, **estimate})
            for task, arrays in task_arrays.items():
                raw = {}
                for model, (logits, targets) in arrays.items():
                    raw[model] = candidate_values[f"{task}/{model}"] if task.startswith("matched_") else categorical_values(logits, targets)
                    raw[model] = diagnostics(raw[model])
                    if task in TWO_CHOICE_TASKS:
                        labels = label_diagnostics(logits, targets)
                        pair_result["label_diagnostics"].setdefault(task, {}).setdefault(protocol, {})[model] = labels
                        for label, metrics in labels.items():
                            csv_labels.append({"pair_id": pair_id, "protocol": protocol, "task": task, "model": model, "label": label, **metrics})
                    for strategy, bins in raw[model]["ece_sensitivity"].items():
                        for count, ece in bins.items():
                            csv_sensitivity.append({"pair_id": pair_id, "protocol": protocol, "task": task, "model": model,
                                                    "strategy": strategy, "bins": count, "ece": ece})
                pair_details["full_pool"].setdefault(task, {})[protocol] = raw
                if task == "matched_mc1_all_candidates":
                    continue
                calibration, held_out_details = held_out_analysis(
                    arrays, questions[task] if task in questions else questions["matched"], task=task, iterations=iterations, seed=seed,
                    standard_rows={model: standard["groups"][protocol, model] for model in MODELS} if task == "mc1" else None)
                pair_result["held_out"].setdefault(task, {})[protocol] = calibration
                pair_details["held_out"].setdefault(task, {})[protocol] = held_out_details
                for model, fit in calibration["fits"].items():
                    csv_temperatures.append({"pair_id": pair_id, "protocol": protocol, "task": task, "model": model,
                                             "calibration_n": calibration["calibration_n"], "test_n": calibration["test_n"],
                                             "temperature": fit["temperature"], "boundary": fit["boundary"] or "",
                                             "calibration_nll": fit["calibration_nll"]})
                for condition, metrics in calibration["estimates"].items():
                    for metric, estimate in metrics.items():
                        csv_calibration.append({"pair_id": pair_id, "protocol": protocol, "task": task, "condition": condition, "metric": metric, **estimate})
                for condition, estimate in calibration.get("mc2_truth_mass", {}).items():
                    csv_calibration.append({"pair_id": pair_id, "protocol": protocol, "task": "mc2", "condition": condition, "metric": "truth_mass", **estimate})
        summary["pairs"][pair_id] = pair_result
        details["pairs"][pair_id] = pair_details
    write_json(summary, out / "summary.json")
    write_json(details, out / "diagnostics.json")
    for filename, rows in (("candidate_control.csv", csv_candidates), ("held_out.csv", csv_calibration),
                           ("temperatures.csv", csv_temperatures), ("label_diagnostics.csv", csv_labels),
                           ("ece_sensitivity.csv", csv_sensitivity)):
        _write_csv(out / filename, rows, list(rows[0]))
    write_figures(summary, details, out)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/day_scale"))
    parser.add_argument("--old-binary-root", type=Path, default=Path("results/cross_family"))
    parser.add_argument("--out", type=Path, default=Path("results/reanalysis"))
    parser.add_argument("--full-out", type=Path, default=Path("results/day_scale/analysis"),
                        help="Where to regenerate the published full-task analysis (default: its committed location).")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args(argv)
    summary = analyze_reanalysis(args.root, args.old_binary_root, args.out, full_out=args.full_out,
                                 iterations=args.iterations, seed=args.seed)
    print(f"Matched {summary['matching']['matched_n']} questions; analyzed {len(summary['pairs'])} pairs, both protocols, four held-out tasks.")
    print(f"Wrote candidate controls, diagnostics and held-out calibration to {args.out}")
    print(f"Regenerated the full-task analysis in {args.full_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
