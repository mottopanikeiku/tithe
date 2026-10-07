from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_jsonl, scores_path
from .metrics import METRIC_NAMES, calibration_bins, metric_summary, two_way_softmax


def _stable_split(question_id: str, *, seed: int, calibration_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}|{question_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "calibration" if value < calibration_fraction else "test"


def _scaled_row(row: dict[str, Any], *, temperature: float) -> dict[str, Any]:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    p_a, p_b = two_way_softmax(
        float(row["raw_logprob_A"]) / temperature,
        float(row["raw_logprob_B"]) / temperature,
    )
    predicted_label = "A" if p_a >= p_b else "B"
    correct_label = str(row["correct_label"])
    p_correct = p_a if correct_label == "A" else p_b
    output = dict(row)
    output.update(
        {
            "prob_A": p_a,
            "prob_B": p_b,
            "normalized_logprob_A": math.log(max(p_a, 1e-12)),
            "normalized_logprob_B": math.log(max(p_b, 1e-12)),
            "predicted_label": predicted_label,
            "correct": predicted_label == correct_label,
            "confidence": max(p_a, p_b),
            "p_correct": p_correct,
            "calibration_temperature": temperature,
        }
    )
    return output


def _nll_for_temperature(rows: list[dict[str, Any]], temperature: float) -> float:
    scaled = [_scaled_row(row, temperature=temperature) for row in rows]
    return float(metric_summary(scaled)["nll"])


def fit_temperature(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("Cannot fit temperature with no rows.")
    low = math.log(0.05)
    high = math.log(100.0)
    best_temperature = 1.0
    best_nll = float("inf")
    for _round in range(4):
        steps = 100
        candidates = [
            math.exp(low + (high - low) * index / (steps - 1))
            for index in range(steps)
        ]
        for temperature in candidates:
            nll = _nll_for_temperature(rows, temperature)
            if nll < best_nll:
                best_nll = nll
                best_temperature = temperature
        center = math.log(best_temperature)
        radius = (high - low) / 8.0
        low = max(math.log(0.01), center - radius)
        high = min(math.log(200.0), center + radius)
    return best_temperature


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_row(
    *,
    protocol: str,
    model_key: str,
    split: str,
    stage: str,
    temperature: float,
    rows: list[dict[str, Any]],
    ece_bins: int,
) -> dict[str, Any]:
    summary = metric_summary(rows, ece_bins=ece_bins)
    return {
        "prompt_protocol": protocol,
        "model_key": model_key,
        "split": split,
        "stage": stage,
        "temperature": temperature,
        **summary,
    }


def _group_scores(
    scores: list[dict[str, Any]],
    *,
    calibration_fraction: float,
    seed: int,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in scores:
        split = _stable_split(
            str(row["question_id"]),
            seed=seed,
            calibration_fraction=calibration_fraction,
        )
        key = (str(row["prompt_protocol"]), str(row["model_key"]), split)
        grouped.setdefault(key, []).append(row)
    return grouped


def _plot_stage2_bars(summary_rows: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in summary_rows
        if row["prompt_protocol"] == "shared_plain_ab_label"
        and row["split"] == "test"
        and row["model_key"] == "instruct"
    ]
    metrics = ["mean_confidence", "overconfidence_gap", "brier", "nll", "ece"]
    labels = ["Confidence", "Overconf.", "Brier", "NLL", "ECE"]
    raw = {row["stage"]: row for row in selected}["raw"]
    calibrated = {row["stage"]: row for row in selected}["calibrated"]

    x_positions = list(range(len(metrics)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(
        [x - width / 2 for x in x_positions],
        [float(raw[metric]) for metric in metrics],
        width,
        label="Raw instruct",
        color="#c45a3a",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        [float(calibrated[metric]) for metric in metrics],
        width,
        label="Temperature scaled",
        color="#1d7a55",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title("Raw and temperature-scaled instruct metrics on held-out test split")
    ax.legend()
    path = output_dir / "stage2_instruct_calibration_repair.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_stage2_reliability(
    *,
    raw_rows: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
    output_dir: Path,
    ece_bins: int,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_bins = calibration_bins(raw_rows, bins=ece_bins)
    calibrated_bins = calibration_bins(calibrated_rows, bins=ece_bins)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1, linestyle="--")
    ax.plot(
        [float(row["mean_confidence"]) for row in raw_bins],
        [float(row["accuracy"]) for row in raw_bins],
        marker="o",
        label="Raw instruct",
        color="#c45a3a",
    )
    ax.plot(
        [float(row["mean_confidence"]) for row in calibrated_bins],
        [float(row["accuracy"]) for row in calibrated_bins],
        marker="o",
        label="Temperature scaled",
        color="#1d7a55",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Stage 2 reliability on held-out test split")
    ax.legend()
    path = output_dir / "stage2_reliability_instruct_shared_prompt.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def run_calibration_stage(
    *,
    run_dir: str | Path,
    report_dir: str | Path,
    seed: int = 20260420,
    calibration_fraction: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, Any]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("Calibration fraction must be between 0 and 1.")
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = read_jsonl(scores_path(run_dir))
    grouped = _group_scores(scores, calibration_fraction=calibration_fraction, seed=seed)

    temperatures: dict[tuple[str, str], float] = {}
    summary_rows: list[dict[str, Any]] = []
    for protocol in sorted({str(row["prompt_protocol"]) for row in scores}):
        for model_key in sorted({str(row["model_key"]) for row in scores}):
            calibration_rows = grouped[(protocol, model_key, "calibration")]
            test_rows = grouped[(protocol, model_key, "test")]
            temperature = fit_temperature(calibration_rows)
            temperatures[(protocol, model_key)] = temperature

            for split, rows in (("calibration", calibration_rows), ("test", test_rows)):
                calibrated_rows = [_scaled_row(row, temperature=temperature) for row in rows]
                summary_rows.append(
                    _summary_row(
                        protocol=protocol,
                        model_key=model_key,
                        split=split,
                        stage="raw",
                        temperature=1.0,
                        rows=rows,
                        ece_bins=ece_bins,
                    )
                )
                summary_rows.append(
                    _summary_row(
                        protocol=protocol,
                        model_key=model_key,
                        split=split,
                        stage="calibrated",
                        temperature=temperature,
                        rows=calibrated_rows,
                        ece_bins=ece_bins,
                    )
                )

    fieldnames = [
        "prompt_protocol",
        "model_key",
        "split",
        "stage",
        "temperature",
        "n",
        *METRIC_NAMES,
    ]
    _write_csv(output_dir / "stage2_temperature_calibration_summary.csv", summary_rows, fieldnames)

    shared_test_raw = grouped[("shared_plain_ab_label", "instruct", "test")]
    shared_temperature = temperatures[("shared_plain_ab_label", "instruct")]
    shared_test_calibrated = [
        _scaled_row(row, temperature=shared_temperature) for row in shared_test_raw
    ]
    plots = [
        _plot_stage2_bars(summary_rows, output_dir),
        _plot_stage2_reliability(
            raw_rows=shared_test_raw,
            calibrated_rows=shared_test_calibrated,
            output_dir=output_dir,
            ece_bins=ece_bins,
        ),
    ]

    manifest = {
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "ece_bins": ece_bins,
        "summary": str(output_dir / "stage2_temperature_calibration_summary.csv"),
        "plots": plots,
        "temperatures": {
            f"{protocol}/{model_key}": temperature
            for (protocol, model_key), temperature in temperatures.items()
        },
    }
    with (output_dir / "stage2_calibration_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> int:
    run_calibration_stage(
        run_dir="runs/qwen2_5_1_5b",
        report_dir="reports/qwen2_5_1_5b",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

