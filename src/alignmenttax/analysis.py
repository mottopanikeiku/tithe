from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_jsonl, scores_path, write_json
from .metrics import (
    METRIC_NAMES,
    calibration_bins,
    metric_summary,
    paired_bootstrap_deltas,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_run_config(run: str | Path) -> dict[str, Any]:
    run_path = Path(run)
    metadata_path = run_path / "run_metadata.json" if run_path.is_dir() else run_path.parent / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        import json

        return json.loads(metadata_path.read_text(encoding="utf-8")).get("config", {})
    except Exception:
        return {}


def _group_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["prompt_protocol"]), str(row["model_key"]))].append(row)
    return dict(grouped)


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_rows(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    ece_bins: int,
    binning: str,
) -> list[dict[str, Any]]:
    protocols = sorted({protocol for protocol, _model in grouped})
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        groups = {model_key: grouped.get((protocol, model_key), []) for model_key in ("base", "instruct")}
        if groups["base"] and groups["instruct"]:
            # Match paired_bootstrap_deltas: compare both models on the same questions.
            shared = {row["question_id"] for row in groups["base"]} & {row["question_id"] for row in groups["instruct"]}
            groups = {key: [row for row in group if row["question_id"] in shared] for key, group in groups.items()}
        summaries: dict[str, dict[str, float | int]] = {}
        for model_key, group in groups.items():
            if group:
                summaries[model_key] = metric_summary(group, ece_bins=ece_bins, binning=binning)
        for metric in ("n", *METRIC_NAMES):
            base_value = summaries.get("base", {}).get(metric)
            instruct_value = summaries.get("instruct", {}).get(metric)
            delta = None
            if metric != "n" and base_value is not None and instruct_value is not None:
                delta = float(instruct_value) - float(base_value)
            rows.append(
                {
                    "prompt_protocol": protocol,
                    "metric": metric,
                    "base": base_value,
                    "instruct": instruct_value,
                    "delta_instruct_minus_base": delta,
                }
            )
    return rows


def _bootstrap_payload(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
    ece_bins: int,
    binning: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iterations": iterations,
        "confidence_level": confidence_level,
        "seed": seed,
        "binning": binning,
        "ece_bins": ece_bins,
        "protocols": {},
    }
    protocols = sorted({protocol for protocol, _model in grouped})
    for protocol_index, protocol in enumerate(protocols):
        base_rows = grouped.get((protocol, "base"), [])
        instruct_rows = grouped.get((protocol, "instruct"), [])
        if not base_rows or not instruct_rows:
            continue
        payload["protocols"][protocol] = paired_bootstrap_deltas(
            base_rows=base_rows,
            instruct_rows=instruct_rows,
            iterations=iterations,
            confidence_level=confidence_level,
            seed=seed + protocol_index,
            ece_bins=ece_bins,
            binning=binning,
        )
    return payload


def _calibration_rows(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    ece_bins: int,
    binning: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (protocol, model_key), group in sorted(grouped.items()):
        for bin_row in calibration_bins(group, bins=ece_bins, binning=binning):
            rows.append(
                {
                    "prompt_protocol": protocol,
                    "model_key": model_key,
                    **bin_row,
                }
            )
    return rows


def _category_rows(
    rows: list[dict[str, Any]],
    *,
    ece_bins: int,
    binning: str,
) -> list[dict[str, Any]]:
    by_protocol_category_model: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["prompt_protocol"]),
            str(row.get("category", "")),
            str(row["model_key"]),
        )
        by_protocol_category_model[key].append(row)

    categories = sorted({category for _protocol, category, _model in by_protocol_category_model})
    protocols = sorted({protocol for protocol, _category, _model in by_protocol_category_model})
    output: list[dict[str, Any]] = []
    for protocol in protocols:
        for category in categories:
            base_group = by_protocol_category_model.get((protocol, category, "base"), [])
            instruct_group = by_protocol_category_model.get((protocol, category, "instruct"), [])
            if not base_group and not instruct_group:
                continue
            base_summary = (
                metric_summary(base_group, ece_bins=ece_bins, binning=binning)
                if base_group
                else {}
            )
            instruct_summary = (
                metric_summary(instruct_group, ece_bins=ece_bins, binning=binning)
                if instruct_group
                else {}
            )
            row: dict[str, Any] = {
                "prompt_protocol": protocol,
                "category": category,
                "n_base": base_summary.get("n"),
                "n_instruct": instruct_summary.get("n"),
                "exploratory": True,
            }
            for metric in METRIC_NAMES:
                base_value = base_summary.get(metric)
                instruct_value = instruct_summary.get(metric)
                row[f"base_{metric}"] = base_value
                row[f"instruct_{metric}"] = instruct_value
                row[f"delta_{metric}"] = (
                    float(instruct_value) - float(base_value)
                    if base_value is not None and instruct_value is not None
                    else None
                )
            output.append(row)
    return output


def _plot_outputs(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    out_dir: Path,
    ece_bins: int,
    binning: str,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []
    for (protocol, model_key), group in sorted(grouped.items()):
        bins = calibration_bins(group, bins=ece_bins, binning=binning)
        if bins:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1, linestyle="--")
            ax.plot(
                [float(row["mean_confidence"]) for row in bins],
                [float(row["accuracy"]) for row in bins],
                marker="o",
            )
            ax.set_title(f"Reliability: {model_key} / {protocol}")
            ax.set_xlabel("Mean confidence")
            ax.set_ylabel("Accuracy")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            path = out_dir / f"reliability_{_safe_name(protocol)}_{_safe_name(model_key)}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(str(path))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([float(row["confidence"]) for row in group], bins=20, range=(0.0, 1.0))
        ax.set_title(f"Confidence: {model_key} / {protocol}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        path = out_dir / f"confidence_{_safe_name(protocol)}_{_safe_name(model_key)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def analyze_run(
    *,
    run: str | Path,
    out: str | Path,
    bootstrap_iterations: int | None = None,
    no_plots: bool = False,
) -> dict[str, Any]:
    scores = read_jsonl(scores_path(run))
    if not scores:
        raise ValueError("No scores found to analyze.")
    config = _load_run_config(run)
    analysis_config = config.get("analysis", {})
    iterations = int(bootstrap_iterations or analysis_config.get("bootstrap_iterations", 10000))
    confidence_level = float(analysis_config.get("confidence_level", 0.95))
    ece_bins = int(analysis_config.get("ece_bins", 10))
    binning = str(analysis_config.get("binning", "equal_frequency"))
    seed = int(analysis_config.get("bootstrap_seed", 20260420))

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_scores(scores)

    summary = _summary_rows(grouped, ece_bins=ece_bins, binning=binning)
    _write_csv(
        output_dir / "summary.csv",
        summary,
        ["prompt_protocol", "metric", "base", "instruct", "delta_instruct_minus_base"],
    )

    bootstrap = _bootstrap_payload(
        grouped,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
        ece_bins=ece_bins,
        binning=binning,
    )
    write_json(bootstrap, output_dir / "paired_bootstrap.json")

    calibration = _calibration_rows(grouped, ece_bins=ece_bins, binning=binning)
    _write_csv(
        output_dir / "calibration_tables.csv",
        calibration,
        [
            "prompt_protocol",
            "model_key",
            "bin",
            "n",
            "min_confidence",
            "max_confidence",
            "mean_confidence",
            "accuracy",
            "gap",
            "ece_contribution",
        ],
    )

    category_rows = _category_rows(scores, ece_bins=ece_bins, binning=binning)
    category_fields = ["prompt_protocol", "category", "n_base", "n_instruct", "exploratory"]
    for metric in METRIC_NAMES:
        category_fields.extend([f"base_{metric}", f"instruct_{metric}", f"delta_{metric}"])
    _write_csv(output_dir / "category_breakdown.csv", category_rows, category_fields)

    plot_paths: list[str] = []
    if not no_plots:
        plot_paths = _plot_outputs(
            grouped,
            out_dir=output_dir,
            ece_bins=ece_bins,
            binning=binning,
        )

    manifest = {
        "score_rows": len(scores),
        "summary_path": str(output_dir / "summary.csv"),
        "bootstrap_path": str(output_dir / "paired_bootstrap.json"),
        "calibration_path": str(output_dir / "calibration_tables.csv"),
        "category_breakdown_path": str(output_dir / "category_breakdown.csv"),
        "plots": plot_paths,
    }
    write_json(manifest, output_dir / "analysis_manifest.json")
    return manifest

