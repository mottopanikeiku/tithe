from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_jsonl, scores_path


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_summary(path: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protocol = str(row["prompt_protocol"])
            metric = str(row["metric"])
            for model_key in ("base", "instruct"):
                value = row.get(model_key)
                if value:
                    values[(protocol, f"{model_key}_{metric}")] = float(value)
            delta = row.get("delta_instruct_minus_base")
            if delta:
                values[(protocol, f"delta_{metric}")] = float(delta)
    return values


def _paired_shared_rows(score_rows: list[dict[str, Any]], protocol: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base = {
        row["question_id"]: row
        for row in score_rows
        if row["prompt_protocol"] == protocol and row["model_key"] == "base"
    }
    instruct = {
        row["question_id"]: row
        for row in score_rows
        if row["prompt_protocol"] == protocol and row["model_key"] == "instruct"
    }
    return [(base[question_id], instruct[question_id]) for question_id in sorted(set(base) & set(instruct))]


def _disagreement_rows(paired_rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    counts = {
        "both_correct": 0,
        "base_only_correct": 0,
        "instruct_only_correct": 0,
        "both_wrong": 0,
    }
    for base, instruct in paired_rows:
        base_correct = bool(base["correct"])
        instruct_correct = bool(instruct["correct"])
        if base_correct and instruct_correct:
            counts["both_correct"] += 1
        elif base_correct and not instruct_correct:
            counts["base_only_correct"] += 1
        elif instruct_correct and not base_correct:
            counts["instruct_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(paired_rows)
    return [
        {"outcome": outcome, "count": count, "share": count / total}
        for outcome, count in counts.items()
    ]


def _overconfident_error_rows(
    paired_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = [instruct for _base, instruct in paired_rows if not bool(instruct["correct"])]
    candidates.sort(key=lambda row: float(row["confidence"]), reverse=True)
    rows: list[dict[str, Any]] = []
    for row in candidates[:limit]:
        rows.append(
            {
                "question_id": row["question_id"],
                "category": row.get("category", ""),
                "question": row["question"],
                "correct_label": row["correct_label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "p_correct": row["p_correct"],
                "choice_A": row["choices"]["A"],
                "choice_B": row["choices"]["B"],
            }
        )
    return rows


def _selective_accuracy_rows(score_rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    rows: list[dict[str, Any]] = []
    for model_key in ("base", "instruct"):
        model_rows = [
            row
            for row in score_rows
            if row["prompt_protocol"] == protocol and row["model_key"] == model_key
        ]
        total = len(model_rows)
        for threshold in thresholds:
            selected = [row for row in model_rows if float(row["confidence"]) >= threshold]
            if selected:
                accuracy = sum(1 for row in selected if bool(row["correct"])) / len(selected)
                mean_confidence = sum(float(row["confidence"]) for row in selected) / len(selected)
            else:
                accuracy = ""
                mean_confidence = ""
            rows.append(
                {
                    "prompt_protocol": protocol,
                    "model_key": model_key,
                    "confidence_threshold": threshold,
                    "coverage": len(selected) / total if total else 0.0,
                    "n": len(selected),
                    "accuracy": accuracy,
                    "mean_confidence": mean_confidence,
                }
            )
    return rows


def _plot_headline(summary: dict[tuple[str, str], float], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocol = "shared_plain_ab_label"
    labels = ["Accuracy", "Mean confidence", "ECE"]
    base_values = [
        summary[(protocol, "base_accuracy")],
        summary[(protocol, "base_mean_confidence")],
        summary[(protocol, "base_ece")],
    ]
    instruct_values = [
        summary[(protocol, "instruct_accuracy")],
        summary[(protocol, "instruct_mean_confidence")],
        summary[(protocol, "instruct_ece")],
    ]
    x_positions = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([x - width / 2 for x in x_positions], base_values, width, label="Base", color="#557a95")
    ax.bar([x + width / 2 for x in x_positions], instruct_values, width, label="Instruct", color="#c45a3a")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Value")
    ax.set_title("Base and instruct metrics on the shared prompt")
    ax.legend()
    path = output_dir / "headline_metrics_shared_prompt.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

    delta_labels = ["Accuracy", "Confidence", "Overconfidence", "ECE"]
    deltas = [
        summary[(protocol, "delta_accuracy")],
        summary[(protocol, "delta_mean_confidence")],
        summary[(protocol, "delta_overconfidence_gap")],
        summary[(protocol, "delta_ece")],
    ]
    colors = ["#1d7a55", "#a33a2b", "#a33a2b", "#a33a2b"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(delta_labels, deltas, color=colors)
    ax.set_ylabel("Instruct - base")
    ax.set_title("Metric differences on the shared prompt (instruct - base)")
    path_delta = output_dir / "delta_metrics_shared_prompt.png"
    fig.tight_layout()
    fig.savefig(path_delta, dpi=170)
    plt.close(fig)
    return [str(path), str(path_delta)]


def _plot_selective_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for model_key, color in (("base", "#557a95"), ("instruct", "#c45a3a")):
        model_rows = [row for row in rows if row["model_key"] == model_key and row["accuracy"] != ""]
        ax.plot(
            [float(row["coverage"]) for row in model_rows],
            [float(row["accuracy"]) for row in model_rows],
            marker="o",
            label=model_key,
            color=color,
        )
    ax.set_xlabel("Coverage after confidence threshold")
    ax.set_ylabel("Accuracy on retained answers")
    ax.set_title("Selective accuracy under confidence thresholds")
    ax.set_xlim(1.02, 0.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    path = output_dir / "selective_accuracy_shared_prompt.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def build_presentation_artifacts(
    *,
    run_dir: str | Path,
    report_dir: str | Path,
    protocol: str = "shared_plain_ab_label",
    error_limit: int = 12,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = read_jsonl(scores_path(run_path))
    summary = _load_summary(output_dir / "summary.csv")
    paired_rows = _paired_shared_rows(scores, protocol)

    disagreement = _disagreement_rows(paired_rows)
    _write_csv(
        output_dir / "paired_disagreement_shared_prompt.csv",
        disagreement,
        ["outcome", "count", "share"],
    )

    errors = _overconfident_error_rows(paired_rows, limit=error_limit)
    _write_csv(
        output_dir / "overconfident_instruct_errors_shared_prompt.csv",
        errors,
        [
            "question_id",
            "category",
            "question",
            "correct_label",
            "predicted_label",
            "confidence",
            "p_correct",
            "choice_A",
            "choice_B",
        ],
    )

    selective = _selective_accuracy_rows(scores, protocol)
    _write_csv(
        output_dir / "selective_accuracy_shared_prompt.csv",
        selective,
        [
            "prompt_protocol",
            "model_key",
            "confidence_threshold",
            "coverage",
            "n",
            "accuracy",
            "mean_confidence",
        ],
    )

    plots = _plot_headline(summary, output_dir)
    plots.append(_plot_selective_accuracy(selective, output_dir))

    manifest = {
        "paired_questions": len(paired_rows),
        "protocol": protocol,
        "artifacts": {
            "disagreement": str(output_dir / "paired_disagreement_shared_prompt.csv"),
            "overconfident_errors": str(output_dir / "overconfident_instruct_errors_shared_prompt.csv"),
            "selective_accuracy": str(output_dir / "selective_accuracy_shared_prompt.csv"),
            "plots": plots,
        },
    }
    with (output_dir / "presentation_artifacts.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> int:
    build_presentation_artifacts(
        run_dir="runs/qwen2_5_1_5b",
        report_dir="reports/qwen2_5_1_5b",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

