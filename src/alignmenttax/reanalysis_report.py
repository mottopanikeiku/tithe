"""SVG figures for the saved-score reanalysis; reads summary and diagnostics only."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .cross_family import MODELS, PROTOCOLS

PROMPTS = dict(zip(PROTOCOLS, ("shared", "native")))
REPRESENTATIONS = (("matched_labels", "#17649b"), ("matched_strings", "#b75b16"))
COLORS = {"base": "#17649b", "instruct": "#b33d25", "uniform": "#777777", "constant": "#7e4c9f",
          "base/raw": "#17649b", "base/temperature": "#278766", "instruct/raw": "#b33d25", "instruct/temperature": "#ba8300"}
CONDITION_NAMES = {"uniform": "uniform", "constant": "constant prior", "base/raw": "base", "base/temperature": "base + T",
                   "instruct/raw": "instruct", "instruct/temperature": "instruct + T"}


def _start(width: int, height: int, title: str, description: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>', f'<desc id="desc">{escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#202020">',
            f'<text x="20" y="26" font-size="17">{escape(title)}</text>']


def _save(pieces: list[str], path: Path) -> None:
    path.write_text("\n".join([*pieces, "</g></svg>"]) + "\n", encoding="utf-8")


def _candidate_control(summary: dict[str, Any], protocol: str, path: Path) -> None:
    pairs = summary["pairs"]
    n = summary["matching"]["matched_n"]
    width, height = 1240, 175 + 54 * len(pairs)
    prompt = PROMPTS[protocol]
    pieces = _start(width, height, f"Same {n} questions, same two answers: instruct minus base ({prompt} prompts)",
                    "Accuracy, ECE and categorical Brier changes with 95% paired-question percentile intervals, for A/B-label "
                    "likelihoods and for the same two answer strings scored under the standard prompt.")
    pieces.append(f'<circle cx="27" cy="49" r="4" fill="{REPRESENTATIONS[0][1]}"/><text x="39" y="53">A/B labels (binary prompt)</text>')
    pieces.append(f'<path d="M 270 44 l 5 5 -5 5 -5 -5 Z" fill="{REPRESENTATIONS[1][1]}"/><text x="282" y="53">Same two answer strings (standard prompt), renormalized over both</text>')
    for panel, (metric, title) in enumerate((("accuracy", "ΔAccuracy (percentage points)"), ("ece", "ΔECE (percentage points)"),
                                             ("brier", "ΔBrier × 100 (categorical)"))):
        left, right = 300 + 310 * panel, 580 + 310 * panel
        values = [100 * estimate[field] for pair in pairs.values() for task, _ in REPRESENTATIONS
                  for estimate in [pair["candidate_control"][protocol]["estimates"][f"{task}/instruct_minus_base"][metric]]
                  for field in ("point", "ci_low", "ci_high")]
        lower, upper = min(0.0, min(values)), max(0.0, max(values))
        padding = max(1.0, (upper - lower) * 0.1)
        lower, upper = lower - padding, upper + padding

        def x(value: float) -> float:
            return left + (value - lower) * (right - left) / (upper - lower)

        pieces.append(f'<text x="{(left + right) / 2}" y="84" text-anchor="middle" font-size="13">{title}</text>')
        pieces.append(f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="96" y2="{height - 64}" stroke="#999" stroke-dasharray="4 3"/>')
        for tick in _ticks(lower, upper):
            pieces.append(f'<text x="{x(tick):.2f}" y="{height - 43}" text-anchor="middle">{tick:g}</text>')
        for index, pair in enumerate(pairs.values()):
            center = 117 + 54 * index
            if panel == 0:
                pieces.append(f'<text x="20" y="{center + 1}">{escape(pair["label"])}</text>')
                pieces.append(f'<text x="20" y="{center + 18}" fill="#666">{escape(pair["family"])} · {pair["parameters_billion"]:g}B</text>')
            for row, (task, color) in enumerate(REPRESENTATIONS):
                estimate = pair["candidate_control"][protocol]["estimates"][f"{task}/instruct_minus_base"][metric]
                y = center - 6 + 16 * row
                low, high, point = (x(100 * estimate[field]) for field in ("ci_low", "ci_high", "point"))
                pieces.append(f'<path d="M {low:.2f} {y} H {high:.2f} M {low:.2f} {y - 4} V {y + 4} M {high:.2f} {y - 4} V {y + 4}" stroke="{color}" fill="none" stroke-width="1.7"/>')
                if row == 0:
                    pieces.append(f'<circle cx="{point:.2f}" cy="{y}" r="4" fill="{color}"/>')
                else:
                    pieces.append(f'<path d="M {point:.2f} {y - 5} l 5 5 -5 5 -5 -5 Z" fill="{color}"/>')
    pieces.append(f'<text x="20" y="{height - 19}" fill="#555">Higher accuracy is better; lower ECE and Brier are better. Prompts and scored continuations still differ between rows. Intervals are pointwise.</text>')
    _save(pieces, path)


def _ticks(lower: float, upper: float) -> list[float]:
    """Round tick values strictly inside the axis, so neighbouring panels never collide."""
    raw = (upper - lower) / 4
    step = min(base * 10 ** np.floor(np.log10(raw)) for base in (1, 2, 5, 10) if base * 10 ** np.floor(np.log10(raw)) >= raw)
    return [float(tick) + 0.0 for tick in np.arange(np.ceil(lower / step) * step, upper, step) if lower < tick < upper]


def _panels(summary: dict[str, Any], kind: str, task: str, protocol: str) -> list[str]:
    prompt = PROMPTS[protocol]
    if kind == "reliability":
        title = f"Reliability, {task}, {prompt} prompts (all questions)"
        lines = ("Equal-width confidence bins; dashed line is perfect calibration. Blue base, red instruct.",
                 "Right: question counts per bin, base/instruct.")
    else:
        title = f"Risk-coverage, {task}, {prompt} prompts (held-out test half)"
        lines = ("Error rate among the most confident test questions as coverage grows; exact confidence ties are averaged.",
                 "Right: AURC, lower is better. T = held-out temperature; with two choices it keeps the ranking up to floating-point ties.")
    pieces = _start(1080, 95 + 245 * ((len(summary["pairs"]) + 2) // 3), title, " ".join(lines))
    pieces.extend(f'<text x="20" y="{45 + 16 * i}">{escape(line)}</text>' for i, line in enumerate(lines))
    return pieces


def _small_multiples(summary: dict[str, Any], details: dict[str, Any], kind: str, task: str, protocol: str, path: Path) -> None:
    pieces = _panels(summary, kind, task, protocol)
    for i, (pair_id, pair) in enumerate(summary["pairs"].items()):
        left, top = 30 + (i % 3) * 360, 100 + (i // 3) * 245

        def x(value: float) -> float:
            return left + 170 * value

        def y(value: float) -> float:
            return top + 160 * (1 - value)

        pieces.append(f'<text x="{left}" y="{top - 15}">{escape(pair["label"])}</text>')
        pieces.append(f'<path d="M{x(0)} {y(1)}V{y(0)}H{x(1)}" fill="none" stroke="#777"/>')
        for tick in (0, 0.5, 1):
            pieces.append(f'<text x="{x(tick)}" y="{y(0) + 15}" text-anchor="middle">{tick:g}</text>')
            pieces.append(f'<text x="{left - 6}" y="{y(tick) + 3}" text-anchor="end">{tick:g}</text>')
        if kind == "reliability":
            pieces.append(f'<path d="M{x(0)} {y(0)}L{x(1)} {y(1)}" stroke="#aaa" stroke-dasharray="3 3"/>')
            data = details["pairs"][pair_id]["full_pool"][task][protocol]
            counts = {}
            for model in MODELS:
                bins = data[model]["reliability"]["equal_width"]
                counts[model] = {row["bin"]: row["n"] for row in bins}
                points = " ".join(f'{x(row["mean_confidence"]):.2f},{y(row["accuracy"]):.2f}' for row in bins)
                pieces.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[model]}"/>')
                for row in bins:
                    pieces.append(f'<circle cx="{x(row["mean_confidence"]):.2f}" cy="{y(row["accuracy"]):.2f}" r="3" fill="{COLORS[model]}"><title>n={row["n"]}</title></circle>')
            for bin_id in range(10):
                text = f'{bin_id / 10:.1f}–{(bin_id + 1) / 10:.1f}: {counts["base"].get(bin_id, 0)}/{counts["instruct"].get(bin_id, 0)}'
                pieces.append(f'<text x="{left + 190}" y="{top + 8 + 15 * bin_id}">{text}</text>')
            axis = "x: mean confidence · y: accuracy"
        else:
            data = details["pairs"][pair_id]["held_out"][task][protocol]
            for j, (condition, item) in enumerate(data.items()):
                points = " ".join(f'{x(row["coverage"]):.2f},{y(row["risk"]):.2f}' for row in item["risk_coverage"]["curve"])
                pieces.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[condition]}" stroke-width="1.5"/>')
                pieces.append(f'<text x="{left + 190}" y="{top + 12 + 22 * j}" fill="{COLORS[condition]}">{escape(CONDITION_NAMES[condition])}: {item["risk_coverage"]["aurc"]:.3f}</text>')
            axis = "x: coverage · y: error rate"
        pieces.append(f'<text x="{left}" y="{y(0) + 34}">{axis}</text>')
    _save(pieces, path)


def write_figures(summary: dict[str, Any], details: dict[str, Any], out: Path) -> None:
    for protocol, prompt in PROMPTS.items():
        _candidate_control(summary, protocol, out / f"candidate_control_{prompt}.svg")
        for task in ("binary", "mc1", "matched_labels", "matched_strings"):
            for kind in ("reliability", "risk_coverage"):
                _small_multiples(summary, details, kind, task, protocol, out / f"{kind}_{task}_{prompt}.svg")
