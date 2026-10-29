from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .cross_family import PROTOCOLS

COLORS = {"base": "#23658b", "instruct": "#b33d25", "uniform": "#777777", "constant": "#7e4c9f",
          "base/raw": "#23658b", "base/temperature": "#278766", "instruct/raw": "#b33d25", "instruct/temperature": "#ba8300"}


def _start(title: str, width: int, height: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{escape(title)}</title>', '<style>text{font-family:system-ui,sans-serif;fill:#222;font-size:11px}</style>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>',
            f'<text x="20" y="24" style="font-size:16px">{escape(title)}</text>']


def _save(pieces: list[str], path: Path) -> None:
    path.write_text("\n".join([*pieces, "</svg>"]) + "\n", encoding="utf-8")


def write_figures(summary: dict[str, Any], out: Path) -> None:
    pairs = list(summary["pairs"])
    for protocol in PROTOCOLS:
        prompt = "shared" if protocol == PROTOCOLS[0] else "native"
        pieces = _start(f"tithe: fixed questions and answers, {prompt} prompts", 800, 120 + 46 * len(pairs))
        pieces.append('<text x="20" y="45">ΔECE, instruct − base (percentage points); paired 95% question-bootstrap intervals</text>')
        pieces.append('<text x="20" y="63">Blue: A/B labels · red: two full strings. Prompts and scored continuations remain different.</text>')
        estimates = [summary["pairs"][pair]["candidate_control"][protocol]["estimates"] for pair in pairs]
        low = min(item[f"matched_{task}/instruct_minus_base"]["ece"]["ci_low"] for item in estimates for task in ("labels", "strings"))
        high = max(item[f"matched_{task}/instruct_minus_base"]["ece"]["ci_high"] for item in estimates for task in ("labels", "strings"))
        low, high = min(low, 0) - 0.02, max(high, 0) + 0.02
        x = lambda value: 210 + 540 * (value - low) / (high - low)
        pieces.append(f'<path d="M{x(0):.2f} 80V{80 + 46 * len(pairs)}" stroke="#999" stroke-dasharray="3 3"/>')
        for i, (pair, item) in enumerate(zip(pairs, estimates)):
            y = 96 + i * 46
            pieces.append(f'<text x="15" y="{y}">{escape(pair)}</text>')
            for j, task in enumerate(("labels", "strings")):
                estimate = item[f"matched_{task}/instruct_minus_base"]["ece"]
                color = COLORS[("base", "instruct")[j]]
                yy = y - 8 + j * 15
                pieces.append(f'<path d="M{x(estimate["ci_low"]):.2f} {yy}H{x(estimate["ci_high"]):.2f}" stroke="{color}" stroke-width="2"/>')
                pieces.append(f'<circle cx="{x(estimate["point"]):.2f}" cy="{yy}" r="3" fill="{color}"/>')
        for value in (low, 0, high):
            pieces.append(f'<text x="{x(value):.2f}" y="{100 + 46 * len(pairs)}" text-anchor="middle">{100 * value:.1f}</text>')
        _save(pieces, out / f"candidate_control_{prompt}.svg")
        for task in ("binary", "mc1", "matched_labels", "matched_strings"):
            for kind in ("reliability", "risk_coverage"):
                title = f"tithe: {task}, {prompt} prompts — {kind.replace('_', ' ')}"
                pieces = _start(title, 1080, 80 + 245 * ((len(pairs) + 2) // 3))
                description = ("Full task pool; equal-width bins. Blue base / red instruct. Counts by bin at right (base/instruct)."
                               if kind == "reliability" else "Held-out test split. Raw / temperature: base blue / green, instruct red / gold; uniform gray, constant purple.")
                pieces.append(f'<text x="20" y="45">{escape(description)}</text>')
                for i, pair in enumerate(pairs):
                    left, top = 30 + (i % 3) * 360, 85 + (i // 3) * 245
                    x = lambda value: left + 170 * value
                    y = lambda value: top + 160 * (1 - value)
                    pieces.append(f'<text x="{left}" y="{top - 15}">{escape(pair)}</text>')
                    pieces.append(f'<path d="M{x(0)} {y(1)}V{y(0)}H{x(1)}" fill="none" stroke="#777"/>')
                    for tick in (0, 0.5, 1):
                        pieces.append(f'<text x="{x(tick)}" y="{y(0) + 15}" text-anchor="middle">{tick:g}</text>')
                        pieces.append(f'<text x="{left - 6}" y="{y(tick) + 3}" text-anchor="end">{tick:g}</text>')
                    if kind == "reliability":
                        pieces.append(f'<path d="M{x(0)} {y(0)}L{x(1)} {y(1)}" stroke="#aaa" stroke-dasharray="3 3"/>')
                        data = summary["pairs"][pair]["full_pool_diagnostics"][task][protocol]
                        counts = {}
                        for model in ("base", "instruct"):
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
                        data = summary["pairs"][pair]["held_out"][task][protocol]["diagnostics"]
                        for condition, item in data.items():
                            points = " ".join(f'{x(row["coverage"]):.2f},{y(row["risk"]):.2f}' for row in item["risk_coverage"]["curve"])
                            pieces.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[condition]}" stroke-width="1.5"/>')
                        for j, (condition, item) in enumerate(data.items()):
                            pieces.append(f'<text x="{left + 190}" y="{top + 12 + 22 * j}">{escape(condition)}: {item["risk_coverage"]["aurc"]:.3f}</text>')
                        axis = "x: coverage · y: error risk · right: AURC"
                    pieces.append(f'<text x="{left}" y="{y(0) + 34}">{axis}</text>')
                _save(pieces, out / f"{kind}_{task}_{prompt}.svg")
