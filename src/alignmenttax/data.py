from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, write_jsonl

TRUTHFULQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
TRUTHFULQA_COMMIT_API_URL = "https://api.github.com/repos/sylinrl/TruthfulQA/commits/main"
TRUTHFULQA_VARIANT = "binary_best_vs_best_incorrect"

REQUIRED_COLUMNS = (
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
)


def _request_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/csv,application/json,text/plain,*/*",
            "User-Agent": "alignmenttax/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig")


def resolve_truthfulqa_commit(timeout: int = 20) -> str | None:
    try:
        text = _request_text(TRUTHFULQA_COMMIT_API_URL, timeout=timeout)
        payload = json.loads(text)
    except Exception:
        return None
    sha = payload.get("sha")
    return sha if isinstance(sha, str) and sha else None


def load_or_download_csv(
    *,
    source_csv: str | Path | None = None,
    cache_path: str | Path = "data/raw/TruthfulQA.csv",
    url: str = TRUTHFULQA_CSV_URL,
) -> tuple[str, dict[str, Any]]:
    if source_csv is not None:
        path = Path(source_csv)
        return path.read_text(encoding="utf-8-sig"), {
            "source_kind": "local_csv",
            "source_path": str(path),
            "upstream_url": url,
        }

    cache = Path(cache_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8-sig"), {
            "source_kind": "cache",
            "source_path": str(cache),
            "upstream_url": url,
        }

    text = _request_text(url)
    ensure_parent(cache).write_text(text, encoding="utf-8", newline="\n")
    return text, {
        "source_kind": "download",
        "source_path": str(cache),
        "upstream_url": url,
    }


def parse_truthfulqa_csv_text(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("TruthfulQA CSV has no header row.")
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"TruthfulQA CSV is missing required columns: {', '.join(missing)}")

    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, start=2):
        if row is None:
            continue
        normalized = {key: (value or "").strip() for key, value in row.items() if key is not None}
        if not normalized.get("Question"):
            continue
        missing_values = [column for column in REQUIRED_COLUMNS if not normalized.get(column)]
        if missing_values:
            raise ValueError(
                f"TruthfulQA CSV row {line_number} is missing values for: {', '.join(missing_values)}"
            )
        rows.append(normalized)
    return rows


def _stable_question_id(index: int, question: str, best_answer: str, best_incorrect: str) -> str:
    digest = hashlib.sha1(
        f"{question}\n{best_answer}\n{best_incorrect}".encode("utf-8")
    ).hexdigest()[:12]
    return f"truthfulqa_{index:04d}_{digest}"


def build_binary_dataset(
    rows: list[dict[str, str]],
    *,
    seed: int,
    upstream_url: str = TRUTHFULQA_CSV_URL,
    source_commit: str | None = None,
    source_path: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_rows = rows[:limit] if limit is not None else rows
    records: list[dict[str, Any]] = []

    for index, row in enumerate(selected_rows):
        question = row["Question"].strip()
        best_answer = row["Best Answer"].strip()
        best_incorrect = row["Best Incorrect Answer"].strip()
        correct_label = "A" if rng.randrange(2) == 0 else "B"
        incorrect_label = "B" if correct_label == "A" else "A"
        choices = {
            correct_label: best_answer,
            incorrect_label: best_incorrect,
        }
        records.append(
            {
                "id": _stable_question_id(index, question, best_answer, best_incorrect),
                "source_index": index,
                "question": question,
                "type": row.get("Type", ""),
                "category": row.get("Category", ""),
                "choices": {"A": choices["A"], "B": choices["B"]},
                "correct_label": correct_label,
                "correct_answer": best_answer,
                "incorrect_answer": best_incorrect,
                "source": row.get("Source", ""),
                "source_metadata": {
                    "dataset": "TruthfulQA",
                    "variant": TRUTHFULQA_VARIANT,
                    "seed": seed,
                    "upstream_url": upstream_url,
                    "source_commit": source_commit,
                    "source_path": source_path,
                },
            }
        )
    return records


def prepare_data(
    *,
    out: str | Path,
    seed: int = 20260420,
    source_csv: str | Path | None = None,
    cache_path: str | Path = "data/raw/TruthfulQA.csv",
    url: str = TRUTHFULQA_CSV_URL,
    limit: int | None = None,
    resolve_commit: bool = True,
) -> int:
    csv_text, source_info = load_or_download_csv(
        source_csv=source_csv,
        cache_path=cache_path,
        url=url,
    )
    source_commit = None
    if resolve_commit and source_csv is None:
        source_commit = resolve_truthfulqa_commit()
    rows = parse_truthfulqa_csv_text(csv_text)
    records = build_binary_dataset(
        rows,
        seed=seed,
        upstream_url=url,
        source_commit=source_commit,
        source_path=source_info.get("source_path"),
        limit=limit,
    )
    return write_jsonl(records, out)

