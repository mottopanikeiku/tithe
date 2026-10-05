from __future__ import annotations

import csv
import hashlib
import io
import random
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, write_jsonl

TRUTHFULQA_COMMIT = "d71c110897f5d31c5d7f309e7bc316c152f6f031"
TRUTHFULQA_CSV_URL = (
    f"https://raw.githubusercontent.com/sylinrl/TruthfulQA/{TRUTHFULQA_COMMIT}/TruthfulQA.csv"
)
# SHA256 of the unmodified CSV bytes at the pinned revision.
TRUTHFULQA_CSV_SHA256 = "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c"
TRUTHFULQA_VARIANT = "binary_best_vs_best_incorrect"

REQUIRED_COLUMNS = (
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
)


def _request_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/csv,application/json,text/plain,*/*",
            "User-Agent": "alignmenttax/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_or_download_csv(
    *,
    source_csv: str | Path | None = None,
    cache_path: str | Path = "data/raw/TruthfulQA.csv",
    url: str = TRUTHFULQA_CSV_URL,
) -> tuple[str, dict[str, Any]]:
    """Read exact CSV bytes and identify the pinned dataset by their SHA256.

    Local files and existing caches have unknown upstream provenance unless their
    checksum matches the pin. They remain usable offline either way.
    """
    if source_csv is not None:
        path = Path(source_csv)
        csv_bytes = path.read_bytes()
        source_kind = "local_csv"
    else:
        path = Path(cache_path)
        if path.exists():
            csv_bytes = path.read_bytes()
            source_kind = "cache"
        else:
            csv_bytes = _request_bytes(url)
            source_kind = "download"

    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    matches_pin = csv_sha256 == TRUTHFULQA_CSV_SHA256
    if source_kind == "download":
        if url == TRUTHFULQA_CSV_URL and not matches_pin:
            raise ValueError("Downloaded TruthfulQA CSV does not match the pinned SHA256.")
        ensure_parent(path).write_bytes(csv_bytes)

    # A filename or requested URL cannot establish where existing bytes came from.
    upstream_url = url if source_kind == "download" else None
    if matches_pin:
        upstream_url = TRUTHFULQA_CSV_URL
    return csv_bytes.decode("utf-8-sig"), {
        "source_kind": source_kind,
        "source_path": str(path),
        "upstream_url": upstream_url,
        "source_commit": TRUTHFULQA_COMMIT if matches_pin else None,
        "csv_sha256": csv_sha256,
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
    upstream_url: str | None = TRUTHFULQA_CSV_URL,
    source_commit: str | None = None,
    source_path: str | None = None,
    csv_sha256: str | None = None,
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
                    "csv_sha256": csv_sha256,
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
) -> int:
    """Write binary records with the original CSV SHA256 in source_metadata."""
    csv_text, source_info = load_or_download_csv(
        source_csv=source_csv,
        cache_path=cache_path,
        url=url,
    )
    rows = parse_truthfulqa_csv_text(csv_text)
    records = build_binary_dataset(
        rows,
        seed=seed,
        upstream_url=source_info["upstream_url"],
        source_commit=source_info["source_commit"],
        source_path=source_info["source_path"],
        csv_sha256=source_info["csv_sha256"],
        limit=limit,
    )
    return write_jsonl(records, out)

