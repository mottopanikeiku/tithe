from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def scores_path(run: str | Path) -> Path:
    path = Path(run)
    if not path.is_dir():
        return path
    candidates = [path / name for name in ("scores.jsonl", "scores.jsonl.gz")]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if len(existing) > 1:
        raise ValueError(f"Both compressed and plain scores exist in {run}.")
    return existing[0] if existing else candidates[0]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    output_path = ensure_parent(path)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_json(data: Any, path: str | Path) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML configs require PyYAML. Install the project dependencies or use a JSON config."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must load to a mapping: {path}")
    return data

