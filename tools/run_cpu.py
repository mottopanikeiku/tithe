"""Run the pinned 0.5B experiment with a sampled RSS and wall-time stop guard."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qwen2_5_0_5b"
BUDGET_BYTES = 1_500_000_000
STOP_BYTES = 1_450_000_000  # Leave room between the sampled guard and the budget.
SECONDS = 7200


def memory_field(path: Path, field: str) -> int:
    for line in path.read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


def main() -> int:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    available = memory_field(Path("/proc/meminfo"), "MemAvailable")
    command = ["nice", "-n", "19", str(ROOT / ".venv/bin/alignmenttax"), "score",
               "--config", "configs/qwen2_5_0_5b.yaml", "--out", str(OUT / "scores.jsonl")]
    metadata = {
        "command": command, "available_memory_before_bytes": available,
        "memory_budget_bytes": BUDGET_BYTES, "rss_stop_threshold_bytes": STOP_BYTES,
        "sample_interval_seconds": 0.01, "wall_time_limit_seconds": SECONDS,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface-hub", "pyyaml")},
        "hardware": platform.processor(), "cpu_threads": 2,
    }
    peak = 0
    started = time.monotonic()
    reason = None
    exit_code = None
    if available < BUDGET_BYTES:
        reason = "insufficient_available_memory"
    else:
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           HF_HOME=str(ROOT / ".hf-cache"), HF_HUB_DISABLE_XET="1",
                           TOKENIZERS_PARALLELISM="false")
        with (OUT / "runtime.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                try:
                    peak = max(peak, memory_field(Path(f"/proc/{process.pid}/status"), "VmRSS"))
                except FileNotFoundError:
                    break
                elapsed = time.monotonic() - started
                if peak >= STOP_BYTES:
                    reason = "rss_stop_threshold_exceeded"
                elif elapsed >= SECONDS:
                    reason = "wall_time_limit_exceeded"
                if reason:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.01)
            exit_code = process.wait()
            if reason is None and exit_code:
                reason = "scoring_process_failed"
    metadata.update(exit_code=exit_code, stopped_reason=reason,
                    sampled_peak_rss_bytes=peak, elapsed_seconds=time.monotonic() - started,
                    completed=reason is None and exit_code == 0)
    (OUT / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if metadata["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
