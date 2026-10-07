"""Score pinned base/instruct pairs on one L4; download weights only in the cloud."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
MINUTES = int(os.environ.get("ALIGNMENTTAX_MINUTES", "90"))
app = modal.App("alignmenttax-pairs")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.6", "accelerate==1.10.1",
        "huggingface-hub==0.36.2", "numpy==2.2.6", "pyyaml==6.0.3",
        "sentencepiece==0.2.1",
    )
    .env({"PYTHONPATH": "/project/src", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
          "HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", remote_path="/project/src", copy=True)
    .add_local_dir(ROOT / "configs", remote_path="/project/configs", copy=True)
)
cache = modal.Volume.from_name("alignmenttax-pair-downloads", create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=2, memory=16384, timeout=MINUTES * 60,
              max_containers=1, volumes={"/cache": cache})
def score_pairs(pair_ids: list[str], limit: int | None, check_parity: bool):
    import datetime as dt
    import gc
    import hashlib
    import importlib.metadata
    import shutil
    import time
    import sys

    import torch
    from alignmenttax.data import prepare_data
    from alignmenttax.io_utils import read_jsonl
    from alignmenttax.scoring import (
        TransformerLabelScorer, build_prompt, configured_protocols, score_run,
    )

    os.chdir("/project")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest = json.loads(Path("configs/cross_family.json").read_text())
    started = time.perf_counter()
    dataset_path = Path(manifest["dataset"]["path"])
    prepared = prepare_data(out=dataset_path, seed=manifest["dataset"]["seed"])
    records = read_jsonl(dataset_path)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    selected = [pair for pair in manifest["pairs"] if pair["experiment"]["pair_id"] in pair_ids]
    if len(selected) != len(pair_ids):
        raise ValueError("Unknown or duplicate pair IDs.")
    for pair in selected:
        pair_started = time.perf_counter()
        pair_id = pair["experiment"]["pair_id"]
        config = {key: manifest[key] for key in ("dataset", "scoring", "analysis")}
        config.update(pair)
        out = Path("results/cross_family") / pair_id
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        parity = []
        if check_parity:
            for model_key, model_config in config["models"].items():
                scorer = TransformerLabelScorer(model_key=model_key, model_config=model_config)
                for protocol in configured_protocols(config):
                    prompts = [build_prompt(item, protocol=protocol, model_key=model_key,
                                            tokenizer=scorer.tokenizer) for item in records[:8]]
                    scalar = [scorer.score_prompt(prompt) for prompt in prompts]
                    batch = scorer.score_prompts(prompts)
                    errors = [abs(s[i] - b[i]) for s, b in zip(scalar, batch) for i in (0, 1)]
                    mismatches = sum((s[0] >= s[1]) != (b[0] >= b[1]) for s, b in zip(scalar, batch))
                    row = {"model_key": model_key, "protocol": protocol, "n": len(prompts),
                           "max_abs_logprob_error": max(errors), "label_mismatches": mismatches,
                           "label_token_counts": scalar[0][2]}
                    parity.append(row)
                    if max(errors) > 0.08 or mismatches:
                        raise ValueError(f"Batch/scalar bf16 parity failed: {row}")
                del scorer
                gc.collect()
                torch.cuda.empty_cache()
        count = score_run(config_path=config_path, out=out / "scores.jsonl", limit=limit)
        elapsed = time.perf_counter() - pair_started
        runtime = {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "gpu_type": "L4", "cpu_cores": 2,
            "memory_gib": 16, "cloud_pair_seconds": elapsed,
            "cloud_elapsed_seconds": time.perf_counter() - started,
            "question_rows": min(limit, len(records)) if limit is not None else len(records),
            "score_rows": count, "dataset_sha256": dataset_hash,
            "dataset_preparation": prepared, "parity_checks": parity,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "huggingface-hub", "numpy", "pyyaml")},
            "python": sys.version,
            "cost_basis": "L4 $0.7992/hour + 2 CPU cores $0.047160/core-hour + 16 GiB $0.007992/GiB-hour",
            "cloud_pair_cost_estimate_usd": elapsed / 3600 * (0.7992 + 2 * 0.047160 + 16 * 0.007992),
        }
        payload = {
            "pair_id": pair_id,
            "scores.jsonl.gz": gzip.compress((out / "scores.jsonl").read_bytes(), mtime=0),
            "run_metadata.json": (out / "run_metadata.json").read_bytes(),
            "runtime.json": (json.dumps(runtime, indent=2, sort_keys=True) + "\n").encode(),
            "dataset.jsonl.gz": gzip.compress(dataset_path.read_bytes(), mtime=0),
        }
        yield payload
        # Retain results, not downloaded weights. Each checkpoint is loaded sequentially.
        shutil.rmtree("/cache/huggingface/hub", ignore_errors=True)
        cache.commit()

@app.local_entrypoint()
def main(pairs: str = "all", limit: int = 0, parity: bool = False,
         out: str = "results/cross_family"):
    manifest = json.loads((ROOT / "configs/cross_family.json").read_text())
    pair_ids = [pair["experiment"]["pair_id"] for pair in manifest["pairs"]] if pairs == "all" else pairs.split(",")
    for payload in score_pairs.remote_gen(pair_ids, limit or None, parity):
        pair_id = payload.pop("pair_id")
        destination = ROOT / out / pair_id
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in payload.items():
            if name == "dataset.jsonl.gz":
                target = ROOT / out / "truthfulqa_binary.jsonl.gz"
            else:
                target = destination / name
            target.write_bytes(content)
        print(f"Saved {pair_id} to {out}/{pair_id}")
