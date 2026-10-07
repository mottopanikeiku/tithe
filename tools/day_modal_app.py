"""Prepare pinned checkpoints on CPU, then score offline on one BF16 GPU."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MINUTES = int(os.environ.get("ALIGNMENTTAX_MINUTES", "25"))
GPU = os.environ.get("ALIGNMENTTAX_GPU", "H100")
GPU_HOURLY_USD = {"L4": 0.7992, "H100": 3.9492}[GPU]
app = modal.App("alignmenttax-day-scale")
volume = modal.Volume.from_name("alignmenttax-day-weights", create_if_missing=True)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub==0.36.2", "datasets==3.6.0", "numpy==2.2.6")
    .env({"HF_HOME": "/cache/huggingface", "HF_XET_NUM_CONCURRENT_RANGE_GETS": "4",
          "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "RAYON_NUM_THREADS": "2",
          "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "configs", remote_path="/project/configs", copy=True)
    .add_local_file(ROOT / "results/cross_family/truthfulqa_binary.jsonl.gz",
                    remote_path="/project/truthfulqa_binary.jsonl.gz", copy=True)
)


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=MINUTES * 60,
              max_containers=1, volumes={"/cache": volume})
def prepare_checkpoints(include_optional: bool):
    import datetime as dt
    import hashlib
    import importlib.metadata
    import time

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    manifest = json.loads(Path("/project/configs/day_scale.json").read_text())
    data_dir = Path("/cache/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    binary_bytes = gzip.decompress(Path("/project/truthfulqa_binary.jsonl.gz").read_bytes())
    if hashlib.sha256(binary_bytes).hexdigest() != "73657092e60bf182a207fa7db9bac5743661d2388bc57c13c2f4bf37eb510430":
        raise ValueError("The previous binary dataset no longer matches its published hash.")
    (data_dir / "truthfulqa_binary.jsonl").write_bytes(binary_bytes)
    pin = manifest["standard_dataset"]
    source = load_dataset(pin["dataset_id"], pin["config"], revision=pin["revision"],
                          split=pin["split"], cache_dir="/cache/datasets")
    if len(source) != 817:
        raise ValueError(f"Expected all 817 standard questions, got {len(source)}.")
    records = []
    for index, item in enumerate(source):
        question = item["question"]
        records.append({
            "question_id": f"truthfulqa_mc_{index:04d}_{hashlib.sha256(question.encode()).hexdigest()[:12]}",
            "question": question, "source_index": index,
            "mc1_choices": item["mc1_targets"]["choices"],
            "mc1_labels": item["mc1_targets"]["labels"],
            "mc2_choices": item["mc2_targets"]["choices"],
            "mc2_labels": item["mc2_targets"]["labels"],
            "source_metadata": pin,
        })
    standard_bytes = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in records)).encode()
    (data_dir / "truthfulqa_standard.jsonl").write_bytes(standard_bytes)
    selected = manifest["pairs"] + (manifest["optional_pairs"] if include_optional else [])
    downloads = []
    allow = ["*.json", "model*.safetensors", "tokenizer.model*", "tokenizer*.model",
             "vocab.bpe", "merges.txt", "LICENSE*", "README.md"]
    for pair in selected:
        for role, config in pair["models"].items():
            model_started = time.perf_counter()
            destination = Path("/cache/models") / config["revision"]
            snapshot_download(config["model_id"], revision=config["revision"],
                              local_dir=destination, allow_patterns=allow, max_workers=4)
            files = [{"name": str(path.relative_to(destination)), "bytes": path.stat().st_size}
                     for path in sorted(destination.rglob("*"))
                     if path.is_file() and ".cache" not in path.parts]
            if not any(item["name"].endswith(".safetensors") for item in files):
                raise ValueError(f"No Transformers weights downloaded for {config['model_id']}.")
            downloads.append({"pair_id": pair["experiment"]["pair_id"], "model_key": role,
                              "model_id": config["model_id"], "revision": config["revision"],
                              "license": config["license"], "files": files,
                              "cloud_download_seconds": time.perf_counter() - model_started})
            volume.commit()
            print(f"Prepared {config['model_id']} at {config['revision']}", flush=True)
    metadata = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "volume": "alignmenttax-day-weights", "cpu_cores": 2, "memory_gib": 4,
                "cloud_preparation_seconds": time.perf_counter() - started,
                "standard_dataset": pin, "standard_question_count": 817,
                "standard_dataset_sha256": hashlib.sha256(standard_bytes).hexdigest(),
                "binary_dataset_sha256": hashlib.sha256(binary_bytes).hexdigest(),
                "dataset_fingerprint": source._fingerprint, "downloads": downloads,
                "packages": {name: importlib.metadata.version(name) for name in
                             ("datasets", "huggingface-hub", "numpy")}}
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    (data_dir / "preparation.json").write_bytes(metadata_bytes)
    volume.commit()
    return {"preparation.json": metadata_bytes,
            "truthfulqa_standard.jsonl.gz": gzip.compress(standard_bytes, mtime=0)}


@app.local_entrypoint()
def download(optional: bool = False, out: str = "results/day_scale"):
    destination = ROOT / out
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in prepare_checkpoints.remote(optional).items():
        (destination / name).write_bytes(content)
    print(f"Saved preparation metadata and all 817 standard questions to {out}")


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.6", "accelerate==1.10.1",
        "huggingface-hub==0.36.2", "numpy==2.2.6", "pyyaml==6.0.3",
        "sentencepiece==0.2.1",
    )
    .env({"PYTHONPATH": "/project/src", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
          "HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "ALIGNMENTTAX_GPU": GPU})
    .add_local_dir(ROOT / "src", remote_path="/project/src", copy=True)
    .add_local_dir(ROOT / "configs", remote_path="/project/configs", copy=True)
)


@app.function(image=gpu_image, gpu=GPU, cpu=2, memory=16384,
              timeout=MINUTES * 60, max_containers=1, volumes={"/cache": volume})
def score_checkpoints(include_optional: bool, pair_ids: list[str]):
    import datetime as dt
    import gc
    import importlib.metadata
    import time
    import sys

    import numpy as np
    import torch
    from alignmenttax.io_utils import read_jsonl
    from alignmenttax.scoring import (
        NATIVE_PROMPT_PROTOCOL, SHARED_PLAIN_PROTOCOL, TransformerLabelScorer,
        _score_row, build_prompt,
    )
    from alignmenttax.standard import (
        CachedContinuationScorer, build_standard_prompt, standard_metric_row,
    )

    os.chdir("/project")
    import hashlib
    source_hash = hashlib.sha256()
    for name in ("standard.py", "scoring.py"):
        source_hash.update(name.encode() + b"\0")
        source_hash.update((Path("src/alignmenttax") / name).read_bytes())
    scorer_source_sha256 = source_hash.hexdigest()
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "accelerate", "huggingface-hub", "numpy", "pyyaml")}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest = json.loads(Path("configs/day_scale.json").read_text())
    settings = manifest["standard_scoring"]
    preparation = json.loads(Path("/cache/data/preparation.json").read_text())
    standard = read_jsonl("/cache/data/truthfulqa_standard.jsonl")
    binary = read_jsonl("/cache/data/truthfulqa_binary.jsonl")
    if len(standard) != 817 or len(binary) != 790:
        raise ValueError("Prepared datasets must contain all 817/790 questions.")
    available = {(item["model_id"], item["revision"]) for item in preparation["downloads"]}
    started = time.perf_counter()
    # Leave three minutes for startup, streaming and shutdown inside the booking.
    soft_deadline = started + max(1, MINUTES - 3) * 60
    pair_times = {}
    decisions = []
    selected = manifest["pairs"] + (manifest["optional_pairs"] if include_optional else [])
    if pair_ids:
        requested = set(pair_ids)
        known = {pair["experiment"]["pair_id"] for pair in selected}
        if requested - known:
            raise ValueError(f"Unknown requested pairs: {sorted(requested - known)}")
        selected = [pair for pair in selected if pair["experiment"]["pair_id"] in requested]
    optional_ids = {pair["experiment"]["pair_id"] for pair in manifest["optional_pairs"]}
    existing_binary = {pair["experiment"]["pair_id"] for pair in manifest["pairs"][:7]}

    def probabilities(values):
        values = np.asarray(values, dtype=np.float64)
        weights = np.exp(values - values.max())
        return weights / weights.sum()

    def audit(cached, model_key, protocol, batch_size):
        checks = []
        reference_count = settings["cache_reference_questions"]
        records = standard[:max(reference_count, batch_size)]
        prompts = [build_standard_prompt(item, model_key=model_key, protocol=protocol,
                                         tokenizer=cached.scorer.tokenizer,
                                         template_date=manifest["scoring"]["template_date"]) for item in records]
        choices = [list(dict.fromkeys(" " + answer for answer in
                                      item["mc1_choices"] + item["mc2_choices"])) for item in records]
        accelerated = cached.score_question_batch(prompts, choices)
        for item, prompt, continuations, actual in zip(
                records[:reference_count], prompts[:reference_count], choices[:reference_count],
                accelerated[:reference_count], strict=True):
            reference = [cached.score_reference(prompt, continuation) for continuation in continuations]
            counts = [len(cached.encode_pair(prompt, continuation)[1]) for continuation in continuations]
            actual_map, reference_map = dict(zip(continuations, actual)), dict(zip(continuations, reference))
            actual1 = [actual_map[" " + answer] for answer in item["mc1_choices"]]
            reference1 = [reference_map[" " + answer] for answer in item["mc1_choices"]]
            actual2 = [actual_map[" " + answer] for answer in item["mc2_choices"]]
            reference2 = [reference_map[" " + answer] for answer in item["mc2_choices"]]
            actual_metrics = standard_metric_row(actual1, item["mc1_labels"], actual2, item["mc2_labels"])
            reference_metrics = standard_metric_row(reference1, item["mc1_labels"],
                                                    reference2, item["mc2_labels"])
            error = max(float(np.max(np.abs(probabilities(actual1) - probabilities(reference1)))),
                        float(np.max(np.abs(probabilities(actual2) - probabilities(reference2)))),
                        abs(actual_metrics["mc2"] - reference_metrics["mc2"]))
            mean_ll_error = max(abs(a - b) / count for a, b, count in
                                zip(actual, reference, counts, strict=True))
            mismatch = int(np.argmax(actual1) != np.argmax(reference1))
            checks.append({"question_id": item["question_id"], "source_index": item["source_index"],
                           "choices": continuations, "cached_loglikelihoods": actual,
                           "reference_loglikelihoods": reference, "continuation_token_counts": counts,
                           "max_abs_probability_error": error,
                           "max_abs_mean_token_loglikelihood_error": mean_ll_error,
                           "mc1_winner_disagreement": mismatch,
                           "cached_metrics": actual_metrics, "reference_metrics": reference_metrics})
        passed = all(row["max_abs_probability_error"] <= settings["cache_reference_max_abs_probability_error"]
                     and row["max_abs_mean_token_loglikelihood_error"] <=
                     settings["cache_reference_max_abs_mean_logprob_error"]
                     and row["mc1_winner_disagreement"] <= settings["cache_reference_max_mc1_disagreements"]
                     for row in checks)
        return {"model_key": model_key, "protocol": protocol, "passed": passed,
                "cached_batch_question_count": len(records), "reference_question_count": len(checks),
                "execution": "prefix_cache" if passed else "full_forward_reference", "checks": checks}

    for pair in selected:
        pair_id = pair["experiment"]["pair_id"]
        optional_pair = pair_id in optional_ids
        if optional_pair:
            remaining = soft_deadline - time.perf_counter()
            reference_seconds = pair_times.get("qwen2_5_32b", float("inf"))
            projected = reference_seconds * pair["experiment"]["parameters_billion"] / 32
            accepted = remaining >= 1.2 * projected + 30
            decisions.append({"pair_id": pair_id, "optional": True, "accepted": accepted,
                              "remaining_cloud_seconds": remaining,
                              "projected_pair_seconds": projected, "headroom_factor": 1.2})
            if not accepted:
                continue
        if any((config["model_id"], config["revision"]) not in available for config in pair["models"].values()):
            raise ValueError(f"CPU preparation is missing a pinned checkpoint for {pair_id}.")
        pair_started = time.perf_counter()
        pair_measured_seconds = 0.0
        for model_key, public_config in pair["models"].items():
            saved_unit = Path("/cache/results/day_scale") / pair_id / f"{model_key}.json.gz"
            if saved_unit.exists():
                saved_bytes = saved_unit.read_bytes()
                saved = json.loads(gzip.decompress(saved_bytes))
                if (saved["config"]["models"][model_key] != public_config or
                        saved["config"]["standard_dataset"] != manifest["standard_dataset"] or
                        saved["config"]["scoring"] != manifest["scoring"] or
                        saved["config"]["standard_scoring"] != manifest["standard_scoring"] or
                        saved["runtime"]["scorer_source_sha256"] != scorer_source_sha256 or
                        saved["runtime"]["packages"] != packages or
                        len(saved["standard"]) != 1634 or
                        len(saved["binary"]) != (0 if pair_id in existing_binary else 1580)):
                    raise ValueError(f"{pair_id}/{model_key}: saved unit does not match this run.")
                pair_measured_seconds += saved["runtime"]["cloud_model_seconds"]
                yield saved_bytes
                print(f"Reused completed {pair_id}/{model_key} from the volume", flush=True)
                continue
            model_started = time.perf_counter()
            local_config = dict(public_config)
            snapshot = str(Path("/cache/models") / public_config["revision"])
            local_config.update(model_id=snapshot, tokenizer_id=snapshot)
            scorer = TransformerLabelScorer(model_key=model_key, model_config=local_config)
            if any(str(device) in ("cpu", "disk") for device in scorer.model.hf_device_map.values()):
                raise ValueError(f"{public_config['model_id']}: CPU/disk offload is not allowed.")
            scorer.model_id = public_config["model_id"]
            cached = CachedContinuationScorer(
                scorer, continuation_batch_size=settings["continuation_batch_size"],
                logit_chunk_size=settings["logit_chunk_size"],
                add_bos_token=manifest["standard_dataset"]["add_bos_token"],
                template_date=manifest["scoring"]["template_date"],
            )
            batch_size = settings["question_batch_size_large" if
                                  pair["experiment"]["parameters_billion"] >= 14 else "question_batch_size_small"]
            standard_rows, binary_rows, numerical_checks = [], [], []
            protocols = [SHARED_PLAIN_PROTOCOL] if model_key == "base" else [
                SHARED_PLAIN_PROTOCOL, NATIVE_PROMPT_PROTOCOL]
            for protocol in protocols:
                check = audit(cached, model_key, protocol, batch_size)
                numerical_checks.append(check)
                standard_rows.extend(cached.standard_records(
                    standard, model_key=model_key, protocol=protocol, batch_size=batch_size,
                    use_reference=not check["passed"],
                ))
                if pair_id not in existing_binary:
                    for item in binary:
                        prompt = build_prompt(item, protocol=protocol, model_key=model_key,
                                              tokenizer=scorer.tokenizer,
                                              template_date=manifest["scoring"]["template_date"])
                        row_started = time.perf_counter()
                        logprob_a, logprob_b, counts = scorer.score_prompt(prompt)
                        binary_rows.append(_score_row(
                            item=item, model_key=model_key, model_id=public_config["model_id"],
                            protocol=protocol, logprob_a=logprob_a, logprob_b=logprob_b,
                            device=str(scorer.device), dtype=scorer.dtype,
                            elapsed_seconds=time.perf_counter() - row_started, label_token_counts=counts,
                            prompt_format="chat_template" if model_key == "instruct" and
                            protocol == NATIVE_PROMPT_PROTOCOL else "plain",
                            model_revision=public_config["revision"],
                        ))
            if model_key == "base":
                standard_rows.extend(dict(row, protocol=NATIVE_PROMPT_PROTOCOL, elapsed_seconds=0.0,
                                          reused_from_protocol=SHARED_PLAIN_PROTOCOL)
                                     for row in list(standard_rows))
                binary_rows.extend(dict(row, prompt_protocol=NATIVE_PROMPT_PROTOCOL, elapsed_seconds=0.0,
                                        reused_from_protocol=SHARED_PLAIN_PROTOCOL)
                                   for row in list(binary_rows))
            elapsed = time.perf_counter() - model_started
            config = {key: manifest[key] for key in
                      ("dataset", "scoring", "analysis", "standard_dataset", "standard_scoring")}
            config.update(pair)
            runtime = {
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name(), "gpu_type": GPU, "cpu_cores": 2,
                "memory_gib": 16, "cloud_model_seconds": elapsed,
                "cloud_elapsed_seconds": time.perf_counter() - started,
                "model_key": model_key, "model_id": public_config["model_id"],
                "model_revision": public_config["revision"], "standard_rows": len(standard_rows),
                "binary_rows": len(binary_rows), "question_batch_size": batch_size,
                "continuation_batch_size": settings["continuation_batch_size"],
                "numerical_checks": numerical_checks, "base_native_rows_reuse_shared_plain": model_key == "base",
                "bf16_reduced_precision_reduction": False,
                "template_date": manifest["scoring"]["template_date"],
                "scorer_source_sha256": scorer_source_sha256,
                "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "packages": packages,
                "python": sys.version,
                "cost_basis": f"{GPU} ${GPU_HOURLY_USD}/hour + 2 CPU cores $0.047160/core-hour + 16 GiB $0.007992/GiB-hour",
            }
            payload = {"pair_id": pair_id, "model_key": model_key,
                       "standard": standard_rows, "binary": binary_rows, "runtime": runtime, "config": config}
            cloud_out = Path("/cache/results/day_scale") / pair_id
            cloud_out.mkdir(parents=True, exist_ok=True)
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            compressed_payload = gzip.compress(payload_bytes, mtime=0)
            (cloud_out / f"{model_key}.json.gz").write_bytes(compressed_payload)
            volume.commit()
            del cached, scorer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            pair_measured_seconds += elapsed
            yield compressed_payload
            print(f"Completed {pair_id}/{model_key}: {len(standard_rows)} standard rows, "
                  f"{len(binary_rows)} new binary rows", flush=True)
        pair_times[pair_id] = max(pair_measured_seconds, time.perf_counter() - pair_started)
    yield gzip.compress(json.dumps({"decisions": decisions, "gpu_type": GPU,
                                   "selected_pairs": [pair["experiment"]["pair_id"] for pair in selected],
                                   "cloud_scoring_seconds": time.perf_counter() - started}).encode(), mtime=0)


@app.local_entrypoint()
def score(optional: bool = False, pairs: str = "", out: str = "results/day_scale"):
    destination = ROOT / out
    destination.mkdir(parents=True, exist_ok=True)
    pending = {}
    pair_ids = [value.strip() for value in pairs.split(",") if value.strip()]
    for compressed in score_checkpoints.remote_gen(optional, pair_ids):
        payload = json.loads(gzip.decompress(compressed))
        if "decisions" in payload:
            (destination / f"scoring_runtime_{GPU.lower()}.json").write_text(json.dumps(payload, indent=2) + "\n")
            continue
        pair_id, role = payload["pair_id"], payload["model_key"]
        pending.setdefault(pair_id, {})[role] = payload
        (destination / f".{pair_id}_{role}.json.gz").write_bytes(compressed)
        if set(pending[pair_id]) != {"base", "instruct"}:
            continue
        pair = pending.pop(pair_id)
        pair_out = destination / pair_id
        pair_out.mkdir(parents=True, exist_ok=True)
        config = payload["config"]
        for task in ("standard", "binary"):
            rows = pair["base"][task] + pair["instruct"][task]
            if not rows:
                continue
            expected = 3268 if task == "standard" else 3160
            if len(rows) != expected:
                raise ValueError(f"{pair_id}: {task} expected {expected} rows, got {len(rows)}.")
            raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
            (pair_out / f"{task}_scores.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
            metadata = {"config": config, "fake": False, "question_rows": expected // 4,
                        "score_rows_total": len(rows), "resume": False, "limit": None}
            (pair_out / f"{task}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (pair_out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (pair_out / "runtime.json").write_text(json.dumps(
            {key: pair[key]["runtime"] for key in ("base", "instruct")}, indent=2) + "\n")
        for key in ("base", "instruct"):
            (destination / f".{pair_id}_{key}.json.gz").unlink()
        print(f"Saved complete pair {pair_id} to {out}/{pair_id}")
