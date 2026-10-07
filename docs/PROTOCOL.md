# Experiment protocol

This is a matched base/instruct comparison, not a causal estimate of reinforcement learning. Model weights and TruthfulQA revisions are pinned; neither the original 1.5B experiment nor its model outputs were published before this change.

## Dataset

`prepare-data` selects TruthfulQA's Best Answer and Best Incorrect Answer for each question. The correct answer is placed in A or B using a seeded random generator. This is a binary derivative, not the official MC1/MC2 evaluation. Each record includes its source index, category, answer order, upstream commit when verified, and SHA256 of the exact source CSV bytes.

The default CSV comes from the immutable URL in `data.py`. Its known checksum must match a fresh download. Existing caches and local files are usable offline, but only matching bytes claim the pinned upstream commit; other inputs retain their actual hash with unknown upstream provenance.

## Scoring

`shared_plain_ab_label` uses identical plain prompts for both models. `native_prompt_ab_label` keeps the base model's plain prompt and applies the instruct tokenizer's chat template. Template errors stop scoring instead of silently changing the prompt. Each row records both the requested `prompt_protocol` and the actual `prompt_format` (`plain`, `chat_template`, or `synthetic` for fake tests), plus `model_revision`.

The scorer evaluates the continuations ` A` and ` B` and normalizes their likelihoods over those two choices. It does not measure an unrestricted probability of giving a truthful answer. Label logits are normalized in float32, including when weights are bf16. Batched execution uses attention masks and logical position IDs; multi-token labels retain the separately tokenized continuation semantics. Models that support suffix-only logits avoid materializing unneeded positions; older architectures use the full-logits forward path. Attention caching is disabled. Models are loaded sequentially and released between models. `--resume` skips completed question/model/protocol keys before inference, including avoiding a model load if all of its keys exist. Resume only with the same configuration and dataset; an output file is not a cross-experiment cache.

`--fake` creates deterministic synthetic scores for tests. Those rows are marked `device=fake`, `dtype=fake`, and `prompt_format=synthetic`; they are not model evidence.

## Analysis

`analyze` reports accuracy, mean restricted-choice confidence, confidence minus accuracy, binary Brier score, negative log likelihood, and equal-frequency expected calibration error. The paired bootstrap resamples question IDs together for both models and reports percentile intervals for instruct-minus-base deltas. Category summaries are exploratory.

`calibrate` assigns questions to calibration or test sets using a hash of the question ID and seed, so the same split is used across models and protocols. It fits a separate positive scalar temperature for each model/protocol by minimizing calibration-set NLL with a bounded logarithmic grid search. Metrics are reported separately on calibration and held-out test questions, before and after scaling. A temperature does not change the predicted label or repair accuracy.

The [cross-family plan](CROSS_FAMILY_PLAN.md) fixes the model pairs and interpretation before scoring. Its vectorized analysis uses NumPy PCG64 with the same paired-resampling and percentile definitions as the original Python implementation, but a different random-number stream. All pairs and protocols share draws over sorted question IDs. The summary records the generator, seed, binning and interval conventions. Compressed `scores.jsonl.gz` files preserve every full score row and are accepted by analysis, calibration and presentation commands.

## Local execution

The CPU configuration is `configs/qwen2_5_0_5b.yaml`. `tools/run_cpu.py` runs it under `nice -n 19`, using two CPU threads, bf16 weights, one loaded model at a time, and a two-hour wall-time limit. It checks available memory before starting and samples RSS every 10 ms, stopping at 1.45 GB to leave headroom below the 1.5 GB budget. This is a sampled stop guard, not a kernel-enforced memory limit. It writes the actual command, environment versions, sampled peak RSS, runtime, and completion status to `results/qwen2_5_0_5b/runtime.json`.

The original pinned 1.5B CPU configuration is retained for another machine; it must not be run under the earlier laptop's limits. Analysis reads saved scores and does not load models.

The [stopped attempt](../results/qwen2_5_0_5b) is retained separately. [NEXT.md](NEXT.md) now proposes label and answer-order checks after the completed cloud comparison.

## Cloud execution

`modal_app.py` runs the pinned cross-family configuration on one L4 with two CPU cores and 16 GiB host memory. The image pins package versions, downloads weights inside the container, and returns full scores plus dataset and runtime metadata. Set `ALIGNMENTTAX_MINUTES` to the job's limit. A small pilot checks scalar/batch likelihood differences under bf16 before the full run; its tolerance and observed differences are saved, not treated as bitwise equality.

The cloud runtime estimates GPU, CPU and memory charges. The separate cost summary includes run wall time and overhead at the same resource rates with a ten-percent allowance. It is a conservative estimate, not an invoice. The stopped local attempt remains historical evidence; it is not mixed with completed cloud scores.

### Execution decision before full scoring

The Qwen2.5-0.5B base pilot found a maximum scalar/batch raw-log-likelihood difference of 0.107038 nats on eight shared-prompt questions, above the preset 0.08 tolerance, with no changed predicted labels in that subset. I did not loosen the tolerance after seeing it. I switched the full configuration to scalar execution (`batch_size: 1`) before scoring the complete pairs. The likelihood definition, bf16 weights, pinned revisions, question set and paired analysis plan did not change. The batching code remains covered by numerical reference tests, but these published measurements do not use it. [The failed pilot](../results/pilot/batch_parity_failure.json) records what was actually checked.

The previous stopped CPU attempt normalized logits in bf16; the current scorer uses float32 normalization of bf16 model logits for both checkpoints. I recompute all rows rather than mix either device or normalization implementation with the historical partial file.

## Earlier calibration studies

I build on [Kadavath et al.](https://arxiv.org/abs/2207.05221), who studied choice-probability calibration and found that temperature adjustment could repair apparent RLHF miscalibration on several evaluations. [Huang, Lu and Zeng](https://arxiv.org/abs/2508.00264) report worse calibration after instruction tuning in matched models on MMLU (their Figure 1). My experiment is a task-specific comparison on binary TruthfulQA with paired intervals and a separate prompt sensitivity check, not the first study of post-training calibration and not a replication of their MMLU setting.
