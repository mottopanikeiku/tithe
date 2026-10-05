# Experiment protocol

This is a matched base/instruct comparison, not a causal estimate of reinforcement learning. Model weights and TruthfulQA revisions are pinned; neither the original 1.5B experiment nor its model outputs were published before this change.

## Dataset

`prepare-data` selects TruthfulQA's Best Answer and Best Incorrect Answer for each question. The correct answer is placed in A or B using a seeded random generator. This is a binary derivative, not the official MC1/MC2 evaluation. Each record includes its source index, category, answer order, upstream commit when verified, and SHA256 of the exact source CSV bytes.

The default CSV comes from the immutable URL in `data.py`. Its known checksum must match a fresh download. Existing caches and local files are usable offline, but only matching bytes claim the pinned upstream commit; other inputs retain their actual hash with unknown upstream provenance.

## Scoring

`shared_plain_ab_label` uses identical plain prompts for both models. `native_prompt_ab_label` keeps the base model's plain prompt and applies the instruct tokenizer's chat template. Template errors stop scoring instead of silently changing the prompt. Each row records both the requested `prompt_protocol` and the actual `prompt_format` (`plain`, `chat_template`, or `synthetic` for fake tests), plus `model_revision`.

The scorer evaluates the continuations ` A` and ` B` and normalizes their likelihoods over those two choices. It does not measure an unrestricted probability of giving a truthful answer. If both continuations are single tokens, only the last-position logits are materialized and the attention cache is disabled. Multi-token continuations are scored token by token. Models are loaded sequentially and released between models. `--resume` skips completed question/model/protocol keys before inference, including avoiding a model load if all of its keys exist. Resume only with the same configuration and dataset; an output file is not a cross-experiment cache.

`--fake` creates deterministic synthetic scores for tests. Those rows are marked `device=fake`, `dtype=fake`, and `prompt_format=synthetic`; they are not model evidence.

## Analysis

`analyze` reports accuracy, mean restricted-choice confidence, confidence minus accuracy, binary Brier score, negative log likelihood, and equal-frequency expected calibration error. The paired bootstrap resamples question IDs together for both models and reports percentile intervals for instruct-minus-base deltas. Category summaries are exploratory.

`calibrate` assigns questions to calibration or test sets using a hash of the question ID and seed, so the same split is used across models and protocols. It fits a separate positive scalar temperature for each model/protocol by minimizing calibration-set NLL with a bounded logarithmic grid search. Metrics are reported separately on calibration and held-out test questions, before and after scaling. A temperature does not change the predicted label or repair accuracy.

## Local execution

The CPU configuration is `configs/qwen2_5_0_5b.yaml`. `tools/run_cpu.py` runs it under `nice -n 19`, using two CPU threads, bf16 weights, one loaded model at a time, and a two-hour wall-time limit. It checks available memory before starting and samples RSS every 10 ms, stopping at 1.45 GB to leave headroom below the 1.5 GB budget. This is a sampled stop guard, not a kernel-enforced memory limit. It writes the actual command, environment versions, sampled peak RSS, runtime, and completion status to `results/qwen2_5_0_5b/runtime.json`.

The original pinned 1.5B configuration is retained for another machine; it must not be run under this laptop's limits. Scoring requires public model downloads but no paid service. Analysis reads saved scores and does not load models.

The stopped attempt and proposed hardware allowance are described in [NEXT.md](NEXT.md).
