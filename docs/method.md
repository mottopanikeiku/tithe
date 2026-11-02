# Method

The study compares nine base/instruct pairs from Qwen2.5 (0.5B to 32B), OLMo-2, SmolLM2 and Mistral. Checkpoint and tokenizer revisions, precision and prompts are in [`configs/day_scale.json`](../configs/day_scale.json).

## Two tasks, one question set

- A/B task: 790 TruthfulQA questions from the pinned CSV. Each gets its Best Answer and Best Incorrect Answer, assigned to A and B with seed 20260420. A zero-shot prompt shows the question and both answers and asks which is more truthful; the model scores the continuations ` A` and ` B`.
- Standard task: all 817 validation questions of `truthfulqa/truthful_qa` (`multiple_choice`) with the lm-evaluation-harness six-example Q/A primer. The model scores each full answer as a sum of token log probabilities, with no length normalization. MC1 and MC2 follow the harness definitions.

To compare them, questions are joined on exact text and the two A/B answers are found among each question's MC1 candidates (outer whitespace and terminal periods stripped; matches must be unique and agree on which answer is correct). That leaves 779 questions. For the full-answer side, the two matched answers' saved likelihoods are renormalized over just those two. No scored text changes. [`matching.json`](../results/reanalysis/matching.json) lists every match and exclusion.

Both checkpoints get the same plain prompt (`shared_plain_ab_label`). The `native_prompt_ab_label` rows wrap the instruct prompt in its chat template; they are a sensitivity check.

## Metrics

- Accuracy is the share of questions where the higher-probability choice is correct. Confidence is that probability.
- ECE uses ten equal-frequency bins; weighted mean absolute gap between accuracy and confidence. Bins are rebuilt inside every bootstrap draw.
- Brier is categorical Brier `sum((p - y)^2)`; NLL is unclipped, in nats.
- All changes are instruct minus base. 10,000 paired question resamples (PCG64, seed 20260420), percentile 95% intervals, shared across models, protocols and metrics. A direction counts only if its interval excludes zero. No multiple-comparison correction.

Rebuilding `results/reanalysis` on a different CPU architecture can change saved floats in their last digits; an arm64 rebuild differs from the committed files only there, and every rounded figure in the README stays the same.

## Temperature scaling

Questions are split in half by a hash of their text. One temperature per model, fit on the calibration half by minimizing NLL in `[0.01, 200]`, is evaluated on the test half. 65 of 144 fits end at the upper bound, including every base model on matched answer strings. [`temperatures.csv`](../results/reanalysis/temperatures.csv) flags them.

## Running the models

The seven A/B pairs ran on one L4 via [`modal_app.py`](../modal_app.py). A pilot on Qwen2.5-0.5B found batched and scalar log likelihoods differing by up to 0.107 nats in BF16, above the 0.08 tolerance, so A/B scoring uses scalar forwards ([`batch_parity_failure.json`](../results/pilot/batch_parity_failure.json)).

The standard task and the two largest Qwen pairs ran through [`tools/day_modal_app.py`](../tools/day_modal_app.py) on L4 and H100 GPUs, with BF16 weights and no quantization or offload. Standard scoring can reuse a question's prefix cache across answers; on a four-question check per model, 21 of 27 comparisons exceeded the numerical tolerance and fell back to full forwards ([`numerical_audit.json`](../results/day_scale/numerical_audit.json)).

The saved runs hold 29,412 standard and 28,440 A/B score rows. Compute cost was about $6 ([`cost.json`](../results/day_scale/cost.json)).
