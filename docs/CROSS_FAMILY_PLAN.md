# Cross-family comparison plan

I wrote this plan before scoring the new pairs on 2026-10-07. I will keep it unchanged when interpreting the results.

## Question and fixed sample

Does an instruction-tuned checkpoint consistently gain truthfulness at the expense of confidence calibration? I will compare seven public, ungated base/instruct pairs: Qwen2.5 0.5B, 1.5B and 7B; OLMo-2-0425 1B and OLMo-2-1124 7B; SmolLM2 1.7B; and Mistral 7B v0.3. Exact model and tokenizer commits are in [the configuration](../configs/cross_family.json). The earlier Qwen2.5-0.5B weight revisions, bf16 precision, dataset seed and scoring rules remain unchanged; only device and batch execution change.

I will use all 790 rows of the repository's pinned TruthfulQA CSV, with its existing Best Answer versus Best Incorrect Answer construction and seeded A/B placement (20260420). This is a binary derivative, not official TruthfulQA MC1 or MC2. There is no question filtering based on outputs, no model selection based on results and no quantization. Both models see the same question IDs and answer order. Missing pairs are an incomplete experiment, not a zero effect.

## Measurements

The primary comparison uses the identical plain prompt for both checkpoints (`shared_plain_ab_label`). The sensitivity comparison uses each instruct tokenizer's native chat template while leaving the base prompt unchanged (`native_prompt_ab_label`). I will report these separately: the latter combines checkpoint and prompt effects. I will score the likelihoods of the continuations ` A` and ` B` in bf16 on one L4, normalizing over those two choices. These probabilities describe a restricted choice, not free-form truthfulness or a model's verbal confidence. Batched execution must agree with scalar execution on a fixed pilot subset; I will retain both log-likelihoods for every row.

Truthfulness is measured by binary accuracy. Calibration's primary summary is ten-bin equal-frequency expected calibration error (ECE). I will also report Brier score, negative log likelihood (NLL), mean confidence, and confidence minus accuracy using the existing definitions. Lower ECE, Brier and NLL are better; Brier and NLL also reflect discrimination, so they are checks, not pure calibration measures.

## Uncertainty and interpretation

For each pair and protocol I will resample the same question indices for base and instruct, with 10,000 paired bootstrap replicates, seed 20260420 and percentile 95% intervals. ECE binning is recomputed within each replicate. All deltas are instruct minus base. The intervals quantify question-sampling uncertainty on this task, not uncertainty over training seeds or model families. They are pointwise, not multiple-comparison-adjusted.

An accuracy interval wholly above zero and an ECE interval wholly above zero support a truthfulness/calibration tradeoff for that pair. Accuracy wholly above zero and ECE wholly below zero support improvement in both. If either relevant interval includes zero, I will call that direction uncertain. Other sign combinations will be reported directly rather than forced into a tradeoff label. I will show every pair, including negative and uncertain results, and describe the pattern across nominal sizes and families without fitting a size trend or treating these selected families as independent population samples.

The universal claim is unsupported unless every measured primary-protocol pair supports it. Any credible improvement in both is a counterexample within this design; inconclusive pairs are not proof of no effect. Native-template sensitivity can change that conclusion and must remain visible. Base/instruct differences include supervised tuning, preference optimization, data and training changes; this comparison cannot isolate reinforcement learning causally.

## Budget and outputs

I will first run a small L4 pilot, then size the full jobs to stay within $2.50 including failures and image/download overhead. No weights will be downloaded to the local machine. I will commit the full scores, exact configurations, dataset hashes, runtime/software metadata, paired intervals, summary table and SVG. Cloud durations will be labeled as cloud execution, not local benchmarks; the cost will be a conservative upper estimate, not a billing receipt.
