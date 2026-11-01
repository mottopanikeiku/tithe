# Method

I compare nine base/instruct pairs: Qwen2.5 0.5/1.5/7/14/32B, OLMo-2 1/7B, SmolLM2 1.7B and Mistral 7B v0.3. [The model configuration](../configs/day_scale.json) specifies checkpoint and tokenizer revisions, precision and scoring settings.

## Questions and answers

The original A/B task uses all 790 questions in the pinned TruthfulQA CSV. For each question, `prepare-data` takes the Best Answer and Best Incorrect Answer and assigns them to A/B with seed 20260420. [`data.py`](../src/alignmenttax/data.py) specifies the source revision and checks the downloaded CSV's SHA256.

The standard task uses all 817 validation questions in `truthfulqa/truthful_qa`, `multiple_choice`. It keeps the dataset's answer order and uses the MC1/MC2 definitions from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/truthfulqa). Dataset and harness revisions are in the configuration.

For the matched comparison, I join questions by exact text, then find the two A/B answers among that question's MC1 candidates. Answer matching strips outer whitespace and terminal periods. Matches must be unique and agree on the correct answer. This gives 779 questions; 777 need the whitespace/period normalization. Three questions and eight answer pairs have no match. [The matching file](../results/reanalysis/matching.json) includes every match and exclusion. Matching does not change any scored text or likelihood.

## Scoring

The A/B prompt lists the question and its two answers. I score the continuations ` A` and ` B`, then normalize their likelihoods over those two choices.

The standard prompt is the harness's six-example Q/A primer. Each continuation is a space followed by the full answer; its score is the sum of token log probabilities, without length normalization. [`standard.py`](../src/alignmenttax/standard.py) implements the harness token-boundary and BOS rules. For the matched comparison, I take the saved likelihoods of the two matched answers and renormalize over those two.

`shared_plain_ab_label` means both checkpoints receive the same plain prompt within each scoring rule. `native_prompt_ab_label` applies the instruct model's chat template while leaving the base prompt plain. In the standard task, the chat template wraps the complete Q/A primer. Date-sensitive templates use 2026-10-07. The protocol names are shared across tasks, but standard rows contain full-answer likelihoods.

Weights are BF16, and label logits are normalized in float32. Scoring loads one checkpoint at a time. A/B scoring uses scalar forwards. Standard scoring can reuse a question's prefix cache across candidate answers, subject to the numerical checks below. Each score row includes the model revision, prompt format, device and dtype. Resume skips completed question/model/protocol keys; synthetic test rows are marked `fake`.

## Metrics and intervals

- Accuracy is the fraction of highest-probability choices that are correct. Confidence is the highest probability after normalization over the offered choices.
- ECE uses ten equal-frequency confidence bins. It is the sample-weighted absolute difference between accuracy and confidence within each bin. Bins are rebuilt in each bootstrap draw.
- The reanalysis uses categorical Brier score, `sum((p - y)^2)`, and unclipped NLL in nats. For two choices, categorical Brier is twice the earlier binary score `(1 - p_correct)^2`. The original binary summaries also clip `p_correct` at `1e-12` for NLL.
- MC1 is best-answer accuracy across all of a question's MC1 candidates. MC2 is the total probability assigned to true answers in its separate candidate set. MC2 has multiple true targets, so the analysis does not calculate MC2 ECE.

All changes are instruct minus base. I use 10,000 paired question resamples with NumPy PCG64 seed 20260420 and percentile 95% intervals. Each draw is shared across models, pairs, protocols and metrics within a task. The matched comparison also shares draws across labels and strings, including their difference of changes. A direction counts only when its interval excludes zero. These are pointwise intervals; the 790- and 817-question tasks are analyzed separately.

[The matched comparison CSV](../results/reanalysis/candidate_control.csv) includes accuracy, confidence, ECE, Brier and NLL for both scoring rules, both prompt protocols and the paired contrasts. [Reliability diagrams](../results/reanalysis/reliability_matched_strings_shared.svg) and [binning comparisons](../results/reanalysis/ece_sensitivity.csv) show the confidence distribution and ECE sensitivity.

## Standard TruthfulQA results

These use all 817 questions and all candidate answers with shared plain prompts. Values are instruct minus base, in percentage points, with 95% intervals. Lower ECE is better. [Full results](../results/day_scale/analysis/summary.csv).

| Pair | ΔMC1 [95% CI] | ΔMC2 [95% CI] | ΔMC1 ECE [95% CI] |
|---|---:|---:|---:|
| [Mistral-7B-v0.3](../results/day_scale/mistral_7b_v0_3) | +14.32 [+11.63, +17.14] | +17.60 [+15.27, +19.95] | -6.66 [-9.56, -3.73] |
| [OLMo-2-1B](../results/day_scale/olmo2_0425_1b) | +8.45 [+5.75, +11.02] | +11.43 [+8.91, +13.92] | -1.63 [-4.48, +1.33] |
| [OLMo-2-7B](../results/day_scale/olmo2_1124_7b) | +10.16 [+7.47, +12.85] | +11.18 [+8.79, +13.57] | -1.43 [-4.26, +1.42] |
| [Qwen2.5-0.5B](../results/day_scale/qwen2_5_0_5b) | +1.47 [-0.37, +3.30] | +2.11 [+0.82, +3.43] | -0.73 [-2.69, +1.21] |
| [Qwen2.5-1.5B](../results/day_scale/qwen2_5_1_5b) | +0.49 [-0.86, +1.72] | +0.24 [-0.70, +1.16] | +1.03 [-0.49, +2.61] |
| [Qwen2.5-7B](../results/day_scale/qwen2_5_7b) | +9.18 [+6.73, +11.75] | +8.45 [+6.48, +10.44] | -3.31 [-6.00, -0.61] |
| [Qwen2.5-14B](../results/day_scale/qwen2_5_14b) | +11.38 [+8.69, +14.08] | +10.57 [+8.74, +12.42] | -5.47 [-8.15, -2.80] |
| [Qwen2.5-32B](../results/day_scale/qwen2_5_32b) | +8.20 [+5.75, +10.65] | +7.89 [+6.08, +9.69] | -3.48 [-6.08, -0.89] |
| [SmolLM2-1.7B](../results/day_scale/smollm2_1_7b) | +1.47 [-0.37, +3.18] | +3.35 [+2.04, +4.65] | -0.35 [-2.41, +1.73] |

## Temperature scaling and diagnostics

The reanalysis splits questions by a hash of their exact text and seed, using half for calibration and half for testing. The split is shared across models, protocols and tasks. A separate positive temperature for each model minimizes calibration-set NLL within `[0.01, 200]`; metrics are then evaluated on the test questions. Temperature changes confidence while preserving the predicted answer.

There are 65 upper-bound fits among 144 fits, including every base-model fit on matched answer strings. [Temperatures](../results/reanalysis/temperatures.csv) include boundary flags. [Held-out estimates](../results/reanalysis/held_out.csv) compare raw and scaled models with uniform and calibration-fitted constant predictors. MC1 candidates are hash-permuted for those baselines because the dataset puts the correct answer first. Test-question intervals keep the split, temperatures and priors fixed.

The additional diagnostics include [A/B prediction rates](../results/reanalysis/label_diagnostics.csv), equal-width and equal-frequency ECE with 5–50 bins, and [risk–coverage curves](../results/reanalysis/risk_coverage_matched_strings_shared.svg). Risk–coverage curves rank predictions by confidence and measure the error rate as more questions are accepted; ties use random ordering. Their areas are point estimates without intervals. MC2 results transfer each model's fitted MC1 temperature to its MC2 answer likelihoods.

## How the scores were collected

The first seven A/B pairs ran on an L4 using [`modal_app.py`](../modal_app.py). An eight-question Qwen2.5-0.5B pilot found a maximum scalar/batch log-likelihood difference of 0.107 nats, above the 0.08 tolerance, so the full A/B runs used scalar forwards. [Pilot measurements](../results/pilot/batch_parity_failure.json).

The standard tasks and the two larger Qwen pairs used [`tools/day_modal_app.py`](../tools/day_modal_app.py). Its `download` function stages weights on CPU before offline GPU scoring. Jobs used H100 and L4 GPUs, without quantization or CPU offload. OLMo-2-7B has an H100 base run and an L4 instruct run. Per-pair runtime files identify the hardware and package versions.

Standard prefix-cache scoring was compared with scalar full forwards on the first four questions of each model/protocol. The limits were 0.05 nats per continuation token, 0.02 in answer probability or MC2 mass, and no changed MC1 winner. Of 27 checks, 21 failed and used full forwards; six used the cache. [Numerical comparisons](../results/day_scale/numerical_audit.json) include both paths. These four-question checks do not cover numerical error across the full dataset.

The saved runs contain 29,412 standard score rows and 28,440 A/B score rows across both protocols. Base-native rows reuse the unchanged base prompt. The [compute estimate](../results/day_scale/cost.json) is $6.12 including failed attempts, CPU/memory and a ten-percent allowance; storage and network charges are excluded. The [earlier incomplete CPU run](../results/qwen2_5_0_5b) is separate from these results.

The [README command](../README.md#rebuild-from-saved-scores) rebuilds the analysis without loading any models. Raw scores, dataset attribution and licenses remain under [`results`](../results).
