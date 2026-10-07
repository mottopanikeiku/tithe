# Larger checkpoints and standard TruthfulQA scores

I am extending the seven-pair comparison to larger checkpoints and the standard answer-string task. I will commit this plan before preparing new score results. I will keep the previous binary results unchanged rather than mixing old and new measurements without explaining them.

## Checkpoints and priorities

The required additions are Qwen2.5-14B and Qwen2.5-32B, each with its matching Instruct checkpoint. Together with Qwen2.5 0.5B, 1.5B and 7B, OLMo-2 1B and 7B, SmolLM2 1.7B and Mistral 7B, this gives nine pairs in four families. The exact repository revisions are in [the manifest](../configs/day_scale.json). I checked Hugging Face model metadata: both new Qwen pairs are ungated and Apache-2.0 licensed; each model has its own license-source link in that file. No weights are included in this repository.

If the measured throughput and remaining budget permit complete additional pairs, I will next add OLMo-2-0325-32B, then Mistral-Small-24B-Base-2501 / Instruct-2501. All four optional checkpoints are also ungated and Apache-2.0 licensed at the listed revisions. A pair is included in the summary only when both checkpoints have all questions and both protocols. I will not add a smaller subset of questions to make an optional checkpoint fit.

## Two tasks, kept separate

1. **Binary derivative:** the same 790 pinned CSV questions, seeded A/B ordering, exact `" A"` and `" B"` continuations, shared plain prompt and native instruct-chat/base-plain sensitivity as the previous comparison. The seven existing pairs retain their L4 measurements. New pairs run on an H100 in BF16, with scalar label scoring so this extension does not silently change the previous batch-size decision. Accuracy, confidence, overconfidence gap, Brier, NLL and 10-bin equal-frequency ECE are reported.
2. **Standard multiple choice:** all 817 validation questions from `truthfulqa/truthful_qa`, `multiple_choice`, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`. Every required pair is scored afresh on an H100 in BF16. The shared prompt is the exact six-example Q/A string from the [harness MC1 definition](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/truthfulqa/truthfulqa_mc1.yaml), despite its `num_fewshot: 0` setting. Answer continuations have the harness space delimiter, and likelihood is the sum of token log probabilities, not a length-normalized value. Token boundaries and BOS defaults follow the pinned harness tokenizer implementation. MC1 is whether the highest-likelihood answer is the best true answer. [MC2](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/truthfulqa/utils.py) is the normalized probability mass assigned to all true answers; I use a stable softmax rather than directly exponentiating very negative likelihoods.

The standard native sensitivity wraps that same complete harness Q/A prompt in the instruct checkpoint's user chat template, with an assistant generation prefix. Base checkpoints remain plain. These sensitivity scores are not the harness's default prompt setting. The inherited protocol identifiers are retained for pairing, but standard rows explicitly contain answer-string likelihood arrays rather than A/B label likelihoods.

I will fix date-sensitive native templates to 2026-10-07 for both the standard and new binary scores. Mistral Small's pinned template otherwise inserts the wall-clock date into its default system message. The configured date will be recorded with each run and included in the identity checked before reusing a completed cloud unit.

For standard MC1, I additionally report maximum answer-set probability, the overconfidence gap, equal-frequency ECE, NLL of the best answer and multiclass Brier `sum((p - y)^2)`. This Brier definition is not the binary derivative's one-probability Brier. MC2 is a truth-mass metric, not a hard classification accuracy; I will not reuse MC1 confidence to claim MC2 calibration.

## Computation and numerical checks

I will download pinned weights into a persistent cloud volume from a CPU container before starting the H100; GPU scoring must be offline. Checkpoints load sequentially without quantization or CPU offload. Standard answer scoring reuses a question's prefix KV cache and deduplicates identical answer strings across MC1 and MC2. It projects only needed prediction positions to vocabulary logits in small chunks.

I will test the cached method against an independent full-forward FP32 toy reference, including variable question/answer lengths, cache expansion, padding and token boundaries. On the first four questions of each real model/protocol I will save cached and full-forward likelihoods and derived scores. The planned BF16 audit limits are 0.05 maximum absolute difference in mean token log likelihood (summed-likelihood difference divided by continuation length), 0.02 maximum answer-probability or MC2-mass difference, and no MC1 winner changes. These are numerical checks on fixed examples, not a proof for every question. If they fail, I will retain the failure and use the full-forward reference path for that model, not increase the threshold. The remaining budget must still support the full task; otherwise I will state the missing prerequisite rather than present incomplete scores as a complete comparison.

For the cached side of this audit I will use the production question-batch size: 16 for smaller checkpoints and four for 14B and above. I will compare the first four questions with scalar full forwards; the additional questions only set the batch shapes and are not selected based on their scores.

The additional compute cap is $2.00. A roughly 25-minute H100 job plus cheap CPU preparation is the starting budget estimate, not a measured result. Optional pairs require enough projected time for their full tasks with 20% headroom; required pairs take priority. Runtime and conservative cost estimates, including unsuccessful attempts, will be saved separately from model metrics.

## Analysis and answer

I will use 10,000 question-paired bootstrap resamples, PCG64 seed 20260420, with 95% percentile intervals. Each draw is shared across base/instruct, metrics, protocols and checkpoint pairs within a task. ECE is recomputed inside every draw with stable confidence ordering and 10 equal-frequency bins. The 790-question binary task and 817-question standard task are never pooled.

I will show base/instruct values and instruct-minus-base intervals for every metric. The headline counts will distinguish resolved truthfulness gains with worse MC1 ECE, simultaneous improvements, simultaneous worsening and uncertain comparisons. MC1 and MC2 will have separate gain counts. A sign is resolved only if its interval excludes zero. These are pointwise intervals with no multiplicity adjustment, for fixed checkpoints and one benchmark—not a causal estimate of reinforcement learning or a claim about model families in general. I will report any disagreement between ECE and proper scoring rules instead of choosing the friendliest metric.

## Execution change after the first H100 run

The first H100 function reached its 24-minute limit after four complete pairs and the OLMo-2-7B base checkpoint. Eleven of the twelve audits in the four complete pairs rejected cached BF16 scoring, so I kept the predeclared scalar full-forward results for those conditions rather than changing the limits. Completed model units are saved before the next checkpoint starts.

My additional compute cap is now $4.00, including that attempt. I will run the remaining checkpoints of 7B and below on an L4 and reserve the H100 for the required 14B and 32B pairs. I will not score the optional OLMo-32B or Mistral-24B pairs. Runtime files record each model's device; the OLMo-7B comparison consequently uses an H100 base result and an L4 instruct result, adding a hardware difference to its limitations. Scoring formulas, BF16 precision and numerical limits are unchanged.

The larger-checkpoint run completed the full Qwen-14B pair and Qwen-32B base, then reached its 19-minute function limit while loading Qwen-32B-Instruct. I have one final H100 booking, at most 20 minutes, for that instruct checkpoint after the L4 work ends; the additional compute cap is $5.50 including all attempts. If a required pair remains missing, analysis requires an explicit `--allow-incomplete` flag and names the omitted pair. That flag never permits fewer questions or incomplete model/protocol groups within an included pair.

The L4 function also reached its 24-minute limit after finishing OLMo-7B-Instruct and the SmolLM2 pair; Mistral-7B remained unfinished. I have one final L4 booking of at most 25 minutes for the full Mistral standard pair, after the Qwen-32B run stops. The combined compute cap is $7.00 including the earlier binary study and all attempts. Mistral's existing binary rows are reused; I will not replace an incomplete standard pair with a shorter question set.
