# tithe

Does instruction tuning hurt calibration? It depends on what you score.

![Calibration changes across nine base/instruct pairs on the same 779 questions: A/B-label scoring mostly worsens ECE, while full-answer scoring mostly improves it](results/reanalysis/candidate_control_shared.svg)

I compared nine base/instruct pairs on the same 779 TruthfulQA questions and the same two answers. Scoring the A/B label, instruction tuning worsens expected calibration error (ECE) in 8 of 9 pairs; scoring the two full answers, it improves ECE in 6 and worsens it in none. These counts use 95% paired-bootstrap intervals that exclude zero.

| Pair | A/B-label ΔECE | Two-answer ΔECE |
|---|---:|---:|
| [Mistral-7B-v0.3](results/day_scale/mistral_7b_v0_3) | +9.0* | -13.4* |
| [OLMo-2-1B](results/day_scale/olmo2_0425_1b) | +12.5* | -9.7* |
| [OLMo-2-7B](results/day_scale/olmo2_1124_7b) | +10.9* | -9.1* |
| [Qwen2.5-0.5B](results/day_scale/qwen2_5_0_5b) | -2.8 | +0.1 |
| [Qwen2.5-1.5B](results/day_scale/qwen2_5_1_5b) | +8.0* | -1.0 |
| [Qwen2.5-7B](results/day_scale/qwen2_5_7b) | +7.2* | -7.0* |
| [Qwen2.5-14B](results/day_scale/qwen2_5_14b) | +10.1* | -9.7* |
| [Qwen2.5-32B](results/day_scale/qwen2_5_32b) | +7.4* | -8.6* |
| [SmolLM2-1.7B](results/day_scale/smollm2_1_7b) | +32.2* | -0.7 |

Change is instruct minus base, in percentage points; positive ΔECE is worse. Both checkpoints use plain prompts. A star marks a 95% interval excluding zero, from 10,000 paired question resamples. [Exact values and intervals](results/reanalysis/candidate_control.csv).

## What's going on

Full-answer confidence starts very high: the base models average 92–94% confidence at only 28–46% accuracy. Instruction tuning moves that confidence by at most 3.5 points while accuracy rises by up to 16 points. Label confidence starts lower, at 61–83%, and usually rises much faster than accuracy. The two scoring rules start from very different kinds of overconfidence.

The comparison uses saved scores from a 790-question A/B task and the 817-question standard TruthfulQA task. I match questions and answer pairs, then normalize each scoring rule over those same two choices. The [method](docs/method.md) covers matching, prompts, metrics, the full standard-task results and temperature scaling.

## Rebuild from saved scores

From the repo root, with [uv](https://docs.astral.sh/uv/) and only the saved scores:

```sh
uv sync --locked
uv run python -m alignmenttax.reanalysis
git diff --exit-code -- results
```

This rebuilds `results/reanalysis` and `results/day_scale/analysis`. The final command checks that the outputs match the saved results.

Fast tests:

```sh
uv run --extra dev pytest -q tests
```

Twelve model-scorer tests skip without PyTorch and Transformers; CI runs them too. The CLI is `tithe`; the Python package remains `alignmenttax`.

## Limitations

- The matched questions and answers still use different prompts and scored text: an A/B prompt with a letter continuation versus a six-example Q/A prompt with a full answer. This comparison cannot separate their effects.
- Base/instruct checkpoints differ in training data and objectives. Their differences do not isolate a causal effect of instruction tuning or reinforcement learning.
- Confidence is normalized over the offered choices. It does not measure confidence in a generated answer.
- Intervals resample questions for fixed checkpoints, with no multiple-comparison correction. Training seeds and numerical variation are outside them; some runs also used different GPUs.
- ECE depends on binning. Brier score, NLL, alternative bins and held-out temperature fits are available in the [analysis](results/reanalysis); temperature fits often reach the upper bound of 200.

## Related work

[Kadavath et al.](https://arxiv.org/abs/2207.05221), the [GPT-4 report](https://arxiv.org/abs/2303.08774) and [Huang, Lu and Zeng](https://arxiv.org/abs/2508.00264) study calibration after post-training. Here, holding the questions and candidate answers fixed exposes how much the measurement depends on the scoring setup.
[TruthfulQA](https://arxiv.org/abs/2109.07958) is by Lin, Hilton and Evans; temperature scaling follows [Guo et al.](https://arxiv.org/abs/1706.04599). [Dataset and code attribution](results/NOTICE.txt).
