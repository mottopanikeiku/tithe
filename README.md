# Alignment Tax

Alignment Tax compares truthfulness and confidence in matched [Qwen](https://huggingface.co/Qwen) base/instruct models using a binary derivative of [TruthfulQA](https://github.com/sylinrl/TruthfulQA).

Does instruction tuning change accuracy or calibration when both models choose between the same truthful and false answers?

[scoring.py](src/alignmenttax/scoring.py) evaluates A/B label likelihoods under shared plain prompts and model-native prompts. [metrics.py](src/alignmenttax/metrics.py) computes paired bootstrap differences; [calibration_stage.py](src/alignmenttax/calibration_stage.py) fits temperatures on separate calibration questions and evaluates held-out questions.

**Result: no completed paired comparison yet.** The real CPU attempt used **Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct**, not the original 1.5B pair. It stopped during base-model scoring at the memory safety threshold. The saved partial scores do not support a claim about instruction tuning.

## What the local run established

| Observation | Recorded value |
|---|---:|
| Prepared TruthfulQA questions | 790 |
| Completed rows | 118, base model only |
| Sampled peak process RSS | 1.450 GB |
| Elapsed time before stop | 56.82 seconds |
| Paired analysis / held-out calibration | Not run |

Sources: [attempt summary](results/qwen2_5_0_5b/attempt_summary.json), [runtime measurements](results/qwen2_5_0_5b/runtime.json), and [partial scores](results/qwen2_5_0_5b/scores.jsonl). These rows are execution evidence, not a performance comparison. No 1.5B results are published.

The CPU runner uses bf16, loads one model at a time, and runs at low priority. It stopped at the 1.45 GB safety threshold, below the 1.5 GB process budget; this does not measure the memory needed to finish. The proposed next step is the unchanged experiment on an existing CPU machine with a larger approved memory allowance. See the [protocol and next-run proposal](docs/PROTOCOL.md#local-execution).

## Reproduce

Linux CPU, no GPU or paid service; public downloads are required. The exact attempted environment is saved below. The memory-limited command reproduces the attempt, **not a promise of a completed comparison**. It checks available RAM and stops on sampled RSS or wall time; analysis runs only if scoring completes.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r results/qwen2_5_0_5b/environment.txt -e .
nice -n 19 .venv/bin/alignmenttax prepare-data --out data/processed/truthfulqa_binary.jsonl
nice -n 19 .venv/bin/python tools/run_cpu.py && nice -n 19 .venv/bin/alignmenttax analyze --run results/qwen2_5_0_5b --out results/qwen2_5_0_5b && nice -n 19 .venv/bin/alignmenttax calibrate --run results/qwen2_5_0_5b --out results/qwen2_5_0_5b
```

Rerunning overwrites the local attempt outputs. Both model configurations pin weight/tokenizer revisions. Dataset records identify the exact CSV hash; unverified local caches do not claim an upstream commit. Completed-row resume skips inference, and rows record their actual prompt format; chat-template errors no longer silently become plain prompts.

## Limitations

- This is a post-training-associated comparison, **not a clean causal isolation of reinforcement learning**.
- One small model pair and one task cannot establish a general alignment tax; the local pair is incomplete.
- Binary Best Answer versus Best Incorrect Answer is not the official TruthfulQA MC1/MC2 evaluation or free-form truthfulness.
- Confidence is normalized over A/B labels, not over all possible model answers; native prompts also change formatting.
- Calibration depends on the split and binning. A scalar temperature cannot repair wrong choices.

## Prior work

The models are Alibaba Qwen's [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and [Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), used through Hugging Face Transformers. TruthfulQA comes from [Lin, Hilton and Evans](https://arxiv.org/abs/2109.07958). Temperature scaling follows the approach of [Guo et al.](https://arxiv.org/abs/1706.04599).
