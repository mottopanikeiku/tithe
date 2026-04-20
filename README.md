# Alignment Tax

This project runs a reproducible TruthfulQA binary multiple-choice experiment to compare a matched Qwen2.5 base/instruct pair:

- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-1.5B-Instruct`

The intended interpretation is cautious: this is a post-training or instruction-tuning-associated comparison, not a clean causal isolation of reinforcement learning.

## Quickstart

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Prepare the current upstream TruthfulQA binary dataset:

```powershell
alignmenttax prepare-data --out data/processed/truthfulqa_binary.jsonl --seed 20260420
```

Score the Qwen pair locally:

```powershell
alignmenttax score --config configs/qwen2_5_1_5b.yaml --out runs/qwen2_5_1_5b/scores.jsonl
```

Analyze the paired results:

```powershell
alignmenttax analyze --run runs/qwen2_5_1_5b --out reports/qwen2_5_1_5b
```

For an offline smoke run that does not download models:

```powershell
alignmenttax score --config configs/qwen2_5_1_5b.yaml --out runs/fake/scores.jsonl --fake --limit 8
alignmenttax analyze --run runs/fake --out reports/fake --bootstrap-iterations 100 --no-plots
```

## Outputs

`prepare-data` writes one JSONL row per TruthfulQA question with randomized `A/B` order, correct label, category/type metadata, and source provenance.

`score` writes one JSONL row per model, prompt protocol, and question. Rows include restricted `A/B` probabilities, raw label log probabilities, normalized label log probabilities, predicted label, correctness, confidence, `p_correct`, device/dtype metadata, and timing.

`analyze` writes:

- `summary.csv`
- `paired_bootstrap.json`
- `calibration_tables.csv`
- `category_breakdown.csv`
- reliability plots and confidence histograms when `matplotlib` is available and plots are enabled

## Sources

- Qwen base model card: https://huggingface.co/Qwen/Qwen2.5-1.5B
- Qwen instruct model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- TruthfulQA upstream repository: https://github.com/sylinrl/TruthfulQA
- TruthfulQA publication summary: https://openai.com/index/truthfulqa/

