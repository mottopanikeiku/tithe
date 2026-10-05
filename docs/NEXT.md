# Next experiment

The local 0.5B bf16 attempt stopped at the sampled RSS safety threshold before finishing the base model. `results/qwen2_5_0_5b/runtime.json` records 1,450,061,824 bytes observed RSS after 56.82 seconds; `attempt_summary.json` records 118 base-only score rows. This is not evidence about the instruct-minus-base difference. No paired analysis or temperature scaling was run.

The most useful next step is to finish the same pinned 0.5B pair, unchanged, on an existing CPU machine with at least 4 GB available memory and permission for a larger per-process RSS budget. Keep one model loaded at a time and use the same bf16 weights, two prompt protocols, randomized binary dataset, paired bootstrap, and held-out temperature scaling. Start a new run rather than treating this interrupted output as a completed experiment.

Estimated paid cost: $0 on an existing machine. Estimated wall time: 15–60 minutes for scoring, plus several minutes for bootstrap and calibration, with a two-hour stop limit. These are planning estimates, not measured full-run results: the only measured execution is the partial attempt above. Budget 3 GB per scoring process as an unverified allowance; measure peak RSS before trusting it. Public model downloads require roughly 2 GB of disk for the pair, plus environment/cache storage; model weight sizes are about 988 MB each according to the upstream safetensors metadata.

Owner decision: approve a different existing machine or a larger memory allowance before rerunning. The current 1.5 GB process limit remains in effect. Quantization or a smaller architecture would be a different experiment and should not be silently substituted for this bf16 pair. The original 1.5B pair must not be loaded on this laptop.

After a completed paired run, commit the full scores, run metadata, summary, paired bootstrap intervals, calibration tables, held-out scaling summary and figures. The README should then state the measured accuracy and calibration differences, including negative or null results, rather than claim an alignment tax in advance.
