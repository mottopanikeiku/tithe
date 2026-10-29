# Next falsification

I compared the original two-label derivative with standard TruthfulQA answer-string likelihoods. Those measurements differ in question pool, candidate set, prompting and scored continuation, and some runs use different GPUs. The [saved-score reanalysis](../results/reanalysis) now fixes the questions and the two answers: on 779 matched questions the reversal remains. It cannot separate the prompt from the scored continuation, because each saved run used one of each.

The next useful check is a controlled format comparison, not another checkpoint size. I would use the same fixed binary questions and answer strings for every condition, score both checkpoints of a pair on the same GPU, and cross the prompt (A/B question versus Q/A primer) with the continuation (label versus whole answer). I would also reverse A/B placement and cross leading-space/bare-letter continuations. I would fix that matrix before scoring, retain every condition and average order-counterbalanced predictions only within each question.

I would keep paired question intervals and report accuracy, ECE, Brier, NLL and confidence separately. Answer-set probabilities are not unrestricted generation probabilities. The cached BF16 numerical failures also make an unchanged scalar reference important: a speed improvement is not useful if it changes the confidence measurement. No runs of this follow-up are included in the current results, and I have not measured its cost.

A separate question is where calibration changes along OLMo's SFT, DPO and RLVR checkpoint sequence. Comparing those stages would help locate changes in one training trajectory; it would still not isolate a causal RL effect without controlled data, objectives and repeated training seeds.
