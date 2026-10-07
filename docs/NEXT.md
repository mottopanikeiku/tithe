# Next falsification

I now compare the original two-label derivative with standard TruthfulQA answer-string likelihoods. Those measurements differ in question pool, candidate set and prompting, and some runs use different GPUs. A difference between their results does not by itself identify which change caused it.

The next useful check is a controlled format comparison, not another checkpoint size. I would use the same fixed binary questions and answer strings for every condition, score both checkpoints of a pair on the same GPU, and compare two-label scoring with whole-answer likelihood scoring. I would also reverse A/B placement and cross leading-space/bare-letter continuations. I would fix that matrix before scoring, retain every condition and average order-counterbalanced predictions only within each question.

I would keep paired question intervals and report accuracy, ECE, Brier, NLL and confidence separately. Answer-set probabilities are not unrestricted generation probabilities. The cached BF16 numerical failures also make an unchanged scalar reference important: a speed improvement is not useful if it changes the confidence measurement. No runs of this follow-up are included in the current results, and I have not measured its cost.

A separate question is where calibration changes along OLMo's SFT, DPO and RLVR checkpoint sequence. Comparing those stages would help locate changes in one training trajectory; it would still not isolate a causal RL effect without controlled data, objectives and repeated training seeds.
