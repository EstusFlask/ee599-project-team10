# RL Experiment Status

## Completed

- Matched-cost comparisons of budget-aware QR-DQN, history ML, and strong fixed policies.
- Multiple attackers and continuous changes, with separate training, validation, and test seeds.
- Paired hierarchical bootstrap 95% CIs for success rates and key differences.
- Two retrained ablations: no history features and gamma=0, each with 3 seeds and 100,000 steps per seed.
- Reused the existing original-budget comparison; no additional algorithm search or tuning.

## Results for the report

At the same mean normalized communication cost of 0.75:

| Method | Weighted defense success |
|---|---:|
| Full RL | 38.15% |
| No history features | 37.86% |
| gamma=0 | 38.00% |
| Original budget mechanism | 31.14% |
| Strong fixed baseline | 39.32% |

- New budget mechanism versus original RL: +7.01 percentage points, with a 95% difference CI of [+5.10, +9.13].
- Full model versus no history: +0.28 percentage points, CI [-2.65, +3.46]; an additional history benefit was not established.
- Full model versus gamma=0: +0.14 percentage points; an additional benefit from long-term TD returns was not established.
  Only 2788/3000 bootstrap replicates covered the target cost; this interval is exploratory.
- Superiority over the strong fixed policy was not established. The results do not support a claim that RL is universally better.

## Remaining work

The essential RL implementation and experiment materials are complete. Include
the method, main results, ablations, and limitations in the final report, and
distinguish them from the team's communication-layer and real-agent experiments.
These numerical simulations measure message integrity, not real LLM task accuracy.
Do not mix them with milestone results from the old environment. Adaptive methods
lack complete coverage at cost 1.0; no extrapolation is performed.

Detailed results:

- [Main experiment and CIs](../results/rl_effect_check/cost_0.75/EFFECT_REPORT.md)
- [Minimal ablations and CIs](../results/rl_effect_check/ablations/ABLATION_REPORT.md)
- Reproduce: `python -m experiment.rl.run_ablations --workers 3`
