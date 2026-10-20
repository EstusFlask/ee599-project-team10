# RL defense experiments

This folder isolates the EE599 reinforcement-learning experiments from the
communication-layer defense system in `core/`. The simulator is deliberately
partially observable: policies never see the hidden attack regime or exact
compromised-channel count.

## Action and observation design

- 12 actions: redundancy `k in {1,3,5,7}` crossed with `none`, `retry`, or
  `rotate-and-retry` recovery.
- Noisy pre-transmission alerts, message length/importance, remaining budget,
  normalized budget capacity, prior action/outcome, and rolling history are
  observable.
- Attacks have quiet/probing/burst regimes, persistent compromise, adaptive
  pressure, detector errors, correlated workloads, and optional regime shifts.
- Cost reflects two full-text carriers plus short hash witnesses. Recovery is
  charged only after an initial quorum failure.

## Compared policy families

- Non-RL: all 12 fixed action combinations, a risk heuristic, and a
  conservative rule policy.
- Value RL: DQN and distributional QR-DQN.
- Actor-critic RL: A2C and PPO.
- Memory-based RL: recurrent PPO with an LSTM policy.
- Robustness variants: TRPO, MaskablePPO, and wide domain randomization.
- Initial RL baseline: the project's original linear Q learner, adapted to the
  same noisy observations and rewards.

Training uses domain randomization. Evaluation uses common random seeds on six
named scenarios at three budgets. Current reports use mean/min/max only; final
confidence intervals are intentionally deferred.

```powershell
conda activate ee599-rl
pytest experiment/rl/tests -q
python -m experiment.rl.train_agents --steps 200000
python -m experiment.rl.train_agents --algorithms ppo dqn qrdqn --steps 400000 --resume --output experiment/results/rl_v2/models
python -m experiment.rl.evaluate_agents
python -m experiment.rl.stress_test
python -m experiment.rl.analyze_results
python -m experiment.rl.analyze_stress

# Recommended robustness run
python -m experiment.rl.train_agents --algorithms qrdqn --steps 300000 `
  --training-profile robust --constraint-weight 0.15 `
  --output experiment/results/rl_robust_w015/models
python -m experiment.rl.train_legacy_q --steps 300000 `
  --training-profile robust --constraint-weight 0.15 `
  --output experiment/results/rl_robust_w015/models
python -m experiment.rl.evaluate_agents `
  --model-dir experiment/results/rl_robust_w015/models `
  --output experiment/results/rl_robust_w015/legacy_comparison `
  --algorithms qrdqn legacy_q
python -m experiment.rl.stress_test `
  --model-dir experiment/results/rl_robust_w015/models `
  --output experiment/results/rl_robust_w015 `
  --algorithms qrdqn

# Complex-scenario budget versus defense-success figure. Supervised ML uses
# dashed lines, RL uses solid lines, and fixed/rule baselines use dotted lines.
python -m experiment.rl.plot_budget_success_comparison

# Original-style success-cost curves plus a matched-cost bar chart. Operating
# points use measured communication cost; ordinary ML is drawn with dashed lines.
python -m experiment.rl.plot_matched_cost_comparison

# Full version: scan budget/gating operating points, build each method's
# measured Pareto frontier, then place every curve on the same 0.60-1.10 grid.
python -m experiment.rl.plot_unified_cost_comparison

# Expanded traditional-ML benchmark. Every ML model and every RL model uses
# the identical complete 25-dimensional observation at inference time.
python -m experiment.rl.benchmark_ml_methods

# Attacker-process audit for the same five complex scenarios and seeds.
python -m experiment.rl.plot_attacker_dynamics

# Readable probe-lull-burst example selected from the actual evaluation traces.
python -m experiment.rl.plot_attack_pattern_example

# Canonical probe-lull-burst mechanism figure using exact regime probabilities.
python -m experiment.rl.plot_attack_stage_profile

# Train and evaluate a policy on 1-5 independent attacker processes.
python -m experiment.rl.train_agents --algorithms qrdqn --steps 300000 `
  --training-profile multi_attacker --constraint-weight 0.15 `
  --output experiment/results/rl_multi_attacker/models
python -m experiment.rl.benchmark_multi_attacker

# Lower-rate adaptation from the original robust policy avoids catastrophic forgetting.
python -m experiment.rl.fine_tune_multi_attacker

# Three-stage curriculum, independent checkpoint selection, and reliability tuning.
python -m experiment.rl.train_curriculum_multi_attacker
python -m experiment.rl.select_multi_attacker_model
python -m experiment.rl.fine_tune_reliability_multi_attacker
python -m experiment.rl.benchmark_multi_attacker `
  --output experiment/results/rl_multi_attacker_final/analysis/comprehensive_benchmark.png `
  --seeds 2101 2202 2303 2404 2505

# Required fair comparison: interpolate every method on the same measured-cost grid.
python -m experiment.rl.benchmark_multi_attacker_matched_cost
```

The eight supervised ML methods can also be evaluated on the exact same
multi-attacker scenario, seed, budget, and activation-rate grid. The following
commands cache models and measured results under the Git-ignored
`experiment/results/rl_multi_attacker_final/analysis/ml_extension/` directory,
then generate three PNG comparisons under `experiment/ignore/all/`:

```powershell
python -m experiment.rl.benchmark_ml_multi_attacker --workers 2
python -m experiment.rl.plot_progress_success_cost
```

The plots highlight the highest ML and RL point estimates at average cost 1.0.
For unknown attacker count, working points are pooled across equally weighted
counts 1, 2, 3, and 5 before constructing the cost frontier.
