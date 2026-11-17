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

Training uses domain randomization. Evaluation uses common random seeds.
The older exploratory reports use mean/min/max; `verify_effect.py` and the
minimal ablations below now include paired hierarchical bootstrap 95% intervals
over training and test seeds.

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

## Lightweight continuation

Keep the existing QR-DQN and baselines. There are no new policy families or
framework dependencies. The small module split is:

- `config.py`: scenarios, reward coefficients, and optional within-episode changes;
- `environment.py`: attack/workload dynamics, observations, transitions, and costs;
- `budget.py`: an optional training callback that adjusts one cost penalty;
- `train_agents.py` / `evaluate_agents.py`: the existing entry points.

Existing imports from `environment` still work. All methods retain the same
25-dimensional policy observation and 12 actions.

Rotation now updates the persistent compromised count/mask, so repairing a
channel can also reduce risk in later rounds. Fresh attacks may compromise it
again. Plain retries do not repair the persistent state. This is environment
version 2: previous tables and saved policies describe the previous dynamics;
rerun every compared method before drawing a new performance conclusion.

Optional adaptive budget training:

```bash
python -m experiment.rl.train_agents --algorithms qrdqn --steps 100000 \
  --training-profile multi_attacker --adaptive-budget --constraint-weight 0.15 \
  --output experiment/results/rl_adaptive_budget/models
```

The shared penalty is updated after completed episodes as
`weight = clip(weight + learning_rate * (mean_cost - target_cost), 0, 5)`.
Training subtracts `weight * (round_cost - target_cost)` from the usual reward.
The target is the episode's available communication budget divided by its
length; `--budget-learning-rate` defaults to `0.005`. Updates start after the
algorithm's random-exploration warmup (`learning_starts`, 8,000 steps for QR-DQN).
`--budget-warmup-steps` can override this explicitly. The final weight and its
history are saved beside the model in `.budget.json` and restored on resume.
Without `--adaptive-budget`, the original reserve-based penalty remains active.
Evaluation always uses the standard reward for every policy and does not tune
the penalty. This is a soft constraint, not a guarantee of zero overspending;
continue reporting measured costs and budget violations.

A small continuous-run test uses the existing evaluator:

```bash
python -m experiment.rl.evaluate_agents --algorithms qrdqn \
  --model-dir experiment/results/rl_adaptive_budget/models \
  --scenarios adaptive --budgets 1.0 --continuous --save-traces \
  --output experiment/results/rl_adaptive_budget/evaluation
```

The run has 1, then 3, then 1 active attackers. During the middle third, the
future budget allowance drops by 25% and detector misses increase by 0.15.
The last third restores both. History and money already spent are retained;
the model cannot see attacker count or the schedule. These are declared stress
settings, not a claim about a real attack distribution. This held-out test does
not by itself train online adaptation. Its scores are not a matched-cost ranking.
`continuous/traces.csv` records each round's success, chosen action, cost, and
remaining budget for checking the response to each change.

Run the focused checks with `python -m pytest experiment/rl/tests -q`.

### Actual effectiveness check

```bash
python -m experiment.rl.verify_effect --phase all --steps 100000 --workers 3
```

This trains original-penalty and adaptive-penalty QR-DQN with three seeds each,
including both the unsuccessful fast update and the slower post-warmup update,
plus SVM/Random Forest baselines with the same 25 features and 100,000 training
samples per seed. All methods run in the same corrected environment. Validation
selects operating points, including inexpensive fixed strategies; those points
are frozen before 12 independent test seeds across six scenarios are evaluated.
At average communication cost 1.0, test interpolation uses only the frozen
points' costs, not their success rates to choose the winner. Fixed/randomized
baseline portfolios are allowed, so a cheap strong fixed policy is not excluded
merely because it uses slightly less than cost 1.0.

Protocol, checkpoints, raw episode results, and `EFFECT_REPORT.md` are saved in
`experiment/results/rl_effect_check/`. Completed phases are reused. A new protocol
needs a new output directory. The report separates RL-versus-baseline evidence
from the incremental effect of adaptive budget penalties, and reports negative
or inconclusive results as such. These are numerical simulator results only.

The measured 100k-step results do **not** show universal RL superiority. At
matched mean cost 0.75, importance-weighted success was 31.14% for original RL,
38.15% for the slow adaptive update, 28.12% for history ML, and 39.32% for the
validation-selected fixed baseline. The slow-versus-original gain was +7.01 percentage points (paired hierarchical
bootstrap interval +5.10 to +9.13), but
the slow-versus-fixed difference was inconclusive. Budget violations were not
eliminated: 3.40% for slow RL versus 8.07% for original RL at this operating point.
Thus the new setting is an optional improvement over the old RL controller,
not evidence that RL beats a strong fixed strategy or enforces hard budgets.

The original fast update is retained only as a failed comparison. At primary
cost 1.0, not every adaptive training seed had validation coverage; no complete
three-seed adaptive result is extrapolated there. The separate cost-0.75 check
was added after validation exposed this coverage issue; it is a secondary
analysis, not a replacement primary result. Reproduce it after the main run:

```bash
python -m experiment.rl.verify_effect --phase select --target-cost 0.75
python -m experiment.rl.verify_effect --phase test --target-cost 0.75 --workers 3
python -m experiment.rl.verify_effect --phase report --target-cost 0.75
```

See `experiment/results/rl_effect_check/cost_0.75/EFFECT_REPORT.md` and
`effect_comparison.png` there. Compare all methods afresh if the environment,
training length, or cost definition changes; these results cannot be mixed with
the old milestone table.

### Minimal final ablations

After the main cost-0.75 verification, run:

```bash
python -m experiment.rl.run_ablations --workers 3
```

Only two variants are retrained, each for the same 100,000 steps and three
training seeds as the full QR-DQN:

- **No history features:** mask previous actions/outcomes/costs, rolling history,
  and protection history during both training and inference. Current alerts,
  workload, remaining budget, progress, and detector settings remain visible.
- **Myopic action values:** set `gamma=0` instead of `0.985`, retaining the full
  observation and the same adaptive budget mechanism.

The full model and original-budget comparison are reused. Attack dynamics,
costs, actions, network architecture, and training steps are unchanged. Operating
points are selected on validation seeds; the same 12 held-out test seeds and six
scenarios give paired differences and 95% hierarchical bootstrap intervals at
mean cost 0.75. No additional algorithm search or parameter sweep is performed.
Intervals crossing zero, negative findings, or insufficient cost coverage are
reported directly. See `experiment/results/rl_effect_check/ablations/ABLATION_REPORT.md`.

The completed ablations gave 38.15% for full RL, 37.86% without history
features, and 38.00% with gamma=0, all at mean cost 0.75. Neither intervention
established a reliable gain from history or long-term TD returns. The gamma=0
interval is exploratory: only 2788/3000 bootstrap replicates covered the target
cost, and no extrapolation was used. These findings limit the claims; they are
not reasons to add another algorithm search. See [RL_STATUS.md](RL_STATUS.md)
for the concise report-ready conclusions and remaining writing tasks.
