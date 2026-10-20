# Numerical Adaptive-Defense Experiment

This folder contains numerical experiments that do not modify the
communication-layer defense implementation. The files listed immediately below
are the earlier CPU-only experiment; the newer multi-algorithm POMDP work is
under `rl/`.

## New RL experiment

The dedicated `ee599-rl` environment and `experiment/rl/` package compare
PPO, A2C, DQN, QR-DQN, and RecurrentPPO against all 12 fixed action choices
plus rule policies. See `rl/README.md` for reproduction commands and
`RL_EXPERIMENT_REPORT.md` for the current results and limitations.

## Milestone numerical experiment

The milestone experiment compares the original fixed, stateless-ML, stateful-ML,
and linear Q-learning controllers under one attacker and three staggered attackers
at matched average communication cost.

Reproduce the numerical experiments and figures from the repository root:

```powershell
conda run -n ee599-rl --no-capture-output python -m experiment.run_milestone_experiments
conda run -n ee599-rl --no-capture-output python -m experiment.plot_milestone_results
```

The runner uses the complete Git history to verify equivalence to the original
four-controller implementation at commit `b9dafc0`. Raw results and execution
metadata are saved under `experiment/results/milestone_b9dafc0/`, which remains
Git-ignored. Generated figures are saved locally under `experiment/ignore/mid/`.

## Files

- `attacker.py`: HMM multi-stage attacker and IBAL adaptive attack schedule.
- `defender.py`: fixed baseline, stateless ML, stateful ML, and Q-learning RL.
- `train.py`: cost, reward, training, matched-budget evaluation, and reporting.
- `run_single_attacker.py`: multi-seed budget sweep, equal-communication-cost
  interpolation, timing, and plots.

## Run

```bash
python experiment/train.py --attacker hmm --profile bursty \
  --budget 1
```

Single stochastic attacker, three seeds, and a same-cost comparison:

```bash
conda run -n CVPR python experiment/run_single_attacker.py
```

Outputs are written to `experiment/results/single_attacker/`. The left plot is
the success-cost curve; the right plot compares all models at exactly the same
interpolated average communication cost. Timing includes supervised data plus
all three trainable models in `training_seconds`, and all four policy
evaluations in `testing_seconds`.

Three independent bursty attackers start near different thirds of each
episode. Start times receive deterministic random jitter in every episode, so
the test schedule is not copied from training:

```bash
conda run -n CVPR python experiment/run_single_attacker.py \
  --attackers 3 --steps 180 --window 30 \
  --train-episodes 1000 --test-episodes 10 \
  --output-dir experiment/results/multi_attacker
```

The stateful ML and RL models receive the same 12 observable features:
short/long attack history, attack rate, time since the last attack, episode
progress, remaining budget, current message length, message importance, and
the same derived interaction terms. Their difference is the learning
objective: supervised ML predicts a local defense label, while RL optimizes
cumulative reward.

Message length is not sampled independently each round. A persistent
three-state workload process moves among short, normal, and long contexts;
length is smoothed between consecutive rounds with bounded random jitter.
This makes communication cost rise and fall over one continuous agent run.

The multi-attacker environment also uses a fixed feedback heuristic: low
recent defense success shifts each active HMM's transition probability toward
probing and burst states. The attacker itself is not trained. This implements
the stated "attack more when the target is easy" rule and makes current defense
actions affect future attack intensity, which is the sequential part optimized
by RL.

All cost and reward parameters are edited only in the `cost` and `reward`
dictionaries at the top of `train.py`.

IBAL adaptive schedule:

```bash
python experiment/train.py --attacker ibal --budget 1
```

## Cost and outcome

The defense sends full text on at most two carriers and sends a 64-byte SHA-256 hex
hash on each remaining witness carrier. For message length `L`:

```text
B(k,L) = min(k,2)*L + max(0,k-2)*64
B_base(L) = B(1,L)
B_ref = B(3,L_ref) - B_base(L_ref)
```

The normalized communication cost is:

```text
C_comm = ((B(k,L)-B_base(L)) + retries*B(k,L)) / B_ref
```

Thus one byte is not one cost unit. `C_comm=1` means the same extra byte volume
as `k=3` at reference length `L_ref`. Initial redundant packets and every retry
are both charged using their transmitted byte counts.

Final failure cost is separate:

```text
C_failure = importance * 1[all attempts fail]
```

The default performs at most one fresh retry after initial quorum failure. The
retry samples another attack at the same hidden attack stage. An attack attempt
fails when:

```text
attack_count >= (k + 1) / 2
```

Estimated downstream accuracy is:

```text
clean_accuracy * integrity_success_rate
```

`train.py` supports three reward modes:

- `budget`: minimize error under the common defense budget.
- `equal_cost`: defense and error both have unit weight.
- `utility`: success reward minus defense and error costs.

The command-line `--budget` currently controls normalized repetition capacity
(`k=1/3/5/7` uses `0/1/2/3` units). Because message lengths and retry counts
differ, final paper comparisons must run a budget sweep and compare/interpolate
the methods at the same reported average `C_comm`, not assume one `--budget`
value gives identical communication cost.

### Cost sources and assumptions

- [Nyberg and Johnson (2023)](https://arxiv.org/abs/2304.11084) define reward
  as defense activation cost plus compromised-target cost. They set defense
  cost to 1 and choose a larger failure cost from attack-path TTC. This supports
  separating defense and failure costs, but not our exact equality assumption.
- [BFTBrain](https://arxiv.org/abs/2408.06432) treats request/reply size,
  communication phases, latency, and execution overhead as factors affecting
  adaptive BFT performance. This supports including message length and repeat
  communication cost.
- [CVSS v3.1](https://www.first.org/cvss/v3-1/specification-document) assigns
  Low/Medium/High security-requirement constants `0.5/1.0/1.5`; these values
  are used as the failure-importance multipliers.

The reference length `L_ref=100` is only a simulation default. In the final
experiment it should be replaced by the median serialized message length from
the selected dataset. The 64-byte witness length comes from the defense's SHA-256
hex digest implementation. Normal failure cost `1` is the declared base unit,
not a value copied from another paper.

## Literature mapping

- HMM hidden stages follow the modeling structure in
  [Liao et al. (2020)](https://doi.org/10.1155/2020/1428056). The numerical
  transition matrices in this folder are experiment parameters, not copied
  values from that paper.
- `ibal` implements the success-threshold update of the maximum attack
  probability in [IBAL](https://arxiv.org/abs/2605.18024): when average learner
  success reaches `eta`, `P_max <- min(1, alpha * P_max)`.
