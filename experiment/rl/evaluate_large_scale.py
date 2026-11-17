"""Matched-cost comparison on the large-scale redundancy-budget environment.

Every family is reduced to its measured (mean cost, weighted success) points
and the upper concave hull (randomizing between operating points is feasible).
Families: every fixed odd k, a history-only trend rule, the tuned estimator
rule, each RL model at several budgets, and optionally a privileged oracle that
sees the hidden attack count.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .large_scale import (
    SCALE_SCENARIOS, ExactEstimatorPolicy, ScaleDefenseEnv, ScaleFixedPolicy,
    TrendRulePolicy,
)

TREND_MARGINS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
BUDGET_SCALES = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5)
COST_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)


class OraclePolicy:
    """Privileged upper bound: reads the hidden attack count and adds a margin."""

    def __init__(self, margin: int) -> None:
        self.margin = margin
        self.env = None

    def reset(self, env) -> None:
        self.env = env

    def predict(self, observation, deterministic: bool = True):
        k = 2 * (self.env.true_attacks + self.margin) + 1
        k = int(np.clip(k, 1, 2 * self.env.config.channels + 1))
        return ScaleDefenseEnv.encode_action(k), None


def run_episode(policy, scenario, budget_scale: float, seed: int,
                env_kwargs: dict | None = None, price: float | None = None) -> dict:
    env = ScaleDefenseEnv(scenario, budget_scale=budget_scale, **(env_kwargs or {}))
    obs, _ = env.reset(seed=seed, options=None if price is None else {"price": price})
    if hasattr(policy, "reset"):
        policy.reset(env)
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    return {
        "scenario": env.config.name, "budget_scale": budget_scale, "seed": seed,
        "weighted_success": metrics["weighted_success_rate"],
        "success": metrics["success_rate"], "mean_cost": metrics["mean_cost"],
        "mean_redundancy": metrics["mean_redundancy"],
        "budget_violation": metrics["budget_violations"] / env.config.horizon,
    }


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for point in sorted(set(points)):
        while len(out) >= 2 and (
            (out[-1][0] - out[-2][0]) * (point[1] - out[-2][1])
            - (out[-1][1] - out[-2][1]) * (point[0] - out[-2][0])
        ) >= 0:
            out.pop()
        out.append(point)
    return out


def value_at(hull: list[tuple[float, float]], cost: float) -> float:
    for (c0, s0), (c1, s1) in zip(hull, hull[1:]):
        if c0 <= cost <= c1:
            return s0 + (s1 - s0) * (cost - c0) / (c1 - c0)
    return float("nan")


class LearnedPolicy:
    """Stable-Baselines3 model (recurrent ones keep their hidden state)."""

    def __init__(self, model, recurrent: bool = False) -> None:
        self.model = model
        self.recurrent = recurrent
        self.reset()

    def reset(self, env=None) -> None:
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def predict(self, observation, deterministic: bool = True):
        if self.recurrent:
            action, self.state = self.model.predict(
                observation, state=self.state, episode_start=self.episode_start,
                deterministic=True,
            )
            self.episode_start[:] = False
            return action, None
        return self.model.predict(observation, deterministic=True)


class SklearnPolicy:
    """Supervised policy: classifier on the same observation, predicts the k index."""

    def __init__(self, estimator) -> None:
        try:  # single-row prediction is far faster without a worker pool
            estimator.set_params(n_jobs=1)
        except ValueError:
            pass
        self.estimator = estimator

    def predict(self, observation, deterministic: bool = True):
        x = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return int(self.estimator.predict(x)[0]), None


def load_sb3(path: Path):
    """Class from the file-name prefix: qrdqn_, dqn_, ppo_, a2c_, recurrent_ppo_."""
    from sb3_contrib import QRDQN, RecurrentPPO
    from stable_baselines3 import A2C, DQN, PPO
    classes = {"recurrent_ppo": RecurrentPPO, "qrdqn": QRDQN, "dqn": DQN, "ppo": PPO, "a2c": A2C}
    name = next(key for key in classes if path.stem.startswith(key))
    return LearnedPolicy(classes[name].load(path, device="cpu"), recurrent=name == "recurrent_ppo")


def build_policies(args, channels: int) -> dict[str, tuple[str, object, dict]]:
    """name -> (family, policy, env kwargs). Each RL model is its own family."""
    policies: dict[str, tuple[str, object, dict]] = {}
    for k in range(1, 2 * channels + 2, 2):
        if not args.only_models:
            policies[f"fixed_k{k}"] = ("fixed", ScaleFixedPolicy(k), {})
    for margin in TREND_MARGINS:
        if not args.only_models:
            policies[f"trend_z{margin:+.1f}"] = ("trend", TrendRulePolicy(channels, margin), {})
    if args.rule_params and not args.only_models:
        for index, item in enumerate(json.loads(args.rule_params.read_text())):
            policies[f"rule_{index}"] = (
                "rule", ExactEstimatorPolicy(channels, delay=args.delay, **item["params"]), {},
            )
    if args.oracle:
        for margin in range(-2, 12):
            policies[f"oracle_m{margin:+d}"] = ("oracle", OraclePolicy(margin), {})
    import joblib
    for path in args.ml_models:
        # Supervised models act on the k index with the estimator observation.
        policies[f"ml_{path.stem}"] = (
            f"ml_{path.stem}", SklearnPolicy(joblib.load(path)),
            {"observation_mode": "estimator", "action_mode": "k",
             "price_conditioned": True},
        )
    for path in args.models:
        policies[f"rl_{path.stem}"] = (
            f"rl_{path.stem}", load_sb3(path),
            {"observation_mode": args.rl_obs_mode, "action_mode": args.rl_action_mode,
             "price_conditioned": args.rl_price_conditioned,
             "receipts": not args.rl_no_receipts,
             "feature_mask": tuple(args.rl_mask_features)},
        )
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCALE_SCENARIOS,
                        default=list(SCALE_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1020)))
    parser.add_argument("--budgets", nargs="+", type=float, default=list(BUDGET_SCALES))
    parser.add_argument("--models", nargs="*", type=Path, default=[])
    parser.add_argument("--rule-params", type=Path,
                        help="chosen_*.json written by tune_rule.py")
    parser.add_argument("--delay", type=int, default=1,
                        help="Rounds before the signed receipt reveals success.")
    parser.add_argument("--rl-obs-mode", choices=("legacy", "estimator"), default="estimator")
    parser.add_argument("--rl-action-mode", choices=("k", "margin"), default="margin")
    parser.add_argument("--rl-price-conditioned", action="store_true",
                        help="RL models were trained with a price input; sweep --prices.")
    parser.add_argument("--prices", nargs="+", type=float,
                        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0])
    parser.add_argument("--rl-no-receipts", action="store_true")
    parser.add_argument("--rl-mask-features", nargs="*", type=int, default=[])
    parser.add_argument("--ml-models", nargs="*", type=Path, default=[],
                        help="joblib files written by train_ml_large_scale.py")
    parser.add_argument("--only-models", action="store_true",
                        help="Skip fixed/trend/rule baselines; evaluate learned models only.")
    parser.add_argument("--fixed-only", action="store_true",
                        help="Only fixed-k policies (used on validation seeds).")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/rl_large_scale/evaluation"))
    args = parser.parse_args()
    scenarios = {
        name: replace(SCALE_SCENARIOS[name], feedback_delay=args.delay)
        for name in args.scenarios
    }
    policies = build_policies(args, SCALE_SCENARIOS[args.scenarios[0]].channels)
    if args.fixed_only:
        policies = {n: v for n, v in policies.items() if v[0] == "fixed"}

    rows = []
    for name, config in scenarios.items():
        for policy_name, (family, policy, env_kwargs) in policies.items():
            # Only RL conditions on the budget; other families have one setting.
            is_rl = family.startswith(("rl_", "ml_"))
            priced = family.startswith("ml_") or args.rl_price_conditioned
            sweep = args.prices if is_rl and priced else (
                args.budgets if is_rl else [None]
            )
            for value in sweep:
                budget = 1.0 if priced or not is_rl else value
                price = value if is_rl and priced else None
                for seed in args.seeds:
                    row = run_episode(policy, config, budget, seed, env_kwargs, price)
                    rows.append({"policy": policy_name, "family": family,
                                 "price": price if price is not None else "", **row})
        print(f"evaluated {name}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.only_models:
        return
    points = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["scenario"], row["family"])
        points[key][(row["policy"], row["budget_scale"], row["price"])].append(row)
    families = list(dict.fromkeys(row["family"] for row in rows))
    comparison = []
    for name in scenarios:
        hulls = {}
        for family in families:
            groups = points.get((name, family))
            if groups:
                hulls[family] = upper_hull([
                    (float(np.mean([r["mean_cost"] for r in rs])),
                     float(np.mean([r["weighted_success"] for r in rs])))
                    for rs in groups.values()
                ])
        print(f"\n== {name}  (gap vs fixed hull, percentage points)")
        for cost in COST_GRID:
            base = value_at(hulls["fixed"], cost)
            if np.isnan(base):
                continue
            line = {"scenario": name, "cost": cost, "fixed": base}
            text = f"cost {cost:4.1f}: fixed {base:.3f}"
            gaps = []
            for family in families:
                if family == "fixed" or family not in hulls:
                    continue
                value = value_at(hulls[family], cost)
                line[family] = value
                if not np.isnan(value):
                    text += f" | {family} {value:.3f} ({100 * (value - base):+5.1f})"
                    if family.startswith("rl_"):
                        gaps.append(100 * (value - base))
            if len(gaps) > 1:
                text += f" | RL mean gap {np.mean(gaps):+5.1f} (n={len(gaps)})"
            comparison.append(line)
            print(text)
    keys = sorted({key for line in comparison for key in line})
    with (args.output / "matched_cost.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    (args.output / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
