"""Factorial stress test over attack persistence, sensor noise, and adaptation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from .environment import SCENARIOS, ScenarioConfig
from .evaluate_agents import load_policies, run_episode, summarize, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("experiment/results/rl_v1/models"))
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_v1"))
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--algorithms", nargs="+", default=["ppo", "a2c", "dqn", "qrdqn", "recurrent_ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[707, 808, 909])
    args = parser.parse_args()
    policies = load_policies(args.model_dir, args.training_seed, args.algorithms)
    # 3^4 grid exposes interactions that named scenario tests can hide.
    axes = {
        "persistence": (0.35, 0.68, 0.94),
        "detector_false_negative": (0.08, 0.28, 0.50),
        "adaptive_strength": (0.0, 0.42, 0.84),
        "budget_scale": (0.65, 1.0, 1.35),
    }
    rows = []
    combinations = list(itertools.product(*axes.values()))
    for policy_name, policy in policies.items():
        for persistence, false_negative, adaptive, budget_scale in combinations:
            label = f"p{persistence:.2f}_fn{false_negative:.2f}_a{adaptive:.2f}"
            cfg = ScenarioConfig(
                name=label,
                horizon=192,
                persistence=persistence,
                detector_false_negative=false_negative,
                adaptive_strength=adaptive,
                regime_switch_probability=0.035 if adaptive > 0.8 else 0.01,
                budget_per_step=SCENARIOS["train_mix"].budget_per_step,
            )
            for seed in args.seeds:
                row = run_episode(
                    policy, cfg, budget_scale, seed,
                    recurrent=policy_name == "recurrent_ppo",
                    maskable=policy_name == "maskable_ppo",
                )
                row.update({
                    "policy": policy_name,
                    "persistence": persistence,
                    "false_negative": false_negative,
                    "adaptive_strength": adaptive,
                })
                rows.append(row)
        print(f"stress-tested {policy_name}", flush=True)
    write_csv(args.output / "stress_episodes.csv", rows)
    write_csv(args.output / "stress_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
