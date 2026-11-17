"""Evaluate RL agents and non-RL controls with common random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import MaskablePPO, QRDQN, RecurrentPPO, TRPO

from .baselines import ConservativePolicy, FixedPolicy, RiskHeuristicPolicy
from .environment import DefenseEnv, RECOVERIES, REDUNDANCIES, SCENARIOS
from .config import ENVIRONMENT_VERSION, continuous_changes
from .legacy_q import LegacyLinearQPolicy


MODEL_CLASSES = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "qrdqn": QRDQN,
    "recurrent_ppo": RecurrentPPO,
    "maskable_ppo": MaskablePPO,
    "trpo": TRPO,
}
ALGORITHM_NAMES = (*MODEL_CLASSES, "legacy_q")


def load_policies(model_dir: Path, seed: int, algorithms: list[str]):
    policies = {
        f"fixed_k{k}_{recovery}": FixedPolicy(k, recovery)
        for k in REDUNDANCIES
        for recovery in RECOVERIES
    }
    policies.update({
        "heuristic": RiskHeuristicPolicy(),
        "conservative": ConservativePolicy(),
    })
    for name in algorithms:
        if name == "legacy_q":
            path = model_dir / f"legacy_q_seed{seed}.pkl"
            if path.exists():
                policies[name] = LegacyLinearQPolicy.load(path)
            else:
                print(f"warning: missing {path}", flush=True)
            continue
        path = model_dir / f"{name}_seed{seed}.zip"
        if path.exists():
            policies[name] = MODEL_CLASSES[name].load(path, device="auto")
        else:
            print(f"warning: missing {path}", flush=True)
    return policies


def run_episode(
    policy, scenario, budget_scale: float, seed: int, recurrent: bool,
    maskable: bool = False,
    continuous: bool = False,
    trace: list[dict] | None = None,
):
    config = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    env = DefenseEnv(
        config, budget_scale=budget_scale,
        changes=continuous_changes(config) if continuous else (),
    )
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    state = None
    episode_start = np.ones((1,), dtype=bool)
    total_reward = 0.0
    action_counts = np.zeros(12, dtype=np.int64)
    while True:
        if maskable:
            action, _ = policy.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
        elif recurrent:
            action, state = policy.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
            episode_start[:] = False
        else:
            action, _ = policy.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if trace is not None:
            trace.append({
                "round": env.t - 1, "scenario": config.name,
                "budget_scale": budget_scale, "seed": seed,
                "attacker_count": info["attacker_count"],
                "success": int(info["success"]), "importance": info["importance"],
                "cost": info["cost"], "redundancy": info["redundancy"],
                "recovery": info["recovery"],
                "recovery_used": int(info["recovery_used"]),
                "budget_remaining": info["budget_remaining"],
            })
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    env.close()
    return {
        "policy": "",
        "scenario": scenario if isinstance(scenario, str) else scenario.name,
        "budget_scale": budget_scale,
        "seed": seed,
        "return": total_reward,
        "success_rate": metrics["success_rate"],
        "weighted_success_rate": metrics["weighted_success_rate"],
        "mean_cost": metrics["mean_cost"],
        "target_mean_cost": metrics["target_mean_cost"],
        "recovery_rate": metrics["recovery_uses"] / env.config.horizon,
        "rotation_rate": metrics["rotations"] / env.config.horizon,
        "budget_violation_rate": metrics["budget_violations"] / env.config.horizon,
        "mean_redundancy": sum(
            count * env.decode_action(i)[0] for i, count in enumerate(action_counts)
        ) / env.config.horizon,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["scenario"], row["budget_scale"])].append(row)
    summary = []
    metrics = (
        "return", "success_rate", "weighted_success_rate", "mean_cost", "target_mean_cost",
        "recovery_rate", "rotation_rate", "budget_violation_rate", "mean_redundancy",
    )
    for (policy, scenario, budget), values in sorted(groups.items()):
        item = {"policy": policy, "scenario": scenario, "budget_scale": budget, "episodes": len(values)}
        for metric in metrics:
            samples = np.asarray([v[metric] for v in values], dtype=float)
            item[f"{metric}_mean"] = float(samples.mean())
            item[f"{metric}_min"] = float(samples.min())
            item[f"{metric}_max"] = float(samples.max())
        summary.append(item)
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("experiment/results/rl_v1/models"))
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_v1"))
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(MODEL_CLASSES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=[s for s in SCENARIOS if s != "train_mix"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.70, 1.0, 1.30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--continuous", action="store_true",
                        help="Change attacker count, budget, and detector noise within each episode.")
    parser.add_argument("--save-traces", action="store_true",
                        help="Also save per-round actions and costs for adaptation analysis.")
    args = parser.parse_args()
    if args.continuous:
        args.output = args.output / "continuous"
    policies = load_policies(args.model_dir, args.training_seed, args.algorithms)
    rows = []
    traces = []
    total = len(policies) * len(args.scenarios) * len(args.budgets) * len(args.seeds)
    done = 0
    for policy_name, policy in policies.items():
        for scenario in args.scenarios:
            for budget in args.budgets:
                for seed in args.seeds:
                    trace = [] if args.save_traces else None
                    row = run_episode(
                        policy, scenario, budget, seed,
                        recurrent=policy_name == "recurrent_ppo",
                        maskable=policy_name == "maskable_ppo",
                        continuous=args.continuous, trace=trace,
                    )
                    row["policy"] = policy_name
                    rows.append(row)
                    if trace is not None:
                        traces.extend({"policy": policy_name, **step} for step in trace)
                    done += 1
        print(f"evaluated {policy_name} ({done}/{total})", flush=True)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "episodes.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    if traces:
        write_csv(args.output / "traces.csv", traces)
    (args.output / "evaluation_config.json").write_text(
        json.dumps({**vars(args), "environment_version": ENVIRONMENT_VERSION},
                   default=str, indent=2), encoding="utf-8"
    )
    ranked = sorted(
        summary,
        key=lambda x: x["return_mean"],
        reverse=True,
    )[:12]
    print(json.dumps(ranked, indent=2), flush=True)


if __name__ == "__main__":
    main()
