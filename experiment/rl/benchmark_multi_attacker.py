"""Evaluate policies against one to five independent attacker processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO

from .baselines import FixedPolicy, RiskHeuristicPolicy
from .environment import DefenseEnv, SCENARIOS


POLICY_ORDER = (
    "QR-DQN single-trained",
    "QR-DQN multi-trained",
    "QR-DQN fine-tuned",
    "QR-DQN selected composite",
    "QR-DQN reliability-tuned",
    "TRPO robust",
    "PPO",
    "Risk heuristic",
    "Fixed k3 + rotate",
    "Fixed k5 + rotate",
    "Fixed k7 + rotate",
)
COLORS = {
    "QR-DQN single-trained": "#4c78a8",
    "QR-DQN multi-trained": "#f28e8c",
    "QR-DQN fine-tuned": "#e45756",
    "QR-DQN selected composite": "#d62728",
    "QR-DQN reliability-tuned": "#ff7f0e",
    "TRPO robust": "#9467bd",
    "PPO": "#17becf",
    "Risk heuristic": "#b279a2",
    "Fixed k3 + rotate": "#f2cf5b",
    "Fixed k5 + rotate": "#59a14f",
    "Fixed k7 + rotate": "#9c755f",
}


def run_episode(policy, scenario, budget_scale, seed):
    env = DefenseEnv(scenario, budget_scale=budget_scale)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    attack_events = 0
    compromised_replica_rounds = 0
    action_counts = np.zeros(12, dtype=np.int64)
    while True:
        attack_events += int(env.true_attacks > 0)
        compromised_replica_rounds += env.true_attacks
        action, _ = policy.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    return {
        "return": total_reward,
        "success_rate": metrics["success_rate"],
        "weighted_success_rate": metrics["weighted_success_rate"],
        "mean_cost": metrics["mean_cost"],
        "budget_violation_rate": metrics["budget_violations"] / env.config.horizon,
        "mean_redundancy": sum(
            count * env.decode_action(index)[0]
            for index, count in enumerate(action_counts)
        ) / env.config.horizon,
        "attack_event_rate": attack_events / env.config.horizon,
        "mean_compromised_replicas": compromised_replica_rounds / env.config.horizon,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["attacker_count"], row["budget_scale"])].append(row)
    metrics = (
        "return", "success_rate", "weighted_success_rate", "mean_cost",
        "budget_violation_rate", "mean_redundancy", "attack_event_rate",
        "mean_compromised_replicas",
    )
    summary = []
    for (policy, attackers, budget), values in groups.items():
        item = {
            "policy": policy,
            "attacker_count": attackers,
            "budget_scale": budget,
            "episodes": len(values),
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = float(np.mean(samples))
            item[f"{metric}_min"] = float(np.min(samples))
            item[f"{metric}_max"] = float(np.max(samples))
        summary.append(item)
    return sorted(summary, key=lambda row: (
        POLICY_ORDER.index(row["policy"]), row["attacker_count"], row["budget_scale"]
    ))


def scenario_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["budget_scale"] == 1.0:
            groups[(row["policy"], row["base_scenario"], row["attacker_count"])].append(row)
    result = []
    for (policy, scenario, attackers), values in groups.items():
        result.append({
            "policy": policy,
            "base_scenario": scenario,
            "attacker_count": attackers,
            "episodes": len(values),
            "weighted_success_rate": float(np.mean([
                row["weighted_success_rate"] for row in values
            ])),
            "return": float(np.mean([row["return"] for row in values])),
            "mean_cost": float(np.mean([row["mean_cost"] for row in values])),
        })
    return result


def composite_ranking(summary, budget_scale=1.0):
    """Rank policies equally across attacker-count and selected budget cells."""
    selected = [
        row for row in summary
        if budget_scale is None or float(row["budget_scale"]) == budget_scale
    ]
    policy_scores = defaultdict(list)
    cells = sorted({
        (int(row["attacker_count"]), float(row["budget_scale"]))
        for row in selected
    })
    for attacker_count, budget in cells:
        count_rows = [
            row for row in selected
            if int(row["attacker_count"]) == attacker_count
            and float(row["budget_scale"]) == budget
        ]
        normalized = {}
        for metric in (
            "weighted_success_rate_mean", "return_mean", "mean_cost_mean"
        ):
            values = np.asarray([float(row[metric]) for row in count_rows])
            low, high = float(values.min()), float(values.max())
            normalized[metric] = {
                row["policy"]: (
                    0.5 if high == low
                    else (float(row[metric]) - low) / (high - low)
                )
                for row in count_rows
            }
        for row in count_rows:
            policy = row["policy"]
            score = (
                0.45 * normalized["weighted_success_rate_mean"][policy]
                + 0.45 * normalized["return_mean"][policy]
                + 0.10 * (1.0 - normalized["mean_cost_mean"][policy])
            )
            policy_scores[policy].append(score)
    rankings = []
    for policy, scores in policy_scores.items():
        rankings.append({
            "policy": policy,
            "mean_count_score": float(np.mean(scores)),
            "worst_count_score": float(np.min(scores)),
            "composite_score": float(0.75 * np.mean(scores) + 0.25 * np.min(scores)),
        })
    return sorted(rankings, key=lambda row: row["composite_score"], reverse=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary, scenario_rows, output):
    selected = [row for row in summary if row["budget_scale"] == 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    metrics = (
        ("weighted_success_rate_mean", "Importance-weighted success", True),
        ("return_mean", "Episode return", False),
        ("mean_cost_mean", "Mean communication cost", False),
    )
    for axis, (metric, title, percentage) in zip(axes, metrics):
        for policy in POLICY_ORDER:
            policy_rows = sorted(
                [row for row in selected if row["policy"] == policy],
                key=lambda row: row["attacker_count"],
            )
            style = "-" if (
                policy.startswith("QR-DQN") or policy in {"TRPO robust", "PPO"}
            ) else ":"
            width = 3.4 if policy == "QR-DQN reliability-tuned" else 1.8
            axis.plot(
                [row["attacker_count"] for row in policy_rows],
                [row[metric] for row in policy_rows],
                marker="o", linewidth=width, linestyle=style,
                color=COLORS[policy], label=policy,
            )
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Independent attackers")
        axis.set_xticks((1, 2, 3, 5))
        axis.grid(True, alpha=0.22)
        if percentage:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Defense performance")
    axes[2].legend(fontsize=8.2, loc="best")
    fig.suptitle(
        "Defense Robustness to Multiple Independent Attackers",
        fontsize=16, weight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Budget scale = 1.0; mean over persistent, adaptive, noisy-OOD, and budget-shock "
        "conditions with five common seeds. Solid lines are learned RL policies.",
        ha="center", fontsize=9.4,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Heatmap: success-rate gain from careful multi-attacker adaptation.
    bases = ("persistent", "adaptive", "noisy_ood", "budget_shock")
    attackers = (1, 2, 3, 5)
    matrix = np.zeros((len(bases), len(attackers)))
    for row_index, base in enumerate(bases):
        for column_index, count in enumerate(attackers):
            new = next(row["weighted_success_rate"] for row in scenario_rows
                       if row["policy"] == "QR-DQN reliability-tuned"
                       and row["base_scenario"] == base
                       and row["attacker_count"] == count)
            old = next(row["weighted_success_rate"] for row in scenario_rows
                       if row["policy"] == "QR-DQN single-trained"
                       and row["base_scenario"] == base
                       and row["attacker_count"] == count)
            matrix[row_index, column_index] = 100 * (new - old)
    heat_fig, heat_ax = plt.subplots(figsize=(8.2, 5.2))
    limit = max(1.0, float(np.abs(matrix).max()))
    image = heat_ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            heat_ax.text(column_index, row_index, f"{matrix[row_index, column_index]:+.1f}",
                         ha="center", va="center", fontsize=11, weight="bold")
    heat_ax.set_xticks(range(len(attackers)), attackers)
    heat_ax.set_yticks(range(len(bases)), [name.replace("_", " ").title() for name in bases])
    heat_ax.set_xlabel("Independent attackers")
    heat_ax.set_title("Reliability-tuned QR-DQN Success Gain (percentage points)", weight="bold")
    heat_fig.colorbar(image, ax=heat_ax, label="Success-rate gain (pp)")
    heat_fig.tight_layout()
    heat_fig.savefig(output.with_name("multi_attacker_gain_heatmap.png"), dpi=220,
                     bbox_inches="tight")
    heat_fig.savefig(output.with_name("multi_attacker_gain_heatmap.pdf"),
                     bbox_inches="tight")
    plt.close(heat_fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-model", type=Path,
        default=Path("experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--multi-model", type=Path,
        default=Path("experiment/results/rl_multi_attacker/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--fine-tuned-model", type=Path,
        default=Path(
            "experiment/results/rl_multi_attacker_finetuned/models/qrdqn_seed2026.zip"
        ),
    )
    parser.add_argument(
        "--selected-model", type=Path,
        default=Path(
            "experiment/results/rl_multi_attacker_curriculum/models/"
            "qrdqn_selected_seed2026.zip"
        ),
    )
    parser.add_argument(
        "--trpo-model", type=Path,
        default=Path("experiment/results/rl_robust_w015/models/trpo_seed2026.zip"),
    )
    parser.add_argument(
        "--reliability-model", type=Path,
        default=Path(
            "experiment/results/rl_multi_attacker_reliability/models/"
            "qrdqn_seed2026.zip"
        ),
    )
    parser.add_argument(
        "--ppo-model", type=Path,
        default=Path("experiment/results/rl_v2/models/ppo_seed2026.zip"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_multi_attacker/analysis/multi_attacker_benchmark.png"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.70, 1.0, 1.30])
    parser.add_argument("--attacker-counts", nargs="+", type=int, default=[1, 2, 3, 5])
    args = parser.parse_args()

    policies = {
        "QR-DQN single-trained": QRDQN.load(args.single_model, device="cpu"),
        "QR-DQN multi-trained": QRDQN.load(args.multi_model, device="cpu"),
        "QR-DQN fine-tuned": QRDQN.load(args.fine_tuned_model, device="cpu"),
        "QR-DQN selected composite": QRDQN.load(args.selected_model, device="cpu"),
        "QR-DQN reliability-tuned": QRDQN.load(args.reliability_model, device="cpu"),
        "TRPO robust": TRPO.load(args.trpo_model, device="cpu"),
        "PPO": PPO.load(args.ppo_model, device="cpu"),
        "Risk heuristic": RiskHeuristicPolicy(),
        "Fixed k3 + rotate": FixedPolicy(3, "rotate"),
        "Fixed k5 + rotate": FixedPolicy(5, "rotate"),
        "Fixed k7 + rotate": FixedPolicy(7, "rotate"),
    }
    base_scenarios = ("persistent", "adaptive", "noisy_ood", "budget_shock")
    rows = []
    for policy_name, policy in policies.items():
        for base_name in base_scenarios:
            for attacker_count in args.attacker_counts:
                scenario = replace(
                    SCENARIOS[base_name],
                    name=f"{base_name}_attackers{attacker_count}",
                    attacker_count=attacker_count,
                )
                for budget in args.budgets:
                    for seed in args.seeds:
                        row = run_episode(policy, scenario, budget, seed)
                        row.update({
                            "policy": policy_name,
                            "base_scenario": base_name,
                            "attacker_count": attacker_count,
                            "budget_scale": budget,
                            "seed": seed,
                        })
                        rows.append(row)
        print(f"evaluated {policy_name}", flush=True)

    summary = summarize(rows)
    by_scenario = scenario_summary(rows)
    make_figure(summary, by_scenario, args.output)
    write_csv(args.output.with_name("multi_attacker_episodes.csv"), rows)
    write_csv(args.output.with_name("multi_attacker_summary.csv"), summary)
    write_csv(args.output.with_name("multi_attacker_by_scenario.csv"), by_scenario)
    write_csv(
        args.output.with_name("multi_attacker_composite_ranking.csv"),
        composite_ranking(summary),
    )
    write_csv(
        args.output.with_name("multi_attacker_all_budget_ranking.csv"),
        composite_ranking(summary, budget_scale=None),
    )
    args.output.with_name("multi_attacker_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
