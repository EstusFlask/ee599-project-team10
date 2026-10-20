"""Matched-communication-cost evaluation for the multi-attacker benchmark."""

from __future__ import annotations

import argparse
import csv
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
from stable_baselines3 import PPO
from sb3_contrib import QRDQN, TRPO

from .baselines import FixedPolicy, RiskHeuristicPolicy
from .environment import DefenseEnv, SCENARIOS


POLICY_ORDER = (
    "QR-DQN original",
    "QR-DQN scratch multi",
    "QR-DQN uniform fine-tuned",
    "QR-DQN curriculum",
    "QR-DQN reliability-tuned",
    "TRPO robust",
    "PPO",
    "Risk heuristic",
    "Fixed k3 + rotate",
    "Fixed k5 + rotate",
    "Fixed k7 + rotate",
)
RL_POLICIES = set(POLICY_ORDER[:7])
COLORS = dict(zip(
    POLICY_ORDER,
    plt.get_cmap("tab20")(np.linspace(0, 0.95, len(POLICY_ORDER))),
))
COLORS["QR-DQN reliability-tuned"] = "#e45756"


def run_gated_episode(policy, scenario, seed, gate, budget_scale):
    env = DefenseEnv(scenario, budget_scale=budget_scale)
    observation, _ = env.reset(seed=seed)
    cheap_action = env.encode_action(1, "none")
    accumulator = (seed % 97) / 97.0
    while True:
        proposed, _ = policy.predict(observation, deterministic=True)
        accumulator += gate
        active = accumulator >= 1.0
        if active:
            accumulator -= 1.0
        action = int(np.asarray(proposed).item()) if active else cheap_action
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    return {
        "mean_cost": metrics["mean_cost"],
        "weighted_success_rate": metrics["weighted_success_rate"],
        "success_rate": metrics["success_rate"],
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_candidates(raw, per_attacker=False):
    groups = defaultdict(list)
    for row in raw:
        key = (
            row["policy"],
            row["attacker_count"] if per_attacker else "all",
            row["budget_scale"], row["gate"],
        )
        groups[key].append(row)
    candidates = []
    for (policy, attacker_count, budget, gate), rows in groups.items():
        candidates.append({
            "policy": policy,
            "attacker_count": attacker_count,
            "budget_scale": budget,
            "gate": gate,
            "episodes": len(rows),
            "mean_cost": float(np.mean([row["mean_cost"] for row in rows])),
            "weighted_success_rate": float(np.mean([
                row["weighted_success_rate"] for row in rows
            ])),
        })
    return candidates


def monotone_frontier(rows):
    ordered = sorted(rows, key=lambda row: (
        float(row["mean_cost"]), -float(row["weighted_success_rate"])
    ))
    collapsed = []
    for row in ordered:
        if collapsed and abs(row["mean_cost"] - collapsed[-1]["mean_cost"]) < 1e-9:
            if row["weighted_success_rate"] > collapsed[-1]["weighted_success_rate"]:
                collapsed[-1] = row
        else:
            collapsed.append(row)
    frontier = []
    best_success = -np.inf
    for row in collapsed:
        if row["weighted_success_rate"] > best_success + 1e-9:
            frontier.append(row)
            best_success = row["weighted_success_rate"]
    return frontier


def interpolate(frontier, cost):
    xs = np.asarray([row["mean_cost"] for row in frontier], dtype=float)
    ys = np.asarray([row["weighted_success_rate"] for row in frontier], dtype=float)
    if len(xs) < 2 or cost < xs.min() or cost > xs.max():
        return None
    return float(np.interp(cost, xs, ys))


def matched_results(overall_candidates, count_candidates, requested_target):
    overall_frontiers = {
        policy: monotone_frontier([
            row for row in overall_candidates if row["policy"] == policy
        ])
        for policy in POLICY_ORDER
    }
    count_frontiers = {
        (policy, count): monotone_frontier([
            row for row in count_candidates
            if row["policy"] == policy and int(row["attacker_count"]) == count
        ])
        for policy in POLICY_ORDER for count in (1, 2, 3, 5)
    }
    all_frontiers = list(overall_frontiers.values()) + list(count_frontiers.values())
    common_min = max(frontier[0]["mean_cost"] for frontier in all_frontiers)
    common_max = min(frontier[-1]["mean_cost"] for frontier in all_frontiers)
    if common_min >= common_max:
        raise RuntimeError(f"no common measured cost interval: {common_min}-{common_max}")
    target = float(np.clip(requested_target, common_min, common_max))
    grid_low = max(common_min, 0.75)
    grid_high = min(common_max, 2.0)
    if grid_low >= grid_high:
        grid_low, grid_high = common_min, common_max
    grid = np.linspace(grid_low, grid_high, 7)

    curves = []
    matched = []
    for policy in POLICY_ORDER:
        for cost in grid:
            value = interpolate(overall_frontiers[policy], float(cost))
            curves.append({
                "policy": policy,
                "mean_cost": float(cost),
                "weighted_success_rate": value,
            })
        for count in (1, 2, 3, 5):
            value = interpolate(count_frontiers[(policy, count)], target)
            matched.append({
                "policy": policy,
                "attacker_count": count,
                "matched_cost": target,
                "weighted_success_rate": value,
            })
    return curves, matched, {
        "common_min_cost": common_min,
        "common_max_cost": common_max,
        "target_cost": target,
    }


def make_figure(curves, matched, coverage, output):
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.2))
    ax_curve, ax_count, ax_rank = axes
    for policy in POLICY_ORDER:
        rows = [row for row in curves if row["policy"] == policy]
        linestyle = "-" if policy in RL_POLICIES else ":"
        width = 3.2 if policy == "QR-DQN reliability-tuned" else 1.7
        ax_curve.plot(
            [row["mean_cost"] for row in rows],
            [row["weighted_success_rate"] for row in rows],
            label=policy, color=COLORS[policy], linestyle=linestyle,
            linewidth=width,
        )
    ax_curve.set_title("A. Unified measured-cost frontier", weight="bold")
    ax_curve.set_xlabel("Average communication cost")
    ax_curve.set_ylabel("Importance-weighted success")
    ax_curve.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_curve.grid(True, alpha=0.22)

    for policy in POLICY_ORDER:
        rows = sorted(
            [row for row in matched if row["policy"] == policy],
            key=lambda row: row["attacker_count"],
        )
        linestyle = "-" if policy in RL_POLICIES else ":"
        width = 3.2 if policy == "QR-DQN reliability-tuned" else 1.7
        ax_count.plot(
            [row["attacker_count"] for row in rows],
            [row["weighted_success_rate"] for row in rows],
            marker="o", label=policy, color=COLORS[policy],
            linestyle=linestyle, linewidth=width,
        )
    target = coverage["target_cost"]
    ax_count.set_title(f"B. Same cost = {target:.3f}", weight="bold")
    ax_count.set_xlabel("Independent attackers")
    ax_count.set_ylabel("Importance-weighted success")
    ax_count.set_xticks((1, 2, 3, 5))
    ax_count.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_count.grid(True, alpha=0.22)

    overall = []
    for policy in POLICY_ORDER:
        values = [
            row["weighted_success_rate"] for row in matched
            if row["policy"] == policy
        ]
        overall.append((policy, float(np.mean(values))))
    overall.sort(key=lambda item: item[1])
    ax_rank.barh(
        [item[0] for item in overall], [item[1] for item in overall],
        color=[COLORS[item[0]] for item in overall], alpha=0.90,
    )
    ax_rank.set_title(f"C. Mean over 1/2/3/5 attackers\nat cost {target:.3f}", weight="bold")
    ax_rank.set_xlabel("Importance-weighted success")
    ax_rank.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_rank.grid(True, axis="x", alpha=0.22)

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=4, fontsize=8.3, framealpha=0.95)
    fig.suptitle(
        "Multi-attacker Defense at Identical Average Communication Cost",
        fontsize=16, weight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.012,
        "Measured operating points only; Pareto-frontier interpolation within the common "
        "coverage interval. Three held-out seeds, four attack conditions, no extrapolation.",
        ha="center", fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_policies(args):
    return {
        "QR-DQN original": QRDQN.load(args.original, device="cpu"),
        "QR-DQN scratch multi": QRDQN.load(args.scratch, device="cpu"),
        "QR-DQN uniform fine-tuned": QRDQN.load(args.fine_tuned, device="cpu"),
        "QR-DQN curriculum": QRDQN.load(args.curriculum, device="cpu"),
        "QR-DQN reliability-tuned": QRDQN.load(args.reliability, device="cpu"),
        "TRPO robust": TRPO.load(args.trpo, device="cpu"),
        "PPO": PPO.load(args.ppo, device="cpu"),
        "Risk heuristic": RiskHeuristicPolicy(),
        "Fixed k3 + rotate": FixedPolicy(3, "rotate"),
        "Fixed k5 + rotate": FixedPolicy(5, "rotate"),
        "Fixed k7 + rotate": FixedPolicy(7, "rotate"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path(
        "experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"))
    parser.add_argument("--scratch", type=Path, default=Path(
        "experiment/results/rl_multi_attacker/models/qrdqn_seed2026.zip"))
    parser.add_argument("--fine-tuned", type=Path, default=Path(
        "experiment/results/rl_multi_attacker_finetuned/models/qrdqn_seed2026.zip"))
    parser.add_argument("--curriculum", type=Path, default=Path(
        "experiment/results/rl_multi_attacker_curriculum/models/qrdqn_selected_seed2026.zip"))
    parser.add_argument("--reliability", type=Path, default=Path(
        "experiment/results/rl_multi_attacker_reliability/models/qrdqn_seed2026.zip"))
    parser.add_argument("--trpo", type=Path, default=Path(
        "experiment/results/rl_robust_w015/models/trpo_seed2026.zip"))
    parser.add_argument("--ppo", type=Path, default=Path(
        "experiment/results/rl_v2/models/ppo_seed2026.zip"))
    parser.add_argument("--output", type=Path, default=Path(
        "experiment/results/rl_multi_attacker_final/analysis/"
        "matched_cost_multi_attacker.png"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3101, 3202, 3303])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.80, 1.20])
    parser.add_argument(
        "--gates", nargs="+", type=float,
        default=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
    )
    # Cost 1.0 is the preregistered reference used by the earlier unified-cost
    # comparison.  Keeping it here avoids choosing a favorable point after
    # looking at the multi-attacker curves.
    parser.add_argument("--target-cost", type=float, default=1.00)
    args = parser.parse_args()

    policies = load_policies(args)
    bases = ("persistent", "adaptive", "noisy_ood", "budget_shock")
    rows = []
    for policy_name, policy in policies.items():
        for base_name in bases:
            for attacker_count in (1, 2, 3, 5):
                scenario = replace(
                    SCENARIOS[base_name],
                    name=f"{base_name}_attackers{attacker_count}",
                    attacker_count=attacker_count,
                )
                for budget in args.budgets:
                    for gate in args.gates:
                        for seed in args.seeds:
                            result = run_gated_episode(
                                policy, scenario, seed, gate, budget
                            )
                            result.update({
                                "policy": policy_name,
                                "base_scenario": base_name,
                                "attacker_count": attacker_count,
                                "budget_scale": budget,
                                "gate": gate,
                                "seed": seed,
                            })
                            rows.append(result)
        print(f"evaluated {policy_name}", flush=True)

    overall_candidates = aggregate_candidates(rows, per_attacker=False)
    count_candidates = aggregate_candidates(rows, per_attacker=True)
    curves, matched, coverage = matched_results(
        overall_candidates, count_candidates, args.target_cost
    )
    make_figure(curves, matched, coverage, args.output)
    write_csv(args.output.with_name("matched_cost_multi_raw.csv"), rows)
    write_csv(args.output.with_name("matched_cost_multi_candidates.csv"), overall_candidates)
    write_csv(args.output.with_name("matched_cost_multi_by_count_candidates.csv"), count_candidates)
    write_csv(args.output.with_name("matched_cost_multi_curves.csv"), curves)
    write_csv(args.output.with_name("matched_cost_multi_by_count.csv"), matched)
    write_csv(args.output.with_name("matched_cost_multi_coverage.csv"), [coverage])
    print({**coverage, "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
