"""Recreate the original success-cost curve plus matched-cost bar chart.

Operating points are produced by budget gating: a fixed fraction of steps use
the candidate policy and the rest use the minimum-cost k=1/no-recovery action.
This gives every method a directly measured success-cost curve in the same
complex scenarios without retraining it for each budget.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .environment import DefenseEnv
from .plot_budget_success_comparison import HARD_SCENARIOS, load_policies


DISPLAY_ORDER = (
    "Fixed baseline",
    "Stateless ML",
    "Stateful ML",
    "Linear Q (initial)",
    "A2C",
    "DQN",
    "PPO",
    "QR-DQN",
    "RecurrentPPO",
    "MaskablePPO robust",
    "TRPO robust",
    "QR-DQN robust (w=0.15)",
)


def selected_policies(result_root: Path, ml_samples: int):
    loaded = load_policies(result_root, ml_samples, seed=2026)
    return {
        "Fixed baseline": loaded["Fixed k3 + rotate"],
        "Stateless ML": loaded["Stateless ML"],
        "Stateful ML": loaded["Stateful ML"],
        "Linear Q (initial)": loaded["Linear Q (initial)"],
        "A2C": loaded["A2C"],
        "DQN": loaded["DQN"],
        "PPO": loaded["PPO"],
        "QR-DQN": loaded["QR-DQN"],
        "RecurrentPPO": loaded["RecurrentPPO"],
        "MaskablePPO robust": loaded["MaskablePPO robust"],
        "TRPO robust": loaded["TRPO robust"],
        "QR-DQN robust (w=0.15)": loaded["QR-DQN robust (w=0.15)"],
    }


def run_gated_episode(
    policy, policy_name: str, scenario: str, seed: int, gate: float,
    budget_scale: float = 1.0,
):
    env = DefenseEnv(scenario, budget_scale=budget_scale)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    phase = (seed % 97) / 97.0
    accumulator = phase
    cheap_action = env.encode_action(1, "none")
    while True:
        if policy_name == "MaskablePPO robust":
            proposed, _ = policy.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
        elif policy_name == "RecurrentPPO":
            proposed, recurrent_state = policy.predict(
                obs, state=recurrent_state, episode_start=episode_start,
                deterministic=True,
            )
            episode_start[:] = False
        else:
            proposed, _ = policy.predict(obs, deterministic=True)
        accumulator += gate
        active = accumulator >= 1.0
        if active:
            accumulator -= 1.0
        action = int(np.asarray(proposed).item()) if active else cheap_action
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    return {
        "policy": policy_name,
        "family": "baseline" if policy_name == "Fixed baseline" else (
            "ml" if policy_name in {"Stateless ML", "Stateful ML"} else "rl"
        ),
        "gate": gate,
        "budget_scale": budget_scale,
        "scenario": scenario,
        "seed": seed,
        "mean_cost": metrics["mean_cost"],
        "weighted_success_rate": metrics["weighted_success_rate"],
        "success_rate": metrics["success_rate"],
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def summarize(rows, gates):
    summary = []
    for policy in DISPLAY_ORDER:
        for gate in gates:
            selected = [
                row for row in rows
                if row["policy"] == policy and row["gate"] == gate
            ]
            cost_mean, cost_ci = mean_ci([row["mean_cost"] for row in selected])
            success_mean, success_ci = mean_ci([
                row["weighted_success_rate"] for row in selected
            ])
            summary.append({
                "policy": policy,
                "family": selected[0]["family"],
                "gate": gate,
                "mean_cost": cost_mean,
                "cost_ci95": cost_ci,
                "weighted_success_rate": success_mean,
                "success_ci95": success_ci,
                "episodes": len(selected),
            })
    return summary


def interpolate_at_cost(rows, target_cost: float):
    ordered = sorted(rows, key=lambda row: row["mean_cost"])
    # Collapse duplicate costs created by policies that ignore the gate.
    collapsed = []
    for row in ordered:
        if collapsed and abs(row["mean_cost"] - collapsed[-1]["mean_cost"]) < 1e-8:
            if row["weighted_success_rate"] > collapsed[-1]["weighted_success_rate"]:
                collapsed[-1] = row
        else:
            collapsed.append(row)
    xs = np.asarray([row["mean_cost"] for row in collapsed])
    if target_cost < xs.min() or target_cost > xs.max() or len(xs) < 2:
        return None
    return {
        "success": float(np.interp(
            target_cost, xs,
            [row["weighted_success_rate"] for row in collapsed],
        )),
        "ci95": float(np.interp(
            target_cost, xs,
            [row["success_ci95"] for row in collapsed],
        )),
        "min_cost": float(xs.min()),
        "max_cost": float(xs.max()),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary, target_cost: float, output: Path):
    colors = dict(zip(
        DISPLAY_ORDER,
        plt.get_cmap("tab20")(np.linspace(0, 0.95, len(DISPLAY_ORDER))),
    ))
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "8", "*")
    fig, axes = plt.subplots(
        1, 2, figsize=(17.0, 6.3),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )
    matched = []
    for index, name in enumerate(DISPLAY_ORDER):
        rows = [row for row in summary if row["policy"] == name]
        rows = sorted(rows, key=lambda row: row["mean_cost"])
        family = rows[0]["family"]
        style = "--" if family == "ml" else "-"
        width = 3.0 if name == "QR-DQN robust (w=0.15)" else 1.7
        xs = np.asarray([row["mean_cost"] for row in rows])
        ys = np.asarray([row["weighted_success_rate"] for row in rows])
        cis = np.asarray([row["success_ci95"] for row in rows])
        axes[0].plot(
            xs, ys, label=name, color=colors[name], linestyle=style,
            marker=markers[index], markersize=4.8, linewidth=width,
        )
        axes[0].fill_between(xs, ys - cis, ys + cis, color=colors[name], alpha=0.08)
        matched.append(interpolate_at_cost(rows, target_cost))

    axes[0].axvline(target_cost, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Average normalized communication cost")
    axes[0].set_ylabel("Importance-weighted defense success rate")
    axes[0].set_title("Success-cost curves (mean and 95% CI)", weight="bold")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(alpha=0.24)
    axes[0].legend(fontsize=7.7, ncol=2, loc="lower right", framealpha=0.94)

    positions = np.arange(len(DISPLAY_ORDER))
    available = [item is not None for item in matched]
    heights = [item["success"] if item else 0.31 for item in matched]
    errors = [item["ci95"] if item else 0.0 for item in matched]
    bars = axes[1].bar(
        positions, heights, yerr=errors, capsize=3,
        color=[colors[name] if ok else "white" for name, ok in zip(DISPLAY_ORDER, available)],
        edgecolor=[colors[name] for name in DISPLAY_ORDER], linewidth=1.3,
    )
    for index, (bar, item) in enumerate(zip(bars, matched)):
        if item is None:
            bar.set_hatch("///")
            axes[1].text(
                index, 0.325, "N/A", ha="center", va="bottom",
                rotation=90, fontsize=7.5, color=colors[DISPLAY_ORDER[index]],
            )
    axes[1].set_xticks(positions, DISPLAY_ORDER, rotation=42, ha="right", fontsize=8)
    axes[1].set_ylabel("Importance-weighted defense success rate")
    axes[1].set_title(f"Matched cost = {target_cost:.3f}", weight="bold")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.30, 1.02)
    axes[1].grid(axis="y", alpha=0.24)

    fig.suptitle("Complex-Attack Defense at Equal Communication Cost", fontsize=15, weight="bold")
    fig.text(
        0.5, 0.012,
        "Operating points vary the defense activation duty cycle; x-axis values are measured costs. "
        "ML curves are dashed. N/A means the policy never reaches the matched cost.",
        ha="center", fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    matched_rows = []
    for name, item in zip(DISPLAY_ORDER, matched):
        matched_rows.append({
            "policy": name,
            "matched_cost": target_cost,
            "weighted_success_rate": "" if item is None else item["success"],
            "ci95": "" if item is None else item["ci95"],
            "status": "not_reached" if item is None else "interpolated",
        })
    return matched_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("experiment/results"))
    parser.add_argument("--output", type=Path, default=Path(
        "experiment/results/rl_robust_w015/analysis/matched_cost_success_complex.png"
    ))
    parser.add_argument("--target-cost", type=float, default=1.0)
    parser.add_argument("--gates", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    parser.add_argument("--ml-samples", type=int, default=40_000)
    args = parser.parse_args()

    policies = selected_policies(args.result_root, args.ml_samples)
    raw = []
    for name in DISPLAY_ORDER:
        policy, _ = policies[name]
        for gate in args.gates:
            for scenario in HARD_SCENARIOS:
                for seed in args.seeds:
                    raw.append(run_gated_episode(policy, name, scenario, seed, gate))
        print(f"evaluated {name}", flush=True)
    summary = summarize(raw, args.gates)
    matched = make_figure(summary, args.target_cost, args.output)
    write_csv(args.output.with_name("matched_cost_success_raw.csv"), raw)
    write_csv(args.output.with_name("matched_cost_success_summary.csv"), summary)
    write_csv(args.output.with_name("matched_cost_success_at_target.csv"), matched)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
