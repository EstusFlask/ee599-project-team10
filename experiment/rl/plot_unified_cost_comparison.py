"""Full comparison on one shared grid of measured communication costs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .plot_budget_success_comparison import HARD_SCENARIOS
from .plot_matched_cost_comparison import (
    DISPLAY_ORDER,
    run_gated_episode,
    selected_policies,
)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    ci = 0.0 if len(values) < 2 else float(
        1.96 * values.std(ddof=1) / math.sqrt(len(values))
    )
    return mean, ci


def aggregate_candidates(raw, budget_scales, gates, policy_names=DISPLAY_ORDER):
    rows = []
    for policy in policy_names:
        for budget_scale in budget_scales:
            for gate in gates:
                selected = [
                    row for row in raw
                    if row["policy"] == policy
                    and row["budget_scale"] == budget_scale
                    and row["gate"] == gate
                ]
                cost, cost_ci = mean_ci([row["mean_cost"] for row in selected])
                success, success_ci = mean_ci([
                    row["weighted_success_rate"] for row in selected
                ])
                rows.append({
                    "policy": policy,
                    "family": selected[0]["family"],
                    "budget_scale": budget_scale,
                    "gate": gate,
                    "mean_cost": cost,
                    "cost_ci95": cost_ci,
                    "weighted_success_rate": success,
                    "success_ci95": success_ci,
                    "episodes": len(selected),
                })
    return rows


def monotone_frontier(rows):
    """Best measured success attainable at or below each observed cost."""
    ordered = sorted(rows, key=lambda row: (row["mean_cost"], -row["weighted_success_rate"]))
    frontier = []
    best = None
    for row in ordered:
        if best is None or row["weighted_success_rate"] > best["weighted_success_rate"]:
            best = row
            frontier.append(row)
    return frontier


def interpolate_frontier(frontier, grid):
    xs = np.asarray([row["mean_cost"] for row in frontier], dtype=float)
    ys = np.asarray([row["weighted_success_rate"] for row in frontier], dtype=float)
    cis = np.asarray([row["success_ci95"] for row in frontier], dtype=float)
    if grid.min() < xs.min() or grid.max() > xs.max():
        return None
    return np.interp(grid, xs, ys), np.interp(grid, xs, cis)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(unified, matched_cost, output):
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
        rows = sorted(
            [row for row in unified if row["policy"] == name],
            key=lambda row: row["communication_cost"],
        )
        xs = np.asarray([row["communication_cost"] for row in rows])
        ys = np.asarray([row["weighted_success_rate"] for row in rows])
        cis = np.asarray([row["ci95"] for row in rows])
        family = rows[0]["family"]
        style = "--" if family == "ml" else "-"
        width = 3.0 if name == "QR-DQN robust (w=0.15)" else 1.8
        axes[0].plot(
            xs, ys, label=name, color=colors[name], linestyle=style,
            marker=markers[index], markersize=5.0, linewidth=width,
        )
        axes[0].fill_between(xs, ys - cis, ys + cis, color=colors[name], alpha=0.08)
        matched.append((
            float(np.interp(matched_cost, xs, ys)),
            float(np.interp(matched_cost, xs, cis)),
        ))

    axes[0].axvline(matched_cost, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Average normalized communication cost")
    axes[0].set_ylabel("Importance-weighted defense success rate")
    axes[0].set_title("Unified success-cost curves (mean and 95% CI)", weight="bold")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(alpha=0.24)
    axes[0].legend(fontsize=7.7, ncol=2, loc="lower right", framealpha=0.94)

    positions = np.arange(len(DISPLAY_ORDER))
    axes[1].bar(
        positions, [item[0] for item in matched],
        yerr=[item[1] for item in matched], capsize=3,
        color=[colors[name] for name in DISPLAY_ORDER], alpha=0.93,
    )
    axes[1].set_xticks(positions, DISPLAY_ORDER, rotation=42, ha="right", fontsize=8)
    axes[1].set_ylabel("Importance-weighted defense success rate")
    axes[1].set_title(f"Matched cost = {matched_cost:.3f}", weight="bold")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.25, 0.82)
    axes[1].grid(axis="y", alpha=0.24)

    fig.suptitle("Complex-Attack Defense on a Unified Cost Grid", fontsize=15, weight="bold")
    fig.text(
        0.5, 0.012,
        "Every method is interpolated onto the same measured-cost grid from its best observed "
        "budget/gating frontier. ML curves are dashed; RL curves are solid.",
        ha="center", fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("experiment/results"))
    parser.add_argument("--output", type=Path, default=Path(
        "experiment/results/rl_robust_w015/analysis/unified_cost_success_complex.png"
    ))
    parser.add_argument("--budget-scales", nargs="+", type=float, default=[0.55, 0.80, 1.05, 1.30, 1.55])
    parser.add_argument("--gates", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
    parser.add_argument("--cost-grid", nargs="+", type=float, default=[0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
    parser.add_argument("--matched-cost", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    parser.add_argument("--ml-samples", type=int, default=40_000)
    args = parser.parse_args()

    policies = selected_policies(args.result_root, args.ml_samples)
    raw = []
    for name in DISPLAY_ORDER:
        policy, _ = policies[name]
        for budget_scale in args.budget_scales:
            for gate in args.gates:
                for scenario in HARD_SCENARIOS:
                    for seed in args.seeds:
                        raw.append(run_gated_episode(
                            policy, name, scenario, seed, gate,
                            budget_scale=budget_scale,
                        ))
        print(f"evaluated {name}", flush=True)

    candidates = aggregate_candidates(raw, args.budget_scales, args.gates)
    grid = np.asarray(args.cost_grid, dtype=float)
    unified = []
    coverage = []
    for name in DISPLAY_ORDER:
        rows = [row for row in candidates if row["policy"] == name]
        frontier = monotone_frontier(rows)
        values = interpolate_frontier(frontier, grid)
        coverage.append({
            "policy": name,
            "frontier_min_cost": frontier[0]["mean_cost"],
            "frontier_max_cost": frontier[-1]["mean_cost"],
            "covers_grid": values is not None,
        })
        if values is None:
            raise RuntimeError(
                f"{name} does not cover common grid {grid.min():.2f}-{grid.max():.2f}; "
                f"frontier is {frontier[0]['mean_cost']:.3f}-{frontier[-1]['mean_cost']:.3f}"
            )
        ys, cis = values
        family = rows[0]["family"]
        for cost, success, ci in zip(grid, ys, cis):
            unified.append({
                "policy": name,
                "family": family,
                "communication_cost": float(cost),
                "weighted_success_rate": float(success),
                "ci95": float(ci),
            })

    make_figure(unified, args.matched_cost, args.output)
    write_csv(args.output.with_name("unified_cost_raw.csv"), raw)
    write_csv(args.output.with_name("unified_cost_candidates.csv"), candidates)
    write_csv(args.output.with_name("unified_cost_coverage.csv"), coverage)
    write_csv(args.output.with_name("unified_cost_summary.csv"), unified)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
