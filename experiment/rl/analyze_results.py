"""Create compact tables and figures without final-stage confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def is_non_rl(name: str) -> bool:
    return name.startswith("fixed_") or name in {"heuristic", "conservative"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiment/results/rl_v1/episodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_v1/analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    metrics = [
        "return", "success_rate", "weighted_success_rate", "mean_cost",
        "budget_violation_rate", "mean_redundancy", "recovery_rate",
    ]
    overall = frame.groupby("policy", as_index=False)[metrics].mean()
    overall["family"] = overall.policy.map(lambda x: "non-RL" if is_non_rl(x) else "RL")
    overall = overall.sort_values("return", ascending=False)
    overall.to_csv(args.output / "overall.csv", index=False)

    cells = frame.groupby(["policy", "scenario", "budget_scale"], as_index=False)[metrics].mean()
    winners = cells.loc[cells.groupby(["scenario", "budget_scale"])["return"].idxmax()].copy()
    winners.to_csv(args.output / "cell_winners.csv", index=False)
    win_counts = winners.groupby("policy").size().sort_values(ascending=False)
    win_counts.rename("wins").to_csv(args.output / "win_counts.csv")

    baseline = cells[cells.policy.map(is_non_rl)].groupby(
        ["scenario", "budget_scale"], as_index=False
    )["return"].max().rename(columns={"return": "best_non_rl_return"})
    rl = cells[~cells.policy.map(is_non_rl)].merge(baseline, on=["scenario", "budget_scale"])
    rl["gain_over_best_non_rl"] = rl["return"] - rl["best_non_rl_return"]
    rl.to_csv(args.output / "rl_vs_best_non_rl.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#377eb8" if family == "RL" else "#999999" for family in overall.family]
    ax.bar(overall.policy, overall["return"], color=colors)
    ax.set_ylabel("Mean episode return")
    ax.set_title("RL and non-RL defense policies across all evaluation cells")
    ax.tick_params(axis="x", rotation=42)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "overall_return.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for family, group in overall.groupby("family"):
        ax.scatter(group.mean_cost, group.weighted_success_rate, s=75, label=family)
        for row in group.itertuples():
            ax.annotate(row.policy, (row.mean_cost, row.weighted_success_rate), fontsize=8)
    ax.set_xlabel("Mean normalized communication cost (lower is better)")
    ax.set_ylabel("Importance-weighted success rate (higher is better)")
    ax.set_title("Reliability-cost trade-off")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "cost_success_tradeoff.png", dpi=180)
    plt.close(fig)

    best_rl = overall[overall.family == "RL"].iloc[0]
    best_non_rl = overall[overall.family == "non-RL"].iloc[0]
    report = f"""# RL v1 result digest

This is an interim screening report. It intentionally reports mean/min/max,
not final-stage confidence intervals.

## Overall ranking

{overall.to_markdown(index=False, floatfmt='.4f')}

The best RL policy by mean episode return is **{best_rl.policy}**
({best_rl['return']:.3f}). The best non-RL policy is **{best_non_rl.policy}**
({best_non_rl['return']:.3f}), for a mean gain of
**{best_rl['return'] - best_non_rl['return']:.3f}**.

## Evaluation-cell wins

{win_counts.to_frame().to_markdown()}
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
