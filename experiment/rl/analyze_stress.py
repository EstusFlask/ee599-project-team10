"""Summarize the full factorial stress test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def is_non_rl(name: str) -> bool:
    return name.startswith("fixed_") or name in {"heuristic", "conservative"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiment/results/rl_v1/stress_episodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_v1/analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    metrics = ["return", "weighted_success_rate", "mean_cost", "budget_violation_rate"]
    cells = frame.groupby(
        ["policy", "persistence", "false_negative", "adaptive_strength", "budget_scale"],
        as_index=False,
    )[metrics].mean()
    overall = cells.groupby("policy", as_index=False)[metrics].mean().sort_values("return", ascending=False)
    overall.to_csv(args.output / "stress_overall.csv", index=False)

    keys = ["persistence", "false_negative", "adaptive_strength", "budget_scale"]
    winners = cells.loc[cells.groupby(keys)["return"].idxmax()].copy()
    wins = winners.groupby("policy").size().sort_values(ascending=False).rename("wins")
    wins.to_csv(args.output / "stress_win_counts.csv")

    baseline = cells[cells.policy.map(is_non_rl)].groupby(keys, as_index=False)["return"].max()
    baseline = baseline.rename(columns={"return": "best_non_rl_return"})
    gains = cells[~cells.policy.map(is_non_rl)].merge(baseline, on=keys)
    gains["gain_over_best_non_rl"] = gains["return"] - gains["best_non_rl_return"]
    gains.to_csv(args.output / "stress_rl_gains.csv", index=False)

    by_policy = gains.groupby("policy")["gain_over_best_non_rl"].agg(
        ["mean", "min", "max", lambda x: int((x > 0).sum())]
    ).rename(columns={"<lambda_0>": "winning_cells"}).sort_values("mean", ascending=False)
    by_policy.to_csv(args.output / "stress_rl_gain_summary.csv")

    sensitivity_rows = []
    for axis in keys:
        grouped = cells.groupby(["policy", axis], as_index=False)[metrics].mean()
        grouped.insert(0, "axis", axis)
        grouped = grouped.rename(columns={axis: "axis_value"})
        sensitivity_rows.append(grouped)
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity.to_csv(args.output / "stress_sensitivity.csv", index=False)

    print("Overall factorial ranking:\n" + overall.to_string(index=False), flush=True)
    print("\nCell wins:\n" + wins.to_string(), flush=True)
    print("\nRL gain over strongest non-RL:\n" + by_policy.to_string(), flush=True)


if __name__ == "__main__":
    main()
