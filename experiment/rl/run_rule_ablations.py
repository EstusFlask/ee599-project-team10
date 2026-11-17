"""Ablate the tuned estimator rule: remove one input or mechanism at a time.

Each variant repeats the identical random search (same seed, 150 candidates)
with one gain pinned or one assumption changed, then reports the held-out gap
to the fixed-k hull.  The full rule is read from the main tuning run.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

VARIANTS = {
    "no_value": ["--fix", "value_gain=0"],
    "no_pace": ["--fix", "pace_gain=0"],
    "no_trend": ["--fix", "trend_gain=0"],
    "no_probe": ["--fix", "probe_every=1000"],
    "margin_only": ["--fix", "value_gain=0", "pace_gain=0", "trend_gain=0", "probe_every=1000"],
    "no_receipt": ["--no-success"],
    "receipt_delay0": ["--delay", "0"],
    "receipt_delay2": ["--delay", "2"],
}
COSTS = (4.0, 8.0, 12.0)


def summarize(files: dict[str, Path], output: Path) -> None:
    rows = []
    for name, path in files.items():
        if not path.exists():
            continue
        gaps = {cost: [] for cost in COSTS}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if float(row["cost"]) in gaps:
                gaps[float(row["cost"])].append(float(row["gap_pp"]))
        rows.append({
            "variant": name,
            **{f"mean_gap_cost{cost}": float(np.mean(v)) if v else float("nan")
               for cost, v in gaps.items()},
        })
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("mean gap to fixed hull over the scenarios with data (pp; + = rule better)")
    for row in rows:
        print(f"{row['variant']:16s}", "  ".join(
            f"cost {cost}: {row[f'mean_gap_cost{cost}']:+5.1f}" for cost in COSTS
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path(
        "experiment/results/rl_large_scale/rule_tuning_zero_start/gaps_receipt_delay1.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/rl_large_scale/rule_ablation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"full": args.full}
    for name, extra in VARIANTS.items():
        target = args.output / f"gaps_{name}.csv"
        if not target.exists():
            subprocess.run(
                [sys.executable, "-m", "experiment.rl.tune_rule", "--candidates", "150",
                 "--tag", name, "--output", str(args.output), *extra],
                check=True, stdout=subprocess.DEVNULL,
            )
        files[name] = target
        print(f"done {name}", flush=True)
    summarize(files, args.output / "summary.csv")


if __name__ == "__main__":
    main()
