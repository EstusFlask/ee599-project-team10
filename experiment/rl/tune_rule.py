"""Random-search the few gains of ExactEstimatorPolicy, then compare at matched cost.

Candidates are scored on training seeds; the (cost, success) hull of those
scores picks which candidates to carry forward; held-out seeds and all four
scenarios give the reported matched-cost gaps against the fixed-k hull.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate_large_scale import COST_GRID, run_episode, upper_hull, value_at
from dataclasses import replace

from .large_scale import SCALE_SCENARIOS, ExactEstimatorPolicy, ScaleFixedPolicy

SEARCH = {
    "alpha": (0.2, 0.9), "margin": (-1.0, 3.0), "value_gain": (0.0, 1.0),
    "pace_gain": (0.0, 3.0), "trend_gain": (0.0, 2.0), "probe_every": (3, 16),
}


def sample(rng: np.random.Generator, fixed: dict | None = None) -> dict:
    values = {k: float(rng.uniform(*v)) for k, v in SEARCH.items()}
    values["probe_every"] = int(round(values["probe_every"]))
    values.update(fixed or {})  # ablations pin a gain instead of searching it
    return values


def score(make, scenarios, seeds):
    rows = [run_episode(make(), s, 1.0, seed) for s in scenarios for seed in seeds]
    return (float(np.mean([r["mean_cost"] for r in rows])),
            float(np.mean([r["weighted_success"] for r in rows])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=150)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(100, 106)))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(range(1000, 1020)))
    parser.add_argument("--delay", type=int, default=1,
                        help="Rounds before the signed receipt reveals success.")
    parser.add_argument("--no-success", action="store_true",
                        help="Receiver cannot tell whether the last round succeeded.")
    parser.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE",
                        help="Pin gains instead of searching them (ablations).")
    parser.add_argument("--tag", help="Output file tag (default from delay / no-success).")
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/rl_large_scale/rule_tuning"))
    args = parser.parse_args()
    fixed = {}
    for item in args.fix:
        name, value = item.split("=")
        if name not in SEARCH:
            raise SystemExit(f"unknown parameter {name!r}; choose from {sorted(SEARCH)}")
        fixed[name] = int(value) if name == "probe_every" else float(value)
    channels = SCALE_SCENARIOS["campaign"].channels
    rng = np.random.default_rng(7)
    train_scenarios = ["independent", "waves", "campaign", "campaign_waves"]
    use_success = not args.no_success
    scenarios = {
        name: replace(config, feedback_delay=args.delay)
        for name, config in SCALE_SCENARIOS.items()
    }
    train_scenarios = [scenarios[name] for name in train_scenarios]

    candidates = []
    for _ in range(args.candidates):
        params = sample(rng, fixed)
        cost, success = score(
            lambda p=params: ExactEstimatorPolicy(channels, use_success=use_success, delay=args.delay, **p),
            train_scenarios, args.train_seeds,
        )
        candidates.append((cost, success, params))
    hull_points = upper_hull([(c, s) for c, s, _ in candidates])
    chosen = [c for c in candidates if (c[0], c[1]) in set(hull_points)]
    print(f"{len(chosen)} train-hull candidates of {len(candidates)}", flush=True)

    summary = []
    for scenario_name, scenario in scenarios.items():
        fixed = []
        for k in range(1, 2 * channels + 2, 2):
            fixed.append(score(lambda k=k: ScaleFixedPolicy(k), [scenario], args.test_seeds))
        fixed_hull = upper_hull(fixed)
        points = [
            score(lambda p=p: ExactEstimatorPolicy(channels, use_success=use_success, delay=args.delay, **p),
                  [scenario], args.test_seeds)
            for _, _, p in chosen
        ]
        rule_hull = upper_hull(points)
        print(f"\n== {scenario} (held-out; gap vs fixed hull, pp)")
        for cost in COST_GRID:
            base, rule = value_at(fixed_hull, cost), value_at(rule_hull, cost)
            if np.isnan(base) or np.isnan(rule):
                continue
            summary.append({"scenario": scenario_name, "cost": cost, "fixed": base,
                            "rule": rule, "gap_pp": 100 * (rule - base)})
            print(f"cost {cost:4.1f}: fixed {base:.3f} | rule {rule:.3f} ({100 * (rule - base):+5.1f})")
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("no_success" if args.no_success else f"receipt_delay{args.delay}")
    with (args.output / f"gaps_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / f"chosen_{tag}.json").write_text(
        json.dumps([{"train_cost": c, "train_success": s, "params": p} for c, s, p in chosen],
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
