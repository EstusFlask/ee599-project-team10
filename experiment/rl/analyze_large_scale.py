"""Final comparison with a frozen fixed baseline and bootstrap intervals.

Fixed-k operating points are chosen on separate validation seeds (their upper
hull vertices), then re-measured on the test seeds, so the baseline is not
selected on the data it is judged on.  Gaps are computed per training seed of
RL and averaged; intervals resample test seeds, paired across all methods.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate_large_scale import COST_GRID, upper_hull, value_at


def load(path: Path):
    """scenario -> policy-point -> (family, seeds[], cost[], success[])."""
    table = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    families = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["policy"], row["price"] or row["budget_scale"])
        seeds, cost, success = table[row["scenario"]][key]
        seeds.append(int(row["seed"]))
        cost.append(float(row["mean_cost"]))
        success.append(float(row["weighted_success"]))
        families[(row["scenario"], key)] = row["family"]
    out = {}
    for scenario, points in table.items():
        out[scenario] = {}
        for key, (seeds, cost, success) in points.items():
            order = np.argsort(seeds)
            out[scenario][key] = (
                families[(scenario, key)], np.asarray(seeds)[order],
                np.asarray(cost)[order], np.asarray(success)[order],
            )
    return out


def hull_vertices(points: dict) -> set:
    means = {key: (cost.mean(), success.mean()) for key, (_, _, cost, success) in points.items()}
    hull = set(upper_hull(list(means.values())))
    return {key for key, value in means.items() if value in hull}


def hull_curve(points: dict, keys, index) -> list:
    return upper_hull([
        (points[key][2][index].mean(), points[key][3][index].mean()) for key in keys
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True, help="episodes.csv on test seeds")
    parser.add_argument("--validation", type=Path, required=True,
                        help="fixed-only episodes.csv on validation seeds")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    test, validation = load(args.test), load(args.validation)
    rng = np.random.default_rng(0)
    rows = []
    for scenario, points in test.items():
        n_seeds = len(next(iter(points.values()))[1])
        frozen = [
            key for key in hull_vertices(
                {k: v for k, v in validation[scenario].items() if v[0] == "fixed"}
            ) if key in points
        ]
        # The main method (QR-DQN, averaged over training seeds) vs. every other model on its own.
        rl_families = sorted({v[0] for v in points.values() if v[0].startswith("rl_qrdqn_seed")})
        ml_families = sorted({
            v[0] for v in points.values()
            if v[0].startswith("ml_") or (v[0].startswith("rl_") and v[0] not in rl_families)
        })
        groups = {
            "rule": [k for k, v in points.items() if v[0] == "rule"],
            **{family: [k for k, v in points.items() if v[0] == family]
               for family in rl_families + ml_families},
        }

        def gaps(index):
            base = hull_curve(points, frozen, index)
            result = {}
            for name, keys in groups.items():
                curve = hull_curve(points, keys, index)
                result[name] = [
                    100 * (value_at(curve, c) - value_at(base, c)) for c in COST_GRID
                ]
            result["fixed"] = [value_at(base, c) for c in COST_GRID]
            return result

        observed = gaps(np.arange(n_seeds))
        draws = [gaps(rng.integers(0, n_seeds, n_seeds)) for _ in range(args.bootstrap)]

        def summarize(name):
            values = np.asarray([d[name] for d in draws], dtype=float)
            return values

        rl_draws = np.stack([summarize(f) for f in rl_families]).mean(axis=0) if rl_families else None
        rl_observed = np.nanmean([observed[f] for f in rl_families], axis=0) if rl_families else None
        for i, cost in enumerate(COST_GRID):
            row = {"scenario": scenario, "cost": cost, "fixed_success": observed["fixed"][i],
                   "rule_gap": observed["rule"][i] if "rule" in observed else float("nan")}
            if rl_families:
                column = rl_draws[:, i]
                column = column[~np.isnan(column)]
                row.update({
                    "rl_gap": rl_observed[i],
                    "rl_ci_low": np.percentile(column, 2.5) if len(column) else float("nan"),
                    "rl_ci_high": np.percentile(column, 97.5) if len(column) else float("nan"),
                    "rl_prob_better": float((column > 0).mean()) if len(column) else float("nan"),
                    "rl_seeds_positive": int(sum(observed[f][i] > 0 for f in rl_families
                                                 if not np.isnan(observed[f][i]))),
                    "rl_seeds": len(rl_families),
                })
            for family in rl_families:  # per-training-seed gaps (used for same-seed ablations)
                row[f"{family}_gap"] = observed[family][i]
            for family in ml_families:
                column = np.asarray([d[family][i] for d in draws], dtype=float)
                column = column[~np.isnan(column)]
                row[f"{family}_gap"] = observed[family][i]
                row[f"{family}_ci_low"] = np.percentile(column, 2.5) if len(column) else float("nan")
                row[f"{family}_ci_high"] = np.percentile(column, 97.5) if len(column) else float("nan")
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("gap = method success minus frozen fixed-k hull at the same mean cost, "
          "percentage points; + means the method is better.")
    print("RL gap is the mean over training seeds; [low, high] is a 95% bootstrap interval "
          "over test seeds; P = share of bootstrap draws with RL better than fixed.\n")
    for scenario in test:
        print(f"== {scenario}")
        for r in rows:
            if r["scenario"] != scenario:
                continue
            for key in sorted(k for k in r if k.endswith("_gap")
                              and k not in ("rl_gap", "rule_gap") and k[:3] in ("ml_", "rl_")):
                base = key[:-4]
                if base + "_ci_low" not in r:  # per-seed QR-DQN gaps are summarized on the RL line
                    continue
                if not np.isnan(r[key]):
                    print(f"cost {r['cost']:4.1f}: {base} {r[key]:+5.1f} "
                          f"[{r[base + '_ci_low']:+5.1f}, {r[base + '_ci_high']:+5.1f}]")
            if "rl_gap" not in r or np.isnan(r["rl_gap"]):
                continue
            print(f"cost {r['cost']:4.1f}: fixed {r['fixed_success']:.3f} | rule {r['rule_gap']:+5.1f}"
                  f" | RL {r['rl_gap']:+5.1f} [{r['rl_ci_low']:+5.1f}, {r['rl_ci_high']:+5.1f}]"
                  f" P={r['rl_prob_better']:.2f} ({r['rl_seeds_positive']}/{r['rl_seeds']} seeds > 0)")


if __name__ == "__main__":
    main()
