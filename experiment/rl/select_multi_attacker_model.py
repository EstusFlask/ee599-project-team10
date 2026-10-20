"""Select a balanced multi-attacker checkpoint on independent validation seeds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from sb3_contrib import QRDQN

from .benchmark_multi_attacker import run_episode
from .environment import SCENARIOS


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", type=Path,
        default=Path("experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--fine-tuned", type=Path,
        default=Path("experiment/results/rl_multi_attacker_finetuned/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--curriculum-dir", type=Path,
        default=Path("experiment/results/rl_multi_attacker_curriculum/models"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_multi_attacker_curriculum/validation"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[606, 707, 808])
    args = parser.parse_args()

    candidates = {
        "Original robust": args.original,
        "Uniform fine-tuned": args.fine_tuned,
        "Curriculum phase 1": args.curriculum_dir / "qrdqn_phase1_preserve_seed2026.zip",
        "Curriculum phase 2": args.curriculum_dir / "qrdqn_phase2_expand_seed2026.zip",
        "Curriculum phase 3": args.curriculum_dir / "qrdqn_phase3_harden_seed2026.zip",
    }
    policies = {
        name: QRDQN.load(path, device="cpu") for name, path in candidates.items()
    }
    bases = ("persistent", "adaptive", "noisy_ood", "budget_shock")
    attacker_counts = (1, 2, 3, 5)
    rows = []
    for policy_name, policy in policies.items():
        for base_name in bases:
            for attacker_count in attacker_counts:
                scenario = replace(
                    SCENARIOS[base_name],
                    name=f"{base_name}_attackers{attacker_count}",
                    attacker_count=attacker_count,
                )
                for seed in args.seeds:
                    row = run_episode(policy, scenario, 1.0, seed)
                    row.update({
                        "policy": policy_name,
                        "base_scenario": base_name,
                        "attacker_count": attacker_count,
                        "seed": seed,
                    })
                    rows.append(row)
        print(f"validated {policy_name}", flush=True)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["attacker_count"])].append(row)
    summary = []
    for (policy, count), values in groups.items():
        summary.append({
            "policy": policy,
            "attacker_count": count,
            "episodes": len(values),
            "weighted_success_rate": float(np.mean([
                value["weighted_success_rate"] for value in values
            ])),
            "return": float(np.mean([value["return"] for value in values])),
            "mean_cost": float(np.mean([value["mean_cost"] for value in values])),
            "budget_violation_rate": float(np.mean([
                value["budget_violation_rate"] for value in values
            ])),
        })

    # Normalize within every attacker-count slice so no easy count dominates.
    scored = defaultdict(list)
    for count in attacker_counts:
        count_rows = [row for row in summary if row["attacker_count"] == count]
        for metric in ("weighted_success_rate", "return", "mean_cost"):
            values = np.asarray([row[metric] for row in count_rows])
            low, high = float(values.min()), float(values.max())
            for row in count_rows:
                normalized = 0.5 if high == low else (row[metric] - low) / (high - low)
                row[f"{metric}_normalized"] = float(normalized)
        for row in count_rows:
            count_score = (
                0.45 * row["weighted_success_rate_normalized"]
                + 0.45 * row["return_normalized"]
                + 0.10 * (1.0 - row["mean_cost_normalized"])
            )
            scored[row["policy"]].append(count_score)

    rankings = []
    for policy, count_scores in scored.items():
        rankings.append({
            "policy": policy,
            "mean_count_score": float(np.mean(count_scores)),
            "worst_count_score": float(np.min(count_scores)),
            "selection_score": float(
                0.75 * np.mean(count_scores) + 0.25 * np.min(count_scores)
            ),
            "model_path": str(candidates[policy]),
        })
    rankings.sort(key=lambda row: row["selection_score"], reverse=True)
    winner = rankings[0]
    selected_path = args.curriculum_dir / "qrdqn_selected_seed2026.zip"
    shutil.copy2(candidates[winner["policy"]], selected_path)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "validation_episodes.csv", rows)
    write_csv(args.output / "validation_by_count.csv", summary)
    write_csv(args.output / "checkpoint_ranking.csv", rankings)
    (args.output / "selected_model.json").write_text(json.dumps({
        **winner,
        "selected_path": str(selected_path),
        "validation_seeds": args.seeds,
        "score_definition": (
            "per-count min-max normalization; 45% success, 45% return, "
            "10% inverse cost; final 75% mean plus 25% worst-count score"
        ),
    }, indent=2), encoding="utf-8")
    print(json.dumps({**winner, "selected_path": str(selected_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
