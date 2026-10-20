"""Run and plot a matched-communication-cost attacker experiment."""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

try:
    from .train import run_experiment
except ImportError:
    from train import run_experiment


MODEL_ORDER = (
    "fixed_baseline",
    "ml_no_state",
    "ml_with_state",
    "q_learning_rl",
)
MODEL_LABELS = {
    "fixed_baseline": "Fixed baseline",
    "ml_no_state": "Stateless ML",
    "ml_with_state": "Stateful ML",
    "q_learning_rl": "Q-learning RL",
}


def parse_csv_numbers(text, cast):
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


def curve(points):
    grouped = defaultdict(list)
    for communication_cost, success_rate in points:
        grouped[round(float(communication_cost), 12)].append(float(success_rate))
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.mean(grouped[x]) for x in xs], dtype=float)
    return xs, ys


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--seeds", default="7,42,99")
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--attackers", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("stealth", "bursty", "persistent"),
        default="bursty",
    )
    parser.add_argument("--clean-accuracy", type=float, default=0.80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "single_attacker",
    )
    args = parser.parse_args()
    budgets = parse_csv_numbers(args.budgets, float)
    seeds = parse_csv_numbers(args.seeds, int)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    wall_start = time.perf_counter()
    total_runs = len(budgets) * len(seeds)
    for run_index, (seed, budget) in enumerate(
        ((seed, budget) for seed in seeds for budget in budgets), start=1
    ):
        experiment_args = SimpleNamespace(
            attacker="hmm",
            profile=args.profile,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            steps=args.steps,
            window=args.window,
            attackers=args.attackers,
            budget=budget,
            clean_accuracy=args.clean_accuracy,
            seed=seed,
            output=None,
        )
        payload = run_experiment(experiment_args, display=False)
        runs.append(payload)
        elapsed = payload["timing"]["total_seconds"]
        print(
            f"[{run_index:02d}/{total_runs}] seed={seed} budget={budget:g} "
            f"{elapsed:.2f}s",
            flush=True,
        )

    points_by_seed_model = defaultdict(list)
    for run in runs:
        seed = run["config"]["seed"]
        for result in run["results"]:
            points_by_seed_model[(seed, result["model"])].append(
                (
                    result["average_communication_cost"],
                    result["integrity_success_rate"],
                )
            )

    curves = {
        key: curve(points) for key, points in points_by_seed_model.items()
    }
    common_low = max(values[0].min() for values in curves.values())
    common_high = min(values[0].max() for values in curves.values())
    if common_low >= common_high:
        raise RuntimeError("budget sweep has no common communication-cost range")
    matched_cost = 1.0 if common_low <= 1.0 <= common_high else (
        common_low + common_high
    ) / 2
    grid = np.linspace(common_low, common_high, 160)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    matched_results = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model_index, model in enumerate(MODEL_ORDER):
        grid_values = []
        matched_values = []
        for seed in seeds:
            xs, ys = curves[(seed, model)]
            grid_values.append(np.interp(grid, xs, ys))
            matched_values.append(float(np.interp(matched_cost, xs, ys)))
        grid_values = np.asarray(grid_values)
        mean_curve = grid_values.mean(axis=0)
        if len(seeds) < 2:
            ci_curve = np.zeros_like(mean_curve)
        else:
            ci_curve = (
                1.96
                * grid_values.std(axis=0, ddof=1)
                / math.sqrt(len(seeds))
            )
        color = colors[model_index % len(colors)]
        axes[0].plot(grid, mean_curve, label=MODEL_LABELS[model], color=color)
        axes[0].fill_between(
            grid,
            mean_curve - ci_curve,
            mean_curve + ci_curve,
            color=color,
            alpha=0.14,
        )
        matched_mean, matched_ci = mean_ci(matched_values)
        matched_results.append(
            {
                "model": model,
                "communication_cost": matched_cost,
                "integrity_success_rate": matched_mean,
                "ci95": matched_ci,
            }
        )

    axes[0].axvline(matched_cost, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Average normalized communication cost")
    axes[0].set_ylabel("Integrity success rate")
    axes[0].set_title("Success-cost curves (mean and 95% CI)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    means = [item["integrity_success_rate"] for item in matched_results]
    cis = [item["ci95"] for item in matched_results]
    labels = [MODEL_LABELS[item["model"]] for item in matched_results]
    positions = np.arange(len(labels))
    axes[1].bar(positions, means, yerr=cis, capsize=4, color=colors[: len(labels)])
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Integrity success rate")
    axes[1].set_title(f"Matched cost = {matched_cost:.3f}")
    axes[1].grid(axis="y", alpha=0.25)
    lower = max(0.0, min(means) - max(cis) - 0.04)
    axes[1].set_ylim(lower, min(1.0, max(means) + max(cis) + 0.04))
    figure.tight_layout()
    figure_path = args.output_dir / "success_cost.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    training_times = [run["timing"]["training_seconds"] for run in runs]
    testing_times = [run["timing"]["testing_seconds"] for run in runs]
    wall_seconds = time.perf_counter() - wall_start
    summary = {
        "attacker": f"{args.attackers}x-hmm-{args.profile}",
        "attacker_count": args.attackers,
        "history_window": args.window,
        "budgets": budgets,
        "seeds": seeds,
        "train_episodes_per_run": args.train_episodes,
        "test_episodes_per_run": args.test_episodes,
        "steps_per_episode": args.steps,
        "common_cost_range": [common_low, common_high],
        "matched_communication_cost": matched_cost,
        "matched_results": matched_results,
        "timing": {
            "mean_training_seconds_per_run": float(np.mean(training_times)),
            "mean_testing_seconds_per_run": float(np.mean(testing_times)),
            "sum_training_seconds": float(np.sum(training_times)),
            "sum_testing_seconds": float(np.sum(testing_times)),
            "wall_seconds": wall_seconds,
        },
        "runs": runs,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"matched_cost={matched_cost:.6f}")
    for result in matched_results:
        print(
            f"{result['model']:<20} "
            f"success={result['integrity_success_rate']:.4%} "
            f"ci95=±{result['ci95']:.4%}"
        )
    print(
        f"mean_train={np.mean(training_times):.3f}s "
        f"mean_test={np.mean(testing_times):.3f}s "
        f"wall={wall_seconds:.3f}s"
    )
    print(figure_path)
    print(summary_path)


if __name__ == "__main__":
    main()
