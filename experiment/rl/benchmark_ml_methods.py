"""Benchmark diverse supervised ML defenders on the unified cost grid."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .environment import DefenseEnv
from .plot_budget_success_comparison import (
    HARD_SCENARIOS,
    generate_ml_dataset,
)
from .plot_unified_cost_comparison import (
    aggregate_candidates,
    interpolate_frontier,
    mean_ci,
    monotone_frontier,
)


ML_ORDER = (
    "Gaussian NB",
    "Logistic Regression",
    "Linear SVM",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoosting",
    "MLP",
)
REFERENCE_ORDER = (
    "Fixed baseline",
    "PPO",
    "TRPO robust",
    "QR-DQN robust (w=0.15)",
)


class SklearnPolicy:
    def __init__(self, estimator, feature_indices=None):
        self.estimator = estimator
        self.feature_indices = feature_indices

    def predict_batch(self, observations):
        x = np.asarray(observations, dtype=np.float32)
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        return self.estimator.predict(x).astype(np.int64)


def build_estimators(seed):
    balanced_tree = dict(
        max_depth=14, min_samples_leaf=16,
        class_weight="balanced", random_state=seed,
    )
    return {
        "Gaussian NB": (GaussianNB(var_smoothing=1e-8), None),
        "Logistic Regression": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, max_iter=500, class_weight="balanced",
                    solver="lbfgs", random_state=seed,
                ),
            ),
            None,
        ),
        "Linear SVM": (
            make_pipeline(
                StandardScaler(),
                LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
            ),
            None,
        ),
        "Decision Tree": (DecisionTreeClassifier(**balanced_tree), None),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=160, max_depth=18, min_samples_leaf=8,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed,
            ),
            None,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=160, max_depth=20, min_samples_leaf=6,
                max_features="sqrt", class_weight="balanced",
                n_jobs=-1, random_state=seed,
            ),
            None,
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=180, max_leaf_nodes=31,
                min_samples_leaf=24, l2_regularization=1.0,
                class_weight="balanced", random_state=seed,
            ),
            None,
        ),
        "MLP": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(96, 48), alpha=2e-4,
                    batch_size=512, learning_rate_init=8e-4,
                    max_iter=100, early_stopping=True,
                    validation_fraction=0.12, n_iter_no_change=8,
                    random_state=seed,
                ),
            ),
            None,
        ),
    }


def train_and_validate(train_samples, validation_samples, seed):
    x_train, y_train = generate_ml_dataset(train_samples, seed)
    x_valid, y_valid = generate_ml_dataset(validation_samples, seed + 100_000)
    expected = DefenseEnv.observation_size
    if x_train.shape[1] != expected or x_valid.shape[1] != expected:
        raise RuntimeError(
            f"ML/RL input mismatch: expected {expected}, got "
            f"train={x_train.shape[1]}, validation={x_valid.shape[1]}"
        )
    policies = {}
    metrics = []
    for name, (estimator, indices) in build_estimators(seed).items():
        if indices is not None:
            raise RuntimeError(f"{name} does not use the complete {expected}-D observation")
        fit_x = x_train if indices is None else x_train[:, indices]
        valid_x = x_valid if indices is None else x_valid[:, indices]
        started = time.perf_counter()
        estimator.fit(fit_x, y_train)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = estimator.predict(valid_x)
        inference_seconds = time.perf_counter() - started
        metrics.append({
            "policy": name,
            "train_samples": len(y_train),
            "validation_samples": len(y_valid),
            "accuracy": accuracy_score(y_valid, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_valid, prediction),
            "macro_f1": f1_score(y_valid, prediction, average="macro", zero_division=0),
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
        })
        policies[name] = SklearnPolicy(estimator, indices)
        print(
            f"trained {name}: accuracy={metrics[-1]['accuracy']:.4f} "
            f"macro_f1={metrics[-1]['macro_f1']:.4f}",
            flush=True,
        )
    return policies, metrics


def evaluate_batch(policy, name, budget_scale, gate, seeds):
    states = []
    for scenario in HARD_SCENARIOS:
        for seed in seeds:
            env = DefenseEnv(scenario, budget_scale=budget_scale)
            obs, _ = env.reset(seed=seed)
            states.append({
                "env": env,
                "obs": obs,
                "scenario": scenario,
                "seed": seed,
                "accumulator": (seed % 97) / 97.0,
            })
    cheap_action = DefenseEnv.encode_action(1, "none")
    for _ in range(max(item["env"].config.horizon for item in states)):
        proposed = policy.predict_batch([item["obs"] for item in states])
        for item, candidate in zip(states, proposed):
            item["accumulator"] += gate
            active = item["accumulator"] >= 1.0
            if active:
                item["accumulator"] -= 1.0
            action = int(candidate) if active else cheap_action
            obs, _, terminated, truncated, info = item["env"].step(action)
            item["obs"] = obs
            if terminated or truncated:
                item["metrics"] = info["episode_metrics"]
        if all("metrics" in item for item in states):
            break
    return [{
        "policy": name,
        "family": "ml",
        "budget_scale": budget_scale,
        "gate": gate,
        "scenario": item["scenario"],
        "seed": item["seed"],
        "mean_cost": item["metrics"]["mean_cost"],
        "weighted_success_rate": item["metrics"]["weighted_success_rate"],
        "success_rate": item["metrics"]["success_rate"],
    } for item in states]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_unified_ml(candidates, grid):
    unified = []
    coverage = []
    for name in ML_ORDER:
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
                f"{name} does not cover {grid.min():.2f}-{grid.max():.2f}"
            )
        ys, cis = values
        for cost, success, ci in zip(grid, ys, cis):
            unified.append({
                "policy": name,
                "family": "ml",
                "communication_cost": float(cost),
                "weighted_success_rate": float(success),
                "ci95": float(ci),
            })
    return unified, coverage


def plot_comparison(unified, output, matched_cost):
    names = (*REFERENCE_ORDER, *ML_ORDER)
    colors = dict(zip(names, plt.get_cmap("tab20")(np.linspace(0, 0.95, len(names)))))
    markers = ("o", "D", "P", "*", "s", "^", "v", "<", ">", "h", "8", "X", "p")
    fig, axes = plt.subplots(
        1, 2, figsize=(18.0, 6.5), gridspec_kw={"width_ratios": (1.5, 1.0)}
    )
    matched = []
    for index, name in enumerate(names):
        rows = sorted(
            [row for row in unified if row["policy"] == name],
            key=lambda row: float(row["communication_cost"]),
        )
        xs = np.asarray([float(row["communication_cost"]) for row in rows])
        ys = np.asarray([float(row["weighted_success_rate"]) for row in rows])
        cis = np.asarray([float(row["ci95"]) for row in rows])
        is_ml = name in ML_ORDER
        axes[0].plot(
            xs, ys, label=name, color=colors[name],
            linestyle="--" if is_ml else "-",
            marker=markers[index], markersize=4.8,
            linewidth=2.6 if name == "QR-DQN robust (w=0.15)" else 1.7,
        )
        axes[0].fill_between(xs, ys - cis, ys + cis, color=colors[name], alpha=0.07)
        matched.append((
            float(np.interp(matched_cost, xs, ys)),
            float(np.interp(matched_cost, xs, cis)),
        ))
    axes[0].axvline(matched_cost, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Average normalized communication cost")
    axes[0].set_ylabel("Importance-weighted defense success rate")
    axes[0].set_title("Traditional ML vs. RL on a unified cost grid", weight="bold")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(alpha=0.24)
    axes[0].legend(fontsize=7.4, ncol=2, loc="lower right", framealpha=0.94)

    positions = np.arange(len(names))
    axes[1].bar(
        positions, [item[0] for item in matched],
        yerr=[item[1] for item in matched], capsize=3,
        color=[colors[name] for name in names], alpha=0.93,
    )
    axes[1].set_xticks(positions, names, rotation=45, ha="right", fontsize=7.8)
    axes[1].set_ylabel("Importance-weighted defense success rate")
    axes[1].set_title(f"Matched cost = {matched_cost:.3f}", weight="bold")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.25, 0.82)
    axes[1].grid(axis="y", alpha=0.24)
    fig.suptitle("Expanded Traditional-ML Benchmark Under Complex Attacks", fontsize=15, weight="bold")
    fig.text(
        0.5, 0.012,
        "Eight supervised ML variants and RL receive the identical 25-D observation vector and share the "
        "same test scenarios and measured-cost grid. ML is dashed; RL is solid.",
        ha="center", fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path(
        "experiment/results/rl_robust_w015/analysis/unified_cost_summary.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "experiment/results/rl_robust_w015/analysis/ml_methods_unified_cost.png"
    ))
    parser.add_argument("--train-samples", type=int, default=60_000)
    parser.add_argument("--validation-samples", type=int, default=15_000)
    parser.add_argument("--budget-scales", nargs="+", type=float, default=[0.55, 0.80, 1.05, 1.30, 1.55])
    parser.add_argument("--gates", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
    parser.add_argument("--cost-grid", nargs="+", type=float, default=[0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
    parser.add_argument("--matched-cost", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    args = parser.parse_args()

    policies, validation_metrics = train_and_validate(
        args.train_samples, args.validation_samples, seed=2026
    )
    raw = []
    for name in ML_ORDER:
        for budget_scale in args.budget_scales:
            for gate in args.gates:
                raw.extend(evaluate_batch(
                    policies[name], name, budget_scale, gate, args.seeds
                ))
        print(f"evaluated {name}", flush=True)

    # Reuse the already evaluated references on the identical common grid.
    with args.reference.open(newline="", encoding="utf-8") as handle:
        reference_all = list(csv.DictReader(handle))
    reference = [row for row in reference_all if row["policy"] in REFERENCE_ORDER]
    candidates = aggregate_candidates(
        raw, args.budget_scales, args.gates, policy_names=ML_ORDER
    )
    grid = np.asarray(args.cost_grid, dtype=float)
    unified_ml, coverage = build_unified_ml(candidates, grid)
    unified = reference + unified_ml
    plot_comparison(unified, args.output, args.matched_cost)

    write_csv(args.output.with_name("ml_methods_validation.csv"), validation_metrics)
    write_csv(args.output.with_name("ml_methods_raw.csv"), raw)
    write_csv(args.output.with_name("ml_methods_candidates.csv"), candidates)
    write_csv(args.output.with_name("ml_methods_coverage.csv"), coverage)
    write_csv(args.output.with_name("ml_methods_unified_cost.csv"), unified_ml)
    args.output.with_name("ml_methods_protocol.json").write_text(json.dumps({
        "observation_dimension": DefenseEnv.observation_size,
        "observation_indices_used_by_every_ml_model": list(range(DefenseEnv.observation_size)),
        "rl_observation_dimension": DefenseEnv.observation_size,
        "hidden_attack_state_available_at_inference": False,
        "hidden_attack_state_used_only_for_supervised_training_labels": True,
        "hard_scenarios": list(HARD_SCENARIOS),
        "seeds": args.seeds,
        "budget_scales": args.budget_scales,
        "gates": args.gates,
        "cost_grid": args.cost_grid,
    }, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
