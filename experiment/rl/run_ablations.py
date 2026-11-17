"""Two essential, retrained ablations for the existing cost-0.75 result."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from .ablations import CURRENT_FEATURES
from .train_agents import train_one
from . import verify_effect as verification


TARGET_COST = 0.75
VARIANTS = ("no_history", "short_sighted")
LABELS = {
    "adaptive_slow": "Full RL",
    "no_history": "No history features",
    "short_sighted": "Myopic RL (gamma=0)",
    "static": "Original budget mechanism (existing comparison)",
}


def train_worker(root, variant, seed, steps):
    torch.set_num_threads(1)
    folder = Path(root) / variant / "models"
    metadata_path = folder / f"metadata_seed{seed}.json"
    settings = {"steps": steps, "use_history": variant != "no_history",
                "discount": 0.0 if variant == "short_sighted" else 0.985}
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if any(saved[key] != value for key, value in settings.items()):
            raise ValueError(f"use a fresh output directory: {metadata_path}")
        return saved
    record = train_one(
        "qrdqn", steps, seed, folder, "cpu", 1,
        training_profile="multi_attacker", constraint_weight=0.15,
        adaptive_budget=True, budget_learning_rate=0.005,
        budget_warmup_steps=8000,
        use_history=settings["use_history"], discount=settings["discount"],
    )
    record["ablation"] = variant
    verification.write_json(metadata_path, record)
    return record


def train(args):
    protocol = {
        "training_seeds": verification.TRAIN_SEEDS,
        "steps_per_seed": args.steps,
        "target_cost": TARGET_COST,
        "no_history": {"retained_features": CURRENT_FEATURES,
                       "mask_during_training_and_evaluation": True},
        "short_sighted": {"discount": 0.0, "same_full_observation": True},
        "unchanged": "Attacks, costs, 25-D input shape, 12 actions, architecture, budget updates, validation/test seeds.",
        "no_further_search": True,
    }
    path = args.output / "ablations" / "protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError("ablation protocol already exists with different settings")
    verification.write_json(path, protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_worker, str(args.output), name, seed, args.steps)
                   for name in VARIANTS for seed in verification.TRAIN_SEEDS]
        records = []
        for future in as_completed(futures):
            records.append(future.result())
            print(f"ablations trained {len(records)}/6", flush=True)
    verification.write_json(args.output / "ablations" / "training.json", records)


def evaluate_worker(root, name, seed):
    verification.TARGET_COST = TARGET_COST
    validation = Path(verification.validation_worker(root, name, seed))
    try:
        choice = {"status": "ready", **verification.select_pair(json.loads(validation.read_text()))}
    except ValueError as error:
        choice = {"status": "unavailable", "reason": str(error)}
    choice.update(family=name, training_seed=seed, target_cost=TARGET_COST)
    path = Path(root) / "ablations" / f"selection_{name}_seed{seed}.json"
    if path.exists() and json.loads(path.read_text()) != choice:
        raise ValueError("ablation operating points are already frozen")
    verification.write_json(path, choice)
    if choice["status"] == "ready":
        verification.test_worker(root, choice)
    return {"ablation": name, "seed": seed, "status": choice["status"]}


def evaluate(args):
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_worker, str(args.output), name, seed)
                   for name in VARIANTS for seed in verification.TRAIN_SEEDS]
        records = []
        for future in as_completed(futures):
            records.append(future.result())
            print(f"ablations evaluated {len(records)}/6: {records[-1]}", flush=True)
    verification.write_json(args.output / "ablations" / "evaluation.json", records)


def load_tensor(root, family):
    endpoint_names = ("low", "high", "guard_low", "guard_high")
    seeds = verification.TEST_SEEDS
    cases = [name for name, _, _ in verification.scenarios()]
    tensor = np.zeros((3, 4, len(seeds), len(cases), 3))
    if family in VARIANTS:
        selections = [json.loads((root / "ablations" / f"selection_{family}_seed{seed}.json").read_text())
                      for seed in verification.TRAIN_SEEDS]
    else:
        selections = [row for row in json.loads((root / "cost_0.75" / "selection.json").read_text())
                      if row["family"] == family]
    if len(selections) != 3 or any(row["status"] != "ready" for row in selections):
        return None
    for train_index, seed in enumerate(verification.TRAIN_SEEDS):
        path = root / "cost_0.75" / "held_out" / f"{family}_seed{seed}.json"
        rows = json.loads(path.read_text())
        expected = {(endpoint, seed, case) for endpoint in endpoint_names for seed in seeds for case in cases}
        observed = {(row["endpoint"], row["evaluation_seed"], row["case"]) for row in rows}
        if observed != expected or len(rows) != len(expected):
            raise ValueError(f"incomplete held-out results: {path}")
        for row in rows:
            tensor[train_index, endpoint_names.index(row["endpoint"]), seeds.index(row["evaluation_seed"]),
                   cases.index(row["case"])] = [row["mean_cost"], row["weighted_success_rate"], row["budget_violation_rate"]]
    return tensor


def estimate(tensor, train_indices, test_indices):
    values = [verification.matched_value(tensor[index][:, test_indices]) for index in train_indices]
    return None if any(value is None for value in values) else np.mean(values, axis=0)


def report(args):
    verification.TARGET_COST = TARGET_COST
    tensors = {}
    unavailable = []
    for family in LABELS:
        tensor = load_tensor(args.output, family)
        if tensor is None:
            unavailable.append(family)
        else:
            tensors[family] = tensor
    point = {}
    for name, tensor in tensors.items():
        value = estimate(tensor, range(3), range(len(verification.TEST_SEEDS)))
        if value is not None:
            point[name] = value
        else:
            unavailable.append(name)
    generator = np.random.default_rng(44001)
    bootstrap = {name: [] for name in point}
    differences = {name: [] for name in point if name != "adaptive_slow"}
    for _ in range(3000):
        training = generator.integers(0, 3, 3)
        testing = generator.integers(0, len(verification.TEST_SEEDS), len(verification.TEST_SEEDS))
        sample = {name: estimate(tensors[name], training, testing) for name in point}
        for name, value in sample.items():
            if value is not None:
                bootstrap[name].append(value[1])
                if name in differences and sample.get("adaptive_slow") is not None:
                    differences[name].append(sample["adaptive_slow"][1] - value[1])
    rows = []
    contrasts = []
    for name, value in point.items():
        rows.append({"method": name, "label": LABELS[name], "success": float(value[1]),
                     "ci95": np.quantile(bootstrap[name], [0.025, 0.975]).tolist(),
                     "valid_bootstrap_replicates": len(bootstrap[name]),
                     "ci_coverage_adequate": len(bootstrap[name]) >= 2850,
                     "budget_violation_rate": float(value[2])})
    for name, values in differences.items():
        if not values:
            continue
        interval = np.quantile(values, [0.025, 0.975]).tolist()
        contrasts.append({"comparison": f"full_minus_{name}",
                          "difference": float(point["adaptive_slow"][1] - point[name][1]),
                          "ci95": interval, "valid_bootstrap_replicates": len(values),
                          "assessment": "inconclusive" if len(values) < 2850 else (
                              "positive" if interval[0] > 0 else "negative" if interval[1] < 0 else "inconclusive"
                          )})
    result = {"matched_cost": TARGET_COST, "rows": rows, "paired_contrasts": contrasts,
              "unavailable": unavailable,
              "intervals": "3000 paired hierarchical bootstrap replicates; 3 training seeds and 12 common test seeds.",
              "scope": "Retrained interventions, fixed settings, validation-only operating-point selection; no test-success selection."}
    directory = args.output / "ablations"
    verification.write_json(directory / "summary.json", result)
    lines = ["# Minimal RL Ablations", "", "## Essential comparisons", "",
             "- The full model reuses the three verified QR-DQN checkpoints.",
             "- No history: mask previous actions, outcomes, costs, rolling history, and protection history during training and inference; retain current alerts, messages, budget, and progress.",
             "- Myopic: change only gamma from 0.985 to 0; keep inputs, actions, and the budget mechanism unchanged.",
             "- Each ablation is retrained with 3 seeds and 100,000 steps per seed; no new algorithms or additional tuning.",
             "- The original budget mechanism is an existing comparison of the complete mechanism, not an ablation of a single coefficient.",
             "- Matched cost is 0.75; evaluation reuses 12 common test seeds and 6 scenarios. CIs use paired hierarchical bootstrap with only 3 training seeds.", "",
             "## Success rates and 95% CIs", "", "| Method | Weighted success | 95% CI | Budget violation rate |", "|---|---:|---:|---:|"]
    for row in rows:
        low, high = row["ci95"]
        qualifier = " (exploratory)" if not row["ci_coverage_adequate"] else ""
        lines.append(f"| {row['label']} | {100*row['success']:.2f}% | {100*low:.2f}–{100*high:.2f}%{qualifier} | {100*row['budget_violation_rate']:.2f}% |")
    lines += ["", "## Full model minus comparison model", ""]
    for row in contrasts:
        low, high = row["ci95"]
        lines.append(f"- {row['comparison']}: {100*row['difference']:+.2f} percentage points, 95% CI [{100*low:+.2f}, {100*high:+.2f}], {row['assessment']}; valid bootstrap replicates: {row['valid_bootstrap_replicates']}/3000.")
    for row in rows:
        if not row["ci_coverage_adequate"]:
            lines.append(
                f"- For {row['label']}, only {row['valid_bootstrap_replicates']}/3000 bootstrap replicates cover the target cost. "
                "The interval is conditional on these replicates and is exploratory, not evidence of statistical significance. No extrapolation or replacement resampling hides coverage failures."
            )
    for name in unavailable:
        lines.append(f"- {LABELS[name]} lacks complete matched-cost coverage; no extrapolation is performed.")
    lines += ["", "## Interpretation and limitations", "",
              "- Intervals crossing zero or favoring the comparison do not establish a benefit; negative results are retained.",
              "- The main experiment has not established superiority over the strongest fixed strategy; ablations do not remove this limitation.",
              "- These numerical simulation results are not real LLM task accuracy and must not be mixed with milestone results from the old environment.",
              "- No further RL algorithms are needed; the remaining work is to include the main results, ablation table, and limitations in the final report."]
    for row in contrasts:
        if row["comparison"] in {"full_minus_no_history", "full_minus_short_sighted"}:
            mechanism = "history features" if row["comparison"] == "full_minus_no_history" else "long-term TD returns"
            if row["assessment"] != "positive":
                lines.append(f"- This experiment did not establish an additional benefit from {mechanism}; do not claim a verified advantage or interpret an interval crossing zero as proof of no effect.")
    (directory / "ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "train", "evaluate", "report"), default="all")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_effect_check"))
    args = parser.parse_args()
    for seed in verification.TRAIN_SEEDS:
        path = args.output / "adaptive_slow" / "models" / f"metadata_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError("run the full-model verification before its ablations")
        if json.loads(path.read_text())["steps"] != args.steps:
            raise ValueError("ablation training steps must match the full-model baseline")
    if args.phase != "train" and not (args.output / "cost_0.75" / "selection.json").exists():
        raise FileNotFoundError("complete the main cost-0.75 verification first")
    phases = {"train": train, "evaluate": evaluate, "report": report}
    for phase in phases if args.phase == "all" else (args.phase,):
        phases[phase](args)


if __name__ == "__main__":
    main()
