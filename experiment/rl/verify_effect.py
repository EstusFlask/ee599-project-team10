"""Small reproducible effectiveness check; validation selects, held-out tests report."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from .config import ENVIRONMENT_VERSION, SCENARIOS, continuous_changes
from .environment import DefenseEnv
from .train_agents import train_one


TRAIN_SEEDS = (2026, 2027, 2028)
VALIDATION_SEEDS = (8101, 8102, 8103)
TEST_SEEDS = tuple(range(9101, 9113))
GATES = (0.0, 0.25, 0.50, 0.75, 1.0)
BUDGETS = (0.70, 1.0, 1.30)
TARGET_COST = 1.0


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analysis_directory(root, target=1.0):
    return Path(root) if target == 1.0 else Path(root) / f"cost_{target:g}"


def scenarios():
    cases = []
    for name in ("adaptive", "noisy_ood"):
        for count in (1, 3):
            config = replace(SCENARIOS[name], attacker_count=count)
            cases.append((f"{name}_{count}_attackers", config, False))
        cases.append((f"{name}_continuous", SCENARIOS[name], True))
    return cases


def train_worker(root, seed, mode, steps):
    torch.set_num_threads(1)
    folder = Path(root) / mode / "models"
    metadata = folder / f"metadata_seed{seed}.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["steps"] == steps:
            return saved
        raise ValueError(f"refusing to overwrite a different training run: {metadata}")
    result = train_one(
        "qrdqn", steps, seed, folder, "cpu", 1,
        training_profile="multi_attacker", constraint_weight=0.15,
        adaptive_budget=mode.startswith("adaptive"),
        budget_learning_rate=0.005 if mode == "adaptive_slow" else 0.05,
        budget_warmup_steps=8000 if mode == "adaptive_slow" else 0,
    )
    result["mode"] = mode
    write_json(metadata, result)
    return result


def run_training(args):
    protocol = {
        "environment_version": ENVIRONMENT_VERSION,
        "steps_per_training": args.steps,
        "training_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "held_out_test_seeds": TEST_SEEDS,
        "target_average_communication_cost": TARGET_COST,
        "budget_scales": BUDGETS, "gates": GATES,
        "scenarios": [
            {"name": name, "config": asdict(config), "continuous": continuous}
            for name, config, continuous in scenarios()
        ],
        "selection": "Select operating points on validation data; freeze before testing.",
        "scope": "Simulator effectiveness, not real-LLM performance.",
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("use a new output folder for a different protocol")
    write_json(protocol_path, protocol)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(train_worker, str(args.output), seed, mode, args.steps)
            for seed in TRAIN_SEEDS for mode in ("static", "adaptive")
        ]
        records = []
        for future in as_completed(pending):
            records.append(future.result())
            print(f"training completed {len(records)}/{len(pending)}", flush=True)
    write_json(args.output / "training.json", {
        "runs": records, "wall_seconds": time.perf_counter() - started,
    })


def run_slow_training(args):
    write_json(args.output / "adaptive_slow_protocol.json", {
        "reason": "Initial validation found random-exploration price overshoot and constant-action collapse.",
        "training_seeds": TRAIN_SEEDS, "steps": args.steps,
        "budget_learning_rate": 0.005, "budget_warmup_steps": 8000,
        "selection_data": "Validation only, same fixed validation/test seed separation.",
    })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(train_worker, str(args.output), seed, "adaptive_slow", args.steps)
                   for seed in TRAIN_SEEDS]
        records = []
        for future in as_completed(pending):
            records.append(future.result())
            print(f"slow adaptive training completed {len(records)}/3", flush=True)
    write_json(args.output / "adaptive_slow_training.json", records)


def ml_worker(root, seed, samples):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from .plot_budget_success_comparison import oracle_action

    folder = Path(root) / "ml"
    metadata = folder / f"metadata_seed{seed}.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["samples"] != samples:
            raise ValueError(f"refusing to overwrite a different ML run: {metadata}")
        return saved
    started = time.perf_counter()
    observations, labels = [], []
    generator = np.random.default_rng(seed + 100_000)
    episode = 0
    while len(labels) < samples:
        env = DefenseEnv(
            "train_mix", domain_randomization=True,
            randomization_profile="multi_attacker", constraint_weight=0.15,
        )
        obs, _ = env.reset(seed=seed + episode)
        while len(labels) < samples:
            observations.append(obs.copy())
            labels.append(oracle_action(env))
            obs, _, terminated, truncated, _ = env.step(int(generator.integers(12)))
            if terminated or truncated:
                break
        env.close()
        episode += 1
    x, y = np.asarray(observations), np.asarray(labels)
    estimators = {
        "svm": make_pipeline(StandardScaler(), LinearSVC(
            C=0.5, class_weight="balanced", dual=False, random_state=seed,
        )),
        "forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=16,
            class_weight="balanced_subsample", n_jobs=1, random_state=seed,
        ),
    }
    folder.mkdir(parents=True, exist_ok=True)
    for name, estimator in estimators.items():
        estimator.fit(x, y)
        joblib.dump(estimator, folder / f"{name}_seed{seed}.joblib")
    result = {
        "seed": seed, "samples": samples, "features": x.shape[1],
        "training_seconds": time.perf_counter() - started,
        "labels": "Expected immediate-reward oracle, hidden state for training labels only.",
        "training_profile": "multi_attacker", "models": list(estimators),
    }
    write_json(metadata, result)
    return result


def run_ml(args):
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(ml_worker, str(args.output), seed, args.steps) for seed in TRAIN_SEEDS]
        records = []
        for future in as_completed(tasks):
            records.append(future.result())
            print(f"ML training completed {len(records)}/3", flush=True)
    write_json(args.output / "ml_training.json", records)


def load_policy(root, name, seed):
    from .baselines import FixedPolicy
    if name.startswith("fixed_"):
        _, k, recovery = name.split("_")
        return FixedPolicy(int(k), recovery), "fixed"
    if name in {"svm", "forest"}:
        import joblib
        return joblib.load(Path(root) / "ml" / f"{name}_seed{seed}.joblib"), "ml"
    from sb3_contrib import QRDQN
    policy = QRDQN.load(Path(root) / name / "models" / f"qrdqn_seed{seed}.zip", device="cpu")
    if name == "no_history":
        from .ablations import CurrentPolicy
        policy = CurrentPolicy(policy)
    return policy, "rl"


def evaluate_point(policy, kind, budget, gate, seeds):
    states = []
    for name, config, continuous in scenarios():
        for seed in seeds:
            env = DefenseEnv(config, budget_scale=budget,
                             changes=continuous_changes(config) if continuous else ())
            obs, _ = env.reset(seed=seed)
            states.append({"env": env, "obs": obs, "case": name, "seed": seed,
                           "phase": (seed % 97) / 97.0, "return": 0.0})
    for _ in range(256):
        observations = np.asarray([item["obs"] for item in states])
        if kind == "fixed":
            proposed = np.full(len(states), policy.action)
        elif kind == "ml":
            proposed = policy.predict(observations)
        else:
            proposed, _ = policy.predict(observations, deterministic=True)
        for item, candidate in zip(states, proposed):
            item["phase"] += gate
            active = item["phase"] >= 1.0
            if active:
                item["phase"] -= 1.0
            action = int(candidate) if active else DefenseEnv.encode_action(1, "none")
            obs, reward, terminated, truncated, info = item["env"].step(action)
            item["return"] += reward
            item["obs"] = obs
            if terminated or truncated:
                item["metrics"] = info["episode_metrics"]
    rows = []
    for item in states:
        metrics = item["metrics"]
        rows.append({
            "case": item["case"], "evaluation_seed": item["seed"],
            "budget_scale": budget, "gate": gate,
            "mean_cost": metrics["mean_cost"],
            "weighted_success_rate": metrics["weighted_success_rate"],
            "success_rate": metrics["success_rate"],
            "budget_violation_rate": metrics["budget_violations"] / 256,
            "return": item["return"],
        })
        item["env"].close()
    return rows


def validation_worker(root, name, seed):
    torch.set_num_threads(1)
    path = Path(root) / "validation" / f"{name}_seed{seed}.json"
    if path.exists():
        return str(path)
    policy, kind = load_policy(root, name, seed)
    rows = []
    budgets = (1.0,) if kind == "fixed" else BUDGETS
    for budget in budgets:
        for gate in GATES:
            values = evaluate_point(policy, kind, budget, gate, VALIDATION_SEEDS)
            rows.append({
                "policy": name, "training_seed": seed,
                "budget_scale": budget, "gate": gate,
                "mean_cost": float(np.mean([r["mean_cost"] for r in values])),
                "weighted_success_rate": float(np.mean([r["weighted_success_rate"] for r in values])),
                "episodes": values,
            })
    write_json(path, rows)
    return str(path)


def run_validation(args):
    jobs = [(name, seed) for seed in TRAIN_SEEDS
            for name in ("static", "adaptive", "adaptive_slow", "svm", "forest")]
    jobs += [(f"fixed_{k}_{recovery}", TRAIN_SEEDS[0])
             for k in (1, 3, 5, 7) for recovery in ("none", "retry", "rotate")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(validation_worker, str(args.output), name, seed) for name, seed in jobs]
        for index, future in enumerate(as_completed(pending), 1):
            print(f"validated {index}/{len(jobs)}: {future.result()}", flush=True)


def select_pair(points):
    # Fixed/randomized baseline portfolios are allowed too. Otherwise an
    # efficient fixed policy just below cost 1 could be unfairly excluded.
    low = [r for r in points if r["mean_cost"] <= TARGET_COST]
    high = [r for r in points if r["mean_cost"] >= TARGET_COST]
    choices = []
    for left in low:
        for right in high:
            if right["mean_cost"] <= left["mean_cost"]:
                continue
            weight = (TARGET_COST - left["mean_cost"]) / (right["mean_cost"] - left["mean_cost"])
            score = (1 - weight) * left["weighted_success_rate"] + weight * right["weighted_success_rate"]
            choices.append((score, left, right))
    if not choices:
        raise ValueError("validation does not bracket the preregistered target cost")
    score, left, right = max(choices, key=lambda item: item[0])
    guard_low = max(
        (r for r in points if r["mean_cost"] <= TARGET_COST - 0.2),
        key=lambda r: r["weighted_success_rate"], default=left,
    )
    guard_high = min(
        (r for r in points if r["mean_cost"] >= TARGET_COST + 0.2),
        key=lambda r: r["mean_cost"], default=right,
    )
    fields = ("policy", "training_seed", "budget_scale", "gate", "mean_cost", "weighted_success_rate")
    return {"validation_success_at_target": score,
            **{name: {k: point[k] for k in fields} for name, point in (
                ("low", left), ("high", right), ("guard_low", guard_low), ("guard_high", guard_high),
            )}}


def run_selection(args):
    points = []
    for path in sorted((args.output / "validation").glob("*.json")):
        points.extend(json.loads(path.read_text()))
    selected = []
    for seed in TRAIN_SEEDS:
        for family in FAMILIES:
            if family == "fixed":
                values = [r for r in points if r["policy"].startswith("fixed_")]
            elif family == "ml":
                values = [r for r in points if r["training_seed"] == seed and r["policy"] in {"svm", "forest"}]
            else:
                values = [r for r in points if r["training_seed"] == seed and r["policy"] == family]
            try:
                selection = {"status": "ready", **select_pair(values)}
            except ValueError as error:
                selection = {"status": "unavailable", "reason": str(error)}
            selected.append({"family": family, "training_seed": seed, **selection})
    directory = analysis_directory(args.output, TARGET_COST)
    write_json(directory / "analysis_protocol.json", {
        "target_cost": TARGET_COST, "validation_seeds": VALIDATION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "primary_cost": 1.0,
        "secondary_cost_note": "Secondary budgets are assessed separately; selection uses validation only.",
    })
    path = directory / "selection.json"
    for item in selected:
        item["target_cost"] = TARGET_COST
    if path.exists() and json.loads(path.read_text()) != selected:
        raise ValueError("selection already frozen; use a new output directory to change protocol")
    write_json(path, selected)
    print(json.dumps(selected, indent=2), flush=True)


def test_worker(root, choice):
    torch.set_num_threads(1)
    family, seed = choice["family"], choice["training_seed"]
    directory = analysis_directory(root, choice.get("target_cost", 1.0))
    path = directory / "held_out" / f"{family}_seed{seed}.json"
    if path.exists():
        return str(path)
    rows = []
    for endpoint in ("low", "high", "guard_low", "guard_high"):
        point = choice[endpoint]
        policy, kind = load_policy(root, point["policy"], point["training_seed"])
        values = evaluate_point(policy, kind, point["budget_scale"], point["gate"], TEST_SEEDS)
        rows.extend({"family": family, "training_seed": seed,
                     "policy": point["policy"], "endpoint": endpoint, **row} for row in values)
    write_json(path, rows)
    return str(path)


def run_test(args):
    selected = json.loads((analysis_directory(args.output, TARGET_COST) / "selection.json").read_text())
    selected = [item for item in selected if item["status"] == "ready"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(test_worker, str(args.output), item) for item in selected]
        for index, future in enumerate(as_completed(pending), 1):
            print(f"tested {index}/{len(selected)}: {future.result()}", flush=True)


FAMILIES = ("fixed", "ml", "static", "adaptive", "adaptive_slow")
LABELS = {
    "fixed": "Fixed baseline (validation-selected)",
    "ml": "History ML (validation-selected SVM/RF)",
    "static": "QR-DQN, original budget penalty",
    "adaptive": "QR-DQN, adaptive budget penalty",
    "adaptive_slow": "QR-DQN, slow budget update after warmup",
}


def matched_value(values):
    """Interpolate frozen points by measured cost, never select on test success."""
    averages = values.mean(axis=(1, 2))
    primary = averages[:2]
    primary_min, primary_max = primary[:, 0].min(), primary[:, 0].max()
    if primary_min <= TARGET_COST <= primary_max:
        # Coverage guards must not replace the validation-optimal primary pair.
        averages = primary
    elif primary_min > TARGET_COST:
        averages = np.concatenate((primary, averages[2:3]))
    else:
        averages = np.concatenate((primary, averages[3:4]))
    averages = averages[np.argsort(averages[:, 0], kind="stable")]
    low_cost, high_cost = averages[0, 0], averages[-1, 0]
    if not low_cost <= TARGET_COST <= high_cost or high_cost <= low_cost:
        return None
    return np.asarray([np.interp(TARGET_COST, averages[:, 0], averages[:, i])
                       for i in range(averages.shape[1])])


def run_report(args):
    directory = analysis_directory(args.output, TARGET_COST)
    selected = json.loads((directory / "selection.json").read_text())
    tensors = {}
    cases = [name for name, _, _ in scenarios()]
    metrics = ("mean_cost", "weighted_success_rate", "budget_violation_rate", "return")
    missing = []
    for family in FAMILIES:
        items = [item for item in selected if item["family"] == family]
        if any(item["status"] != "ready" for item in items):
            missing.append({"family": family, "reason": f"Not all training seeds bracket validation cost {TARGET_COST:g}."})
            continue
        endpoint_names = ("low", "high", "guard_low", "guard_high")
        values = np.zeros((3, 4, len(TEST_SEEDS), len(cases), len(metrics)))
        for train_index, seed in enumerate(TRAIN_SEEDS):
            path = directory / "held_out" / f"{family}_seed{seed}.json"
            rows = json.loads(path.read_text())
            expected = {(endpoint, seed, case) for endpoint in endpoint_names
                        for seed in TEST_SEEDS for case in cases}
            observed = {(row["endpoint"], row["evaluation_seed"], row["case"]) for row in rows}
            if observed != expected or len(rows) != len(expected):
                raise ValueError(f"incomplete or duplicated held-out episodes: {path}")
            for row in rows:
                endpoint = endpoint_names.index(row["endpoint"])
                values[train_index, endpoint, TEST_SEEDS.index(row["evaluation_seed"]),
                       cases.index(row["case"])] = [row[key] for key in metrics]
        tensors[family] = values

    def estimate(tensor, train_indices, test_indices, case_indices):
        result = []
        for train_index in train_indices:
            subset = tensor[train_index][:, test_indices][:, :, case_indices]
            value = matched_value(subset)
            if value is None:
                return None
            result.append(value)
        return np.mean(result, axis=0)

    points = {}
    for family, tensor in tensors.items():
        value = estimate(tensor, range(3), range(len(TEST_SEEDS)), range(len(cases)))
        if value is None:
            missing.append({"family": family, "reason": f"Held-out measured costs do not bracket {TARGET_COST:g}; no extrapolation."})
        else:
            points[family] = value
    generator = np.random.default_rng(44001)
    bootstrap = {family: [] for family in points}
    comparisons = (
        ("adaptive_minus_static", "adaptive", "static"),
        ("static_minus_ml", "static", "ml"),
        ("adaptive_minus_ml", "adaptive", "ml"),
        ("adaptive_slow_minus_static", "adaptive_slow", "static"),
        ("adaptive_slow_minus_fixed", "adaptive_slow", "fixed"),
        ("adaptive_slow_minus_ml", "adaptive_slow", "ml"),
    )
    paired = {name: [] for name, _, _ in comparisons}
    for _ in range(3000):
        train_indices = generator.integers(0, 3, 3)
        test_indices = generator.integers(0, len(TEST_SEEDS), len(TEST_SEEDS))
        sample = {}
        for family in points:
            value = estimate(tensors[family], train_indices, test_indices, range(len(cases)))
            if value is not None:
                sample[family] = value[1]
                bootstrap[family].append(value[1])
        for name, left, right in comparisons:
            if left in sample and right in sample:
                paired[name].append(sample[left] - sample[right])

    rows = []
    for family, value in points.items():
        interval = np.quantile(bootstrap[family], [0.025, 0.975]).tolist()
        rows.append({"family": family, "label": LABELS[family],
                     "matched_cost": TARGET_COST, "weighted_success_rate": float(value[1]),
                     "bootstrap_ci95": interval,
                     "budget_violation_rate": float(value[2]),
                     "valid_bootstrap_replicates": len(bootstrap[family])})
    differences = {}
    for name, samples in paired.items():
        if not samples:
            continue
        left, right = name.split("_minus_")
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        differences[name] = {
            "difference": float(points[left][1] - points[right][1]),
            "bootstrap_ci95": interval, "valid_bootstrap_replicates": len(samples),
            "assessment": "inconclusive" if len(samples) < 2850 else (
                "positive" if interval[0] > 0 else "negative" if interval[1] < 0 else "inconclusive"
            ),
        }
    subsets = {}
    for group in ("stationary", "continuous"):
        indices = [i for i, name in enumerate(cases) if name.endswith("continuous") == (group == "continuous")]
        subsets[group] = {}
        for family, tensor in tensors.items():
            value = estimate(tensor, range(3), range(len(TEST_SEEDS)), indices)
            subsets[group][family] = None if value is None else float(value[1])
    payload = {
        "environment_version": ENVIRONMENT_VERSION,
        "target_cost": TARGET_COST, "rows": rows, "paired_differences": differences,
        "subsets": subsets, "unavailable": missing,
        "inference": "Hierarchical paired bootstrap over 3 training seeds and 12 evaluation seeds; six scenarios fixed.",
        "scope": "Validation-selected operating points, held-out simulator tests, cost interpolation only.",
    }
    write_json(directory / "effect_summary.json", payload)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    short = {"fixed": "Fixed", "ml": "History ML", "static": "QR-DQN\noriginal\npenalty",
             "adaptive": "QR-DQN\nfast\nadaptive", "adaptive_slow": "QR-DQN\nslow\nadaptive"}
    for index, row in enumerate(rows):
        mean, (low, high) = row["weighted_success_rate"], row["bootstrap_ci95"]
        axes[0].bar(index, 100 * mean, color=("#888888", "#e49b31", "#347db1", "#3b9a6d", "#9255a6")[index])
        axes[0].errorbar(index, 100 * mean, yerr=[[100 * max(0, mean-low)], [100 * max(0, high-mean)]],
                         color="black", capsize=4)
    axes[0].set_xticks(range(len(rows)), [short[row["family"]] for row in rows])
    axes[0].set_ylabel("Importance-weighted success (%)")
    axes[0].set_title(f"Held-out results at average cost {TARGET_COST:g}")
    axes[0].grid(axis="y", alpha=0.2)
    for index, (name, row) in enumerate(differences.items()):
        mean, (low, high) = row["difference"], row["bootstrap_ci95"]
        axes[1].errorbar(100 * mean, index,
                         xerr=[[100 * max(0, mean-low)], [100 * max(0, high-mean)]],
                         fmt="o", capsize=4, color="#347db1")
    comparison_labels = {
        "static_minus_ml": "Original RL - ML",
        "adaptive_minus_static": "Fast RL - Original RL",
        "adaptive_minus_ml": "Fast RL - ML",
        "adaptive_slow_minus_static": "Slow RL - Original RL",
        "adaptive_slow_minus_fixed": "Slow RL - Fixed",
        "adaptive_slow_minus_ml": "Slow RL - ML",
    }
    axes[1].set_yticks(range(len(differences)), [comparison_labels[name] for name in differences])
    axes[1].axvline(0, linestyle="--", color="gray")
    axes[1].set_xlabel("Paired difference (percentage points)")
    axes[1].set_title("95% hierarchical bootstrap intervals")
    figure.tight_layout()
    figure.savefig(directory / "effect_comparison.png", dpi=180)
    plt.close(figure)

    lines = ["# RL Effectiveness Verification", "", "## Protocol", "",
             "- Original-budget and adaptive-budget QR-DQN each use 3 training seeds and 100,000 steps per seed.",
             "- After the fast penalty update degraded validation performance, a slower post-warmup version was trained with 3 additional seeds; the failed results are retained.",
             "- History ML uses the same 25-dimensional observations and training sample count; SVM/RF selection uses validation data.",
             "- Fixed baselines are selected on validation data from all 12 actions and fixed randomized mixtures; calibrated mixtures are also allowed for ML.",
             "- Validation and test seeds are independent; testing uses 12 seeds and 6 fixed scenarios, including continuous attacker arrivals and departures.",
             f"- Target cost is {TARGET_COST:g}. Interpolation prefers the primary points frozen on validation data; frozen guard points are used only when the primary points no longer cover the target. No point selection by test success and no extrapolation.",
             "- 95% intervals use paired hierarchical bootstrap over training and test seeds, with only 3 training seeds.", "",
             "## Held-out testing", "", "| Method | Matched-cost success | 95% CI |", "|---|---:|---:|"]
    for row in rows:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['label']} | {100*row['weighted_success_rate']:.2f}% | {100*low:.2f}–{100*high:.2f}% |")
    lines += ["", "## Paired comparisons", ""]
    for name, value in differences.items():
        low, high = value["bootstrap_ci95"]
        lines.append(f"- {name}: {100*value['difference']:+.2f} percentage points, CI [{100*low:+.2f}, {100*high:+.2f}]; {value['assessment']}.")
    for item in missing:
        lines.append(f"- {item['family']}: {item['reason']}")
    if "adaptive_slow_minus_static" in differences:
        value = differences["adaptive_slow_minus_static"]
        lines += ["", "## Conclusions", "",
                  f"The slow budget update changes matched-cost success by {100*value['difference']:+.2f} percentage points relative to original RL; assessment: {value['assessment']}."]
        baseline = differences.get("adaptive_slow_minus_fixed")
        if baseline and baseline["assessment"] != "positive":
            lines.append("This does not establish that RL outperforms the strongest fixed strategy; the fixed-versus-RL difference must still be reported.")
        by_family = {row["family"]: row for row in rows}
        if "static" in by_family and "adaptive_slow" in by_family:
            lines.append(
                f"Episode budget violation rates at matched cost: original RL {100*by_family['static']['budget_violation_rate']:.2f}%, "
                f"slow update {100*by_family['adaptive_slow']['budget_violation_rate']:.2f}%; the budget constraint remains soft."
            )
    lines += ["", "These results apply only to this simulator and budget point; they do not measure real LLM task accuracy.",
              "All methods share the corrected rotation dynamics; this experiment tests budget adjustment and does not attribute the environment correction to improved RL performance."]
    (directory / "EFFECT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    global TARGET_COST
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "train", "train-slow", "ml", "validate", "select", "test", "report"), default="all")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--target-cost", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_effect_check"))
    args = parser.parse_args()
    TARGET_COST = args.target_cost
    phases = {"train": run_training, "train-slow": run_slow_training,
              "ml": run_ml, "validate": run_validation,
              "select": run_selection, "test": run_test, "report": run_report}
    for name in phases if args.phase == "all" else (args.phase,):
        phases[name](args)


if __name__ == "__main__":
    main()
