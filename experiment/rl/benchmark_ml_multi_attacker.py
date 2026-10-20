"""Extend the existing ML benchmark to the exact multi-attacker RL test grid."""

from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .benchmark_ml_methods import ML_ORDER, build_estimators
from .benchmark_multi_attacker_matched_cost import run_gated_episode
from .environment import DefenseEnv, SCENARIOS
from .plot_budget_success_comparison import generate_ml_dataset


ROOT = Path(__file__).resolve().parents[1] / "results/rl_multi_attacker_final/analysis"
CACHE = ROOT / "ml_extension"
BASES = ("persistent", "adaptive", "noisy_ood", "budget_shock")
COUNTS = (1, 2, 3, 5)
SEEDS = (3101, 3202, 3303)
BUDGETS = (0.8, 1.2)
GATES = tuple(i / 10 for i in range(11))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_file(name):
    return CACHE / (name.lower().replace(" ", "_") + ".joblib")


def train_models():
    CACHE.mkdir(parents=True, exist_ok=True)
    dataset = CACHE / "supervised_data_seed2026.npz"
    if not dataset.exists():
        print("Generating 60,000 training and 15,000 validation observations (same original ML protocol).", flush=True)
        started = time.perf_counter()
        x_train, y_train = generate_ml_dataset(60_000, 2026)
        print(f"Training data generated in {time.perf_counter()-started:.1f}s", flush=True)
        x_valid, y_valid = generate_ml_dataset(15_000, 102026)
        np.savez_compressed(dataset, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
    with np.load(dataset) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_valid, y_valid = data["x_valid"], data["y_valid"]
    assert x_train.shape == (60_000, 25) and x_valid.shape == (15_000, 25)
    metrics = []
    for name, (estimator, indices) in build_estimators(2026).items():
        assert indices is None, "All ML policies must use all 25 observable features"
        started = time.perf_counter()
        if model_file(name).exists():
            estimator = joblib.load(model_file(name))
        else:
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 2
            estimator.fit(x_train, y_train)
            # Small-batch prediction is faster without per-call worker creation.
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1
            joblib.dump(estimator, model_file(name), compress=3)
        prediction = estimator.predict(x_valid)
        row = dict(policy=name, train_samples=len(y_train), validation_samples=len(y_valid),
                   accuracy=accuracy_score(y_valid, prediction),
                   balanced_accuracy=balanced_accuracy_score(y_valid, prediction),
                   macro_f1=f1_score(y_valid, prediction, average="macro", zero_division=0),
                   fit_and_validation_seconds=time.perf_counter()-started)
        metrics.append(row)
        print(f"Trained {name}: validation accuracy={row['accuracy']:.4f}; {row['fit_and_validation_seconds']:.1f}s", flush=True)
    write_csv(CACHE / "ml_validation.csv", metrics)


class ScalarPolicy:
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, observation, deterministic=True):
        return self.estimator.predict(np.asarray(observation, dtype=np.float32).reshape(1, -1))[0], None


def evaluate_model(name):
    output = model_file(name).with_suffix(".csv")
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1056
        return name, len(rows), 0.0
    started = time.perf_counter()
    estimator = joblib.load(model_file(name))
    states = []
    for base in BASES:
        for count in COUNTS:
            config = replace(SCENARIOS[base], name=f"{base}_attackers{count}", attacker_count=count)
            for budget in BUDGETS:
                for gate in GATES[1:]:
                    for seed in SEEDS:
                        env = DefenseEnv(config, budget_scale=budget)
                        observation, _ = env.reset(seed=seed)
                        states.append(dict(env=env, observation=observation, base_scenario=base,
                                           attacker_count=count, budget_scale=budget, gate=gate, seed=seed))
    assert len(states) == 960
    phase = np.array([(s["seed"] % 97) / 97.0 for s in states], dtype=np.float64)
    gates = np.array([s["gate"] for s in states], dtype=np.float64)
    cheap_action = DefenseEnv.encode_action(1, "none")
    horizon = states[0]["env"].config.horizon
    for step in range(horizon):
        observations = np.array([s["observation"] for s in states], dtype=np.float32)
        proposals = estimator.predict(observations)
        phase += gates
        active = phase >= 1.0
        phase[active] -= 1.0
        actions = np.where(active, proposals, cheap_action)
        for state, action in zip(states, actions):
            observation, _, terminated, truncated, info = state["env"].step(int(action))
            state["observation"] = observation
            assert not truncated and terminated == (step == horizon - 1)
            if terminated:
                state["metrics"] = info["episode_metrics"]
        if (step + 1) % 64 == 0:
            print(f"Evaluating {name}: {step+1}/{horizon} steps for 960 episodes; {time.perf_counter()-started:.0f}s", flush=True)
    rows = [dict(mean_cost=s["metrics"]["mean_cost"],
                 weighted_success_rate=s["metrics"]["weighted_success_rate"],
                 success_rate=s["metrics"]["success_rate"], policy=name,
                 base_scenario=s["base_scenario"], attacker_count=s["attacker_count"],
                 budget_scale=s["budget_scale"], gate=s["gate"], seed=s["seed"])
            for s in states]
    # At gate=0 every method takes exactly the same cheap action; reuse those trajectories.
    with (ROOT / "matched_cost_multi_raw.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "QR-DQN original" and float(row["gate"]) == 0:
                row["policy"] = name
                rows.append(row)
    assert len(rows) == 1056
    # Check batching against the original sequential evaluator for every attacker count.
    for count in COUNTS:
        config = replace(SCENARIOS["persistent"], name=f"persistent_attackers{count}", attacker_count=count)
        scalar = run_gated_episode(ScalarPolicy(estimator), config, SEEDS[0], 0.5, BUDGETS[0])
        batched = next(r for r in rows if r["base_scenario"] == "persistent"
                       and int(r["attacker_count"]) == count and int(r["seed"]) == SEEDS[0]
                       and float(r["gate"]) == 0.5 and float(r["budget_scale"]) == BUDGETS[0])
        for metric in ("mean_cost", "weighted_success_rate", "success_rate"):
            assert np.isclose(float(batched[metric]), scalar[metric], atol=1e-12, rtol=0), (name, count, metric)
    write_csv(output, rows)
    return name, len(rows), time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train_models()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_model, name): name for name in ML_ORDER}
        for future in as_completed(futures):
            name, episodes, seconds = future.result()
            print(f"Completed {name}: {episodes} evaluation rows, {seconds:.1f}s", flush=True)
    combined = []
    for name in ML_ORDER:
        with model_file(name).with_suffix(".csv").open(encoding="utf-8") as handle:
            combined.extend(csv.DictReader(handle))
    assert len(combined) == 8448
    write_csv(CACHE / "ml_multi_attacker_raw.csv", combined)
    protocol = dict(training_seed=2026, train_samples=60000, validation_samples=15000,
                    training_profile="robust single-attacker; unchanged original supervised ML protocol",
                    observation_dimensions=25, hidden_attack_state_visible_at_inference=False,
                    ml_policies=list(ML_ORDER), test_scenarios=list(BASES), attacker_counts=list(COUNTS),
                    evaluation_seeds=list(SEEDS), budgets=list(BUDGETS), gates=list(GATES),
                    evaluation_rows=len(combined), gate_zero_rows_reused=768,
                    new_simulated_episodes=7680,
                    scalar_batch_equivalence_checks=32,
                    source_rl_data="../matched_cost_multi_raw.csv")
    (CACHE / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"All ML methods evaluated: {CACHE / 'ml_multi_attacker_raw.csv'}", flush=True)


if __name__ == "__main__":
    main()
