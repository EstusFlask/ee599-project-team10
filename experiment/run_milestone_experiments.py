"""Reproduce the early four-controller experiment with one or three attackers."""

from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import ast
import json
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import defender
from .train import make_supervised_data, run_experiment


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/milestone_b9dafc0"
BUDGETS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SEEDS = (7, 42, 99)
SETTINGS = dict(attacker="hmm", profile="bursty", train_episodes=1000,
                test_episodes=10, steps=180, window=30, clean_accuracy=0.8)
ORIGINAL_PREDICT = defender.MLdefender_withstate.predict


def fast_knn_predict(self, observation=None):
    """Same ordered distance sum and stable neighbor voting as the scalar k-NN."""
    if not self.samples:
        action = self.actions[0]
    else:
        if not hasattr(self, "_milestone_sample_matrix"):
            self._milestone_sample_matrix = np.asarray(self.samples, dtype=np.float64)
        target = np.asarray(self._features(observation), dtype=np.float64)
        delta = self._milestone_sample_matrix - target
        distances = np.zeros(len(self.samples), dtype=np.float64)
        for feature in range(delta.shape[1]):
            distances += delta[:, feature] ** 2
        nearest = np.argsort(distances, kind="stable")[:self.neighbors]
        action = Counter(self.labels[int(index)] for index in nearest).most_common(1)[0][0]
    return defender._nearest_action(action, defender._allowed(observation, self.actions))


def verify_early_code_and_acceleration():
    for name in ("attacker.py", "defender.py", "train.py", "run_single_attacker.py"):
        historical = subprocess.check_output(
            ["git", "show", f"b9dafc0:expriment/{name}"], cwd=ROOT.parent
        ).decode("utf-8")
        actual = (ROOT / name).read_text(encoding="utf-8")
        assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(historical)), name
    observations, labels = make_supervised_data(
        episodes=40, steps=100, method="hmm", profile="bursty", window=30,
        seed=2026, budget=1, attackers=3,
    )
    model = defender.MLdefender_withstate(window=30).train(observations, labels)
    test_observations = random.Random(173).sample(observations, 100)
    for observation in test_observations:
        assert ORIGINAL_PREDICT(model, observation) == fast_knn_predict(model, observation)
    # Also check the exact distance vector used for tie-preserving neighbor selection.
    for observation in test_observations[:10]:
        target = model._features(observation)
        scalar = [sum((a-b)**2 for a,b in zip(sample,target)) for sample in model.samples]
        delta = np.asarray(model.samples) - np.asarray(target)
        vector = np.zeros(len(model.samples))
        for feature in range(delta.shape[1]):
            vector += delta[:, feature] ** 2
        assert np.array_equal(vector, np.asarray(scalar))
    print("Verified historical AST equivalence, 100 k-NN action checks and 10 exact distance-vector checks.", flush=True)


def one_run(attackers, seed, budget):
    path = RESULTS / f"attackers{attackers}" / f"seed{seed}_budget{budget:g}.json"
    if path.exists():
        return attackers, seed, budget, json.loads(path.read_text(encoding="utf-8"))["timing"]["total_seconds"], "cached"
    defender.MLdefender_withstate.predict = fast_knn_predict
    args = SimpleNamespace(**SETTINGS, attackers=attackers, seed=seed, budget=budget, output=None)
    payload = run_experiment(args, display=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return attackers, seed, budget, payload["timing"]["total_seconds"], "completed"


def main():
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    verify_early_code_and_acceleration()
    jobs = [(attackers, seed, budget) for attackers in (1,3) for seed in SEEDS for budget in BUDGETS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one_run, *job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            attackers, seed, budget, seconds, status = future.result()
            print(f"[{i:02d}/{len(jobs)}] attackers={attackers} seed={seed} budget={budget:g}: {status}, {seconds:.1f}s", flush=True)
    protocol = dict(source_commit="b9dafc0", settings=SETTINGS, attacker_counts=[1,3],
                    seeds=SEEDS, nominal_budgets=BUDGETS, runs=len(jobs),
                    controller_evaluation_episodes=len(jobs)*4*SETTINGS["test_episodes"],
                    same_training_configuration_across_attacker_counts=True,
                    code_ast_equivalent_to_historical_commit=True,
                    knn_optimization="Exact float64 left-to-right squared-distance accumulation and stable neighbor ordering; unchanged labels, k and features.",
                    knn_action_equivalence_checks=100, knn_exact_distance_checks=10,
                    wall_seconds=time.perf_counter()-started)
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Completed all 42 runs in {protocol['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
