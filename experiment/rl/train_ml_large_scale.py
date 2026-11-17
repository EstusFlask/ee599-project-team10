"""Supervised ML defenders for the large-scale environment.

Same observation as the RL agent (estimator features including the price).  The
label of a state is the oracle action: the k in 1,3,...,2N+1 that maximizes
``importance * (+1 | -failure_penalty) - price * cost`` given the hidden attack
count.  States come from rollouts that follow the oracle half of the time and a
random k otherwise, so observations cover both calm and attacked histories.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from .benchmark_ml_methods import build_estimators
from .large_scale import SCALE_SCENARIOS, ScaleDefenseEnv


def oracle_action(env: ScaleDefenseEnv) -> int:
    """Best k index for the current round when the true attack count is known."""
    ks = 2 * np.arange(env.num_actions) + 1
    costs = np.asarray([env._cost(int(k), env.message_length) for k in ks])
    success = env.true_attacks <= (ks - 1) // 2
    utility = env.importance * np.where(success, 1.0, -env.failure_penalty) - env.price * costs
    return int(np.argmax(utility))


def generate(samples: int, seed: int, scenario: str, delay: int):
    rng = np.random.default_rng(seed + 700_000)
    config = replace(SCALE_SCENARIOS[scenario], feedback_delay=delay)
    observations, labels = [], []
    episode = 0
    while len(labels) < samples:
        env = ScaleDefenseEnv(
            config, domain_randomization=True, observation_mode="estimator",
            action_mode="k", price_conditioned=True,
        )
        obs, _ = env.reset(seed=seed + episode)
        episode += 1
        done = False
        while not done and len(labels) < samples:
            label = oracle_action(env)
            observations.append(obs.copy())
            labels.append(label)
            action = label if rng.random() < 0.5 else int(rng.integers(env.action_space.n))
            obs, _, done, _, _ = env.step(action)
    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario", default="campaign_waves")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/rl_large_scale_v6_ml/models"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y = generate(args.samples, args.seed, args.scenario, args.delay)
    xv, yv = generate(args.validation_samples, args.seed + 50_000, args.scenario, args.delay)
    report = []
    for name, (estimator, _) in build_estimators(args.seed).items():
        started = time.perf_counter()
        estimator.fit(x, y)
        accuracy = float(accuracy_score(yv, estimator.predict(xv)))
        slug = name.lower().replace(" ", "_")
        joblib.dump(estimator, args.output / f"{slug}.joblib")
        report.append({"model": name, "validation_accuracy": accuracy,
                       "training_seconds": time.perf_counter() - started})
        print(json.dumps(report[-1]), flush=True)
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
