"""Reliability-weighted final adaptation for the curriculum QR-DQN."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sb3_contrib import QRDQN

from .train_agents import make_training_env


def linear_schedule(initial, final):
    return lambda progress_remaining: (
        final + (initial - final) * float(progress_remaining)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "experiment/results/rl_multi_attacker_curriculum/models/"
            "qrdqn_selected_seed2026.zip"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "experiment/results/rl_multi_attacker_reliability/models/"
            "qrdqn_seed2026.zip"
        ),
    )
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--failure-penalty", type=float, default=2.4)
    parser.add_argument("--constraint-weight", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    attacker_probabilities = (0.30, 0.22, 0.18, 0.15, 0.15)
    env = make_training_env(
        args.seed + 31, 1, training_profile="multi_attacker",
        constraint_weight=args.constraint_weight,
        attacker_count_probabilities=attacker_probabilities,
        failure_penalty=args.failure_penalty,
    )
    model = QRDQN.load(args.source, env=env, device=args.device)
    model.replay_buffer.reset()
    model.num_timesteps = 0
    model.learning_starts = 8_000
    model.learning_rate = 4e-5
    model.lr_schedule = linear_schedule(4e-5, 1e-5)
    model.exploration_initial_eps = 0.05
    model.exploration_final_eps = 0.015
    model.exploration_fraction = 0.50
    model.exploration_schedule = linear_schedule(0.05, 0.015)

    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=True, progress_bar=False)
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()
    metadata = {
        "source_model": str(args.source),
        "output_model": str(args.output),
        "steps": args.steps,
        "seed": args.seed,
        "attacker_probabilities": attacker_probabilities,
        "failure_penalty": args.failure_penalty,
        "constraint_weight": args.constraint_weight,
        "learning_rate_initial": 4e-5,
        "learning_rate_final": 1e-5,
        "training_seconds": elapsed,
        "device_used": str(model.device),
    }
    args.output.with_name("reliability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
