"""Curriculum adaptation that preserves single-attacker competence."""

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


PHASES = (
    {
        "name": "phase1_preserve",
        "steps": 80_000,
        "attacker_probabilities": (0.55, 0.30, 0.15, 0.0, 0.0),
        "learning_rate": 7e-5,
        "exploration_initial": 0.12,
        "exploration_final": 0.04,
    },
    {
        "name": "phase2_expand",
        "steps": 120_000,
        "attacker_probabilities": (0.35, 0.25, 0.20, 0.12, 0.08),
        "learning_rate": 5e-5,
        "exploration_initial": 0.08,
        "exploration_final": 0.025,
    },
    {
        "name": "phase3_harden",
        "steps": 100_000,
        "attacker_probabilities": (0.30, 0.22, 0.18, 0.15, 0.15),
        "learning_rate": 3e-5,
        "exploration_initial": 0.05,
        "exploration_final": 0.02,
    },
)


def linear_schedule(initial, final):
    return lambda progress_remaining: (
        final + (initial - final) * float(progress_remaining)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_multi_attacker_curriculum/models"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--constraint-weight", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    first_env = make_training_env(
        args.seed, 1, training_profile="multi_attacker",
        constraint_weight=args.constraint_weight,
        attacker_count_probabilities=PHASES[0]["attacker_probabilities"],
    )
    model = QRDQN.load(args.source, env=first_env, device=args.device)
    model.replay_buffer.reset()
    model.num_timesteps = 0
    model.learning_starts = 8_000
    records = []
    total_started = time.perf_counter()

    for phase_index, phase in enumerate(PHASES, 1):
        if phase_index == 1:
            env = first_env
        else:
            env = make_training_env(
                args.seed + phase_index - 1, 1,
                training_profile="multi_attacker",
                constraint_weight=args.constraint_weight,
                attacker_count_probabilities=phase["attacker_probabilities"],
            )
            model.set_env(env)
        model.learning_rate = phase["learning_rate"]
        model.lr_schedule = linear_schedule(
            phase["learning_rate"], phase["learning_rate"] * 0.40
        )
        model.exploration_initial_eps = phase["exploration_initial"]
        model.exploration_final_eps = phase["exploration_final"]
        model.exploration_fraction = 0.55
        model.exploration_schedule = linear_schedule(
            phase["exploration_initial"], phase["exploration_final"]
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=phase["steps"],
            reset_num_timesteps=phase_index == 1,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        checkpoint = args.output / f"qrdqn_{phase['name']}_seed{args.seed}.zip"
        model.save(checkpoint)
        env.close()
        record = {
            **phase,
            "phase": phase_index,
            "elapsed_seconds": elapsed,
            "cumulative_timesteps": int(model.num_timesteps),
            "checkpoint": str(checkpoint),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    metadata = {
        "source_model": str(args.source),
        "seed": args.seed,
        "constraint_weight": args.constraint_weight,
        "device_used": str(model.device),
        "total_training_seconds": time.perf_counter() - total_started,
        "phases": records,
    }
    (args.output / "curriculum_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
