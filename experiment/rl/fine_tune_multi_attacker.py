"""Low-learning-rate adaptation of the robust QR-DQN to 1-5 attackers."""

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
from stable_baselines3.common.utils import get_linear_fn

from .train_agents import make_training_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/results/rl_multi_attacker_finetuned/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--constraint-weight", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    env = make_training_env(
        args.seed, 1, training_profile="multi_attacker",
        constraint_weight=args.constraint_weight,
    )
    model = QRDQN.load(args.source, env=env, device=args.device)
    # Saved SB3 models do not include the replay buffer.  Restart the training
    # clock so the buffer receives a genuine warm-up before gradient updates.
    model.replay_buffer.reset()
    model.num_timesteps = 0
    model.learning_starts = 8_000
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_linear_fn(
        args.learning_rate, args.learning_rate * 0.30, 1.0,
    )
    model.exploration_initial_eps = 0.15
    model.exploration_final_eps = 0.03
    model.exploration_fraction = 0.35
    model.exploration_schedule = get_linear_fn(0.15, 0.03, 0.35)

    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=True, progress_bar=False)
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()
    metadata = {
        "source_model": str(args.source),
        "output_model": str(args.output),
        "adaptation_steps": args.steps,
        "seed": args.seed,
        "learning_rate_initial": args.learning_rate,
        "learning_rate_final": args.learning_rate * 0.30,
        "constraint_weight": args.constraint_weight,
        "training_profile": "multi_attacker",
        "training_seconds": elapsed,
        "device_used": str(model.device),
    }
    args.output.with_name("fine_tune_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
