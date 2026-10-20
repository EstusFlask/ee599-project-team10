"""Train the original linear Q-learning baseline under the common POMDP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .legacy_q import LegacyLinearQPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-profile", choices=("standard", "robust"), default="robust")
    parser.add_argument("--constraint-weight", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("experiment/results/rl_robust_w015/models"))
    args = parser.parse_args()
    started = time.perf_counter()
    policy = LegacyLinearQPolicy(seed=args.seed)
    policy.train(args.steps, args.seed, args.training_profile, args.constraint_weight)
    path = args.output / f"legacy_q_seed{args.seed}.pkl"
    policy.save(path)
    print(json.dumps({
        "algorithm": "legacy_q", "steps": args.steps, "seed": args.seed,
        "training_profile": args.training_profile,
        "constraint_weight": args.constraint_weight,
        "training_seconds": time.perf_counter() - started,
        "model_path": str(path),
    }), flush=True)


if __name__ == "__main__":
    main()
