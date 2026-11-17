"""Train an RL agent on the large-scale redundancy-budget environment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .budget import DEFAULT_BUDGET_LEARNING_RATE, BudgetController, BudgetPenaltyCallback
from .large_scale import SCALE_SCENARIOS, ScaleDefenseEnv
from .train_agents import build_model

ALGORITHMS = ("qrdqn", "dqn", "ppo", "a2c", "recurrent_ppo")
ON_POLICY = {"ppo", "a2c", "recurrent_ppo"}


def make_training_env(
    seed: int, scenario: str, constraint_weight: float, delay: int,
    observation_mode: str, action_mode: str, price_conditioned: bool = False,
    receipts: bool = True, feature_mask: tuple[int, ...] = (), n_envs: int = 1,
):
    config = replace(SCALE_SCENARIOS[scenario], feedback_delay=delay)

    def factory():
        return Monitor(ScaleDefenseEnv(
            config, domain_randomization=True, constraint_weight=constraint_weight,
            observation_mode=observation_mode, action_mode=action_mode,
            price_conditioned=price_conditioned, receipts=receipts,
            feature_mask=feature_mask,
        ))

    return make_vec_env(factory, n_envs=n_envs, seed=seed)


def train(
    steps: int, seed: int, output: Path, scenario: str, device: str,
    constraint_weight: float, adaptive_budget: bool, budget_learning_rate: float,
    delay: int, observation_mode: str, action_mode: str, discount: float,
    hidden: int, price_conditioned: bool = False,
    receipts: bool = True, feature_mask: tuple[int, ...] = (),
    algo: str = "qrdqn",
) -> dict:
    n_envs = 8 if algo in ON_POLICY else 1
    env = make_training_env(
        seed, scenario, constraint_weight, delay, observation_mode, action_mode,
        price_conditioned, receipts, feature_mask, n_envs,
    )
    if algo == "qrdqn":
        model = QRDQN(
            "MlpPolicy", env=env, seed=seed, device=device, verbose=0,
            buffer_size=200_000, learning_starts=8_000, batch_size=256,
            learning_rate=3e-4, gamma=discount, train_freq=4, gradient_steps=2,
            target_update_interval=2_000, exploration_fraction=0.30,
            exploration_final_eps=0.04,
            policy_kwargs=dict(net_arch=[hidden, hidden], n_quantiles=50),
        )
    else:
        # Other backbones reuse the project's existing hyper-parameters (train_agents.py).
        model = build_model(algo, env, seed, "cpu" if algo in ON_POLICY else device, discount)
    controller = (
        BudgetController(weight=constraint_weight, learning_rate=budget_learning_rate)
        if adaptive_budget else None
    )
    started = time.perf_counter()
    model.learn(
        total_timesteps=steps, progress_bar=False,
        callback=BudgetPenaltyCallback(controller, getattr(model, "learning_starts", 0))
        if controller else None,
    )
    elapsed = time.perf_counter() - started
    path = output / f"{algo}_seed{seed}"
    model.save(path)
    metadata = {
        "algorithm": algo, "steps": steps, "seed": seed, "scenario": scenario,
        "feedback_delay": delay, "observation_mode": observation_mode,
        "action_mode": action_mode, "discount": discount, "hidden": hidden,
        "price_conditioned": price_conditioned, "receipts": receipts,
        "feature_mask": list(feature_mask),
        "device_used": str(model.device), "training_seconds": elapsed,
        "constraint_weight": constraint_weight, "adaptive_budget": adaptive_budget,
        "final_constraint_weight": controller.weight if controller else constraint_weight,
        "budget_learning_rate": budget_learning_rate if controller else None,
        "model_path": str(path.with_suffix(".zip")),
    }
    if controller:
        path.with_suffix(".budget.json").write_text(
            json.dumps({"weight": controller.weight, "history": controller.history}, indent=2),
            encoding="utf-8",
        )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(metadata), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGORITHMS, default="qrdqn")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--scenario", default="campaign_waves")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budget-learning-rate", type=float, default=DEFAULT_BUDGET_LEARNING_RATE)
    parser.add_argument("--delay", type=int, default=1,
                        help="Rounds before the signed receipt reveals success.")
    parser.add_argument("--obs-mode", choices=("legacy", "estimator"), default="estimator")
    parser.add_argument("--action-mode", choices=("k", "margin"), default="margin")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--no-receipts", action="store_true",
                        help="Ablation: success is never revealed to the policy.")
    parser.add_argument("--mask-features", nargs="*", type=int, default=[],
                        help="Ablation: zero these estimator-observation indices.")
    parser.add_argument("--price-conditioned", action="store_true",
                        help="Charge a random per-episode price per unit cost and show it to the policy.")
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/rl_large_scale_v3/models"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        train(args.steps, seed, args.output, args.scenario, args.device,
              args.constraint_weight, args.adaptive_budget, args.budget_learning_rate,
              args.delay, args.obs_mode, args.action_mode, args.discount, args.hidden,
              args.price_conditioned, not args.no_receipts, tuple(args.mask_features),
              args.algo)


if __name__ == "__main__":
    main()
