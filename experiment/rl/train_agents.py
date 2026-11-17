"""Train several distinct RL families on the randomized defense POMDP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, QRDQN, RecurrentPPO, TRPO

from .environment import DefenseEnv
from .config import DEFAULT_REWARD, ENVIRONMENT_VERSION
from .budget import DEFAULT_BUDGET_LEARNING_RATE, BudgetController, BudgetPenaltyCallback
from .ablations import CurrentObservation


ALGORITHMS = (
    "ppo", "a2c", "dqn", "qrdqn", "recurrent_ppo", "maskable_ppo", "trpo"
)


def make_training_env(
    seed: int,
    n_envs: int,
    training_profile: str = "standard",
    constraint_weight: float = 0.0,
    attacker_count_probabilities: tuple[float, ...] | None = None,
    failure_penalty: float = DEFAULT_REWARD.failure_penalty,
    adaptive_budget: bool = False,
    use_history: bool = True,
):
    def factory():
        env = DefenseEnv(
            "train_mix",
            domain_randomization=True,
            randomization_profile=training_profile,
            constraint_weight=constraint_weight,
            attacker_count_probabilities=attacker_count_probabilities,
            failure_penalty=failure_penalty,
            budget_penalty_mode="lagrangian" if adaptive_budget else "reserve",
        )
        return Monitor(env if use_history else CurrentObservation(env))

    return make_vec_env(factory, n_envs=n_envs, seed=seed)


def build_model(name: str, env, seed: int, device: str, discount: float = 0.985):
    common = dict(env=env, seed=seed, device=device, verbose=0)
    policy_kwargs = dict(net_arch=[128, 128], activation_fn=torch.nn.ReLU)
    if name == "ppo":
        return PPO(
            "MlpPolicy", n_steps=512, batch_size=256, n_epochs=8,
            learning_rate=3e-4, gamma=discount, gae_lambda=0.93,
            ent_coef=0.012, policy_kwargs=policy_kwargs, **common,
        )
    if name == "a2c":
        return A2C(
            "MlpPolicy", n_steps=64, learning_rate=4e-4, gamma=discount,
            gae_lambda=0.93, ent_coef=0.015, policy_kwargs=policy_kwargs, **common,
        )
    if name == "dqn":
        return DQN(
            "MlpPolicy", buffer_size=120_000, learning_starts=8_000,
            batch_size=256, learning_rate=3e-4, gamma=discount,
            train_freq=4, gradient_steps=2, target_update_interval=2_000,
            exploration_fraction=0.30, exploration_final_eps=0.04,
            policy_kwargs=policy_kwargs, **common,
        )
    if name == "qrdqn":
        return QRDQN(
            "MlpPolicy", buffer_size=120_000, learning_starts=8_000,
            batch_size=256, learning_rate=3e-4, gamma=discount,
            train_freq=4, gradient_steps=2, target_update_interval=2_000,
            exploration_fraction=0.30, exploration_final_eps=0.04,
            policy_kwargs=dict(net_arch=[128, 128], n_quantiles=50), **common,
        )
    if name == "recurrent_ppo":
        return RecurrentPPO(
            "MlpLstmPolicy", n_steps=256, batch_size=256, n_epochs=6,
            learning_rate=3e-4, gamma=discount, gae_lambda=0.93,
            ent_coef=0.012,
            policy_kwargs=dict(
                net_arch=dict(pi=[96], vf=[96]), lstm_hidden_size=128,
                n_lstm_layers=1, shared_lstm=False,
            ),
            **common,
        )
    if name == "maskable_ppo":
        return MaskablePPO(
            "MlpPolicy", n_steps=512, batch_size=256, n_epochs=8,
            learning_rate=3e-4, gamma=discount, gae_lambda=0.93,
            ent_coef=0.018, policy_kwargs=policy_kwargs, **common,
        )
    if name == "trpo":
        return TRPO(
            "MlpPolicy", n_steps=512, batch_size=256, gamma=discount,
            gae_lambda=0.93, learning_rate=8e-4, n_critic_updates=10,
            target_kl=0.015, policy_kwargs=policy_kwargs, **common,
        )
    raise ValueError(f"unknown algorithm: {name}")


def train_one(
    name: str, steps: int, seed: int, output: Path, device: str, n_envs: int,
    resume: bool = False, training_profile: str = "standard",
    constraint_weight: float = 0.0,
    adaptive_budget: bool = False,
    budget_learning_rate: float = DEFAULT_BUDGET_LEARNING_RATE,
    budget_warmup_steps: int | None = None,
    use_history: bool = True,
    discount: float = 0.985,
):
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    algo_envs = 1 if name in {"dqn", "qrdqn"} else n_envs
    env = make_training_env(
        seed, algo_envs, training_profile=training_profile,
        constraint_weight=constraint_weight,
        adaptive_budget=adaptive_budget,
        use_history=use_history,
    )
    path = output / f"{name}_seed{seed}"
    started = time.perf_counter()
    # SB3's small MLP actor-critic implementations are faster on CPU; replay-
    # buffer value methods benefit from CUDA once gradient updates begin.
    effective_device = (
        "cpu" if device == "auto" and name in {
            "ppo", "a2c", "recurrent_ppo", "maskable_ppo", "trpo"
        }
        else device
    )
    if resume and path.with_suffix(".zip").exists():
        classes = {
            "ppo": PPO, "a2c": A2C, "dqn": DQN,
            "qrdqn": QRDQN, "recurrent_ppo": RecurrentPPO,
            "maskable_ppo": MaskablePPO, "trpo": TRPO,
        }
        model = classes[name].load(path, env=env, device=effective_device)
        previous_steps = int(model.num_timesteps)
        if model.gamma != discount:
            env.close()
            raise ValueError("changing the discount requires a fresh ablation run")
    else:
        model = build_model(name, env, seed, effective_device, discount=discount)
        previous_steps = 0
    controller = None
    effective_warmup = (
        int(getattr(model, "learning_starts", 0))
        if budget_warmup_steps is None else budget_warmup_steps
    )
    if adaptive_budget:
        budget_path = path.with_suffix(".budget.json")
        initial_weight = constraint_weight
        history = []
        if resume and previous_steps and budget_path.exists():
            saved = json.loads(budget_path.read_text())
            initial_weight = saved["weight"]
            history = saved["history"]
        controller = BudgetController(
            weight=initial_weight, learning_rate=budget_learning_rate, history=history,
        )
    model.learn(
        total_timesteps=steps,
        progress_bar=False,
        reset_num_timesteps=not resume,
        callback=BudgetPenaltyCallback(controller, effective_warmup) if controller else None,
    )
    elapsed = time.perf_counter() - started
    model.save(path)
    if controller is not None:
        path.with_suffix(".budget.json").write_text(json.dumps({
            "weight": controller.weight,
            "learning_rate": controller.learning_rate,
            "max_weight": controller.max_weight,
            "warmup_steps": effective_warmup,
            "history": controller.history,
        }, indent=2), encoding="utf-8")
    env.close()
    metadata = {
        "environment_version": ENVIRONMENT_VERSION,
        "algorithm": name,
        "steps": steps,
        "previous_steps": previous_steps,
        "total_steps": int(model.num_timesteps),
        "seed": seed,
        "device_requested": device,
        "device_used": str(model.device),
        "training_seconds": elapsed,
        "parallel_environments": algo_envs,
        "training_profile": training_profile,
        "use_history": use_history,
        "discount": discount,
        "constraint_weight": constraint_weight,
        "adaptive_budget": adaptive_budget,
        "final_constraint_weight": controller.weight if controller else constraint_weight,
        "budget_learning_rate": budget_learning_rate if controller else None,
        "budget_warmup_steps": effective_warmup if controller else None,
        "model_path": str(path.with_suffix(".zip")),
    }
    print(json.dumps(metadata), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--discount", type=float, default=0.985)
    parser.add_argument(
        "--training-profile", choices=("standard", "robust", "multi_attacker"),
        default="standard"
    )
    parser.add_argument("--constraint-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-budget", action="store_true",
                        help="Adjust the communication-cost penalty after completed episodes.")
    parser.add_argument("--budget-learning-rate", type=float, default=DEFAULT_BUDGET_LEARNING_RATE)
    parser.add_argument("--budget-warmup-steps", type=int,
                        help="Skip initial random exploration (default: algorithm learning_starts).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        folder = "rl_adaptive_budget" if args.adaptive_budget else "rl_v1"
        args.output = Path("experiment/results") / folder / "models"
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for algorithm in args.algorithms:
        records.append(train_one(
            algorithm, args.steps, args.seed, args.output, args.device, args.n_envs,
            resume=args.resume, training_profile=args.training_profile,
            constraint_weight=args.constraint_weight,
            adaptive_budget=args.adaptive_budget,
            budget_learning_rate=args.budget_learning_rate,
            budget_warmup_steps=args.budget_warmup_steps,
            use_history=not args.no_history, discount=args.discount,
        ))
    (args.output / "training_metadata.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
