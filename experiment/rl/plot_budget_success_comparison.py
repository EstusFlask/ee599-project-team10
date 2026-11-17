"""Plot budget versus defense success for baselines, supervised ML, and RL.

All policies are evaluated in the same difficult named scenarios and with the
same random seeds.  The two supervised policies imitate a one-step expected-
utility oracle using only observable features at inference time.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import MaskablePPO, QRDQN, RecurrentPPO, TRPO

from .baselines import ConservativePolicy, FixedPolicy, RiskHeuristicPolicy
from .environment import DefenseEnv
from .evaluate_agents import run_episode
from .legacy_q import LegacyLinearQPolicy


HARD_SCENARIOS = ("persistent", "adaptive", "multi", "noisy_ood", "budget_shock")
STATELESS_FEATURES = np.asarray([0, 1, 2, 3, 4, 5, 22, 23, 24])


class TreePolicy:
    def __init__(self, model: DecisionTreeClassifier, feature_indices=None):
        self.model = model
        self.feature_indices = feature_indices

    def predict(self, obs, deterministic: bool = True):
        values = np.asarray(obs, dtype=np.float32)
        if self.feature_indices is not None:
            values = values[self.feature_indices]
        action = int(self.model.predict(values.reshape(1, -1))[0])
        return np.asarray(action), None


def _binomial_probability(n: int, p: float, value: int) -> float:
    return math.comb(n, value) * p**value * (1.0 - p) ** (n - value)


def _retry_success_probability(env: DefenseEnv, quorum: int, rotate: bool) -> float:
    attacks = env.true_attacks
    probability = 0.0
    if rotate:
        if env.config.attacker_count > 1:
            # Rotation/background operate on channel masks, so a channel
            # repaired and immediately attacked again is counted only once.
            attacked_p = 1.0 - env.config.rotate_clear_probability * (1.0 - 0.035)
            clean = env.config.max_attack - attacks
            for survivors in range(attacks + 1):
                ps = _binomial_probability(attacks, attacked_p, survivors)
                for background in range(clean + 1):
                    if survivors + background <= quorum:
                        probability += ps * _binomial_probability(clean, 0.035, background)
            return probability
        survivor_p = 1.0 - env.config.rotate_clear_probability
        for survivors in range(attacks + 1):
            ps = _binomial_probability(attacks, survivor_p, survivors)
            for background in range(env.config.max_attack + 1):
                pb = _binomial_probability(env.config.max_attack, 0.035, background)
                if min(env.config.max_attack, survivors + background) <= quorum:
                    probability += ps * pb
        return probability
    extra_p = (0.03, 0.16, 0.40)[env.regime]
    for survivors in range(attacks + 1):
        ps = _binomial_probability(attacks, env.config.persistence, survivors)
        available = env.config.max_attack - survivors
        for extra in range(available + 1):
            pe = _binomial_probability(available, extra_p, extra)
            if min(env.config.max_attack, survivors + extra) <= quorum:
                probability += ps * pe
    return probability


def oracle_action(env: DefenseEnv) -> int:
    """Best immediate action under the simulator's expected reward."""
    remaining_steps = max(1, env.config.horizon - env.t)
    reserve = max(0.0, env.budget_remaining) / remaining_steps
    scores = []
    for action in range(env.action_space.n):
        redundancy, recovery = env.decode_action(action)
        quorum = (redundancy - 1) // 2
        initial_success = env.true_attacks <= quorum
        base_cost = env._communication_cost(redundancy)
        cost = base_cost
        if initial_success:
            success_probability = 1.0
        elif recovery == "none":
            success_probability = 0.0
        else:
            success_probability = _retry_success_probability(
                env, quorum, rotate=recovery == "rotate"
            )
            cost += base_cost + (env.reward_config.rotation_cost if recovery == "rotate" else 0.0)
        utility = env.importance * (
            success_probability - env.failure_penalty * (1.0 - success_probability)
        )
        score = utility - env.reward_config.communication_weight * cost
        if env.budget_penalty_mode == "lagrangian":
            target = env.config.budget_per_step * env.budget_scale
            score -= env.constraint_weight * (cost - target)
        else:
            score -= env.constraint_weight * max(0.0, cost - reserve)
        if env.budget_remaining - cost < 0:
            score -= env.reward_config.budget_violation_base + min(
                env.reward_config.budget_debt_cap,
                (cost - env.budget_remaining) * env.reward_config.budget_debt_weight,
            )
        scores.append(score)
    return int(np.argmax(scores))


def generate_ml_dataset(samples: int, seed: int):
    observations = []
    labels = []
    completed = 0
    episode = 0
    generator = np.random.default_rng(seed + 900_000)
    while completed < samples:
        env = DefenseEnv(
            "train_mix", domain_randomization=True,
            randomization_profile="robust", constraint_weight=0.15,
        )
        obs, _ = env.reset(seed=seed + episode)
        while completed < samples:
            observations.append(obs.copy())
            labels.append(oracle_action(env))
            action = int(generator.integers(env.action_space.n))
            obs, _, terminated, truncated, _ = env.step(action)
            completed += 1
            if terminated or truncated:
                break
        episode += 1
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def train_ml_policies(samples: int, seed: int):
    x, y = generate_ml_dataset(samples, seed)
    common = dict(max_depth=12, min_samples_leaf=24, class_weight="balanced", random_state=seed)
    stateless = DecisionTreeClassifier(**common).fit(x[:, STATELESS_FEATURES], y)
    stateful = DecisionTreeClassifier(**common).fit(x, y)
    return TreePolicy(stateless, STATELESS_FEATURES), TreePolicy(stateful)


def load_policies(root: Path, ml_samples: int, seed: int):
    stateless_ml, stateful_ml = train_ml_policies(ml_samples, seed)
    v2 = root / "rl_v2" / "models"
    robust35 = root / "rl_robust_v1" / "models"
    robust15 = root / "rl_robust_w015" / "models"
    policies = OrderedDict([
        ("Fixed k1 + rotate", (FixedPolicy(1, "rotate"), "baseline")),
        ("Fixed k3 + rotate", (FixedPolicy(3, "rotate"), "baseline")),
        ("Risk heuristic", (RiskHeuristicPolicy(), "baseline")),
        ("Conservative rule", (ConservativePolicy(), "baseline")),
        ("Stateless ML", (stateless_ml, "ml")),
        ("Stateful ML", (stateful_ml, "ml")),
        ("Linear Q (initial)", (LegacyLinearQPolicy.load(robust15 / "legacy_q_seed2026.pkl"), "rl")),
        ("A2C", (A2C.load(v2 / "a2c_seed2026.zip", device="cpu"), "rl")),
        ("DQN", (DQN.load(v2 / "dqn_seed2026.zip", device="cpu"), "rl")),
        ("PPO", (PPO.load(v2 / "ppo_seed2026.zip", device="cpu"), "rl")),
        ("QR-DQN", (QRDQN.load(v2 / "qrdqn_seed2026.zip", device="cpu"), "rl")),
        ("RecurrentPPO", (RecurrentPPO.load(v2 / "recurrent_ppo_seed2026.zip", device="cpu"), "rl")),
        ("MaskablePPO robust", (MaskablePPO.load(robust35 / "maskable_ppo_seed2026.zip", device="cpu"), "rl")),
        ("TRPO robust", (TRPO.load(robust35 / "trpo_seed2026.zip", device="cpu"), "rl")),
        ("QR-DQN robust (w=0.35)", (QRDQN.load(robust35 / "qrdqn_seed2026.zip", device="cpu"), "rl")),
        ("QR-DQN robust (w=0.15)", (QRDQN.load(robust15 / "qrdqn_seed2026.zip", device="cpu"), "rl")),
    ])
    return policies


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(summary: list[dict], output: Path) -> None:
    by_policy = {}
    families = {}
    for row in summary:
        by_policy.setdefault(row["policy"], []).append(row)
        families[row["policy"]] = row["family"]

    left_names = (
        "Fixed k1 + rotate", "Fixed k3 + rotate", "Risk heuristic",
        "Conservative rule", "Stateless ML", "Stateful ML",
        "QR-DQN robust (w=0.15)",
    )
    right_names = (
        "Fixed k1 + rotate", "Linear Q (initial)", "A2C", "DQN", "PPO",
        "QR-DQN", "RecurrentPPO", "MaskablePPO robust", "TRPO robust",
        "QR-DQN robust (w=0.35)", "QR-DQN robust (w=0.15)",
    )
    all_names = list(dict.fromkeys((*left_names, *right_names)))
    colors = dict(zip(all_names, plt.get_cmap("tab20")(np.linspace(0, 0.95, len(all_names)))))
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*")
    marker_map = {name: markers[i % len(markers)] for i, name in enumerate(all_names)}

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.2), sharey=True)
    panels = (
        (axes[0], left_names, "(a) Baselines and supervised ML"),
        (axes[1], right_names, "(b) Reinforcement-learning methods"),
    )
    for ax, names, title in panels:
        for name in names:
            rows = sorted(by_policy[name], key=lambda item: item["budget_scale"])
            family = families[name]
            linestyle = "--" if family == "ml" else ":" if family == "baseline" else "-"
            width = 3.2 if name == "QR-DQN robust (w=0.15)" else 1.8
            alpha = 1.0 if name == "QR-DQN robust (w=0.15)" else 0.86
            ax.plot(
                [row["budget_scale"] for row in rows],
                [row["weighted_success_rate"] for row in rows],
                label=name, color=colors[name], linestyle=linestyle,
                marker=marker_map[name], markersize=5.2, linewidth=width, alpha=alpha,
            )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Budget scale (× nominal budget)")
        ax.grid(True, alpha=0.23)
        ax.legend(fontsize=8.2, loc="lower right", framealpha=0.94)
    axes[0].set_ylabel("Importance-weighted defense success rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(0.30, 1.01)
    fig.suptitle("Defense Success vs. Budget Under Complex Attacks", fontsize=15, weight="bold")
    fig.text(
        0.5, 0.015,
        "Mean over persistent, adaptive, multi-compromise, noisy-OOD, and budget-shock scenarios; "
        "three common evaluation seeds. ML curves are dashed; RL curves are solid.",
        ha="center", fontsize=9.3,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("experiment/results"))
    parser.add_argument("--output", type=Path, default=Path(
        "experiment/results/rl_robust_w015/analysis/budget_vs_defense_success_complex.png"
    ))
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.55, 0.70, 0.85, 1.0, 1.15, 1.30, 1.45])
    parser.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    parser.add_argument("--ml-samples", type=int, default=40_000)
    args = parser.parse_args()

    policies = load_policies(args.result_root, args.ml_samples, seed=2026)
    raw_rows = []
    for policy_name, (policy, family) in policies.items():
        for budget in args.budgets:
            for scenario in HARD_SCENARIOS:
                for seed in args.seeds:
                    row = run_episode(
                        policy, scenario, budget, seed,
                        recurrent=policy_name == "RecurrentPPO",
                        maskable=policy_name == "MaskablePPO robust",
                    )
                    row.update({"policy": policy_name, "family": family})
                    raw_rows.append(row)
        print(f"evaluated {policy_name}", flush=True)

    summary = []
    for policy_name, (_, family) in policies.items():
        for budget in args.budgets:
            selected = [
                row for row in raw_rows
                if row["policy"] == policy_name and row["budget_scale"] == budget
            ]
            summary.append({
                "policy": policy_name,
                "family": family,
                "budget_scale": budget,
                "weighted_success_rate": float(np.mean([
                    row["weighted_success_rate"] for row in selected
                ])),
                "success_rate": float(np.mean([row["success_rate"] for row in selected])),
                "mean_cost": float(np.mean([row["mean_cost"] for row in selected])),
                "episodes": len(selected),
            })
    write_csv(args.output.with_name("budget_success_complex_raw.csv"), raw_rows)
    write_csv(args.output.with_name("budget_success_complex_summary.csv"), summary)
    plot_curves(summary, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
