"""Visualize the attacker process used by the unified-cost benchmark.

The simulator contains one aggregate hidden Markov attacker process.  At each
round it samples how many of the five vulnerable replicas are compromised.
Because attack pressure adapts to recent high-redundancy defense, trajectories
are policy-dependent rather than a fixed replay trace.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

# The Windows Conda stack links Intel OpenMP through both PyTorch and NumPy.
# Single-threaded inference avoids oversubscription; the compatibility flag is
# needed before either runtime is imported on this environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sb3_contrib import QRDQN

from .baselines import FixedPolicy
from .environment import DefenseEnv, SCENARIOS


SCENARIOS_TO_PLOT = (
    "persistent",
    "adaptive",
    "multi",
    "noisy_ood",
    "budget_shock",
)
SCENARIO_LABELS = {
    "persistent": "Persistent",
    "adaptive": "Adaptive",
    "multi": "Multi-compromise",
    "noisy_ood": "Noisy OOD",
    "budget_shock": "Budget shock",
}
POLICY_ORDER = ("Low defense", "Robust QR-DQN", "High defense")
POLICY_COLORS = {
    "Low defense": "#4c78a8",
    "Robust QR-DQN": "#e45756",
    "High defense": "#54a24b",
}
REGIME_LABELS = {0: "Quiet", 1: "Probing", 2: "Burst"}


def run_episode(policy, policy_name: str, scenario: str, seed: int):
    env = DefenseEnv(scenario, budget_scale=1.0)
    observation, _ = env.reset(seed=seed)
    rows = []
    while True:
        rows.append({
            "policy": policy_name,
            "scenario": scenario,
            "seed": seed,
            "round": env.t,
            "regime": env.regime,
            "true_attacks": env.true_attacks,
            "alert_count": env.alert_count,
            "recent_high_protection": env.recent_high_protection,
        })
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(
            int(np.asarray(action).item())
        )
        if terminated or truncated:
            return rows


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw):
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["policy"], row["scenario"], row["seed"])].append(row)

    episode_rows = []
    for (policy, scenario, seed), rows in grouped.items():
        attacks = np.asarray([row["true_attacks"] for row in rows])
        alerts = np.asarray([row["alert_count"] for row in rows])
        regimes = np.asarray([row["regime"] for row in rows])
        episode_rows.append({
            "policy": policy,
            "scenario": scenario,
            "seed": seed,
            "rounds": len(rows),
            "attacked_rounds": int(np.sum(attacks > 0)),
            "multi_compromise_rounds": int(np.sum(attacks >= 2)),
            "total_compromised_replica_draws": int(np.sum(attacks)),
            "mean_compromised_replicas": float(np.mean(attacks)),
            "attack_event_rate": float(np.mean(attacks > 0)),
            "alert_event_rate": float(np.mean(alerts > 0)),
            "missed_attack_rate": float(np.mean((attacks > 0) & (alerts == 0))),
            "false_alert_rate": float(np.mean((attacks == 0) & (alerts > 0))),
            "quiet_share": float(np.mean(regimes == 0)),
            "probing_share": float(np.mean(regimes == 1)),
            "burst_share": float(np.mean(regimes == 2)),
        })

    summary = []
    for policy in POLICY_ORDER:
        for scenario in SCENARIOS_TO_PLOT:
            selected = [
                row for row in episode_rows
                if row["policy"] == policy and row["scenario"] == scenario
            ]
            summary.append({
                "policy": policy,
                "scenario": scenario,
                "episodes": len(selected),
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in (
                        "attacked_rounds",
                        "multi_compromise_rounds",
                        "total_compromised_replica_draws",
                        "mean_compromised_replicas",
                        "attack_event_rate",
                        "alert_event_rate",
                        "missed_attack_rate",
                        "false_alert_rate",
                        "quiet_share",
                        "probing_share",
                        "burst_share",
                    )
                },
            })
    return episode_rows, summary


def rolling_mean(values, window: int = 16):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def make_figure(raw, summary, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    ax_time, ax_events, ax_regimes, ax_sensor = axes.flat
    scenario_colors = dict(zip(
        SCENARIOS_TO_PLOT,
        plt.get_cmap("tab10")(np.linspace(0, 0.8, len(SCENARIOS_TO_PLOT))),
    ))

    # A: temporal attack intensity for the representative learned policy.
    for scenario in SCENARIOS_TO_PLOT:
        by_round = []
        for round_index in range(SCENARIOS[scenario].horizon):
            values = [
                row["true_attacks"] for row in raw
                if row["policy"] == "Robust QR-DQN"
                and row["scenario"] == scenario
                and row["round"] == round_index
            ]
            by_round.append(float(np.mean(values)))
        ax_time.plot(
            range(len(by_round)), rolling_mean(by_round),
            label=SCENARIO_LABELS[scenario], color=scenario_colors[scenario],
            linewidth=2.0,
        )
    ax_time.set_title("A. Attack intensity over an episode (16-round mean)", weight="bold")
    ax_time.set_xlabel("Round")
    ax_time.set_ylabel("Compromised replicas per round")
    ax_time.set_xlim(0, 255)
    ax_time.set_ylim(0, 5)
    ax_time.grid(True, alpha=0.22)
    ax_time.legend(fontsize=8.5, ncol=2)

    # B: policy-dependent number of rounds containing an attack event.
    x = np.arange(len(SCENARIOS_TO_PLOT))
    width = 0.24
    for policy_index, policy in enumerate(POLICY_ORDER):
        values = [
            next(row["attacked_rounds"] for row in summary
                 if row["policy"] == policy and row["scenario"] == scenario)
            for scenario in SCENARIOS_TO_PLOT
        ]
        ax_events.bar(
            x + (policy_index - 1) * width, values, width,
            label=policy, color=POLICY_COLORS[policy], alpha=0.90,
        )
    ax_events.set_title("B. Rounds with >=1 compromised replica", weight="bold")
    ax_events.set_ylabel("Attacked rounds per 256-round episode")
    ax_events.set_xticks(x, [SCENARIO_LABELS[name] for name in SCENARIOS_TO_PLOT], rotation=18)
    ax_events.set_ylim(0, 256)
    ax_events.grid(True, axis="y", alpha=0.22)
    ax_events.legend(fontsize=8.5)

    # C: occupancy of the three hidden Markov regimes.
    bottom = np.zeros(len(SCENARIOS_TO_PLOT))
    regime_colors = ("#72b7b2", "#f2cf5b", "#e45756")
    for regime_key, color in zip(("quiet_share", "probing_share", "burst_share"), regime_colors):
        values = np.asarray([
            next(row[regime_key] for row in summary
                 if row["policy"] == "Robust QR-DQN" and row["scenario"] == scenario)
            for scenario in SCENARIOS_TO_PLOT
        ])
        label = regime_key.replace("_share", "").title()
        ax_regimes.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax_regimes.set_title("C. Hidden attacker-regime occupancy", weight="bold")
    ax_regimes.set_ylabel("Share of rounds")
    ax_regimes.set_xticks(x, [SCENARIO_LABELS[name] for name in SCENARIOS_TO_PLOT], rotation=18)
    ax_regimes.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_regimes.set_ylim(0, 1)
    ax_regimes.legend(fontsize=8.5, ncol=3, loc="upper center")

    # D: true events versus the noisy observation available to every policy.
    sensor_metrics = (
        ("attack_event_rate", "True attack", "#4c78a8"),
        ("alert_event_rate", "Observed alert", "#f58518"),
        ("missed_attack_rate", "Attack but no alert", "#e45756"),
    )
    sensor_width = 0.24
    for metric_index, (metric, label, color) in enumerate(sensor_metrics):
        values = [
            next(row[metric] for row in summary
                 if row["policy"] == "Robust QR-DQN" and row["scenario"] == scenario)
            for scenario in SCENARIOS_TO_PLOT
        ]
        ax_sensor.bar(x + (metric_index - 1) * sensor_width, values, sensor_width,
                      label=label, color=color, alpha=0.90)
    ax_sensor.set_title("D. True attack versus noisy detector signal", weight="bold")
    ax_sensor.set_ylabel("Share of rounds")
    ax_sensor.set_xticks(x, [SCENARIO_LABELS[name] for name in SCENARIOS_TO_PLOT], rotation=18)
    ax_sensor.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_sensor.set_ylim(0, 1)
    ax_sensor.grid(True, axis="y", alpha=0.22)
    ax_sensor.legend(fontsize=8.5)

    fig.suptitle("Attacker Dynamics in the Unified-Cost Benchmark", fontsize=16, weight="bold")
    fig.text(
        0.5, 0.015,
        "One aggregate hidden-Markov attacker; each round samples 0-5 compromised replicas. "
        "256 rounds/episode; scenarios and seeds (111, 222, 333) match the comparison figure.",
        ha="center", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scenario_rows():
    rows = []
    for name in SCENARIOS_TO_PLOT:
        config = SCENARIOS[name]
        rows.append({
            "scenario": name,
            "horizon": config.horizon,
            "max_compromised_replicas": config.max_attack,
            "persistence": config.persistence,
            "adaptive_strength": config.adaptive_strength,
            "detector_false_positive": config.detector_false_positive,
            "detector_false_negative": config.detector_false_negative,
            "regime_switch_probability": config.regime_switch_probability,
            "budget_per_step": config.budget_per_step,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("experiment/results/rl_robust_w015/models/qrdqn_seed2026.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment/results/rl_robust_w015/analysis/attacker_dynamics.png"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    args = parser.parse_args()

    policies = {
        "Low defense": FixedPolicy(1, "none"),
        "Robust QR-DQN": QRDQN.load(args.model, device="cpu"),
        "High defense": FixedPolicy(7, "rotate"),
    }
    raw = []
    for policy_name, policy in policies.items():
        for scenario in SCENARIOS_TO_PLOT:
            for seed in args.seeds:
                raw.extend(run_episode(policy, policy_name, scenario, seed))

    episode_rows, summary = summarize(raw)
    make_figure(raw, summary, args.output)
    write_csv(args.output.with_name("attacker_dynamics_raw.csv"), raw)
    write_csv(args.output.with_name("attacker_dynamics_episodes.csv"), episode_rows)
    write_csv(args.output.with_name("attacker_dynamics_summary.csv"), summary)
    write_csv(args.output.with_name("attacker_scenario_parameters.csv"), scenario_rows())
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
