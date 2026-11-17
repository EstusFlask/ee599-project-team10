"""Run policies through the real DefenseEngine with a mock (LLM-free) adversary.

The attack process (how many channels are compromised each round, message length and
importance) comes from ``ScaleDefenseEnv`` used purely as an attacker simulator.  Policies
never see it: they only get the vote result, the delayed receipt, length and importance.
A round counts as a success when the winning hash is the hash of the sent message.

  python -m integration.run_demo --rule-params <chosen.json> --model <qrdqn.zip> --prices 0.05 0.2 0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment.rl.large_scale import (
    SCALE_SCENARIOS, ExactEstimatorPolicy, ScaleDefenseEnv,
)
from integration.adapter import AdaptiveAdapter
from integration.adaptive_engine import AdaptiveDefenseEngine
from integration.controller import FixedController, QRDQNController, RuleController
from integration.mock_adversary import MockAdversary


def make_message(length: int, index: int) -> str:
    head = f"M{index}|"
    return (head + "x" * length)[:max(1, length)]


def run_episode(make_controller, scenario: str, seed: int, horizon: int = 256):
    config = replace(SCALE_SCENARIOS[scenario], feedback_delay=1, horizon=horizon)
    simulator = ScaleDefenseEnv(config)  # attacker / workload process only
    simulator.reset(seed=seed)
    chooser = random.Random(seed)
    engine = AdaptiveDefenseEngine()
    adapter = AdaptiveAdapter(
        engine, MockAdversary(), [], "Victim", make_controller(),
        attack_model=lambda k: chooser.sample(range(k), min(simulator.true_attacks, k)),
        importance=lambda: simulator.importance,
    )
    agree = 0
    for t in range(horizon):
        length = max(1, int(round(simulator.message_length)))
        adapter.secure_transmit(make_message(length, t), "Agent_A", "Victim")
        last = adapter.history[-1]
        agree += last["verified"] == (last["attacked"] <= (last["k"] - 1) // 2)
        simulator.step(0)  # advance the attack dynamics only
    rows = adapter.history
    weight = sum(r["importance"] for r in rows)
    return {
        "success": sum(r["importance"] * r["verified"] for r in rows) / weight,
        "cost": float(np.mean([r["bytes"] / max(128.0, 2.0 * r["length"]) for r in rows])),
        "mean_k": float(np.mean([r["k"] for r in rows])),
        "agreement": agree / horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCALE_SCENARIOS,
                        default=["independent", "campaign_waves"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1006)))
    parser.add_argument("--fixed", nargs="*", type=int, default=[17, 33, 51])
    parser.add_argument("--rule-params", type=Path, help="chosen_*.json from tune_rule.py")
    parser.add_argument("--rule-indices", nargs="*", type=int, default=[],
                        help="which tuned parameter sets to run (default: three spread over the list)")
    parser.add_argument("--model", type=Path, help="price-conditioned QR-DQN .zip")
    parser.add_argument("--prices", nargs="*", type=float, default=[0.05, 0.2, 0.8])
    parser.add_argument("--output", type=Path,
                        default=Path("experiment/results/integration_demo"))
    args = parser.parse_args()
    channels = SCALE_SCENARIOS["campaign"].channels

    factories = {f"fixed k={k}": (lambda k=k: FixedController(k)) for k in args.fixed}
    if args.rule_params:
        chosen = json.loads(args.rule_params.read_text())
        picks = args.rule_indices or sorted({0, len(chosen) // 2, len(chosen) - 1})
        for i in picks:
            params = chosen[i]["params"]
            factories[f"tuned rule #{i}"] = (
                lambda p=params: RuleController(
                    ExactEstimatorPolicy(channels, delay=1, **p), channels=channels)
            )
    if args.model:
        from sb3_contrib import QRDQN
        model = QRDQN.load(args.model, device="cpu")
        for price in args.prices:
            factories[f"QR-DQN price={price:g}"] = (
                lambda price=price: QRDQNController(model, price=price, channels=channels)
            )

    rows = []
    for scenario in args.scenarios:
        for name, factory in factories.items():
            runs = [run_episode(factory, scenario, s) for s in args.seeds]
            rows.append({
                "scenario": scenario, "policy": name,
                "mean_k": np.mean([r["mean_k"] for r in runs]),
                "cost": np.mean([r["cost"] for r in runs]),
                "success": np.mean([r["success"] for r in runs]),
                "agreement_with_majority_rule": np.mean([r["agreement"] for r in runs]),
            })
            print(f"{scenario:15s} {name:22s} k={rows[-1]['mean_k']:5.1f} "
                  f"cost={rows[-1]['cost']:6.2f} success={rows[-1]['success']:.3f} "
                  f"agree={rows[-1]['agreement_with_majority_rule']:.3f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "demo.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
