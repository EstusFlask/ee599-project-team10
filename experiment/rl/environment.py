"""Gymnasium environment for adaptive redundant-channel defense.

The environment intentionally models a POMDP.  The policy observes a noisy
pre-transmission risk signal and recent outcomes, while the attack regime,
channel persistence, and exact number of compromised replicas remain hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REDUNDANCIES = (1, 3, 5, 7)
RECOVERIES = ("none", "retry", "rotate")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "train_mix"
    horizon: int = 256
    budget_per_step: float = 0.72
    max_attack: int = 5
    detector_false_positive: float = 0.08
    detector_false_negative: float = 0.18
    persistence: float = 0.72
    rotate_clear_probability: float = 0.72
    adaptive_strength: float = 0.35
    regime_switch_probability: float = 0.0
    length_shift: float = 0.0
    importance_shift: float = 0.0
    # Number of independent attacker processes.  One preserves the original
    # aggregate-count simulator exactly; values >1 maintain an independent
    # hidden regime per attacker and union their compromised-replica masks.
    attacker_count: int = 1


SCENARIOS: dict[str, ScenarioConfig] = {
    "train_mix": ScenarioConfig(),
    "mild": ScenarioConfig(
        name="mild", persistence=0.45, adaptive_strength=0.05,
        detector_false_negative=0.10, budget_per_step=0.55,
    ),
    "bursty": ScenarioConfig(
        name="bursty", persistence=0.60, adaptive_strength=0.15,
        budget_per_step=0.68,
    ),
    "persistent": ScenarioConfig(
        name="persistent", persistence=0.92, adaptive_strength=0.25,
        detector_false_negative=0.25, budget_per_step=0.78,
    ),
    "adaptive": ScenarioConfig(
        name="adaptive", persistence=0.80, adaptive_strength=0.80,
        detector_false_negative=0.25, regime_switch_probability=0.035,
        budget_per_step=0.82,
    ),
    "multi": ScenarioConfig(
        name="multi", persistence=0.78, max_attack=5,
        adaptive_strength=0.48, budget_per_step=0.92,
        importance_shift=0.08,
    ),
    "noisy_ood": ScenarioConfig(
        name="noisy_ood", persistence=0.82,
        detector_false_positive=0.28, detector_false_negative=0.42,
        adaptive_strength=0.55, regime_switch_probability=0.06,
        length_shift=0.20, budget_per_step=0.78,
    ),
    "budget_shock": ScenarioConfig(
        name="budget_shock", persistence=0.78, adaptive_strength=0.45,
        budget_per_step=0.50, regime_switch_probability=0.025,
    ),
    "multi_attacker_2": ScenarioConfig(
        name="multi_attacker_2", attacker_count=2, persistence=0.78,
        adaptive_strength=0.48, detector_false_negative=0.25,
        regime_switch_probability=0.025, budget_per_step=0.82,
    ),
    "multi_attacker_3": ScenarioConfig(
        name="multi_attacker_3", attacker_count=3, persistence=0.82,
        adaptive_strength=0.55, detector_false_negative=0.30,
        regime_switch_probability=0.035, budget_per_step=0.82,
    ),
    "multi_attacker_5": ScenarioConfig(
        name="multi_attacker_5", attacker_count=5, persistence=0.86,
        adaptive_strength=0.65, detector_false_negative=0.35,
        regime_switch_probability=0.045, budget_per_step=0.82,
    ),
}


class DefenseEnv(gym.Env):
    """Adaptive defense with redundancy and conditional recovery actions.

    Action = Cartesian product of {1,3,5,7} replicas and
    {none, same-channel retry, channel rotation + retry}.  Recovery is only
    charged and executed after an initial quorum failure.
    """

    metadata = {"render_modes": []}
    observation_size = 25

    def __init__(
        self,
        scenario: str | ScenarioConfig = "train_mix",
        horizon: int | None = None,
        budget_scale: float = 1.0,
        domain_randomization: bool = False,
        randomization_profile: str = "standard",
        constraint_weight: float = 0.0,
        attacker_count_probabilities: tuple[float, ...] | None = None,
        failure_penalty: float = 1.65,
    ) -> None:
        super().__init__()
        if isinstance(scenario, str):
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown scenario {scenario!r}")
            scenario = SCENARIOS[scenario]
        if horizon is not None:
            scenario = replace(scenario, horizon=int(horizon))
        if budget_scale <= 0:
            raise ValueError("budget_scale must be positive")
        self.base_config = scenario
        self.budget_scale = float(budget_scale)
        self.domain_randomization = bool(domain_randomization)
        if randomization_profile not in {"standard", "robust", "multi_attacker"}:
            raise ValueError(
                "randomization_profile must be 'standard', 'robust', or "
                "'multi_attacker'"
            )
        if constraint_weight < 0:
            raise ValueError("constraint_weight must be non-negative")
        if failure_penalty <= 0:
            raise ValueError("failure_penalty must be positive")
        self.randomization_profile = randomization_profile
        self.constraint_weight = float(constraint_weight)
        self.failure_penalty = float(failure_penalty)
        if attacker_count_probabilities is not None:
            probabilities = np.asarray(attacker_count_probabilities, dtype=np.float64)
            if probabilities.shape != (5,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
                raise ValueError(
                    "attacker_count_probabilities must contain five non-negative "
                    "weights with a positive sum"
                )
            probabilities = probabilities / probabilities.sum()
            self.attacker_count_probabilities = tuple(float(value) for value in probabilities)
        else:
            self.attacker_count_probabilities = None
        self.action_space = spaces.Discrete(len(REDUNDANCIES) * len(RECOVERIES))
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_size, dtype=np.float32),
            high=np.ones(self.observation_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.config = scenario
        self._episode_metrics: dict[str, float] = {}

    @staticmethod
    def decode_action(action: int) -> tuple[int, str]:
        if not 0 <= int(action) < 12:
            raise ValueError(f"action must be in [0, 11], got {action}")
        return REDUNDANCIES[int(action) // 3], RECOVERIES[int(action) % 3]

    @staticmethod
    def encode_action(redundancy: int, recovery: str) -> int:
        return REDUNDANCIES.index(int(redundancy)) * 3 + RECOVERIES.index(recovery)

    def _randomized_config(self) -> ScenarioConfig:
        cfg = self.base_config
        if not self.domain_randomization:
            return cfg
        if self.randomization_profile in {"robust", "multi_attacker"}:
            return replace(
                cfg,
                budget_per_step=float(self.np_random.uniform(0.42, 1.00)),
                detector_false_positive=float(self.np_random.uniform(0.01, 0.35)),
                detector_false_negative=float(self.np_random.uniform(0.05, 0.55)),
                persistence=float(self.np_random.uniform(0.30, 0.97)),
                rotate_clear_probability=float(self.np_random.uniform(0.40, 0.94)),
                adaptive_strength=float(self.np_random.uniform(0.0, 1.0)),
                regime_switch_probability=float(self.np_random.uniform(0.0, 0.08)),
                length_shift=float(self.np_random.uniform(-0.15, 0.30)),
                importance_shift=float(self.np_random.uniform(-0.10, 0.12)),
                attacker_count=(
                    int(self.np_random.choice(
                        np.arange(1, 6),
                        p=self.attacker_count_probabilities,
                    ))
                    if (
                        self.randomization_profile == "multi_attacker"
                        and self.attacker_count_probabilities is not None
                    )
                    else int(self.np_random.integers(1, 6))
                    if self.randomization_profile == "multi_attacker"
                    else cfg.attacker_count
                ),
            )
        # Wide training distribution; held-out OOD scenarios exceed parts of it.
        return replace(
            cfg,
            budget_per_step=float(self.np_random.uniform(0.50, 0.95)),
            detector_false_positive=float(self.np_random.uniform(0.02, 0.22)),
            detector_false_negative=float(self.np_random.uniform(0.08, 0.35)),
            persistence=float(self.np_random.uniform(0.38, 0.90)),
            rotate_clear_probability=float(self.np_random.uniform(0.55, 0.92)),
            adaptive_strength=float(self.np_random.uniform(0.0, 0.70)),
            regime_switch_probability=float(self.np_random.uniform(0.0, 0.045)),
            length_shift=float(self.np_random.uniform(-0.10, 0.12)),
            importance_shift=float(self.np_random.uniform(-0.08, 0.08)),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.config = self._randomized_config()
        if self.config.attacker_count < 1:
            raise ValueError("attacker_count must be at least one")
        if options and "budget_scale" in options:
            self.budget_scale = float(options["budget_scale"])
        self.t = 0
        self.budget_total = (
            self.config.horizon * self.config.budget_per_step * self.budget_scale
        )
        self.budget_remaining = self.budget_total
        self.regime = int(self.np_random.choice(3, p=(0.70, 0.25, 0.05)))
        self.attacker_regimes = np.asarray(
            [self.regime] if self.config.attacker_count == 1 else self.np_random.choice(
                3, size=self.config.attacker_count, p=(0.70, 0.25, 0.05)
            ),
            dtype=np.int8,
        )
        self.compromised = 0
        self.compromised_mask = np.zeros(self.config.max_attack, dtype=bool)
        self.length_state = int(self.np_random.choice(3, p=(0.30, 0.50, 0.20)))
        self.importance_state = int(self.np_random.choice(3, p=(0.30, 0.50, 0.20)))
        self.recent_high_protection = 0.0
        self.history = np.zeros((8, 4), dtype=np.float32)
        self.prev_action = 0
        self.prev_success = 1.0
        self.prev_alert = 0.0
        self.prev_cost = 0.0
        self._episode_metrics = {
            "successes": 0.0,
            "importance_total": 0.0,
            "importance_success": 0.0,
            "cost": 0.0,
            "recovery_uses": 0.0,
            "rotations": 0.0,
            "budget_violations": 0.0,
        }
        self._sample_context()
        return self._observation(), {"scenario": self.config.name}

    def _transition_regime(self) -> None:
        transitions = np.asarray(
            ((0.91, 0.08, 0.01), (0.14, 0.73, 0.13), (0.05, 0.18, 0.77)),
            dtype=np.float64,
        )
        next_regimes = []
        for attacker_regime in self.attacker_regimes:
            row = transitions[int(attacker_regime)].copy()
            if (
                self.config.regime_switch_probability > 0
                and self.np_random.random() < self.config.regime_switch_probability
            ):
                row = np.full(3, 1 / 3)
            pressure = self.config.adaptive_strength * self.recent_high_protection
            moved = min(row[0] * 0.75, pressure * 0.20)
            row[0] -= moved
            row[1] += moved * 0.35
            row[2] += moved * 0.65
            next_regimes.append(int(self.np_random.choice(3, p=row / row.sum())))
        self.attacker_regimes = np.asarray(next_regimes, dtype=np.int8)
        # Existing retry logic consumes one aggregate severity indicator.
        self.regime = int(self.attacker_regimes.max())

    def _sample_attack_count(self) -> int:
        if self.config.attacker_count > 1:
            fresh_mask = np.zeros(self.config.max_attack, dtype=bool)
            for attacker_regime in self.attacker_regimes:
                base_p = (0.025, 0.22, 0.62)[int(attacker_regime)]
                adaptive = (
                    self.config.adaptive_strength
                    * self.recent_high_protection
                    * 0.22
                )
                probability = float(np.clip(base_p + adaptive, 0.0, 0.94))
                fresh_mask |= self.np_random.random(self.config.max_attack) < probability
            if self.np_random.random() < self.config.persistence:
                kept_mask = self.compromised_mask & (
                    self.np_random.random(self.config.max_attack) < self.config.persistence
                )
                fresh_mask |= kept_mask
            self.compromised_mask = fresh_mask
            self.compromised = int(fresh_mask.sum())
            return self.compromised

        # Preserve the original single-attacker random-number path and behavior.
        base_p = (0.025, 0.22, 0.62)[self.regime]
        adaptive = self.config.adaptive_strength * self.recent_high_protection * 0.22
        p = float(np.clip(base_p + adaptive, 0.0, 0.94))
        fresh = int(self.np_random.binomial(self.config.max_attack, p))
        if self.np_random.random() < self.config.persistence:
            kept = int(self.np_random.binomial(self.compromised, self.config.persistence))
            fresh = max(fresh, kept)
        self.compromised = min(self.config.max_attack, fresh)
        return self.compromised

    def _sample_context(self) -> None:
        self._transition_regime()
        length_transition = np.asarray(
            ((0.76, 0.21, 0.03), (0.16, 0.69, 0.15), (0.04, 0.21, 0.75))
        )
        self.length_state = int(self.np_random.choice(3, p=length_transition[self.length_state]))
        lengths = (64.0, 192.0, 640.0)
        jitter = float(self.np_random.lognormal(0.0, 0.22))
        self.message_length = lengths[self.length_state] * jitter * (1.0 + self.config.length_shift)
        imp_transition = np.asarray(
            ((0.74, 0.23, 0.03), (0.16, 0.70, 0.14), (0.04, 0.24, 0.72))
        )
        shifted = imp_transition[self.importance_state].copy()
        shifted[2] = np.clip(shifted[2] + self.config.importance_shift, 0.01, 0.90)
        shifted[0] = max(0.01, shifted[0] - self.config.importance_shift)
        shifted /= shifted.sum()
        self.importance_state = int(self.np_random.choice(3, p=shifted))
        self.importance = (0.55, 1.0, 1.75)[self.importance_state]
        self.true_attacks = self._sample_attack_count()
        attacked_signal = self.np_random.binomial(
            self.true_attacks, 1.0 - self.config.detector_false_negative
        )
        clean = max(0, self.config.max_attack - self.true_attacks)
        false_signal = self.np_random.binomial(clean, self.config.detector_false_positive)
        self.alert_count = int(np.clip(attacked_signal + false_signal, 0, self.config.max_attack))

    def _communication_cost(self, redundancy: int) -> float:
        # Mirrors the real implementation: two full messages, then short hashes.
        full = min(redundancy, 2) * self.message_length
        hashes = max(0, redundancy - 2) * 64.0
        baseline = self.message_length
        return float((full + hashes) / max(128.0, baseline * 2.0))

    def action_masks(self) -> np.ndarray:
        """Observable budget safety mask used by MaskablePPO.

        Expensive actions remain available when risk/importance warrants them,
        but an episode cannot spend its future reserve indiscriminately.
        """
        remaining_steps = max(1, self.config.horizon - self.t)
        allowance = max(0.0, self.budget_remaining) / remaining_steps
        importance_factor = 0.85 + 0.75 * (self.importance / 1.75)
        cap = max(self._communication_cost(1), allowance * importance_factor * 1.55)
        masks = np.zeros(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            redundancy, recovery = self.decode_action(action)
            base = self._communication_cost(redundancy)
            quorum = (redundancy - 1) // 2
            failure_proxy = float(np.clip((self.alert_count - quorum + 1) / 3.0, 0, 1))
            expected = base
            if recovery != "none":
                expected += failure_proxy * base
            if recovery == "rotate":
                expected += failure_proxy * 0.22
            masks[action] = expected <= cap
        masks[self.encode_action(1, "none")] = True
        return masks

    @staticmethod
    def _quorum_success(redundancy: int, attacks: int) -> bool:
        return attacks <= (redundancy - 1) // 2

    def _retry_attacks(self, rotate: bool) -> int:
        if rotate:
            survivors = self.np_random.binomial(
                self.true_attacks, 1.0 - self.config.rotate_clear_probability
            )
            background = self.np_random.binomial(self.config.max_attack, 0.035)
            return int(min(self.config.max_attack, survivors + background))
        survivors = self.np_random.binomial(self.true_attacks, self.config.persistence)
        extra = self.np_random.binomial(
            max(0, self.config.max_attack - survivors),
            (0.03, 0.16, 0.40)[self.regime],
        )
        return int(min(self.config.max_attack, survivors + extra))

    def _observation(self) -> np.ndarray:
        k, recovery = self.decode_action(self.prev_action)
        if self.t == 0:
            hist_mean = np.zeros(4, dtype=np.float32)
            hist_recent = np.zeros(4, dtype=np.float32)
        else:
            width = min(self.t, len(self.history))
            hist_mean = self.history[-width:].mean(axis=0)
            hist_recent = self.history[-1]
        recovery_one_hot = [float(recovery == name) for name in RECOVERIES]
        obs = np.asarray(
            [
                np.clip(np.log1p(self.message_length) / np.log(1000.0), 0, 1),
                self.importance / 1.75,
                self.alert_count / max(1, self.config.max_attack),
                float(self.alert_count > 0),
                np.clip(self.budget_remaining / max(self.budget_total, 1e-6), 0, 1),
                self.t / max(1, self.config.horizon),
                self.prev_success,
                self.prev_alert,
                np.clip(self.prev_cost / 4.0, 0, 1),
                k / 7.0,
                *recovery_one_hot,
                *hist_mean.tolist(),
                *hist_recent.tolist(),
                self.recent_high_protection,
                self.config.detector_false_positive,
                self.config.detector_false_negative,
                np.clip(
                    self.config.budget_per_step * self.budget_scale / 1.30,
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )
        if obs.shape != (self.observation_size,):
            raise RuntimeError(f"observation shape mismatch: {obs.shape}")
        return np.clip(obs, 0.0, 1.0)

    def step(self, action: int):
        redundancy, recovery = self.decode_action(action)
        initial_success = self._quorum_success(redundancy, self.true_attacks)
        final_success = initial_success
        recovery_used = False
        attacks_after_recovery = self.true_attacks
        tx_cost = self._communication_cost(redundancy)
        rotation_cost = 0.0
        if not initial_success and recovery != "none":
            recovery_used = True
            attacks_after_recovery = self._retry_attacks(recovery == "rotate")
            final_success = self._quorum_success(redundancy, attacks_after_recovery)
            tx_cost += self._communication_cost(redundancy)
            if recovery == "rotate":
                rotation_cost = 0.22
                tx_cost += rotation_cost

        remaining_steps = max(1, self.config.horizon - self.t)
        reserve_allowance = max(0.0, self.budget_remaining) / remaining_steps
        self.budget_remaining -= tx_cost
        budget_violation = self.budget_remaining < 0
        utility = self.importance * (
            1.0 if final_success else -self.failure_penalty
        )
        reward = utility - 0.20 * tx_cost
        reward -= self.constraint_weight * max(0.0, tx_cost - reserve_allowance)
        if budget_violation:
            reward -= 0.35 + min(1.5, -self.budget_remaining * 0.05)

        self._episode_metrics["successes"] += float(final_success)
        self._episode_metrics["importance_total"] += self.importance
        self._episode_metrics["importance_success"] += self.importance * float(final_success)
        self._episode_metrics["cost"] += tx_cost
        self._episode_metrics["recovery_uses"] += float(recovery_used)
        self._episode_metrics["rotations"] += float(recovery_used and recovery == "rotate")
        self._episode_metrics["budget_violations"] += float(budget_violation)

        hist_row = np.asarray(
            [float(final_success), self.alert_count / max(1, self.config.max_attack),
             min(tx_cost / 4.0, 1.0), redundancy / 7.0], dtype=np.float32
        )
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = hist_row
        self.prev_action = int(action)
        self.prev_success = float(final_success)
        self.prev_alert = self.alert_count / max(1, self.config.max_attack)
        self.prev_cost = tx_cost
        self.recent_high_protection = 0.88 * self.recent_high_protection + 0.12 * float(redundancy >= 5)
        self.t += 1
        terminated = self.t >= self.config.horizon
        truncated = False
        info: dict[str, Any] = {
            "success": bool(final_success),
            "initial_success": bool(initial_success),
            "importance": self.importance,
            "cost": tx_cost,
            "redundancy": redundancy,
            "recovery": recovery,
            "recovery_used": recovery_used,
            "true_attacks": self.true_attacks,
            "attacks_after_recovery": attacks_after_recovery,
            "alert_count": self.alert_count,
            "budget_remaining": self.budget_remaining,
            "budget_violation": budget_violation,
            "regime": self.regime,
            "attacker_count": self.config.attacker_count,
            "attacker_regimes": tuple(int(value) for value in self.attacker_regimes),
        }
        if terminated:
            total = max(1.0, float(self.config.horizon))
            imp_total = max(1e-8, self._episode_metrics["importance_total"])
            info["episode_metrics"] = {
                **self._episode_metrics,
                "success_rate": self._episode_metrics["successes"] / total,
                "weighted_success_rate": self._episode_metrics["importance_success"] / imp_total,
                "mean_cost": self._episode_metrics["cost"] / total,
            }
            obs = np.zeros(self.observation_size, dtype=np.float32)
        else:
            self._sample_context()
            obs = self._observation()
        return obs, float(reward), terminated, truncated, info
