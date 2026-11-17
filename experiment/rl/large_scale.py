"""Large-scale redundancy-budget environment (tens of channels and attackers).

The task is reduced to a number comparison: a round succeeds when the number of
compromised channels ``n`` is at most ``(k - 1) // 2``.  The policy only picks
the redundancy ``k`` (odd, 1..2N+1).  Costs, reward shape, importance and
workload follow ``environment.py``; recovery actions are dropped.

What the policy can see matches the real engine: after a round with ``k >= 3``
the vote reveals the size of the minority hash group, ``min(n_k, k - n_k)``,
where ``n_k = min(n, k)`` (all-poisoned rounds look quiet).  It never sees the
attacker count, the attack regime, or ``n`` itself.  An optional noisy
pre-transmission alert (``use_alert``) is off by default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT_REWARD, RewardConfig


# Stage model aligned with the milestone report (Liao et al. 2020 style three-stage HMM;
# the numbers are experimental settings).  A stage-z attacker acts with probability
# ATTACK_PROBABILITY[z] and then compromises min(3, 1 + Poisson(POISSON_RATE[z])) channels.
TRANSITIONS = np.asarray(
    ((0.90, 0.09, 0.01), (0.10, 0.75, 0.15), (0.05, 0.10, 0.85)), dtype=np.float64
)
ATTACK_PROBABILITY = np.asarray((0.02, 0.20, 0.75))
POISSON_RATE = np.asarray((0.0, 0.5, 2.0))
MAX_PER_ATTACKER = 3
MESSAGE_LENGTHS = (50.0, 100.0, 200.0)
IMPORTANCE_LEVELS = (0.5, 1.0, 1.5)
REGIME_PRIOR = (0.425, 0.274, 0.301)  # stationary distribution of TRANSITIONS
HISTORY = 8
# Multi-scale memory beyond the raw window: EMA rates (about 2, 5, 20 rounds).
DISSENT_RATES = (0.5, 0.2, 0.05)
FAIL_RATES = (0.2, 0.03)
K_RATES = (0.2, 0.03)
PEAK_WINDOW = 32
LONG_FEATURES = len(DISSENT_RATES) + len(FAIL_RATES) + len(K_RATES) + 3
SCALE_COMMUNICATION_WEIGHT = 0.05
BUDGET_NORMALIZER = 10.0
# 'margin' actions: k = 2*ceil(n_fast + m*sqrt(n_fast+1)) + 1, plus absolute k.
MARGIN_ACTIONS = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
ABSOLUTE_ACTIONS = (1, 3, 5, 7, 9, 13, 17, 21, 27, 35, 45, 61)
ESTIMATOR_RATES = (0.5, 0.2, 0.05)
ESTIMATOR_FEATURES = 22
# Price-conditioned training: each episode charges ``price`` per unit cost and the
# policy sees log(price); sweeping the price at test time traces the cost frontier.
PRICE_RANGE = (0.01, 5.0)


@dataclass(frozen=True)
class ScaleScenario:
    name: str = "scale_train"
    horizon: int = 256
    channels: int = 50
    min_attackers: int = 0
    max_attackers: int = 40
    initial_attackers: int = 20
    # Chance per round that the attacker population jumps (join/leave wave).
    wave_probability: float = 0.0
    # Chance that an attacker copies the shared campaign regime each round.
    campaign_coupling: float = 0.0
    persistence: float = 0.72
    budget_per_step: float = 4.0
    detector_false_positive: float = 0.08
    detector_false_negative: float = 0.30
    use_alert: bool = False
    # Rounds before a signed receipt reveals whether a round succeeded (0 = at once).
    feedback_delay: int = 0
    importance_shift: float = 0.0
    length_shift: float = 0.0


SCALE_SCENARIOS: dict[str, ScaleScenario] = {
    "independent": ScaleScenario(name="independent"),
    "waves": ScaleScenario(name="waves", wave_probability=0.03),
    "campaign": ScaleScenario(name="campaign", campaign_coupling=0.7),
    "campaign_waves": ScaleScenario(
        name="campaign_waves", campaign_coupling=0.7, wave_probability=0.03,
    ),
}


def _step_regimes(rng: np.random.Generator, regimes: np.ndarray) -> np.ndarray:
    cumulative = TRANSITIONS[regimes].cumsum(axis=-1)
    draw = rng.random(regimes.shape)
    return np.minimum((draw[..., None] > cumulative).sum(axis=-1), 2).astype(np.int8)


class CountEstimator:
    """Estimate the compromised count n from votes and delayed signed receipts.

    A round with redundancy k and minority size d reveals n exactly (success,
    k>=3: n = d; failure with d>0: n = k - d; k=1 success: n = 0) or a lower
    bound (failure with d=0: n >= k; k=1 failure: n >= 1).  Success arrives
    ``delay`` rounds late; until then a round is provisionally read as a
    success (n = d) and corrected when its receipt arrives.  With
    ``use_success=False`` receipts never arrive.
    """

    WINDOW = 96

    def __init__(self, rates=(0.5, 0.05), delay: int = 0, use_success: bool = True) -> None:
        self.rates = tuple(rates)
        self.delay = int(delay)
        self.use_success = use_success
        self.reset()

    def reset(self) -> None:
        # One [k, dissent, success-or-None] record per played round.
        self.log: deque[list] = deque(maxlen=self.WINDOW)
        self.ema = [3.0] * len(self.rates)
        self.values: list[float] = []
        self.age = 0
        self.since_vote = 0

    @staticmethod
    def _sample(k: int, dissent: int, success, fast: float):
        """Return (value, informative) for one round given what is known."""
        if success is None:
            return (float(dissent), True) if k >= 3 else (fast, False)
        if success:
            return (float(dissent) if k >= 3 else 0.0), True
        if k <= 1:
            return max(1.0, fast), True
        return (float(k - dissent) if dissent > 0 else max(float(k), fast)), True

    def update(self, k: int, dissent: int, revealed) -> None:
        """Add the round just played; ``revealed`` is the success of the round
        ``delay`` rounds ago (None if no receipt has arrived)."""
        self.log.append([int(k), int(dissent), None])
        if self.use_success and revealed is not None:
            index = len(self.log) - 1 - self.delay
            if index >= 0:
                self.log[index][2] = bool(revealed)
        self.since_vote = 0 if k >= 3 else self.since_vote + 1
        ema = [3.0] * len(self.rates)
        values: list[float] = []
        age = 0
        for record_k, record_d, success in self.log:
            value, informative = self._sample(record_k, record_d, success, ema[0])
            if informative:
                ema = [(1 - r) * e + r * value for r, e in zip(self.rates, ema)]
                values.append(value)
                age = 0
            else:
                age += 1
        self.ema, self.values, self.age = ema, values, age

    def trailing(self, predicate) -> int:
        count = 0
        for value in reversed(self.values):
            if not predicate(value):
                break
            count += 1
        return count


class ScaleDefenseEnv(gym.Env):
    """Choose the redundancy ``k`` each round without seeing the attackers."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str | ScaleScenario = "campaign_waves",
        horizon: int | None = None,
        budget_scale: float = 1.0,
        domain_randomization: bool = False,
        constraint_weight: float = 0.0,
        failure_penalty: float | None = None,
        reward_config: RewardConfig = DEFAULT_REWARD,
        communication_weight: float = SCALE_COMMUNICATION_WEIGHT,
        observation_mode: str = "legacy",
        action_mode: str = "k",
        price_conditioned: bool = False,
        receipts: bool = True,
        feature_mask: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.price_conditioned = bool(price_conditioned)
        # Ablations: withhold success receipts / zero selected estimator features.
        self.receipts = bool(receipts)
        self.feature_mask = tuple(feature_mask)
        self.price = 0.5
        if observation_mode not in {"legacy", "estimator"}:
            raise ValueError("observation_mode must be 'legacy' or 'estimator'")
        if action_mode not in {"k", "margin"}:
            raise ValueError("action_mode must be 'k' or 'margin'")
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        if isinstance(scenario, str):
            if scenario not in SCALE_SCENARIOS:
                raise ValueError(f"unknown scenario {scenario!r}")
            scenario = SCALE_SCENARIOS[scenario]
        if horizon is not None:
            scenario = replace(scenario, horizon=int(horizon))
        if budget_scale <= 0:
            raise ValueError("budget_scale must be positive")
        if constraint_weight < 0:
            raise ValueError("constraint_weight must be non-negative")
        self.base_config = scenario
        self.config = scenario
        self.budget_scale = float(budget_scale)
        self.domain_randomization = bool(domain_randomization)
        self.constraint_weight = float(constraint_weight)
        self.failure_penalty = float(
            reward_config.failure_penalty if failure_penalty is None else failure_penalty
        )
        self.reward_config = reward_config
        self.communication_weight = float(communication_weight)
        self.num_actions = (
            scenario.channels + 1 if action_mode == "k"
            else len(MARGIN_ACTIONS) + len(ABSOLUTE_ACTIONS)
        )
        self.observation_size = (
            10 + 4 * HISTORY + LONG_FEATURES if observation_mode == "legacy"
            else ESTIMATOR_FEATURES
        )
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.observation_size,), dtype=np.float32
        )
        self._episode_metrics: dict[str, float] = {}

    @staticmethod
    def redundancy(action: int) -> int:
        return 2 * int(action) + 1

    @staticmethod
    def encode_action(redundancy: int) -> int:
        if redundancy < 1 or redundancy % 2 == 0:
            raise ValueError("redundancy must be a positive odd number")
        return (redundancy - 1) // 2

    @staticmethod
    def _cost(redundancy: int, length: float) -> float:
        # Two full-text carriers, then 64-byte hash witnesses (as in core/engine.py).
        full = min(redundancy, 2) * length
        hashes = max(0, redundancy - 2) * 64.0
        return float((full + hashes) / max(128.0, length * 2.0))

    def _randomized_config(self) -> ScaleScenario:
        cfg = self.base_config
        if not self.domain_randomization:
            return cfg
        rng = self.np_random
        return replace(
            cfg,
            budget_per_step=float(rng.uniform(2.0, 7.0)),
            persistence=float(rng.uniform(0.30, 0.95)),
            # Stationary attackers (no campaign, no waves) must be common, not a 1% corner.
            campaign_coupling=0.0 if rng.random() < 0.35 else float(rng.uniform(0.0, 0.9)),
            wave_probability=0.0 if rng.random() < 0.40 else float(rng.uniform(0.0, 0.06)),
            # Start with at least one attacker so randomized episodes are not trivially empty.
            initial_attackers=int(rng.integers(max(1, cfg.min_attackers), cfg.max_attackers + 1)),
            detector_false_negative=float(rng.uniform(0.10, 0.50)),
            detector_false_positive=float(rng.uniform(0.02, 0.20)),
            length_shift=float(rng.uniform(-0.10, 0.25)),
            importance_shift=float(rng.uniform(-0.08, 0.10)),
        )

    def set_constraint_weight(self, weight: float) -> None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("constraint weight must be finite and non-negative")
        self.constraint_weight = float(weight)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.config = self._randomized_config()
        if options and "budget_scale" in options:
            self.budget_scale = float(options["budget_scale"])
        cfg = self.config
        rng = self.np_random
        self.t = 0
        if self.price_conditioned:
            requested = (options or {}).get("price")
            if requested is not None:
                self.price = float(requested)
            elif self.domain_randomization:
                self.price = float(np.exp(rng.uniform(*np.log(PRICE_RANGE))))
            # The price, not a budget, is what limits spending in this mode.
            cfg = self.config = replace(cfg, budget_per_step=1000.0)
        self.budget_total = cfg.horizon * cfg.budget_per_step * self.budget_scale
        self.budget_remaining = self.budget_total
        self.campaign = int(rng.choice(3, p=REGIME_PRIOR))
        self.regimes = rng.choice(3, size=cfg.initial_attackers, p=REGIME_PRIOR).astype(np.int8)
        self.compromised = 0
        self.length_state = int(rng.choice(3, p=(0.30, 0.50, 0.20)))
        self.importance_state = int(rng.choice(3, p=(0.30, 0.50, 0.20)))
        self.history = np.zeros((HISTORY, 4), dtype=np.float32)
        self.prev_k = 1
        self.prev_success = 1.0
        self.success_log: list[bool] = []
        self.estimator = CountEstimator(
            ESTIMATOR_RATES, delay=cfg.feedback_delay, use_success=self.receipts
        )
        self.prev_cost = 0.0
        self.prev_dissent = 0.0
        self.prev_observed = 0.0
        self.ema_dissent = np.zeros(len(DISSENT_RATES))
        self.ema_fail = np.zeros(len(FAIL_RATES))
        self.ema_k = np.zeros(len(K_RATES))
        self.since_observed = 0
        self.recent_dissent: deque[float] = deque(maxlen=PEAK_WINDOW)
        self._episode_metrics = {
            "successes": 0.0, "importance_total": 0.0, "importance_success": 0.0,
            "cost": 0.0, "budget_violations": 0.0, "redundancy": 0.0,
        }
        self._sample_context()
        return self._observation(), {"scenario": cfg.name}

    def _update_attackers(self) -> None:
        cfg, rng = self.config, self.np_random
        if cfg.wave_probability > 0 and rng.random() < cfg.wave_probability:
            factor = float(rng.choice((0.4, 0.6, 1.6, 2.5)))
            current = len(self.regimes)
            target = round(current * factor)
            if factor > 1:
                # Growth must also work from zero: at least a few new arrivals.
                target = max(target, current + int(rng.integers(2, 6)))
            target = int(np.clip(target, cfg.min_attackers, cfg.max_attackers))
            if target > len(self.regimes):
                added = rng.choice(3, size=target - len(self.regimes), p=REGIME_PRIOR)
                self.regimes = np.concatenate((self.regimes, added)).astype(np.int8)
            else:
                self.regimes = self.regimes[:target]
        self.campaign = int(_step_regimes(rng, np.asarray([self.campaign]))[0])
        own = _step_regimes(rng, self.regimes)
        if cfg.campaign_coupling > 0:
            follow = rng.random(len(own)) < cfg.campaign_coupling
            own = np.where(follow, np.int8(self.campaign), own).astype(np.int8)
        self.regimes = own

    def _sample_attacks(self) -> int:
        cfg, rng = self.config, self.np_random
        active = rng.random(len(self.regimes)) < ATTACK_PROBABILITY[self.regimes]
        per_attacker = np.minimum(
            MAX_PER_ATTACKER, 1 + rng.poisson(POISSON_RATE[self.regimes])
        )
        fresh = int((active * per_attacker).sum())
        if rng.random() < cfg.persistence:
            fresh = max(fresh, int(rng.binomial(self.compromised, cfg.persistence)))
        self.compromised = min(cfg.channels, fresh)
        return self.compromised

    def _sample_context(self) -> None:
        cfg, rng = self.config, self.np_random
        self._update_attackers()
        length_transition = np.asarray(
            ((0.76, 0.21, 0.03), (0.16, 0.69, 0.15), (0.04, 0.21, 0.75))
        )
        self.length_state = int(rng.choice(3, p=length_transition[self.length_state]))
        jitter = float(rng.lognormal(0.0, 0.22))
        self.message_length = MESSAGE_LENGTHS[self.length_state] * jitter * (
            1.0 + cfg.length_shift
        )
        importance_transition = np.asarray(
            ((0.74, 0.23, 0.03), (0.16, 0.70, 0.14), (0.04, 0.24, 0.72))
        )
        shifted = importance_transition[self.importance_state].copy()
        shifted[2] = np.clip(shifted[2] + cfg.importance_shift, 0.01, 0.90)
        shifted[0] = max(0.01, shifted[0] - cfg.importance_shift)
        shifted /= shifted.sum()
        self.importance_state = int(rng.choice(3, p=shifted))
        self.importance = IMPORTANCE_LEVELS[self.importance_state]
        self.true_attacks = self._sample_attacks()
        self.alert_count = 0
        if cfg.use_alert:
            self.alert_count = int(
                rng.binomial(self.true_attacks, 1.0 - cfg.detector_false_negative)
                + rng.binomial(cfg.channels - self.true_attacks, cfg.detector_false_positive)
            )

    def _normalized_count(self, value: float) -> float:
        return float(np.log1p(value) / np.log1p(self.config.channels))

    def _normalized_redundancy(self, k: int) -> float:
        return float(np.log1p(k) / np.log1p(2 * self.config.channels + 1))

    def _action_to_k(self, action: int) -> int:
        action = int(action)
        if self.action_mode == "k":
            return self.redundancy(action)
        if action >= len(MARGIN_ACTIONS):
            return ABSOLUTE_ACTIONS[action - len(MARGIN_ACTIONS)]
        fast = self.estimator.ema[0]
        need = fast + MARGIN_ACTIONS[action] * np.sqrt(fast + 1.0)
        return int(np.clip(2 * np.ceil(max(need, 0.0)) + 1, 1, 2 * self.config.channels + 1))

    def _estimator_observation(self) -> np.ndarray:
        cfg, est = self.config, self.estimator
        n_max = float(cfg.channels)
        slow = est.ema[-1]
        remaining_rounds = max(1, cfg.horizon - self.t)
        remaining_fraction = self.budget_remaining / max(self.budget_total, 1e-6)
        price = 128.0 / max(128.0, 2.0 * self.message_length)
        pace = remaining_fraction - (1.0 - self.t / max(1, cfg.horizon))
        obs = [
            est.values[-1] / n_max if est.values else 0.0,
            min(1.0, est.age / 16.0),
            *(value / n_max for value in est.ema),
            0.5 + 0.5 * float(np.clip((est.ema[0] - est.ema[-1]) / 10.0, -1, 1)),
            max(est.values[-PEAK_WINDOW:], default=0.0) / n_max,
            min(1.0, est.trailing(lambda v: v > slow + 2.0) / 16.0),
            min(1.0, est.trailing(lambda v: v <= slow) / 16.0),
            float(self.ema_fail[0]),
            self.prev_k / (2.0 * n_max + 1.0) if self.t else 0.0,
            self.prev_success,
            self.prev_observed,
            min(1.0, est.since_vote / 16.0),
            price,
            self.importance / 1.75,
            min(1.0, self.importance / price / 17.5),
            remaining_fraction,
            self.t / max(1, cfg.horizon),
            0.5 + 0.5 * float(np.clip(pace, -1, 1)),
            self.budget_remaining / remaining_rounds / BUDGET_NORMALIZER,
            (
                np.log(self.price / PRICE_RANGE[0]) / np.log(PRICE_RANGE[1] / PRICE_RANGE[0])
                if self.price_conditioned
                else cfg.budget_per_step * self.budget_scale / BUDGET_NORMALIZER
            ),
        ]
        values = np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0)
        if self.feature_mask:
            values[list(self.feature_mask)] = 0.0
        return values

    def _observation(self) -> np.ndarray:
        if self.observation_mode == "estimator":
            return self._estimator_observation()
        cfg = self.config
        obs = np.asarray(
            [
                np.clip(np.log1p(self.message_length) / np.log(1000.0), 0, 1),
                self.importance / 1.75,
                self._normalized_count(self.alert_count) if cfg.use_alert else 0.0,
                np.clip(self.budget_remaining / max(self.budget_total, 1e-6), 0, 1),
                self.t / max(1, cfg.horizon),
                np.clip(cfg.budget_per_step * self.budget_scale / BUDGET_NORMALIZER, 0, 1),
                self.prev_success,
                self._normalized_redundancy(self.prev_k) if self.t else 0.0,
                self.prev_dissent,
                self.prev_observed,
                *self.history.reshape(-1).tolist(),
                *self.ema_dissent.tolist(),
                *self.ema_fail.tolist(),
                *self.ema_k.tolist(),
                min(1.0, self.since_observed / PEAK_WINDOW),
                max(self.recent_dissent, default=0.0),
                # Short minus long vote-dissent memory: above 0.5 means attacks rising.
                float(np.clip(0.5 + 2.0 * (self.ema_dissent[0] - self.ema_dissent[2]), 0, 1)),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def step(self, action: int):
        cfg = self.config
        k = self._action_to_k(action)
        success = self.true_attacks <= (k - 1) // 2
        cost = self._cost(k, self.message_length)
        self.budget_remaining -= cost
        violation = self.budget_remaining < 0
        utility = self.importance * (1.0 if success else -self.failure_penalty)
        if self.price_conditioned:
            reward = utility - self.price * cost
            violation = False
        else:
            reward = utility - self.communication_weight * cost
            target = cfg.budget_per_step * self.budget_scale
            reward -= self.constraint_weight * (cost - target)
        if violation:
            reward -= self.reward_config.budget_violation_base + min(
                self.reward_config.budget_debt_cap,
                -self.budget_remaining * self.reward_config.budget_debt_weight,
            )

        compromised_used = min(self.true_attacks, k)
        observed = k >= 3
        dissent = min(compromised_used, k - compromised_used) if observed else 0
        dissent_norm = self._normalized_count(dissent) if observed else 0.0
        metrics = self._episode_metrics
        metrics["successes"] += float(success)
        metrics["importance_total"] += self.importance
        metrics["importance_success"] += self.importance * float(success)
        metrics["cost"] += cost
        metrics["budget_violations"] += float(violation)
        metrics["redundancy"] += k

        self.history = np.roll(self.history, -1, axis=0)
        self.success_log.append(bool(success))
        revealed = self.t - cfg.feedback_delay
        visible = float(self.success_log[revealed]) if revealed >= 0 and self.receipts else None
        self.history[-1] = (1.0 if visible is None else visible, dissent_norm, float(observed),
                            self._normalized_redundancy(k))
        self.estimator.update(k, dissent, None if visible is None else bool(visible))
        self.prev_k = k
        self.prev_success = 1.0 if visible is None else visible
        self.prev_cost = cost
        self.prev_dissent = dissent_norm
        self.prev_observed = float(observed)
        if observed:
            rates = np.asarray(DISSENT_RATES)
            self.ema_dissent = (1 - rates) * self.ema_dissent + rates * dissent_norm
            self.recent_dissent.append(dissent_norm)
            self.since_observed = 0
        else:
            self.since_observed += 1
        failure_rates = np.asarray(FAIL_RATES)
        if visible is not None:
            self.ema_fail = (1 - failure_rates) * self.ema_fail + failure_rates * (1.0 - visible)
        k_rates = np.asarray(K_RATES)
        self.ema_k = (1 - k_rates) * self.ema_k + k_rates * self._normalized_redundancy(k)
        self.t += 1
        terminated = self.t >= cfg.horizon
        info: dict[str, Any] = {
            "success": bool(success), "importance": self.importance, "cost": cost,
            "redundancy": k, "true_attacks": self.true_attacks,
            "attackers": int(len(self.regimes)), "dissent": int(dissent),
            "budget_remaining": self.budget_remaining,
            "budget_violation": bool(violation),
        }
        if terminated:
            total = float(cfg.horizon)
            importance_total = max(1e-8, metrics["importance_total"])
            info["episode_metrics"] = {
                **metrics,
                "success_rate": metrics["successes"] / total,
                "weighted_success_rate": metrics["importance_success"] / importance_total,
                "mean_cost": metrics["cost"] / total,
                "mean_redundancy": metrics["redundancy"] / total,
                "target_mean_cost": self.budget_total / total,
            }
            return np.zeros(self.observation_size, dtype=np.float32), float(reward), True, False, info
        self._sample_context()
        return self._observation(), float(reward), False, False, info


class ScaleFixedPolicy:
    def __init__(self, redundancy: int) -> None:
        self.action = ScaleDefenseEnv.encode_action(redundancy)

    def predict(self, observation, deterministic: bool = True):
        return self.action, None


class TrendRulePolicy:
    """Non-RL history baseline: EMA of observed vote dissent plus a safety margin.

    Uses only the observation.  The count is recovered from the normalized
    feature; unobserved rounds (k = 1) keep the last estimate and force k >= 3
    every ``probe_every`` rounds so the estimate does not go stale.
    """

    def __init__(self, channels: int, margin: float, alpha: float = 0.5,
                 probe_every: int = 6) -> None:
        self.channels = channels
        self.margin = margin
        self.alpha = alpha
        self.probe_every = probe_every
        self.reset()

    def reset(self, env=None) -> None:
        self.estimate = 3.0
        self.rounds = 0

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation)
        dissent_norm, observed = obs[8], obs[9]
        if observed > 0.5:
            count = np.expm1(dissent_norm * np.log1p(self.channels))
            self.estimate = (1 - self.alpha) * self.estimate + self.alpha * count
        if self.rounds and obs[6] < 0.5:
            # A failed round proves n > (k_prev - 1) / 2 even when the vote hid it.
            previous_k = np.expm1(obs[7] * np.log1p(2 * self.channels + 1))
            self.estimate = max(self.estimate, 0.6 * (previous_k + 1.0))
        self.rounds += 1
        need = self.estimate + self.margin * np.sqrt(self.estimate + 1.0)
        k = int(2 * np.ceil(need) + 1)
        if self.rounds % self.probe_every == 0:
            k = max(k, 3)
        k = int(np.clip(k, 1, 2 * self.channels + 1))
        return ScaleDefenseEnv.encode_action(k), None


class ExactEstimatorPolicy:
    """Non-RL rule: ``CountEstimator`` forecast plus a margin with a few gains.

    The margin grows with message value (importance / marginal cost) and with
    budget pace.  Reads only the legacy observation.
    """

    PARAMETERS = ("alpha", "margin", "value_gain", "pace_gain", "trend_gain", "probe_every")

    def __init__(self, channels: int, alpha: float = 0.5, margin: float = 1.0,
                 value_gain: float = 0.0, pace_gain: float = 0.0,
                 trend_gain: float = 0.0, probe_every: int = 6,
                 use_success: bool = True, delay: int = 0) -> None:
        self.channels = channels
        self.margin = margin
        self.value_gain = value_gain
        self.pace_gain = pace_gain
        self.trend_gain = trend_gain
        self.probe_every = int(probe_every)
        self.estimator = CountEstimator((alpha, 0.05), delay=delay, use_success=use_success)
        self.reset()

    def reset(self, env=None) -> None:
        self.rounds = 0
        self.estimator.reset()

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation)
        if self.rounds:
            k_prev = int(round(np.expm1(obs[7] * np.log1p(2 * self.channels + 1))))
            dissent = int(round(np.expm1(obs[8] * np.log1p(self.channels))))
            self.estimator.update(k_prev, dissent, bool(obs[6] > 0.5))
        self.rounds += 1
        fast, slow = self.estimator.ema
        length = np.expm1(obs[0] * np.log(1000.0))
        price = 128.0 / max(128.0, 2.0 * length)
        value = (obs[1] * 1.75) / price
        pace = obs[3] - (1.0 - obs[4])
        margin = (
            self.margin + self.value_gain * np.log(max(value, 1e-3) / 3.0)
            + self.pace_gain * pace
        )
        forecast = fast + self.trend_gain * max(0.0, fast - slow)
        need = forecast + margin * np.sqrt(forecast + 1.0)
        k = int(2 * np.ceil(max(need, 0.0)) + 1)
        if (self.estimator.since_vote >= self.probe_every
                or self.rounds % self.probe_every == 0):
            k = max(k, 3)
        return ScaleDefenseEnv.encode_action(int(np.clip(k, 1, 2 * self.channels + 1))), None
