"""Controllers that choose ``k`` for each message from signals a real deployment has.

Inputs per round: message length, message importance, the vote result of the previous
rounds (size of the minority hash group), and a signed receipt telling the sender whether
a past round delivered the original text (``delay`` rounds late, not charged).
Nothing here reads the attacker count, the attack stage, or any simulator internals.

``AdaptiveController`` mirrors the bookkeeping and the 22 estimator features of
``experiment.rl.large_scale.ScaleDefenseEnv`` so a trained policy can be used as is
(``tests/test_integration.py`` checks the two observations are identical).
"""

from __future__ import annotations

import numpy as np

from experiment.rl.large_scale import (
    ABSOLUTE_ACTIONS, ESTIMATOR_RATES, FAIL_RATES, MARGIN_ACTIONS, PEAK_WINDOW,
    PRICE_RANGE, CountEstimator,
)

COST_FLOOR_BYTES = 128.0  # cost = bytes / max(128, 2 * length), as in the simulator


def normalized_cost(bytes_sent: float, message_length: float) -> float:
    return float(bytes_sent / max(COST_FLOOR_BYTES, 2.0 * message_length))


class FixedController:
    """Baseline: the same ``k`` for every message."""

    def __init__(self, k: int) -> None:
        self.k = k

    def decide(self, message_length: float, importance: float) -> int:
        return self.k

    def feedback(self, k, dissent, success, bytes_sent, message_length) -> None:
        pass


class AdaptiveController:
    """Shared bookkeeping (vote history, delayed receipts, spend) for learned policies."""

    def __init__(self, channels: int = 50, horizon: int = 256, delay: int = 1,
                 budget_per_step: float = 1000.0, use_receipts: bool = True) -> None:
        self.channels = channels
        self.horizon = horizon
        self.delay = delay
        self.budget_per_step = budget_per_step
        self.use_receipts = use_receipts
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.estimator = CountEstimator(
            ESTIMATOR_RATES, delay=self.delay, use_success=self.use_receipts
        )
        self.success_log: list[bool] = []
        self.ema_fail = np.zeros(len(FAIL_RATES))
        self.prev_k = 1
        self.prev_success = 1.0
        self.prev_observed = 0.0
        self.prev_dissent = 0
        self.budget_total = self.horizon * self.budget_per_step
        self.budget_remaining = self.budget_total

    def feedback(self, k: int, dissent: int, success: bool, bytes_sent: float,
                 message_length: float) -> None:
        """Record one finished round.  ``dissent`` is the size of the minority hash group
        (``len(traitors)`` from ``DefenseEngine.resolve``); ``success`` is the receipt."""
        observed = k >= 3  # with k = 1 there is no vote, hence no information
        dissent = int(dissent) if observed else 0
        self.budget_remaining -= normalized_cost(bytes_sent, message_length)
        self.success_log.append(bool(success))
        revealed = self.t - self.delay
        visible = float(self.success_log[revealed]) if revealed >= 0 and self.use_receipts else None
        self.estimator.update(k, dissent, None if visible is None else bool(visible))
        self.prev_k, self.prev_dissent = k, dissent
        self.prev_success = 1.0 if visible is None else visible
        self.prev_observed = float(observed)
        if visible is not None:
            rates = np.asarray(FAIL_RATES)
            self.ema_fail = (1 - rates) * self.ema_fail + rates * (1.0 - visible)
        self.t += 1


class RuleController(AdaptiveController):
    """The tuned non-RL rule (``ExactEstimatorPolicy``); it keeps its own estimator."""

    def __init__(self, policy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy = policy

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "policy") and hasattr(self.policy, "reset"):
            self.policy.reset()

    def decide(self, message_length: float, importance: float) -> int:
        n = float(self.channels)
        obs = np.zeros(10 + 4 * 8 + 10, dtype=np.float32)  # legacy layout read by the rule
        obs[0] = np.clip(np.log1p(message_length) / np.log(1000.0), 0, 1)
        obs[1] = importance / 1.75
        obs[3] = 1.0 - self.t / self.horizon  # on budget pace: the pace term is zero
        obs[4] = self.t / self.horizon
        obs[6] = self.prev_success
        obs[7] = np.log1p(self.prev_k) / np.log1p(2 * n + 1) if self.t else 0.0
        obs[8] = np.log1p(self.prev_dissent) / np.log1p(n)
        action, _ = self.policy.predict(obs)
        return 2 * int(np.asarray(action).item()) + 1


class QRDQNController(AdaptiveController):
    """A price-conditioned estimator-feature policy trained on ``ScaleDefenseEnv``
    (stable-baselines3 model with ``predict``), margin actions."""

    def __init__(self, model, price: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.price = float(price)

    def observation(self, message_length: float, importance: float) -> np.ndarray:
        est, n = self.estimator, float(self.channels)
        slow = est.ema[-1]
        remaining_rounds = max(1, self.horizon - self.t)
        remaining_fraction = self.budget_remaining / max(self.budget_total, 1e-6)
        marginal = COST_FLOOR_BYTES / max(COST_FLOOR_BYTES, 2.0 * message_length)
        pace = remaining_fraction - (1.0 - self.t / max(1, self.horizon))
        features = [
            est.values[-1] / n if est.values else 0.0,
            min(1.0, est.age / 16.0),
            *(value / n for value in est.ema),
            0.5 + 0.5 * float(np.clip((est.ema[0] - est.ema[-1]) / 10.0, -1, 1)),
            max(est.values[-PEAK_WINDOW:], default=0.0) / n,
            min(1.0, est.trailing(lambda v: v > slow + 2.0) / 16.0),
            min(1.0, est.trailing(lambda v: v <= slow) / 16.0),
            float(self.ema_fail[0]),
            self.prev_k / (2.0 * n + 1.0) if self.t else 0.0,
            self.prev_success,
            self.prev_observed,
            min(1.0, est.since_vote / 16.0),
            marginal,
            importance / 1.75,
            min(1.0, importance / marginal / 17.5),
            remaining_fraction,
            self.t / max(1, self.horizon),
            0.5 + 0.5 * float(np.clip(pace, -1, 1)),
            self.budget_remaining / remaining_rounds / 10.0,
            np.log(self.price / PRICE_RANGE[0]) / np.log(PRICE_RANGE[1] / PRICE_RANGE[0]),
        ]
        return np.clip(np.asarray(features, dtype=np.float32), 0.0, 1.0)

    def decide(self, message_length: float, importance: float) -> int:
        action, _ = self.model.predict(self.observation(message_length, importance),
                                       deterministic=True)
        action = int(np.asarray(action).item())
        if action >= len(MARGIN_ACTIONS):
            return ABSOLUTE_ACTIONS[action - len(MARGIN_ACTIONS)]
        fast = self.estimator.ema[0]
        need = fast + MARGIN_ACTIONS[action] * np.sqrt(fast + 1.0)
        return int(np.clip(2 * np.ceil(max(need, 0.0)) + 1, 1, 2 * self.channels + 1))
