"""Non-learning and supervised comparison policies."""

from __future__ import annotations

import numpy as np

from .environment import DefenseEnv


class FixedPolicy:
    def __init__(self, redundancy: int, recovery: str = "none") -> None:
        self.action = DefenseEnv.encode_action(redundancy, recovery)

    def predict(self, observation, deterministic: bool = True):
        return self.action, None


class RiskHeuristicPolicy:
    """Strong hand-written baseline using only observable information."""

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation)
        length, importance, alert, _, budget, progress = obs[:6]
        risk = 1.25 * alert + 0.50 * importance + 0.25 * (1.0 - budget)
        if risk < 0.48:
            redundancy = 1
        elif risk < 0.83:
            redundancy = 3
        elif risk < 1.22:
            redundancy = 5
        else:
            redundancy = 7
        if budget < 0.16 and importance < 0.72:
            redundancy = max(1, redundancy - 2)
        if alert > 0.58 and importance > 0.55:
            recovery = "rotate"
        elif alert > 0.18 and budget > 0.20:
            recovery = "retry"
        else:
            recovery = "none"
        return DefenseEnv.encode_action(redundancy, recovery), None


class ConservativePolicy:
    """A non-RL production-style policy favoring reliability."""

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation)
        importance, alert, budget = obs[1], obs[2], obs[4]
        redundancy = 3
        if alert > 0.25 or importance > 0.72:
            redundancy = 5
        if alert > 0.62 and budget > 0.12:
            redundancy = 7
        recovery = "rotate" if alert > 0.35 else "retry"
        return DefenseEnv.encode_action(redundancy, recovery), None
