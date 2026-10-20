"""Adapter that trains the project's original linear Q learner on the POMDP.

The old learner selects only redundancy.  Its historical behavior retried a
failed transmission, so action ``k`` is mapped to ``(k, retry)`` here.  This
keeps the initial RL baseline recognizable while giving it the same noisy
observations, rewards, attacks, and evaluation seeds as the newer agents.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from experiment.defender import ACTIONS, RLdefender, defense_cost

from .environment import DefenseEnv


class LegacyLinearQPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.model = RLdefender(seed=seed)
        self.env: DefenseEnv | None = None

    def reset(self, env: DefenseEnv | None = None) -> None:
        self.model.reset()
        self.env = env

    def _state(self, obs: np.ndarray) -> dict:
        length = math.expm1(float(obs[0]) * math.log(1000.0))
        importance_value = float(obs[1]) * 1.75
        importance = (
            "low" if importance_value < 0.78
            else "normal" if importance_value < 1.38
            else "high"
        )
        if self.env is not None:
            rounds_left = max(1, self.env.config.horizon - self.env.t)
            remaining_total = max(0.0, self.env.budget_remaining)
        else:
            progress = min(0.999, float(obs[5]))
            rounds_left = max(1, round(256 * (1.0 - progress)))
            per_step_capacity = float(obs[-1]) * 1.30
            remaining_total = float(obs[4]) * per_step_capacity * 256
        max_now = min(3, int(math.floor(remaining_total + 1e-8)))
        allowed = [k for k in ACTIONS if defense_cost(k) <= max_now]
        return {
            "history": list(self.model.states),
            "remaining_budget": remaining_total / rounds_left,
            "progress": float(obs[5]),
            "message_length": length,
            "importance": importance,
            "allowed_actions": allowed or [1],
        }

    def predict(self, obs, deterministic: bool = True):
        redundancy = self.model.predict(self._state(np.asarray(obs)), explore=not deterministic)
        action = DefenseEnv.encode_action(redundancy, "retry")
        return np.asarray(action), None

    def train(
        self,
        total_steps: int,
        seed: int,
        training_profile: str = "robust",
        constraint_weight: float = 0.15,
    ) -> None:
        completed = 0
        episode = 0
        while completed < total_steps:
            env = DefenseEnv(
                "train_mix", domain_randomization=True,
                randomization_profile=training_profile,
                constraint_weight=constraint_weight,
            )
            obs, _ = env.reset(seed=seed + episode)
            self.reset(env)
            fraction = completed / max(1, total_steps)
            self.model.epsilon = max(0.02, 0.25 * (1.0 - fraction))
            while completed < total_steps:
                state = self._state(obs)
                redundancy = self.model.predict(state, explore=True)
                action = env.encode_action(redundancy, "retry")
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.model.observe(info["alert_count"])
                done = terminated or truncated
                next_state = {"history": list(self.model.states)} if done else self._state(next_obs)
                self.model.update(next_state, reward, done=done)
                completed += 1
                obs = next_obs
                if done:
                    break
            episode += 1
        self.model.epsilon = 0.0
        self.env = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "LegacyLinearQPolicy":
        with path.open("rb") as handle:
            policy = pickle.load(handle)
        if not isinstance(policy, cls):
            raise TypeError(f"unexpected legacy policy type: {type(policy)!r}")
        return policy
