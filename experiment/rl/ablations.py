"""Only the observation intervention needed by the no-history ablation."""

import gymnasium as gym
import numpy as np


# Current workload, alert, resource state, time, detector rates, and capacity.
# Remove previous action/outcome/cost, rolling history, and protection history.
CURRENT_FEATURES = (0, 1, 2, 3, 4, 5, 22, 23, 24)


def mask_history(observation):
    values = np.asarray(observation)
    if values.shape[-1] != 25:
        raise ValueError("the no-history ablation expects 25 observation features")
    masked = np.zeros_like(values)
    masked[..., CURRENT_FEATURES] = values[..., CURRENT_FEATURES]
    return masked


class CurrentObservation(gym.ObservationWrapper):
    def observation(self, observation):
        return mask_history(observation)


class CurrentPolicy:
    """Apply the identical observation mask during evaluation."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, observation, deterministic=True):
        return self.policy.predict(mask_history(observation), deterministic=deterministic)
