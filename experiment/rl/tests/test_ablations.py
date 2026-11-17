import numpy as np
import torch

from experiment.rl.ablations import CURRENT_FEATURES, CurrentObservation, CurrentPolicy, mask_history
from experiment.rl.environment import DefenseEnv
from experiment.rl.train_agents import build_model


def test_mask_removes_history_but_retains_current_risk_and_resources():
    before = np.ones((2, 25), dtype=np.float32)
    before[0, list(CURRENT_FEATURES)] = 0.4
    before[1, list(CURRENT_FEATURES)] = 0.4
    before[1, 6:22] = 0.1
    masked = mask_history(before)
    np.testing.assert_array_equal(masked[0], masked[1])
    np.testing.assert_array_equal(masked[:, CURRENT_FEATURES], before[:, CURRENT_FEATURES])
    assert not masked[:, 6:22].any()
    assert before[0, 6] == 1.0  # Input is not modified in-place.


def test_no_history_does_not_change_environment_dynamics_for_fixed_actions():
    full = DefenseEnv(horizon=5)
    reduced = CurrentObservation(DefenseEnv(horizon=5))
    obs, _ = full.reset(seed=17)
    masked, _ = reduced.reset(seed=17)
    np.testing.assert_array_equal(masked, mask_history(obs))
    for action in (0, 5, 11, 2, 4):
        original = full.step(action)
        ablated = reduced.step(action)
        np.testing.assert_array_equal(ablated[0], mask_history(original[0]))
        assert ablated[1:] == original[1:]


def test_training_and_inference_use_the_same_observation_intervention():
    class CapturePolicy:
        def predict(self, observation, deterministic=True):
            self.seen = observation
            return np.zeros(observation.shape[0], dtype=int), None

    underlying = CapturePolicy()
    policy = CurrentPolicy(underlying)
    values = np.ones((3, 25), dtype=np.float32)
    policy.predict(values)
    np.testing.assert_array_equal(underlying.seen, mask_history(values))


def test_zero_discount_keeps_architecture_and_initial_weights_identical():
    env = DefenseEnv(horizon=5)
    full = build_model("qrdqn", env, 7, "cpu")
    short = build_model("qrdqn", env, 7, "cpu", discount=0.0)
    assert full.gamma == 0.985
    assert short.gamma == 0.0
    for key, value in full.policy.state_dict().items():
        torch.testing.assert_close(value, short.policy.state_dict()[key])
    env.close()
