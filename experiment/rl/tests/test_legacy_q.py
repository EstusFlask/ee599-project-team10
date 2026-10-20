import numpy as np

from experiment.rl.environment import DefenseEnv
from experiment.rl.legacy_q import LegacyLinearQPolicy


def test_legacy_q_adapter_produces_valid_action():
    env = DefenseEnv(horizon=4)
    obs, _ = env.reset(seed=11)
    policy = LegacyLinearQPolicy(seed=11)
    policy.reset(env)
    action, state = policy.predict(obs)
    assert state is None
    assert env.action_space.contains(int(np.asarray(action).item()))
    assert env.decode_action(int(action))[1] == "retry"
