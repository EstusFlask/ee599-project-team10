import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from experiment.rl.budget import BudgetController, BudgetPenaltyCallback
from experiment.rl.environment import DefenseEnv


def test_budget_price_tracks_excess_and_stays_bounded():
    controller = BudgetController(weight=0.1, learning_rate=0.5, max_weight=1.0)
    assert controller.update(1.2, 1.0) == pytest.approx(0.2)
    assert controller.update(0.8, 1.0) == pytest.approx(0.1)
    assert controller.update(10.0, 1.0) == 1.0
    assert controller.update(0.0, 10.0) == 0.0


def test_budget_callback_updates_wrapped_parallel_envs():
    env = make_vec_env(lambda: Monitor(DefenseEnv(
        horizon=4, budget_scale=0.01, budget_penalty_mode="lagrangian",
    )), n_envs=2, seed=7)
    controller = BudgetController(weight=0.1)
    model = DQN(
        "MlpPolicy", env, learning_starts=100, buffer_size=100,
        policy_kwargs={"net_arch": [8]}, seed=7, device="cpu",
    )
    try:
        model.learn(total_timesteps=16, callback=BudgetPenaltyCallback(controller))
        assert len(controller.history) >= 2
        assert controller.weight > 0.1
        assert env.get_attr("constraint_weight") == pytest.approx([controller.weight] * 2)
    finally:
        env.close()


def test_budget_price_does_not_learn_from_random_warmup():
    env = make_vec_env(lambda: Monitor(DefenseEnv(
        horizon=4, budget_scale=0.01, budget_penalty_mode="lagrangian",
    )), seed=7)
    controller = BudgetController(weight=0.1)
    model = DQN(
        "MlpPolicy", env, learning_starts=100, buffer_size=100,
        policy_kwargs={"net_arch": [8]}, seed=7, device="cpu",
    )
    try:
        model.learn(total_timesteps=16,
                    callback=BudgetPenaltyCallback(controller, warmup_steps=100))
        assert controller.weight == 0.1
        assert controller.history == []
    finally:
        env.close()
