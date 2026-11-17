from dataclasses import replace
import pytest

from experiment.rl.benchmark_ml_methods import build_estimators
from experiment.rl.environment import DefenseEnv
from experiment.rl.config import ScenarioConfig
from experiment.rl.plot_budget_success_comparison import _retry_success_probability


def test_all_primary_ml_methods_use_full_rl_observation():
    assert DefenseEnv.observation_size == 25
    estimators = build_estimators(seed=7)
    assert len(estimators) >= 8
    assert all(indices is None for _, indices in estimators.values())


def test_ml_oracle_uses_channel_union_for_multi_attacker_rotation():
    config = replace(ScenarioConfig(), attacker_count=3, max_attack=2,
                     rotate_clear_probability=0.6)
    env = DefenseEnv(config)
    env.reset(seed=7)
    env.true_attacks = 2
    # Each attacked channel is still bad if it survives or is reinfected.
    bad_probability = 0.4 + 0.6 * 0.035
    assert _retry_success_probability(env, quorum=1, rotate=True) == pytest.approx(
        1.0 - bad_probability ** 2
    )
