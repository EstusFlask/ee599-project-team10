from experiment.rl.benchmark_ml_methods import build_estimators
from experiment.rl.environment import DefenseEnv


def test_all_primary_ml_methods_use_full_rl_observation():
    assert DefenseEnv.observation_size == 25
    estimators = build_estimators(seed=7)
    assert len(estimators) >= 8
    assert all(indices is None for _, indices in estimators.values())
