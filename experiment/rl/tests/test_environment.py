import numpy as np
import pytest

from experiment.rl.environment import DefenseEnv, RECOVERIES, REDUNDANCIES, SCENARIOS


def test_action_round_trip():
    for k in REDUNDANCIES:
        for recovery in RECOVERIES:
            action = DefenseEnv.encode_action(k, recovery)
            assert DefenseEnv.decode_action(action) == (k, recovery)


def test_seed_is_deterministic():
    a = DefenseEnv(horizon=12, domain_randomization=True)
    b = DefenseEnv(horizon=12, domain_randomization=True)
    oa, _ = a.reset(seed=19)
    ob, _ = b.reset(seed=19)
    np.testing.assert_array_equal(oa, ob)
    for action in (0, 4, 11, 2, 7):
        xa = a.step(action)
        xb = b.step(action)
        np.testing.assert_array_equal(xa[0], xb[0])
        assert xa[1:] == xb[1:]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_smoke(scenario):
    env = DefenseEnv(scenario=scenario, horizon=9)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    total_cost = 0.0
    for action in range(9):
        obs, reward, terminated, truncated, info = env.step(action % 12)
        assert np.isfinite(reward)
        assert info["cost"] > 0
        total_cost += info["cost"]
        if not terminated:
            assert env.observation_space.contains(obs)
    assert terminated and not truncated
    assert info["episode_metrics"]["cost"] == pytest.approx(total_cost)


def test_budget_decreases_by_reported_cost():
    env = DefenseEnv(horizon=3)
    env.reset(seed=3)
    before = env.budget_remaining
    _, _, _, _, info = env.step(11)
    assert env.budget_remaining == pytest.approx(before - info["cost"])


def test_budget_capacity_is_observable_from_first_step():
    low = DefenseEnv(horizon=3, budget_scale=0.65)
    high = DefenseEnv(horizon=3, budget_scale=1.35)
    low_obs, _ = low.reset(seed=17)
    high_obs, _ = high.reset(seed=17)
    np.testing.assert_array_equal(low_obs[:-1], high_obs[:-1])
    assert low_obs[-1] < high_obs[-1]


def test_action_mask_is_valid_and_budget_sensitive():
    env = DefenseEnv(horizon=8, budget_scale=0.35)
    env.reset(seed=23)
    masks = env.action_masks()
    assert masks.dtype == np.bool_
    assert masks.shape == (12,)
    assert masks[DefenseEnv.encode_action(1, "none")]
    assert 1 <= int(masks.sum()) < 12


@pytest.mark.parametrize("attacker_count", [2, 3, 5])
def test_independent_multi_attacker_processes(attacker_count):
    config = SCENARIOS[f"multi_attacker_{attacker_count}"]
    env = DefenseEnv(config, horizon=24)
    obs, _ = env.reset(seed=41)
    assert env.observation_space.contains(obs)
    assert len(env.attacker_regimes) == attacker_count
    for _ in range(24):
        obs, _, terminated, _, info = env.step(0)
        assert info["attacker_count"] == attacker_count
        assert len(info["attacker_regimes"]) == attacker_count
        assert 0 <= info["true_attacks"] <= config.max_attack
        if terminated:
            break


def test_multi_attacker_training_profile_randomizes_count():
    counts = set()
    env = DefenseEnv(
        "train_mix", domain_randomization=True,
        randomization_profile="multi_attacker",
    )
    for seed in range(20):
        env.reset(seed=seed)
        counts.add(env.config.attacker_count)
    assert counts.issubset({1, 2, 3, 4, 5})
    assert len(counts) >= 4


def test_weighted_attacker_count_sampling():
    env = DefenseEnv(
        "train_mix", domain_randomization=True,
        randomization_profile="multi_attacker",
        attacker_count_probabilities=(1, 0, 0, 0, 0),
    )
    for seed in range(8):
        env.reset(seed=seed)
        assert env.config.attacker_count == 1


def test_training_failure_penalty_does_not_change_observation():
    standard = DefenseEnv(horizon=4)
    reliability = DefenseEnv(horizon=4, failure_penalty=2.4)
    standard_obs, _ = standard.reset(seed=33)
    reliability_obs, _ = reliability.reset(seed=33)
    np.testing.assert_array_equal(standard_obs, reliability_obs)
    standard_step = standard.step(0)
    reliability_step = reliability.step(0)
    np.testing.assert_array_equal(standard_step[0], reliability_step[0])
    assert standard_step[2:] == reliability_step[2:]
    assert reliability_step[1] <= standard_step[1]
