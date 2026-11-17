import numpy as np
import pytest

from experiment.rl.large_scale import (
    SCALE_SCENARIOS, ScaleDefenseEnv, ScaleFixedPolicy, TrendRulePolicy,
)


def test_actions_are_odd_redundancies_up_to_two_n_plus_one():
    env = ScaleDefenseEnv("campaign")
    channels = SCALE_SCENARIOS["campaign"].channels
    assert env.action_space.n == channels + 1
    assert env.redundancy(0) == 1
    assert env.redundancy(env.action_space.n - 1) == 2 * channels + 1
    assert env.encode_action(7) == 3
    with pytest.raises(ValueError):
        env.encode_action(4)


def test_success_is_majority_comparison():
    env = ScaleDefenseEnv("independent")
    env.reset(seed=3)
    env.true_attacks = 4
    assert env.step(env.encode_action(9))[4]["success"]
    env.true_attacks = 5
    assert not env.step(env.encode_action(9))[4]["success"]


def test_vote_feedback_is_censored_and_hides_attackers():
    env = ScaleDefenseEnv("independent")
    env.reset(seed=5)
    env.true_attacks = 6
    obs, *_ = env.step(env.encode_action(1))
    assert obs[9] == 0.0 and obs[8] == 0.0  # k=1: no vote, nothing observed
    env.true_attacks = 6
    info = env.step(env.encode_action(5))[4]
    assert info["dissent"] == min(5, 5 - 5)  # every used channel poisoned looks quiet
    env.true_attacks = 2
    assert env.step(env.encode_action(9))[4]["dissent"] == 2


def test_observation_is_bounded_and_has_declared_size():
    env = ScaleDefenseEnv("campaign_waves", domain_randomization=True)
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        assert obs.shape == (env.observation_size,)
        assert np.all((obs >= 0) & (obs <= 1))
        obs, _, done, _, _ = env.step(env.action_space.sample())


def test_cost_matches_two_carrier_hash_witness_formula():
    assert ScaleDefenseEnv._cost(1, 192.0) == pytest.approx(0.5)
    assert ScaleDefenseEnv._cost(3, 192.0) == pytest.approx((384 + 64) / 384)
    assert ScaleDefenseEnv._cost(11, 192.0) == pytest.approx((384 + 9 * 64) / 384)


def test_waves_change_the_attacker_count_but_independent_does_not():
    def counts(name):
        env = ScaleDefenseEnv(name, horizon=400)
        env.reset(seed=11)
        seen = set()
        for _ in range(399):
            _, _, done, _, info = env.step(env.encode_action(3))
            seen.add(info["attackers"])
            if done:
                break
        return seen

    assert len(counts("independent")) == 1
    assert len(counts("waves")) > 1


def test_baseline_policies_return_valid_actions():
    env = ScaleDefenseEnv("campaign")
    obs, _ = env.reset(seed=2)
    policy = TrendRulePolicy(env.config.channels, margin=1.0)
    for _ in range(30):
        action, _ = policy.predict(obs)
        assert 0 <= action < env.action_space.n
        obs, *_ = env.step(action)
    assert ScaleFixedPolicy(9).predict(obs)[0] == 4


def test_feedback_delay_reveals_success_late():
    from dataclasses import replace

    def first_visible(delay):
        env = ScaleDefenseEnv(replace(SCALE_SCENARIOS["independent"], feedback_delay=delay))
        env.reset(seed=4)
        env.true_attacks = 40  # k=1 fails, so a visible failure shows as 0.0
        obs, *_ = env.step(env.encode_action(1))
        first = obs[6]
        env.true_attacks = 0
        obs, *_ = env.step(env.encode_action(1))
        return first, obs[6]

    assert first_visible(0) == (0.0, 1.0)
    assert first_visible(1) == (1.0, 0.0)  # failure of round 0 appears one round later


def test_attackers_can_drop_to_zero_and_return():
    from dataclasses import replace

    env = ScaleDefenseEnv(replace(
        SCALE_SCENARIOS["waves"], initial_attackers=3, wave_probability=0.5, horizon=600,
    ))
    env.reset(seed=8)
    counts = []
    for _ in range(599):
        _, _, done, _, info = env.step(env.encode_action(1))
        counts.append(info["attackers"])
        if done:
            break
    assert 0 in counts
    first_zero = counts.index(0)
    assert max(counts[first_zero:]) > 0


def test_estimator_and_margin_modes_have_declared_shapes_and_bounds():
    from dataclasses import replace

    config = replace(SCALE_SCENARIOS["campaign_waves"], feedback_delay=1)
    env = ScaleDefenseEnv(
        config, domain_randomization=True, observation_mode="estimator", action_mode="margin",
    )
    assert env.observation_size == 22 and env.action_space.n == 19
    obs, _ = env.reset(seed=3)
    done = False
    while not done:
        assert obs.shape == (22,) and np.all((obs >= 0) & (obs <= 1))
        obs, _, done, _, info = env.step(env.action_space.sample())
        assert info["redundancy"] % 2 == 1 and 1 <= info["redundancy"] <= 101


def test_count_estimator_recovers_exact_n_and_corrects_after_receipt():
    from experiment.rl.large_scale import CountEstimator

    est = CountEstimator(rates=(1.0,), delay=1)
    est.update(9, 3, None)  # provisional: read as success, n = d = 3
    assert est.values[-1] == 3.0
    est.update(9, 3, False)  # receipt: round 0 failed -> n = k - d = 6
    assert est.values[0] == 6.0


def test_price_conditioned_env_charges_price_and_exposes_it():
    env = ScaleDefenseEnv(
        "campaign", observation_mode="estimator", action_mode="margin",
        price_conditioned=True,
    )
    obs, _ = env.reset(seed=1, options={"price": 0.5})
    low = obs[21]
    env.true_attacks = 0
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(env.importance - 0.5 * info["cost"])
    obs2, _ = env.reset(seed=1, options={"price": 2.0})
    assert obs2[21] > low and not info["budget_violation"]
