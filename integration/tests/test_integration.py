import random
from dataclasses import replace

import numpy as np
import pytest

from experiment.rl.large_scale import SCALE_SCENARIOS, ScaleDefenseEnv
from integration.adapter import AdaptiveAdapter
from integration.adaptive_engine import AdaptiveDefenseEngine
from integration.controller import FixedController, QRDQNController, normalized_cost
from integration.mock_adversary import MockAdversary


def make_adapter(k, attacked):
    engine = AdaptiveDefenseEngine()
    adapter = AdaptiveAdapter(
        engine, MockAdversary(), [], victim_name="Victim",
        controller=FixedController(k), attack_model=lambda _k: attacked,
    )
    return adapter, engine


def test_honest_hash_majority_succeeds_even_when_both_text_carriers_are_poisoned():
    message = ("The answer is B. " * 5).strip()
    adapter, engine = make_adapter(9, [0, 1, 2])  # carriers 0 and 1 poisoned, honest 6 of 9
    final, traitors = adapter.secure_transmit(message, "Agent_A", "Victim")
    assert engine.integrity_verified()           # the winning hash is the sender's hash
    assert final != message                      # no honest text arrived, and the engine says so
    assert "DATA_INTEGRITY_VIOLATION" in final
    assert len(traitors) == 3


def test_poisoned_hash_majority_fails():
    adapter, engine = make_adapter(9, [0, 1, 2, 3, 4])  # 5 of 9 channels poisoned
    adapter.secure_transmit("m" * 80, "Agent_A", "Victim")
    assert not engine.integrity_verified()


@pytest.mark.parametrize("k,n", [(1, 0), (1, 1), (3, 1), (3, 2), (9, 4), (9, 5), (21, 10), (21, 11)])
def test_engine_outcome_matches_the_simulator_majority_rule(k, n):
    rng = random.Random(k * 100 + n)
    attacked = rng.sample(range(k), min(n, k))
    adapter, engine = make_adapter(k, attacked)
    _, traitors = adapter.secure_transmit("m" * 120, "Agent_A", "Victim")
    assert engine.integrity_verified() == (min(n, k) <= (k - 1) // 2)
    if k >= 3:  # the vote shows the minority group size, as the simulator assumes
        assert len(traitors) == min(min(n, k), k - min(n, k))


def test_bytes_follow_the_two_carrier_hash_witness_cost_model():
    adapter, engine = make_adapter(11, [])
    adapter.secure_transmit("x" * 150, "Agent_A", "Victim")
    assert engine.last_bytes == 2 * 150 + 9 * 64
    assert normalized_cost(engine.last_bytes, 150) == pytest.approx(
        ScaleDefenseEnv._cost(11, 150.0)
    )


class ConstantModel:
    def predict(self, observation, deterministic=True):
        return 7 + 3, None  # absolute k = 13


def test_controller_observation_is_identical_to_the_training_environment():
    config = replace(SCALE_SCENARIOS["campaign_waves"], feedback_delay=1)
    env = ScaleDefenseEnv(config, observation_mode="estimator", action_mode="margin",
                          price_conditioned=True)
    obs, _ = env.reset(seed=11, options={"price": 0.3})
    controller = QRDQNController(ConstantModel(), price=0.3, channels=config.channels,
                                 horizon=config.horizon, delay=1)
    assert np.allclose(controller.observation(env.message_length, env.importance), obs, atol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(150):
        length = env.message_length
        action = int(rng.integers(env.action_space.n))
        obs, _, done, _, info = env.step(action)
        controller.feedback(info["redundancy"], info["dissent"], info["success"],
                            info["cost"] * max(128.0, 2.0 * length), length)
        if done:
            break
        assert np.allclose(controller.observation(env.message_length, env.importance), obs,
                           atol=1e-4), f"mismatch at round {controller.t}"
