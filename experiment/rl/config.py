"""Shared scenarios and reward settings for the small defense experiment."""

from dataclasses import dataclass


REDUNDANCIES = (1, 3, 5, 7)
RECOVERIES = ("none", "retry", "rotate")
# Rotation now repairs persistent state; old numerical scores are not comparable.
ENVIRONMENT_VERSION = 2


@dataclass(frozen=True)
class RewardConfig:
    failure_penalty: float = 1.65
    communication_weight: float = 0.20
    rotation_cost: float = 0.22
    budget_violation_base: float = 0.35
    budget_debt_weight: float = 0.05
    budget_debt_cap: float = 1.5


DEFAULT_REWARD = RewardConfig()


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "train_mix"
    horizon: int = 256
    budget_per_step: float = 0.72
    max_attack: int = 5
    detector_false_positive: float = 0.08
    detector_false_negative: float = 0.18
    persistence: float = 0.72
    rotate_clear_probability: float = 0.72
    adaptive_strength: float = 0.35
    regime_switch_probability: float = 0.0
    length_shift: float = 0.0
    importance_shift: float = 0.0
    attacker_count: int = 1


@dataclass(frozen=True)
class ScenarioChange:
    """An exogenous change applied before observing the specified round."""

    round: int
    attacker_count: int | None = None
    budget_per_step: float | None = None
    detector_false_negative: float | None = None

    def __post_init__(self):
        if self.round < 0:
            raise ValueError("change round must be non-negative")
        if self.attacker_count is not None and self.attacker_count < 1:
            raise ValueError("attacker_count must be at least one")
        if self.budget_per_step is not None and self.budget_per_step <= 0:
            raise ValueError("budget_per_step must be positive")
        if self.detector_false_negative is not None and not 0 <= self.detector_false_negative <= 1:
            raise ValueError("detector_false_negative must be in [0, 1]")


SCENARIOS: dict[str, ScenarioConfig] = {
    "train_mix": ScenarioConfig(),
    "mild": ScenarioConfig(
        name="mild", persistence=0.45, adaptive_strength=0.05,
        detector_false_negative=0.10, budget_per_step=0.55,
    ),
    "bursty": ScenarioConfig(
        name="bursty", persistence=0.60, adaptive_strength=0.15,
        budget_per_step=0.68,
    ),
    "persistent": ScenarioConfig(
        name="persistent", persistence=0.92, adaptive_strength=0.25,
        detector_false_negative=0.25, budget_per_step=0.78,
    ),
    "adaptive": ScenarioConfig(
        name="adaptive", persistence=0.80, adaptive_strength=0.80,
        detector_false_negative=0.25, regime_switch_probability=0.035,
        budget_per_step=0.82,
    ),
    "multi": ScenarioConfig(
        name="multi", persistence=0.78, max_attack=5,
        adaptive_strength=0.48, budget_per_step=0.92,
        importance_shift=0.08,
    ),
    "noisy_ood": ScenarioConfig(
        name="noisy_ood", persistence=0.82,
        detector_false_positive=0.28, detector_false_negative=0.42,
        adaptive_strength=0.55, regime_switch_probability=0.06,
        length_shift=0.20, budget_per_step=0.78,
    ),
    "budget_shock": ScenarioConfig(
        name="budget_shock", persistence=0.78, adaptive_strength=0.45,
        budget_per_step=0.50, regime_switch_probability=0.025,
    ),
    "multi_attacker_2": ScenarioConfig(
        name="multi_attacker_2", attacker_count=2, persistence=0.78,
        adaptive_strength=0.48, detector_false_negative=0.25,
        regime_switch_probability=0.025, budget_per_step=0.82,
    ),
    "multi_attacker_3": ScenarioConfig(
        name="multi_attacker_3", attacker_count=3, persistence=0.82,
        adaptive_strength=0.55, detector_false_negative=0.30,
        regime_switch_probability=0.035, budget_per_step=0.82,
    ),
    "multi_attacker_5": ScenarioConfig(
        name="multi_attacker_5", attacker_count=5, persistence=0.86,
        adaptive_strength=0.65, detector_false_negative=0.35,
        regime_switch_probability=0.045, budget_per_step=0.82,
    ),
}


def continuous_changes(config: ScenarioConfig) -> tuple[ScenarioChange, ...]:
    """Small held-out evaluation: attackers arrive, then leave; budget recovers."""
    if config.horizon < 3:
        raise ValueError("continuous evaluation needs at least three rounds")
    return (
        ScenarioChange(0, attacker_count=1),
        ScenarioChange(
            config.horizon // 3, attacker_count=3,
            budget_per_step=config.budget_per_step * 0.75,
            detector_false_negative=min(0.80, config.detector_false_negative + 0.15),
        ),
        ScenarioChange(
            2 * config.horizon // 3, attacker_count=1,
            budget_per_step=config.budget_per_step,
            detector_false_negative=config.detector_false_negative,
        ),
    )
