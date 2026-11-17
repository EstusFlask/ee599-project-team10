"""Optional training-only adjustment of one shared communication-cost price."""

from dataclasses import dataclass, field
import math

from stable_baselines3.common.callbacks import BaseCallback


DEFAULT_BUDGET_LEARNING_RATE = 0.005


@dataclass
class BudgetController:
    weight: float = 0.15
    learning_rate: float = DEFAULT_BUDGET_LEARNING_RATE
    max_weight: float = 5.0
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.weight, self.learning_rate, self.max_weight
        )):
            raise ValueError("budget settings must be finite")
        if self.learning_rate <= 0 or self.max_weight <= 0:
            raise ValueError("budget learning rate and maximum weight must be positive")
        if not 0 <= self.weight <= self.max_weight:
            raise ValueError("initial budget weight must be between zero and max_weight")

    def update(self, mean_cost: float, target_cost: float) -> float:
        self.weight = min(self.max_weight, max(
            0.0, self.weight + self.learning_rate * (mean_cost - target_cost)
        ))
        self.history.append({
            "mean_cost": float(mean_cost), "target_cost": float(target_cost),
            "weight": self.weight,
        })
        return self.weight


class BudgetPenaltyCallback(BaseCallback):
    def __init__(self, controller: BudgetController, warmup_steps: int = 0):
        super().__init__()
        self.controller = controller
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("budget warmup steps must be non-negative")

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_constraint_weight", self.controller.weight)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            return True
        metrics = [
            info["episode_metrics"] for info in self.locals.get("infos", [])
            if "episode_metrics" in info
        ]
        if metrics:
            mean_cost = sum(item["mean_cost"] for item in metrics) / len(metrics)
            target = sum(item["target_mean_cost"] for item in metrics) / len(metrics)
            weight = self.controller.update(mean_cost, target)
            self.training_env.env_method("set_constraint_weight", weight)
            self.logger.record("budget/weight", weight)
            self.logger.record("budget/mean_cost", mean_cost)
            self.logger.record("budget/target_cost", target)
        return True
