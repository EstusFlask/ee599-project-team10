import numpy as np
import pytest

from experiment.rl.verify_effect import matched_value, select_pair


def test_matched_effect_uses_cost_interpolation_and_refuses_extrapolation():
    values = np.zeros((2, 3, 2, 2))
    values[0, ..., 0] = 0.5
    values[1, ..., 0] = 1.5
    values[0, ..., 1] = 0.2
    values[1, ..., 1] = 0.8
    np.testing.assert_allclose(matched_value(values), [1.0, 0.5])
    values[1, ..., 0] = 0.8
    assert matched_value(values) is None


def test_validation_selection_includes_efficient_baseline_portfolios():
    common = {"training_seed": 7, "budget_scale": 1.0, "gate": 0.5}
    points = [
        {**common, "policy": "svm", "mean_cost": 0.7, "weighted_success_rate": 0.6},
        {**common, "policy": "forest", "mean_cost": 1.3, "weighted_success_rate": 0.8},
    ]
    choice = select_pair(points)
    assert choice["low"]["policy"] == "svm"
    assert choice["high"]["policy"] == "forest"
    assert choice["validation_success_at_target"] == pytest.approx(0.7)


def test_cost_guards_do_not_replace_validation_optimal_primary_pair():
    values = np.zeros((4, 2, 2, 2))
    values[:, ..., 0] = np.asarray([0.5, 1.5, 0.7, 1.2])[:, None, None]
    values[:, ..., 1] = np.asarray([0.2, 0.8, 0.0, 0.0])[:, None, None]
    np.testing.assert_allclose(matched_value(values), [1.0, 0.5])
