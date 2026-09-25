from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from phase_evonet.baselines import (
    _composition_statistics,
    _correct_balanced_probability,
    _select_and_gate,
    _weighted_average_precision_prepared,
)


def test_composition_statistics_are_deterministic_and_normalized() -> None:
    entropy, maximum, mean_z, std_z = _composition_statistics('{"Li": 2, "O": 1}')
    expected_entropy = -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)
    assert np.isclose(entropy, expected_entropy)
    assert np.isclose(maximum, 2 / 3)
    assert np.isclose(mean_z, (2 * 3 + 8) / 3)
    assert std_z > 0


def test_prior_correction_is_monotonic_and_restores_balanced_half_probability() -> None:
    values = _correct_balanced_probability(np.array([0.1, 0.5, 0.9]), 0.01)
    assert np.all(np.diff(values) > 0)
    assert np.isclose(values[1], 0.01)
    assert np.all((values >= 0) & (values <= 1))


def test_weighted_average_precision_matches_sklearn_with_ties() -> None:
    y = np.array([1, 0, 1, 0, 0], dtype=float)
    score = np.array([0.8, 0.8, 0.4, 0.2, 0.2], dtype=float)
    weight = np.array([2, 1, 3, 1, 4], dtype=float)
    expected = average_precision_score(y, score, sample_weight=weight)
    actual = _weighted_average_precision_prepared(y, score, weight)
    assert np.isclose(actual, expected)


def test_selection_and_relation_gate_use_frozen_validation_rules() -> None:
    metrics = pd.DataFrame(
        [
            {"model_name": "current_hull_distance_hgb", "role": "validation", "average_precision": 0.10, "brier": 0.02, "ece_10": 0.03, "top_decile_enrichment": 2.0},
            {"model_name": "static_phase_context_hgb", "role": "validation", "average_precision": 0.12, "brier": 0.019, "ece_10": 0.02, "top_decile_enrichment": 2.2},
        ]
    )
    config = {
        "population": {"validation_role": "validation"},
        "evaluation": {"strongest_baseline_order": ["highest_validation_average_precision"]},
        "gate": {
            "relation_model": "static_phase_context_hgb",
            "hull_reference_model": "current_hull_distance_hgb",
            "numerical_tolerance": 1e-12,
        },
    }
    selected, gate = _select_and_gate(metrics, config)
    assert selected["model_name"] == "static_phase_context_hgb"
    assert gate["passed"]
    assert gate["validation_average_precision_difference"] > 0
    assert gate["validation_brier_difference"] < 0
