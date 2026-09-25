from __future__ import annotations

import numpy as np
import pandas as pd

from phase_evonet.v2_capacity_matched import (
    _classification_metrics,
    _fit_calibrated,
    _semantically_equal,
    _weighted_ap,
    _weighted_ap_preparation,
    exposure_rescale_probability,
)


def _tiny_config() -> dict:
    return {
        "population": {"calibration_fold_unit": "canonical_lineage_id"},
        "shared_budget": {
            "learning_rate": 0.05,
            "max_leaf_nodes": 5,
            "min_samples_leaf": 2,
            "l2_regularization": 1.0,
            "max_bins": 32,
            "early_stopping": False,
            "class_weight": "balanced",
            "execution_seed": 42,
            "calibration_folds": 2,
            "calibration_C": 1000.0,
            "calibration_method": "sigmoid_lineage_group_crossfit_on_training_only",
        },
    }


def test_exposure_rescaling_is_monotone_and_identity() -> None:
    probability = np.asarray([0.0, 0.01, 0.2, 0.8])
    identity = exposure_rescale_probability(probability, 369, 369)
    longer = exposure_rescale_probability(probability, 369, 413)
    assert np.allclose(identity, probability, atol=1e-12)
    assert np.all(longer >= identity)
    assert np.all((longer >= 0) & (longer <= 1))


def test_weighted_average_precision_matches_unweighted_metric() -> None:
    y = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0])
    score = np.asarray([0.1, 0.9, 0.3, 0.8, 0.2])
    prepared = _weighted_ap_preparation(score)
    weighted = _weighted_ap(y, np.ones(len(y)), prepared)
    expected = _classification_metrics(y.astype(np.int8), score)["average_precision"]
    assert np.isclose(weighted, expected)


def test_group_crossfit_calibration_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(42)
    lineages = np.repeat([f"L{index:03d}" for index in range(40)], 2)
    x1 = rng.normal(size=len(lineages))
    x2 = rng.normal(size=len(lineages))
    event = ((x1 + 0.5 * x2 + rng.normal(scale=0.4, size=len(lineages))) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "canonical_lineage_id": lineages,
            "x1": x1,
            "x2": x2,
            "event": event,
        }
    )
    train = frame.iloc[:60].reset_index(drop=True)
    validation = frame.iloc[60:].reset_index(drop=True)
    first = _fit_calibrated(train, validation, "event", ["x1", "x2"], _tiny_config(), 8)
    second = _fit_calibrated(train, validation, "event", ["x1", "x2"], _tiny_config(), 8)
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert np.all((first[0] >= 0) & (first[0] <= 1))
    assert np.all((first[1] >= 0) & (first[1] <= 1))
    assert first[2]["calibration_fold_assignment_sha256"] == second[2]["calibration_fold_assignment_sha256"]


def test_group_crossfit_never_splits_a_lineage() -> None:
    rng = np.random.default_rng(7)
    lineages = np.repeat([f"G{index:03d}" for index in range(30)], 3)
    frame = pd.DataFrame(
        {
            "canonical_lineage_id": lineages,
            "x": rng.normal(size=len(lineages)),
            "event": np.tile([0, 1, 0], 30),
        }
    )
    oof, prediction, artifact = _fit_calibrated(
        frame,
        frame.iloc[:10].copy(),
        "event",
        ["x"],
        _tiny_config(),
        5,
    )
    assert len(oof) == len(frame)
    assert len(prediction) == 10
    assert artifact["training_lineages"] == 30
    assert artifact["calibration_folds"] == 2


def test_semantic_report_comparison_tolerates_csv_float_round_trip_only() -> None:
    assert _semantically_equal(
        {"value": 0.010257900781976656, "checks": {"passed": False}},
        {"value": 0.0102579007819766, "checks": {"passed": False}},
    )
    assert not _semantically_equal({"value": 0.1}, {"value": 0.2})
    assert not _semantically_equal({"passed": False}, {"passed": 0})
