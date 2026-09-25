import numpy as np
import pandas as pd

from phase_evonet.r3.versioned_benchmark import (
    _weighted_ap,
    ece_equal_width,
    point_metrics,
    prediction_vector_hash,
    resolve_budget,
)


def test_budget_resolution_is_frozen_and_ceiling_based():
    assert resolve_budget(100, 50) == 50
    assert resolve_budget("top_1pct", 101) == 2
    assert resolve_budget("top_5pct", 101) == 6


def test_ece_equal_width_handles_probability_one_in_final_bin():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert ece_equal_width(y, p) == 0.0


def test_prediction_hash_is_order_independent_and_byte_sensitive():
    frame = pd.DataFrame({"panel_unit_id": ["b", "a"], "probability": [0.2, 0.4]})
    observed = prediction_vector_hash(frame)
    assert observed == prediction_vector_hash(frame.iloc[::-1])
    frame.loc[0, "probability"] += 1e-14
    assert observed != prediction_vector_hash(frame)


def test_point_metrics_use_lexical_tie_break_for_budgets():
    frame = pd.DataFrame({
        "panel_unit_id": ["b", "a", "c"], "model_id": ["M0"] * 3,
        "label_definition": ["exact_zero"] * 3, "label_version": ["v1"] * 3,
        "event": [False, True, False], "probability": [0.5, 0.5, 0.1],
        "energy_change_eV_per_atom": [0.0, 1.0, -1.0],
    })
    rows = pd.DataFrame(point_metrics(frame))
    top = rows[(rows.metric == "precision_at_budget") & (rows.budget == "100")].iloc[0]
    assert top.resolved_budget == 3
    one = frame.iloc[:0].copy()
    assert len(rows) == 15


def test_point_metrics_binary_core_is_finite():
    frame = pd.DataFrame({
        "panel_unit_id": ["a", "b", "c", "d"], "model_id": ["M0"] * 4,
        "label_definition": ["exact_zero"] * 4, "label_version": ["v1"] * 4,
        "event": [False, True, False, True], "probability": [0.1, 0.9, 0.2, 0.8],
        "energy_change_eV_per_atom": [-1.0, 1.0, -0.5, 0.5],
    })
    rows = pd.DataFrame(point_metrics(frame))
    core = rows[rows.metric.isin(["average_precision", "brier_score", "roc_auc", "expected_calibration_error"])]
    assert np.isfinite(core.value).all()


def test_weighted_ap_keeps_equal_score_rows_in_one_threshold_group():
    y = np.array([False, True, False, True])
    score = np.full(4, 0.2)
    weights = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]])
    observed = _weighted_ap(y, score, np.array(["a", "b", "c", "d"]), weights)
    assert np.allclose(observed, [0.5, 0.25])
