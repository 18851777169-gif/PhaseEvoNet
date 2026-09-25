import pandas as pd
import pytest

from phase_evonet.r3.versioned_benchmark_protocol import (
    assert_label_free_columns,
    canonical_panel_id,
    canonical_prediction_vector_hash,
    deterministic_top_k,
    exact_model_intersection,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"canonical_lineage_id": lineage, "source_snapshot": "2023-11-01", "thermo_type": "R2SCAN", "model_name": model, "probability": score}
            for lineage, score in (("lineage-a", 0.4), ("lineage-b", 0.2))
            for model in ("M0", "M1")
        ]
    )


def test_panel_id_is_deterministic_and_workflow_partitioned():
    one = canonical_panel_id("lineage-a", "2023-11-01", "R2SCAN")
    assert one == canonical_panel_id("lineage-a", "2023-11-01", "R2SCAN")
    assert one != canonical_panel_id("lineage-a", "2023-11-01", "GGA_GGA+U")


def test_exact_model_intersection_requires_complete_unique_scores():
    frame = _predictions()
    selected = exact_model_intersection(frame, ("M0", "M1"))
    assert len(selected) == 4
    with pytest.raises(ValueError, match="not unique"):
        exact_model_intersection(pd.concat([frame, frame.iloc[[0]]]), ("M0", "M1"))


def test_exact_model_intersection_removes_incomplete_candidates():
    frame = _predictions().drop(index=3)
    selected = exact_model_intersection(frame, ("M0", "M1"))
    assert selected["canonical_lineage_id"].unique().tolist() == ["lineage-a"]


def test_prediction_vector_hash_is_order_independent_but_score_sensitive():
    frame = pd.DataFrame({"panel_unit_id": ["b", "a"], "probability": [0.2, 0.4]})
    digest = canonical_prediction_vector_hash(frame)
    assert digest == canonical_prediction_vector_hash(frame.iloc[::-1])
    changed = frame.copy()
    changed.loc[0, "probability"] = 0.2000000000001
    assert digest != canonical_prediction_vector_hash(changed)


def test_label_columns_are_rejected_but_evaluation_method_is_not():
    assert_label_free_columns(["panel_unit_id", "evaluation_method", "probability"])
    for column in ("event", "future_stability", "rebuilt_unified_flip", "stability_label"):
        with pytest.raises(ValueError, match="prohibited"):
            assert_label_free_columns(["panel_unit_id", column])


def test_top_k_ties_use_lexical_panel_id():
    frame = pd.DataFrame({"panel_unit_id": ["b", "a", "c"], "probability": [0.5, 0.5, 0.1]})
    assert deterministic_top_k(frame, 2)["panel_unit_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="nonnegative"):
        deterministic_top_k(frame, -1)
