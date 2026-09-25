from __future__ import annotations

import pandas as pd

from phase_evonet.r3.competitor_identity_resolution import (
    COMPOSITION_ATOL,
    build_crosswalk,
    composition_delta,
    compositions_concordant,
    synthetic_bootstrap_diagnostics,
)


def _lineage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "canonical_lineage_id": "L1",
                "snapshot_id": "s1",
                "material_id": "m1",
                "composition_reduced_json": '{"A":1.0,"B":2.0}',
                "lineage_confidence": "A1",
                "high_confidence": True,
            },
            {
                "canonical_lineage_id": "L1",
                "snapshot_id": "s2",
                "material_id": "m2",
                "composition_reduced_json": '{"A":1.0,"B":2.0}',
                "lineage_confidence": "A1",
                "high_confidence": True,
            },
            {
                "canonical_lineage_id": "L2",
                "snapshot_id": "s2",
                "material_id": "m3",
                "composition_reduced_json": '{"A":1.0,"B":2.0}',
                "lineage_confidence": "C",
                "high_confidence": False,
            },
        ]
    )


def _change(**updates: object) -> pd.DataFrame:
    row = {
        "transition_id": bytes.fromhex("01" * 16),
        "candidate_lineage_id": "candidate",
        "competitor_contextual_id": "context",
        "competitor_canonical_id": "strict",
        "source_snapshot": "s1",
        "target_snapshot": "s2",
        "source_material_id": "m1",
        "target_material_id": "m2",
        "source_task_id": "t1",
        "target_task_id": "t2",
        "source_entry_id": "e1",
        "target_entry_id": "e2",
        "source_workflow": "GGA",
        "target_workflow": "R2SCAN",
        "thermo_type": "mixed",
        "composition_signature": '{"A":0.333333333333,"B":0.666666666667}',
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_composition_uses_fractional_tolerance_not_string_equality() -> None:
    left = '{"A":0.333333333333,"B":0.666666666667}'
    right = '{"A":1.0,"B":2.0}'
    assert composition_delta(left, right) < COMPOSITION_ATOL
    assert compositions_concordant(left, right)


def test_direct_target_side_mapping_and_cross_side_concordance() -> None:
    crosswalk, audit = build_crosswalk(_change(), _lineage())
    row = crosswalk.iloc[0]
    assert row.identity_side == "target"
    assert row.identity_snapshot == "s2"
    assert row.material_id == "m2"
    assert row.p2_canonical_lineage_id == "L1"
    assert bool(row.cross_side_lineage_concordant)
    assert row.mapping_status == "mapped"
    assert audit["maximum_join_expansion"] == 1.0


def test_source_side_is_used_when_target_material_is_absent() -> None:
    frame = _change(target_material_id=None, target_task_id=None, target_entry_id=None, target_workflow=None)
    crosswalk, _ = build_crosswalk(frame, _lineage())
    assert crosswalk.iloc[0].identity_side == "source"
    assert crosswalk.iloc[0].material_id == "m1"
    assert crosswalk.iloc[0].mapping_method == "direct_material"


def test_task_bridge_is_only_used_when_selected_material_is_missing() -> None:
    frame = _change(
        source_material_id=None,
        target_material_id=None,
        target_task_id=None,
        target_entry_id=None,
        target_workflow=None,
    )
    bridge = pd.DataFrame([{"snapshot_id": "s1", "task_id": "t1", "material_id": "m1"}])
    crosswalk, audit = build_crosswalk(frame, _lineage(), task_bridge=bridge)
    assert crosswalk.iloc[0].mapping_method == "task_bridge"
    assert crosswalk.iloc[0].mapping_status == "mapped"
    assert audit["task_bridge_mapped"] == 1


def test_ambiguous_task_bridge_is_isolated_not_imputed() -> None:
    frame = _change(
        source_material_id=None,
        target_material_id=None,
        target_task_id=None,
        target_entry_id=None,
        target_workflow=None,
    )
    bridge = pd.DataFrame(
        [
            {"snapshot_id": "s1", "task_id": "t1", "material_id": "m1"},
            {"snapshot_id": "s1", "task_id": "t1", "material_id": "m9"},
        ]
    )
    crosswalk, audit = build_crosswalk(frame, _lineage(), task_bridge=bridge)
    assert crosswalk.iloc[0].mapping_status == "conflict"
    assert "ambiguous_task_bridge" in crosswalk.iloc[0].unresolved_reason
    assert audit["task_bridge_ambiguous"] == 1


def test_cross_side_lineage_disagreement_is_a_conflict() -> None:
    frame = _change(target_material_id="m3")
    crosswalk, audit = build_crosswalk(frame, _lineage())
    assert crosswalk.iloc[0].mapping_status == "conflict"
    assert crosswalk.iloc[0].cross_side_lineage_concordant is False or not bool(
        crosswalk.iloc[0].cross_side_lineage_concordant
    )
    assert audit["cross_side_disagreements"] == 1


def test_id2_partitions_lineage_by_thermo_and_workflow() -> None:
    first, _ = build_crosswalk(_change(), _lineage())
    second, _ = build_crosswalk(_change(target_workflow="GGA"), _lineage())
    third, _ = build_crosswalk(_change(thermo_type="GGA"), _lineage())
    assert first.iloc[0].id2_lineage_thermo_workflow != second.iloc[0].id2_lineage_thermo_workflow
    assert first.iloc[0].id2_lineage_thermo_workflow != third.iloc[0].id2_lineage_thermo_workflow
    assert first.iloc[0].id3_lineage_only == second.iloc[0].id3_lineage_only == third.iloc[0].id3_lineage_only


def test_synthetic_contextual_bootstrap_diagnostic_is_deterministic() -> None:
    first = synthetic_bootstrap_diagnostics(seed=42)
    second = synthetic_bootstrap_diagnostics(seed=42)
    pd.testing.assert_frame_equal(first, second)
    one = first.set_index("graph").loc["one_to_one_contextual"]
    assert one.full_synthetic_gini == 0.0
    assert one.without_replacement_gini_min == 0.0
    assert one.without_replacement_gini_max == 0.0
    assert one.old_multinomial_gini_low > 0.0
    assert not bool(one.full_point_inside_old_interval)
    assert bool(first.diagnostic_pass.all())
