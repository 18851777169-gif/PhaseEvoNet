from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from phase_evonet.v2_development import (
    _context_pair_tables,
    _entry_conversion,
    _material_conversion,
    _task_conversion,
    build_cause_specific_targets,
    build_interval_exposure,
    feature_sets,
)


def _tid(value: int) -> bytes:
    return value.to_bytes(16, "big")


def test_official_identifier_roundtrips_preserve_numeric_identity() -> None:
    material = _material_conversion("mp-43")
    task = _task_conversion("mp-1476022")
    entry = _entry_conversion("mp-43-GGA+U")
    alpha_material = _material_conversion(material["canonical_alpha_material_id"])
    alpha_task = _task_conversion(task["canonical_alpha_task_id"])
    alpha_entry = _entry_conversion(entry["canonical_alpha_entry_id"])
    assert material["canonical_material_numeric_id"] == 43
    assert alpha_material["canonical_material_numeric_id"] == 43
    assert task["canonical_task_numeric_id"] == alpha_task["canonical_task_numeric_id"]
    assert entry["canonical_alpha_entry_id"] == alpha_entry["canonical_alpha_entry_id"]


def test_interval_exposure_uses_actual_frozen_days() -> None:
    labels = pd.DataFrame(
        {
            "transition_id": [_tid(1), _tid(2)],
            "canonical_lineage_id": ["L1", "L2"],
            "assigned_role": ["train", "validation"],
            "source_snapshot": ["2022-10-28", "2023-11-01"],
            "target_snapshot": ["2023-11-01", "2024-12-18"],
        }
    )
    allowed = [
        {
            "role": "train",
            "source_snapshot": "2022-10-28",
            "target_snapshot": "2023-11-01",
            "exposure_days": 369,
        },
        {
            "role": "validation",
            "source_snapshot": "2023-11-01",
            "target_snapshot": "2024-12-18",
            "exposure_days": 413,
        },
    ]
    result = build_interval_exposure(labels, allowed)
    assert set(result["exposure_days"]) == {369, 413}
    assert (result["exposure_months"] > 12).all()
    assert (result["log_exposure_years_offset"].abs() < 0.13).all()


def test_context_history_never_uses_target_as_source_feature() -> None:
    stats = pd.DataFrame(
        [
            ("2022-10-28", "GGA_GGA+U", "Li-O", 2, 1, 1, 1),
            ("2023-11-01", "GGA_GGA+U", "Li-O", 3, 2, 1, 2),
            ("2024-12-18", "GGA_GGA+U", "Li-O", 4, 3, 2, 3),
        ],
        columns=[
            "snapshot_id",
            "thermo_type",
            "phase_context_chemsys",
            "context_entry_count",
            "context_competitor_count",
            "context_stable_count",
            "context_near_hull_10meV_count",
        ],
    )
    contexts = pd.DataFrame(
        {
            "source_snapshot": ["2022-10-28", "2023-11-01"],
            "thermo_type": ["GGA_GGA+U", "GGA_GGA+U"],
            "phase_context_chemsys": ["Li-O", "Li-O"],
        }
    )
    sets = {
        ("2022-10-28", "GGA_GGA+U", "Li-O"): frozenset({"a"}),
        ("2023-11-01", "GGA_GGA+U", "Li-O"): frozenset({"a", "b"}),
        ("2024-12-18", "GGA_GGA+U", "Li-O"): frozenset({"a", "b", "c"}),
    }
    history, targets, audit = _context_pair_tables(
        contexts,
        stats,
        sets,
        {"2022-10-28": "2023-11-01", "2023-11-01": "2024-12-18"},
        {"2022-10-28": None, "2023-11-01": "2022-10-28"},
        {"2022-10-28": 0, "2023-11-01": 369},
    )
    first = history.loc[history["source_snapshot"] == "2022-10-28"].iloc[0]
    second = history.loc[history["source_snapshot"] == "2023-11-01"].iloc[0]
    assert first["source_history_snapshot_available"] == 0
    assert first["source_competitor_arrivals_since_previous"] == 0
    assert second["source_prior_context_competitor_count"] == 1
    assert second["source_competitor_arrivals_since_previous"] == 1
    assert "target_snapshot" not in history.columns
    assert targets["competitor_arrival_count"].tolist() == [1, 1]
    assert audit["source_context_missing"] == 0
    assert audit["target_context_missing"] == 0


def test_cause_specific_targets_are_separate_and_lineage_aligned() -> None:
    labels = pd.DataFrame(
        {
            "transition_id": [_tid(1), _tid(2), _tid(3)],
            "canonical_lineage_id": ["L1", "L2", "L3"],
            "assigned_role": ["train", "train", "train"],
            "identity_confidence": ["A1", "A1", "A1"],
            "source_snapshot": ["2022-10-28"] * 3,
            "target_snapshot": ["2023-11-01"] * 3,
            "source_reported_is_stable": [True, False, True],
            "target_reported_is_stable": [False, True, True],
            "source_reported_energy_above_hull": [0.0, 0.1, 0.0],
            "target_reported_energy_above_hull": [0.1, 0.0, 0.0],
            "delta_reported_energy_above_hull": [0.1, -0.1, 0.0],
            "source_reported_formation_energy_per_atom": [-1.0, -1.0, -1.0],
            "target_reported_formation_energy_per_atom": [-0.9, -1.1, -1.0],
            "delta_reported_formation_energy_per_atom": [0.1, -0.1, 0.0],
            "rebuilt_unified_flip": [True, True, False],
            "rebuilt_unified_label_transition": [
                "stable_to_unstable",
                "unstable_to_stable",
                "no_flip",
            ],
            "attribution_id": [_tid(101), _tid(102), None],
        }
    )
    source = pd.DataFrame(
        {
            "transition_id": [_tid(1), _tid(2), _tid(3)],
            "canonical_lineage_id": ["L1", "L2", "L3"],
            "assigned_role": ["train"] * 3,
            "source_snapshot": ["2022-10-28"] * 3,
            "thermo_type": ["GGA_GGA+U"] * 3,
            "phase_context_chemsys": ["Li-O"] * 3,
            "source_is_stable": [1, 0, 1],
        }
    )
    context = pd.DataFrame(
        {
            "source_snapshot": ["2022-10-28"],
            "target_snapshot": ["2023-11-01"],
            "thermo_type": ["GGA_GGA+U"],
            "phase_context_chemsys": ["Li-O"],
            "source_competitor_inventory_count": [2],
            "target_competitor_inventory_count": [3],
            "competitor_arrival_count": [1],
            "competitor_removal_count": [0],
            "hull_relevant_competitor_arrival_event": [True],
            "competitor_inventory_revision_event": [True],
        }
    )
    attribution = pd.DataFrame(
        {
            "attribution_id": [_tid(101), _tid(102)],
            "transition_id": [_tid(1), _tid(2)],
            "source_snapshot": ["2022-10-28"] * 2,
            "target_snapshot": ["2023-11-01"] * 2,
            "competitor_inventory_contribution": [0.1, 0.0],
            "uncorrected_energy_contribution": [0.0, -0.1],
            "compatibility_correction_contribution": [0.0, 0.0],
            "candidate_identity_contribution": [0.0, 0.0],
            "dominant_channel": ["competitor_inventory", "uncorrected_energy"],
            "candidate_identity_changed": [False, False],
            "source_decomposition_phase_keys_json": [json.dumps(["a"]), json.dumps(["b"])],
            "target_decomposition_phase_keys_json": [json.dumps(["a", "c"]), json.dumps(["b"])],
        }
    )
    result, counts = build_cause_specific_targets(
        labels, source, context, attribution, 1.0e-12
    )
    assert counts["stable_to_unstable_events"] == 1
    assert counts["unstable_to_stable_events"] == 1
    assert counts["attribution_rows"] == 2
    assert result["conditional_displacement_eligible"].sum() == 2
    assert result.loc[result["transition_id"] == _tid(1), "stable_to_unstable_competing_cause"].item() == "competitor_inventory"
    assert result.loc[result["transition_id"] == _tid(3), "decomposition_set_change_event"].isna().item()


def test_model_matrix_is_capacity_matched_and_training_disabled() -> None:
    matrix = yaml.safe_load(
        Path("configs/analysis/v2_1_model_matrix.yaml").read_text(encoding="utf-8")
    )
    assert matrix["training_authorized"] is False
    assert set(matrix["models"]) == {"M0", "M1", "M2", "M3", "M4", "M5"}
    shared = matrix["shared_budget"]
    for name, model in matrix["models"].items():
        assert model["algorithm_class"] == shared["algorithm_class"]
        assert model["effective_total_boosting_iterations"] == shared["effective_total_boosting_iterations"]
    sets = feature_sets()
    assert set(sets["M3"]) < set(sets["M4"])
    assert set(sets["M2"]) < set(sets["M4"])
    assert "source_energy_above_hull" in sets["M0"]
    assert all(not name.startswith("target_") for names in sets.values() for name in names)
