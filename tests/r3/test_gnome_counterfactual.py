from __future__ import annotations

import json

import pandas as pd
import pytest

from phase_evonet.r3.competitor_cascade import solve_hull_distance
from phase_evonet.r3.gnome_counterfactual import (
    SCOPES,
    _retained_value_hash,
    _threshold_reversal,
    freeze_matched_design,
    select_primary_population,
)


def _entry(uid: bytes, composition: dict[str, float], energy: float) -> dict[str, object]:
    return {
        "unified_entry_id": uid,
        "composition_json": json.dumps(composition),
        "num_atoms": sum(composition.values()),
        "uncorrected_energy": energy,
        "correction": 0.0,
        "corrected_energy": energy,
        "corrected_energy_per_atom": energy / sum(composition.values()),
        "source_workflow": "R2SCAN",
        "is_competitor": True,
        "energy_above_hull": 0.0,
    }


def test_removing_competitor_reverses_only_database_state_label():
    candidate = _entry(b"c" * 16, {"Li": 1, "O": 1}, -1.8)
    elemental_li = _entry(b"l" * 16, {"Li": 1}, 0.0)
    elemental_o = _entry(b"o" * 16, {"O": 1}, 0.0)
    arrival = _entry(b"a" * 16, {"Li": 1, "O": 1}, -2.0)
    full, status, _, _ = solve_hull_distance(
        candidate, [elemental_li, elemental_o, arrival], ["Li", "O"]
    )
    removed, removed_status, _, _ = solve_hull_distance(
        candidate, [elemental_li, elemental_o], ["Li", "O"]
    )
    assert status == removed_status == "feasible"
    assert full == pytest.approx(0.1)
    assert removed == pytest.approx(0.0)
    assert _threshold_reversal(full, removed, 0.010, 1e-8)


def test_primary_scope_never_includes_c_or_u():
    assert SCOPES["A_ONLY"] == {"A"}
    assert SCOPES["A_PLUS_B"] == {"A", "B"}
    assert not (SCOPES["A_ONLY"] | SCOPES["A_PLUS_B"]) & {"C", "U"}


def test_threshold_boundaries_are_frozen():
    assert _threshold_reversal(0.02, 0.009999, 0.010, 1e-8)
    assert not _threshold_reversal(0.02, 0.010, 0.010, 1e-8)
    assert _threshold_reversal(0.02, 0.0, 0.0, 1e-8)
    assert not _threshold_reversal(0.02, 2e-8, 0.0, 1e-8)


def test_retained_value_hash_is_order_independent_and_energy_sensitive():
    first = _entry(b"a" * 16, {"Li": 1}, -1.0)
    second = _entry(b"b" * 16, {"O": 1}, -2.0)
    assert _retained_value_hash([first, second]) == _retained_value_hash([second, first])
    changed = dict(second)
    changed["correction"] = 0.1
    assert _retained_value_hash([first, second]) != _retained_value_hash([first, changed])


def test_primary_population_selection_is_strict():
    base = {
        "transition_id": b"1" * 16,
        "direction": "stable_to_unstable",
        "survives_10meV": True,
        "identity_confidence": "A1",
        "candidate_identity_unchanged": True,
        "same_workflow": True,
        "same_phase_context": True,
        "target_unified_entry_id": b"2" * 16,
    }
    frame = pd.DataFrame([base, {**base, "transition_id": b"3" * 16, "same_workflow": False}])
    selected = select_primary_population(frame, expected_rows=1)
    assert selected.transition_id.tolist() == [(b"1" * 16).hex()]


def test_matching_is_deterministic_and_uses_only_preoutcome_covariates():
    relation = pd.DataFrame(
        [
            {
                "transition_id": "t1", "competitor_contextual_id": "a", "change_type": "target_only_arrival",
                "target_unified_entry_id_hex": "01" * 16, "target_snapshot": "2025-09-25",
                "thermo_type_change": "R2SCAN", "phase_context_chemsys_change": "Li-O",
                "provenance_class": "A", "raw_record_found": True,
                "raw_provenance_present": True, "source_manifest_match": True,
            },
            {
                "transition_id": "t1", "competitor_contextual_id": "c", "change_type": "target_only_arrival",
                "target_unified_entry_id_hex": "02" * 16, "target_snapshot": "2025-09-25",
                "thermo_type_change": "R2SCAN", "phase_context_chemsys_change": "Li-O",
                "provenance_class": "U", "raw_record_found": True,
                "raw_provenance_present": True, "source_manifest_match": True,
            },
        ]
    )
    primary = pd.DataFrame(
        [{
            "transition_id": "t1", "source_snapshot": "2024-12-18",
            "target_snapshot": "2025-09-25", "thermo_type": "R2SCAN",
            "phase_context_chemsys": "Li-O",
        }]
    )
    target_key = ("2025-09-25", "R2SCAN", "Li-O")
    source_key = ("2024-12-18", "R2SCAN", "Li-O")
    context_cache = {
        target_key: [
            {**_entry(bytes.fromhex("01" * 16), {"Li": 1, "O": 1}, -2.0), "is_competitor": True},
            {**_entry(bytes.fromhex("02" * 16), {"Li": 1, "O": 1}, -1.9), "is_competitor": True},
        ]
    }
    contexts = {source_key: [{"is_competitor": True}, {"is_competitor": False}]}
    config = {
        "counterfactual": {"exact_zero_tolerance_eV_per_atom": 1e-8},
        "population": {"primary_threshold_eV_per_atom": 0.01},
        "matching": {
            "prohibited_variables": ["cascade_size", "removal_effect"],
            "exact_strata": ["target_snapshot", "thermo_type", "chemical_dimensionality", "entry_stability_class_at_arrival"],
            "numeric_covariates": ["composition_complexity", "source_context_entry_count", "source_context_competitor_count", "provenance_completeness"],
            "distance_caliper": 4.0,
            "with_replacement": True,
        },
    }
    first, _ = freeze_matched_design(relation, primary, context_cache, contexts, config)
    second, _ = freeze_matched_design(relation, primary, context_cache, contexts, config)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "matched"]
    config["matching"]["numeric_covariates"].append("removal_effect")
    with pytest.raises(RuntimeError, match="prohibited"):
        freeze_matched_design(relation, primary, context_cache, contexts, config)
