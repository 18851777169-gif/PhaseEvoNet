from __future__ import annotations

import pandas as pd

from phase_evonet.reported_transitions import (
    classify_transition,
    deduplicate_reported_states,
    transition_id,
)


def thermo_row(source: str, row: int, stable: bool, hull: float) -> dict:
    return {
        "snapshot_id": "2022-10-28",
        "material_id": "mp-1",
        "thermo_id": "mp-1_GGA_GGA+U",
        "thermo_type": "GGA_GGA+U",
        "is_stable": stable,
        "energy_above_hull": hull,
        "formation_energy_per_atom": -1.25,
        "builder_database_version": "2022.10.28",
        "builder_run_id": None,
        "builder_emmet_version": "0.37.0",
        "builder_pymatgen_version": "2022.4.19",
        "source_object_sha256": source,
        "source_key": f"thermo/{source}.jsonl.gz",
        "source_row_number": row,
    }


def test_exact_reported_duplicates_are_ledgered_deterministically() -> None:
    frame = pd.DataFrame(
        [thermo_row("b" * 64, 2, True, 0.0), thermo_row("a" * 64, 1, True, 0.0)]
    )
    selected, duplicates, ambiguities = deduplicate_reported_states(frame)
    assert len(selected) == 1
    assert selected.iloc[0].source_object_sha256 == "a" * 64
    assert selected.iloc[0].source_record_count == 2
    assert bool(selected.iloc[0].usable)
    assert len(duplicates) == 1
    assert bool(duplicates.iloc[0].reported_values_equal)
    assert ambiguities == []


def test_conflicting_reported_values_are_not_usable() -> None:
    frame = pd.DataFrame(
        [thermo_row("a" * 64, 1, True, 0.0), thermo_row("b" * 64, 2, False, 0.02)]
    )
    selected, duplicates, ambiguities = deduplicate_reported_states(frame)
    assert not bool(selected.iloc[0].usable)
    assert not bool(duplicates.iloc[0].reported_values_equal)
    assert len(ambiguities) == 1


def test_transition_classification_preserves_censoring() -> None:
    assert classify_transition(True, True, True, True, True, False) == (
        "observed",
        "stable_to_unstable",
        True,
    )
    assert classify_transition(True, False, True, False, True, None) == (
        "source_only",
        None,
        None,
    )
    assert classify_transition(False, False, False, False, None, None) == (
        "no_reported_thermo",
        None,
        None,
    )


def test_transition_id_is_thermo_type_specific_and_deterministic() -> None:
    edge = bytes.fromhex("00" * 16)
    assert transition_id(edge, "R2SCAN") == transition_id(edge, "R2SCAN")
    assert transition_id(edge, "R2SCAN") != transition_id(edge, "GGA_GGA+U")
