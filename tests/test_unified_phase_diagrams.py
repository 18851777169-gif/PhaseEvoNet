from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

import phase_evonet.context_phase_diagrams as context_phase_diagrams
from phase_evonet.context_phase_diagrams import (
    _AuditedMixingScheme,
    _StructureEntryCache,
    _has_exact_r2scan_target,
    _mixed_context_assignment,
    _mixed_shard_paths,
    _reference_audit,
    _reconcile_reported_energy,
    _resolve_mixing_duplicate_alias,
    contextual_entry_id,
    solve_homogeneous_contexts,
)
from phase_evonet.unified_phase_diagrams import deduplicate_entries


def entry_row(
    entry_id: str,
    composition: dict[str, float],
    energy: float,
    *,
    correction: float = 0.0,
    source: str = "a",
    row: int = 1,
) -> dict:
    return {
        "snapshot_id": "synthetic-1",
        "material_id": f"material-{entry_id}",
        "thermo_id": f"thermo-{entry_id}",
        "thermo_type": "SYNTHETIC_COMPATIBILITY",
        "entry_label": "synthetic",
        "entry_id": entry_id,
        "task_id": entry_id,
        "energy": energy,
        "correction": correction,
        "composition_json": json.dumps(composition, sort_keys=True),
        "energy_adjustments_json": "[]",
        "parameters_json": "{}",
        "run_type": "synthetic",
        "hubbards_json": "{}",
        "potcar_spec_json": "[]",
        "entry_data_json": "{}",
        "source_object_sha256": source * 64,
        "source_key": f"synthetic/{source}.jsonl",
        "source_row_number": row,
    }


def test_unified_entry_id_is_context_specific_and_deterministic() -> None:
    first = contextual_entry_id("s1", "workflow-a", "Li-F", "entry-1")
    assert len(first) == 16
    assert first == contextual_entry_id("s1", "workflow-a", "Li-F", "entry-1")
    assert first != contextual_entry_id("s2", "workflow-a", "Li-F", "entry-1")
    assert first != contextual_entry_id("s1", "workflow-b", "Li-F", "entry-1")
    assert first != contextual_entry_id("s1", "workflow-a", "Li", "entry-1")


def test_empty_oxidation_state_list_is_normalized_for_compatibility() -> None:
    row = entry_row("Li", {"Li": 1}, -1.0)
    row["parameters_json"] = json.dumps({"run_type": "GGA"})
    row["entry_data_json"] = json.dumps({"oxidation_states": []})
    row["structure_json"] = json.dumps(
        Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]]).as_dict()
    )
    cache = _StructureEntryCache(limit=1)
    entry = cache.get(row)
    assert entry.data["oxidation_states"] == {}
    assert cache.warning_counts == {
        "empty oxidation_states list normalized to empty mapping": 1
    }


def test_nonempty_oxidation_state_list_is_rejected() -> None:
    row = entry_row("Li", {"Li": 1}, -1.0)
    row["parameters_json"] = json.dumps({"run_type": "GGA"})
    row["entry_data_json"] = json.dumps({"oxidation_states": [1]})
    row["structure_json"] = json.dumps(
        Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]]).as_dict()
    )
    with pytest.raises(ValueError, match="oxidation_states list must be empty"):
        _StructureEntryCache(limit=1).get(row)


def test_exact_duplicates_are_selected_once_and_fully_ledgered() -> None:
    frame = pd.DataFrame(
        [
            entry_row("Li", {"Li": 1}, -1.1, correction=0.1, source="b", row=2),
            entry_row("Li", {"Li": 1}, -1.0, correction=0.0, source="a", row=1),
        ]
    )
    selected, duplicates, ambiguities = deduplicate_entries(frame)
    assert len(selected) == 1
    assert selected.iloc[0].source_object_sha256 == "a" * 64
    assert selected.iloc[0].corrected_energy == -1.0
    assert selected.iloc[0].source_record_count == 2
    assert not bool(selected.iloc[0].duplicate_conflict)
    assert len(duplicates) == 1
    assert bool(duplicates.iloc[0].corrected_energy_equal)
    assert ambiguities == []


def test_conflicting_duplicates_are_explicitly_ambiguous() -> None:
    frame = pd.DataFrame(
        [
            entry_row("Li", {"Li": 1}, -1.0, source="a"),
            entry_row("Li", {"Li": 1}, -0.9, source="b", row=2),
        ]
    )
    selected, duplicates, ambiguities = deduplicate_entries(frame)
    assert bool(selected.iloc[0].duplicate_conflict)
    assert not bool(duplicates.iloc[0].corrected_energy_equal)
    assert duplicates.iloc[0].reason == "conflicting_duplicate"
    assert len(ambiguities) == 1


def test_binary_reference_phase_diagram_and_decomposition() -> None:
    raw = pd.DataFrame(
        [
            entry_row("Li", {"Li": 1}, -1.0),
            entry_row("F", {"F": 1}, -2.0, row=2),
            entry_row("LiF-stable", {"Li": 1, "F": 1}, -4.0, row=3),
            entry_row("LiF-unstable", {"Li": 2, "F": 2}, -7.0, row=4),
        ]
    )
    selected, _, _ = deduplicate_entries(raw)
    selected["_space"] = selected["composition_json"].map(
        lambda value: tuple(sorted(json.loads(value)))
    )
    phase, decomposition, errors, _ = solve_homogeneous_contexts(
        selected, stable_tolerance=1e-8
    )
    assert errors == []
    indexed = phase[phase["is_target"]].set_index("entry_id")
    assert indexed.loc["LiF-stable", "energy_above_hull"] == 0.0
    assert indexed.loc["LiF-stable", "formation_energy_per_atom"] == -0.5
    assert indexed.loc["LiF-unstable", "energy_above_hull"] == 0.25
    assert indexed.loc["LiF-unstable", "phase_context_chemsys"] == "F-Li"
    unstable_id = indexed.loc["LiF-unstable", "unified_entry_id"]
    unstable_decomposition = decomposition[decomposition["unified_entry_id"] == unstable_id]
    assert len(unstable_decomposition) == 1
    assert unstable_decomposition.iloc[0].component_entry_id == "LiF-stable"
    assert unstable_decomposition.iloc[0].amount == 1.0


def test_mixed_r2scan_detection_requires_exact_selected_target() -> None:
    rows = [
        {
            "entry_label": "R2SCAN",
            "energy_type": "R2SCAN",
            "chemsys": "F-Li-O",
        },
        {
            "entry_label": "GGA",
            "energy_type": "R2SCAN",
            "chemsys": "F-Li",
        },
    ]
    assert not _has_exact_r2scan_target(rows, "F-Li")
    rows.append(
        {
            "entry_label": "R2SCAN",
            "energy_type": "R2SCAN",
            "chemsys": "F-Li",
        }
    )
    assert _has_exact_r2scan_target(rows, "F-Li")


def test_reported_energy_reconciliation_is_general_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected, _, _ = deduplicate_entries(
        pd.DataFrame([entry_row("Li2", {"Li": 2}, -2.0, correction=-0.2)])
    )
    selected["_space"] = [("Li",)]
    reported = pd.DataFrame(
        [
            {
                "thermo_id": "thermo-Li2",
                "thermo_type": "SYNTHETIC_COMPATIBILITY",
                "energy_type": "synthetic",
                "chemsys": "Li",
                "energy_per_atom": -1.11,
                "uncorrected_energy_per_atom": -1.01,
            }
        ]
    )
    monkeypatch.setattr(context_phase_diagrams, "_thermo_selection", lambda *_: reported)
    resolved, errors, ledger = _reconcile_reported_energy(
        root=Path("."), snapshot="synthetic-1",
        thermo_type="SYNTHETIC_COMPATIBILITY", selected=selected,
        tolerance=1e-6,
    )
    assert errors == []
    assert resolved.iloc[0].energy == pytest.approx(-2.02)
    assert resolved.iloc[0].corrected_energy == pytest.approx(-2.22)
    assert len(ledger) == 1
    assert ledger[0]["entry_id"] == "Li2"
    assert ledger[0]["resolution"] == "frozen_thermo_top_level_energy_pair"


def test_reported_energy_with_inconsistent_correction_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected, _, _ = deduplicate_entries(
        pd.DataFrame([entry_row("Li2", {"Li": 2}, -2.0, correction=-0.2)])
    )
    selected["_space"] = [("Li",)]
    reported = pd.DataFrame(
        [
            {
                "thermo_id": "thermo-Li2",
                "thermo_type": "SYNTHETIC_COMPATIBILITY",
                "energy_type": "synthetic",
                "chemsys": "Li",
                "energy_per_atom": -1.12,
                "uncorrected_energy_per_atom": -1.00,
            }
        ]
    )
    monkeypatch.setattr(context_phase_diagrams, "_thermo_selection", lambda *_: reported)
    unresolved, errors, ledger = _reconcile_reported_energy(
        root=Path("."), snapshot="synthetic-1",
        thermo_type="SYNTHETIC_COMPATIBILITY", selected=selected,
        tolerance=1e-6,
    )
    assert unresolved.iloc[0].energy == -2.0
    assert unresolved.iloc[0].corrected_energy == -2.2
    assert ledger == []
    assert len(errors) == 1
    assert errors[0]["reason"] == "reported_entry_energy_inconsistent"


def test_mixing_duplicate_alias_requires_one_structural_representative() -> None:
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    target = ComputedStructureEntry(
        structure, -1.0, entry_id="target", parameters={"run_type": "GGA"}
    )
    representative = ComputedStructureEntry(
        structure.copy(), -2.0, entry_id="representative", parameters={"run_type": "GGA"}
    )
    scheme = MaterialsProjectDFTMixingScheme(check_potcar=False)
    resolved = _resolve_mixing_duplicate_alias(target, [representative], scheme)
    assert resolved is not None
    alias, source, method = resolved
    assert alias.entry_id == "target"
    assert alias.energy == -2.0
    assert source.entry_id == "representative"
    assert method == "unique_processed_structure_representative"
    second = ComputedStructureEntry(
        structure.copy(), -3.0, entry_id="second", parameters={"run_type": "GGA"}
    )
    assert _resolve_mixing_duplicate_alias(target, [representative, second], scheme) is None


def test_mixing_state_counterpart_resolves_ambiguous_structure_matches() -> None:
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    target = ComputedStructureEntry(
        structure, -1.0, entry_id="target-gga", parameters={"run_type": "GGA"}
    )
    r2scan = ComputedStructureEntry(
        structure.copy(), -2.0, entry_id="kept-r2", parameters={"run_type": "R2SCAN"}
    )
    competing_match = ComputedStructureEntry(
        structure.copy(), -3.0, entry_id="other-gga", parameters={"run_type": "GGA"}
    )
    state = pd.DataFrame(
        [{"entry_id_1": "target-gga", "entry_id_2": "kept-r2"}]
    )
    scheme = MaterialsProjectDFTMixingScheme(check_potcar=False)
    resolved = _resolve_mixing_duplicate_alias(
        target, [r2scan, competing_match], scheme, state
    )
    assert resolved is not None
    alias, representative, method = resolved
    assert alias.entry_id == "target-gga"
    assert representative.entry_id == "kept-r2"
    assert method == "mixing_state_cross_workflow_counterpart"


def test_audited_mixing_scheme_sorts_upstream_entry_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    a = ComputedStructureEntry(
        structure, -1.0, entry_id="a", parameters={"run_type": "GGA"}
    )
    b = ComputedStructureEntry(
        structure.copy(), -2.0, entry_id="b", parameters={"run_type": "GGA"}
    )
    r1 = ComputedStructureEntry(
        structure.copy(), -3.0, entry_id="r1", parameters={"run_type": "R2SCAN"}
    )
    r2 = ComputedStructureEntry(
        structure.copy(), -4.0, entry_id="r2", parameters={"run_type": "R2SCAN"}
    )

    def unsorted_upstream(self, entries, verbose=False):
        return [b, a], [r2, r1]

    monkeypatch.setattr(
        MaterialsProjectDFTMixingScheme,
        "_filter_and_sort_entries",
        unsorted_upstream,
    )
    scheme = _AuditedMixingScheme(check_potcar=False)
    first, second = scheme._filter_and_sort_entries([r2, b, r1, a])
    assert [entry.entry_id for entry in first] == ["a", "b"]
    assert [entry.entry_id for entry in second] == ["r1", "r2"]


def test_mixed_context_shards_are_deterministic_and_complete() -> None:
    mixed = pd.DataFrame(
        [
            {"entry_id": "nao", "thermo_id": "t-nao", "entry_label": "GGA", "_space": ("Na", "O")},
            {"entry_id": "li", "thermo_id": "t-li", "entry_label": "GGA", "_space": ("Li",)},
            {"entry_id": "lio", "thermo_id": "t-lio", "entry_label": "R2SCAN", "_space": ("Li", "O")},
        ]
    )
    selection = pd.DataFrame(
        [
            {"thermo_id": "t-li", "selected_entry_id": "li", "selected_entry_label": "GGA", "energy_type": "GGA", "chemsys": "Li"},
            {"thermo_id": "t-lio", "selected_entry_id": "lio", "selected_entry_label": "R2SCAN", "energy_type": "R2SCAN", "chemsys": "Li-O"},
            {"thermo_id": "t-nao", "selected_entry_id": "nao", "selected_entry_label": "GGA", "energy_type": "GGA", "chemsys": "Na-O"},
        ]
    )
    gga = pd.DataFrame(
        {"_space": [("Li",), ("Li",), ("O",), ("Li", "O"), ("Na", "O")]}
    )
    r2scan = pd.DataFrame(
        {"_space": [("Li", "O"), ("Li", "O"), ("Na",)]}
    )
    first, first_meta = _mixed_context_assignment(
        mixed_selected=mixed,
        gga_selected=gga,
        r2_selected=r2scan,
        selection=selection,
        shard_count=2,
    )
    second, second_meta = _mixed_context_assignment(
        mixed_selected=mixed.iloc[::-1].reset_index(drop=True),
        gga_selected=gga.iloc[::-1].reset_index(drop=True),
        r2_selected=r2scan.iloc[::-1].reset_index(drop=True),
        selection=selection.iloc[::-1].reset_index(drop=True),
        shard_count=2,
    )
    assert first == second
    assert set(first) == {"Li", "Li-O", "Na-O"}
    assert set(first.values()) <= {0, 1}
    assert first_meta == second_meta
    assert first_meta["context_count"] == 3
    assert first_meta["mixing_contexts"] == 1
    assert len(first_meta["assignment_sha256"]) == 64


def test_frozen_target_selection_uses_energy_label_and_excludes_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        [
            {"thermo_id": "t1", "energy_type": "r2scan", "chemsys": "Li"},
            {"thermo_id": "t2", "energy_type": "GGA", "chemsys": "F"},
        ]
    )
    embedded = pd.DataFrame(
        [
            {
                "thermo_id": "t1", "entry_label": "GGA", "entry_id": "sibling",
                "composition_json": json.dumps({"Li": 1}),
            },
            {
                "thermo_id": "t1", "entry_label": "R2SCAN", "entry_id": "selected",
                "composition_json": json.dumps({"Li": 1}),
            },
            {
                "thermo_id": "t2", "entry_label": "GGA", "entry_id": "fluorine",
                "composition_json": json.dumps({"F": 1}),
            },
        ]
    )
    monkeypatch.setattr(context_phase_diagrams, "_thermo_selection", lambda *_: source)
    monkeypatch.setattr(
        context_phase_diagrams, "_thermo_embedded_target_rows", lambda *_: embedded
    )
    selection = context_phase_diagrams._thermo_target_selection(
        Path("."), "s1", "GGA_GGA+U_R2SCAN"
    )
    assert dict(zip(selection["thermo_id"], selection["selected_entry_id"], strict=True)) == {
        "t1": "selected",
        "t2": "fluorine",
    }

    mixed = embedded.assign(
        energy_type=["r2scan", "r2scan", "GGA"],
        _space=[("Li",), ("Li",), ("F",)],
    )
    chosen = context_phase_diagrams._mixed_target_rows(mixed, selection)
    assert set(chosen["entry_id"]) == {"selected", "fluorine"}
    assert "sibling" not in set(chosen["entry_id"])


def test_only_declared_compatibility_failures_become_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = pd.DataFrame(
        [
            {
                "thermo_id": "t1", "selected_entry_id": "target",
                "selected_entry_label": "R2SCAN", "chemsys": "Li-O",
                "source_object_sha256": "a" * 64,
                "source_key": "raw.jsonl", "source_row_number": 7,
            }
        ]
    )
    monkeypatch.setattr(
        context_phase_diagrams, "_thermo_target_selection", lambda *_: selection
    )
    blocking, exclusions = context_phase_diagrams._classify_compatibility_exclusions(
        root=Path("."), snapshot="s1", thermo_type="GGA_GGA+U_R2SCAN",
        ambiguities=[
            {
                "entry_id": "target", "phase_context_chemsys": "Li-O",
                "reason": "unresolved_mixing_structure_duplicate",
                "processed_target_present": False,
                "mixing_state_candidate_count": 0,
                "structural_match_count": 2,
                "candidate_representative_ids": ["b", "a"],
            },
            {
                "phase_context_chemsys": "Na-O", "reason": "mixed_context_failure",
                "error": "KeyError: technical",
            },
        ],
        config={
            "reference_test": {
                "allowed_exclusion_reasons": [
                    "no_unique_processed_representative",
                    "homogeneous_source_unavailable",
                ],
                "eligibility_rule_version": "test-rule",
            }
        },
    )
    assert [row["reason"] for row in blocking] == ["mixed_context_failure"]
    assert len(exclusions) == 1
    assert exclusions[0]["reason"] == "no_unique_processed_representative"
    assert exclusions[0]["candidate_representative_ids_json"] == '["a","b"]'
    assert exclusions[0]["source_row_number"] == 7


def test_mixed_shard_paths_are_stable_and_nonoverlapping(tmp_path: Path) -> None:
    first = _mixed_shard_paths(tmp_path, "2022-10-28", 0, 8)
    repeated = _mixed_shard_paths(tmp_path, "2022-10-28", 0, 8)
    second = _mixed_shard_paths(tmp_path, "2022-10-28", 1, 8)
    assert first == repeated
    assert first["stats"] != second["stats"]
    assert first["phase"].parent == tmp_path / "snapshot=2022-10-28"


def test_mixed_reference_gate_uses_context_reconstruction_not_global_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_uid = contextual_entry_id("s1", "GGA_GGA+U_R2SCAN", "Li", "first")
    second_uid = contextual_entry_id("s1", "GGA_GGA+U_R2SCAN", "F", "second")
    phase = pd.DataFrame(
        [
            {
                "unified_entry_id": first_uid, "thermo_id": "t1",
                "entry_label": "R2SCAN", "entry_id": "first",
                "phase_context_chemsys": "Li", "is_target": True,
                "compatibility_mode": "regenerated_context_mixing",
                "corrected_energy_per_atom": -2.0,
                "energy_above_hull": 0.2, "formation_energy_per_atom": -1.0,
            },
            {
                "unified_entry_id": second_uid, "thermo_id": "t2",
                "entry_label": "GGA", "entry_id": "second",
                "phase_context_chemsys": "F", "is_target": True,
                "compatibility_mode": "homogeneous_gga_mirror",
                "corrected_energy_per_atom": -1.5,
                "energy_above_hull": 0.1, "formation_energy_per_atom": -0.8,
            },
        ]
    )
    decomposition = pd.DataFrame(
        [
            {"unified_entry_id": first_uid, "amount": 1.0, "component_energy_per_atom": -2.2},
            {"unified_entry_id": second_uid, "amount": 1.0, "component_energy_per_atom": -1.6},
        ]
    )
    source = pd.DataFrame(
        [
            {
                "thermo_id": "t1", "energy_type": "R2SCAN", "chemsys": "Li",
                "selected_entry_id": "first", "selected_entry_label": "R2SCAN",
                "energy_above_hull": 0.5, "formation_energy_per_atom": -0.5,
            },
            {
                "thermo_id": "t2", "energy_type": "GGA", "chemsys": "F",
                "selected_entry_id": "second", "selected_entry_label": "GGA",
                "energy_above_hull": 0.1, "formation_energy_per_atom": -0.8,
            },
        ]
    )
    homogeneous_reference = pd.DataFrame(
        [
            {
                "unified_entry_id": contextual_entry_id(
                    "s1", "GGA_GGA+U", "F", "second"
                ),
                "entry_id": "second",
                "phase_context_chemsys": "F",
                "corrected_energy_per_atom": -1.5,
                "energy_above_hull": 0.1,
                "formation_energy_per_atom": -0.8,
            }
        ]
    )
    monkeypatch.setattr(context_phase_diagrams, "_thermo_target_selection", lambda *_: source)
    monkeypatch.setattr(
        context_phase_diagrams,
        "_homogeneous_mirror_reference",
        lambda **_: homogeneous_reference,
    )
    summary, _ = _reference_audit(
        snapshot="s1", thermo_type="GGA_GGA+U_R2SCAN", root=Path("."),
        phase=phase, decomposition=decomposition,
        config={
            "seed": 42,
            "reference_test": {
                "absolute_tolerance_eV_per_atom": 1e-6,
                "sample_per_snapshot_thermo_type": 100,
                "eligibility_rule_version": "test-rule",
                "maximum_exclusion_fraction": 0.0005,
            },
        },
    )
    assert summary["match_rate"] == 1.0
    assert summary["target_coverage_rate"] == 1.0
    assert summary["decomposition_reconstruction_match_rate"] == 1.0
    assert summary["mirror_source_match_rate"] == 1.0
    assert summary["mirror_reference_mode"] == "exact_homogeneous_context_artifact"
    assert summary["mirror_matched_states"] == 1
    assert summary["serialized_source_match_rate_diagnostic"] == 0.5

    mismatched_reference = homogeneous_reference.copy()
    mismatched_reference.loc[0, "energy_above_hull"] = 0.2
    monkeypatch.setattr(
        context_phase_diagrams,
        "_homogeneous_mirror_reference",
        lambda **_: mismatched_reference,
    )
    failed_summary, _ = _reference_audit(
        snapshot="s1", thermo_type="GGA_GGA+U_R2SCAN", root=Path("."),
        phase=phase, decomposition=decomposition,
        config={
            "seed": 42,
            "reference_test": {
                "absolute_tolerance_eV_per_atom": 1e-6,
                "sample_per_snapshot_thermo_type": 100,
                "eligibility_rule_version": "test-rule",
                "maximum_exclusion_fraction": 0.0005,
            },
        },
    )
    assert failed_summary["match_rate"] == 0.0
    assert failed_summary["mirror_source_match_rate"] == 0.0


def test_mixed_reference_sample_uses_contextual_id_with_repeated_and_unmatched_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_uid = contextual_entry_id("s1", "GGA_GGA+U_R2SCAN", "Li", "r2")
    sibling_uid = contextual_entry_id("s1", "GGA_GGA+U_R2SCAN", "Li", "gga")
    phase = pd.DataFrame(
        [
            {
                "unified_entry_id": selected_uid,
                "thermo_id": "shared-thermo",
                "entry_label": "R2SCAN",
                "entry_id": "r2",
                "phase_context_chemsys": "Li",
                "is_target": True,
                "compatibility_mode": "regenerated_context_mixing",
                "corrected_energy_per_atom": -2.0,
                "energy_above_hull": 0.2,
                "formation_energy_per_atom": -1.0,
            },
            {
                "unified_entry_id": sibling_uid,
                "thermo_id": "shared-thermo",
                "entry_label": "GGA",
                "entry_id": "gga",
                "phase_context_chemsys": "Li",
                "is_target": True,
                "compatibility_mode": "regenerated_context_mixing",
                "corrected_energy_per_atom": -1.5,
                "energy_above_hull": 0.1,
                "formation_energy_per_atom": -0.8,
            },
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "unified_entry_id": selected_uid,
                "amount": 1.0,
                "component_energy_per_atom": -2.2,
            },
            {
                "unified_entry_id": sibling_uid,
                "amount": 1.0,
                "component_energy_per_atom": -1.6,
            },
        ]
    )
    source = pd.DataFrame(
        [
            {
                "thermo_id": "shared-thermo",
                "energy_type": "R2SCAN",
                "chemsys": "Li",
                "selected_entry_id": "r2",
                "selected_entry_label": "R2SCAN",
                "energy_above_hull": 0.2,
                "formation_energy_per_atom": -1.0,
            },
            {
                "thermo_id": "missing-a",
                "energy_type": "R2SCAN",
                "chemsys": "F",
                "selected_entry_id": "missing-entry-a",
                "selected_entry_label": "R2SCAN",
                "energy_above_hull": 0.3,
                "formation_energy_per_atom": -0.4,
            },
            {
                "thermo_id": "missing-b",
                "energy_type": "GGA",
                "chemsys": "O",
                "selected_entry_id": "missing-entry-b",
                "selected_entry_label": "GGA",
                "energy_above_hull": 0.4,
                "formation_energy_per_atom": -0.5,
            },
        ]
    )
    monkeypatch.setattr(context_phase_diagrams, "_thermo_target_selection", lambda *_: source)

    summary, sample = _reference_audit(
        snapshot="s1",
        thermo_type="GGA_GGA+U_R2SCAN",
        root=Path("."),
        phase=phase,
        decomposition=decomposition,
        config={
            "seed": 42,
            "reference_test": {
                "absolute_tolerance_eV_per_atom": 1e-6,
                "sample_per_snapshot_thermo_type": 100,
                "eligibility_rule_version": "test-rule",
                "maximum_exclusion_fraction": 1.0,
            },
        },
        compatibility_exclusions=[
            {"thermo_id": "missing-a", "reason": "no_unique_processed_representative", "rule_version": "test-rule"},
            {"thermo_id": "missing-b", "reason": "homogeneous_source_unavailable", "rule_version": "test-rule"},
        ],
    )

    assert summary["match_rate"] == 1.0
    assert summary["raw_target_coverage_rate"] == pytest.approx(1 / 3)
    assert summary["eligible_target_coverage_rate"] == 1.0
    assert summary["excluded_source_states"] == 2
    assert summary["decomposition_reconstruction_states"] == 2
    assert len(sample) == 2
    assert {row["unified_entry_id"] for row in sample} == {
        selected_uid.hex(),
        sibling_uid.hex(),
    }
    json.dumps(sample)
    diagnostics = pd.Series(
        [row["serialized_source_matched"] for row in sample], dtype="boolean"
    )
    assert diagnostics.notna().sum() == 1
    assert bool(diagnostics.dropna().iloc[0])
