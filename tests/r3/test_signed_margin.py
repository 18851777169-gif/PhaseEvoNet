from __future__ import annotations

import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from phase_evonet.r3.energy_amplitude import (
    _records_with_elemental_terminals,
    _terminal_workflow_for_context,
)
from phase_evonet.r3.signed_margin import compute_signed_margin


def _entry(entry_id: str, formula: str, energy: float) -> ComputedEntry:
    return ComputedEntry(Composition(formula), energy, entry_id=entry_id)


def test_binary_stable_intermediate_has_negative_signed_margin() -> None:
    entries = [
        _entry("Li", "Li", -1.0),
        _entry("F", "F", -2.0),
        _entry("LiF", "LiF", -4.0),
    ]
    result = compute_signed_margin(entries, "LiF", stored_energy_above_hull_eV_per_atom=0.0)
    assert result.signed_energy_eV_per_atom == pytest.approx(-0.5)
    assert result.stability_margin_eV_per_atom == pytest.approx(0.5)
    assert result.is_stable_by_tolerance is True
    assert result.validation_comparable is True
    assert result.absolute_validation_error_eV_per_atom == pytest.approx(0.0)


def test_binary_unstable_intermediate_has_positive_signed_margin() -> None:
    entries = [
        _entry("Li", "Li", -1.0),
        _entry("F", "F", -2.0),
        _entry("LiF", "LiF", -4.0),
        _entry("Li2F", "Li2F", -4.2),
    ]
    result = compute_signed_margin(entries, "Li2F")
    assert result.signed_energy_eV_per_atom is not None
    assert result.signed_energy_eV_per_atom > 0
    assert result.is_stable_by_tolerance is False
    assert result.validation_comparable is True


def test_ternary_stable_interior_phase_is_supported() -> None:
    entries = [
        _entry("Li", "Li", -1.0),
        _entry("F", "F", -2.0),
        _entry("O", "O", -1.5),
        _entry("LiFO", "LiFO", -7.5),
    ]
    result = compute_signed_margin(entries, "LiFO")
    assert result.signed_energy_eV_per_atom == pytest.approx(-1.0)
    assert len(result.explicit_loo_decomposition) == 3


def test_duplicate_composition_polymorph_is_explicit_semantic_exception() -> None:
    entries = [
        _entry("Li", "Li", -1.0),
        _entry("F", "F", -2.0),
        _entry("LiF-stable", "LiF", -4.0),
        _entry("LiF-polymorph", "LiF", -3.5),
    ]
    result = compute_signed_margin(entries, "LiF-polymorph")
    assert result.official_signed_energy_eV_per_atom == pytest.approx(-0.25)
    assert result.explicit_loo_energy_eV_per_atom == pytest.approx(0.25)
    assert result.signed_energy_eV_per_atom == pytest.approx(0.25)
    assert result.validation_comparable is False
    assert result.validation_exception == "PYMATGEN_SAME_COMPOSITION_EXCLUSION_SEMANTICS"


def test_elemental_endpoint_is_explicit_special_case() -> None:
    result = compute_signed_margin([_entry("Li", "Li", -1.0)], "Li")
    assert result.signed_energy_eV_per_atom == pytest.approx(0.0)
    assert result.explicit_loo_energy_eV_per_atom is None
    assert result.validation_exception == "ELEMENTAL_ENDPOINT_WITHOUT_ALTERNATIVE_TERMINAL"
    assert result.solver_status == "PASS_ELEMENTAL_ENDPOINT_SPECIAL"


def test_near_zero_numerical_boundary_preserves_raw_sign() -> None:
    entries = [
        _entry("Li", "Li", 0.0),
        _entry("F", "F", 0.0),
        _entry("LiF-near", "LiF", 1e-8),
    ]
    result = compute_signed_margin(
        entries, "LiF-near", numerical_tolerance_eV_per_atom=1e-8
    )
    assert result.signed_energy_eV_per_atom == pytest.approx(5e-9)
    assert result.is_stable_by_tolerance is True


def test_candidate_removal_changes_hull_and_is_not_silently_zero() -> None:
    entries = [
        _entry("Li", "Li", 0.0),
        _entry("F", "F", 0.0),
        _entry("LiF", "LiF", -1.0),
    ]
    result = compute_signed_margin(entries, "LiF")
    assert result.signed_energy_eV_per_atom == pytest.approx(-0.5)
    assert result.stability_margin_eV_per_atom == pytest.approx(0.5)


def test_nonunique_boundary_decomposition_is_canonical_and_ledgerable() -> None:
    entries = [
        _entry("Li", "Li", 0.0),
        _entry("F", "F", 0.0),
        _entry("Li2F2", "Li2F2", 0.0),
        _entry("LiF2", "LiF2", 0.0),
        _entry("Li2F", "Li2F", 0.0),
        _entry("LiF-candidate", "LiF", 0.0),
    ]
    result = compute_signed_margin(entries, "LiF-candidate")
    assert result.signed_energy_eV_per_atom == pytest.approx(0.0, abs=1e-8)
    component_ids = [component.entry_id for component in result.explicit_loo_decomposition]
    assert component_ids == sorted(component_ids)
    assert result.solver_status.startswith("PASS")


def test_multielement_context_reloads_frozen_elemental_terminals() -> None:
    context = [{"unified_entry_id": b"compound", "entry_id": "LiF"}]
    terminals = {
        ("S1", "GGA", "F"): [{"unified_entry_id": b"fluorine", "entry_id": "F2"}],
        ("S1", "GGA", "Li"): [{"unified_entry_id": b"lithium", "entry_id": "Li"}],
    }

    augmented = _records_with_elemental_terminals(
        ("S1", "GGA", "F-Li"), context, terminals
    )

    assert [row["entry_id"] for row in augmented] == ["LiF", "F2", "Li"]
    assert _records_with_elemental_terminals(
        ("S1", "GGA", "Li"), terminals[("S1", "GGA", "Li")], terminals
    ) == terminals[("S1", "GGA", "Li")]


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"compatibility_mode": "homogeneous_gga_mirror"}], "GGA_GGA+U"),
        (
            [
                {
                    "compatibility_mode": "regenerated_context_mixing",
                    "source_workflow": "R2SCAN",
                    "correction": 2.0,
                }
            ],
            "GGA_GGA+U",
        ),
        (
            [
                {
                    "compatibility_mode": "regenerated_context_mixing",
                    "source_workflow": "R2SCAN",
                    "correction": 0.0,
                }
            ],
            "R2SCAN",
        ),
    ],
)
def test_mixed_context_terminal_workflow_follows_frozen_energy_reference(
    records: list[dict[str, object]], expected: str
) -> None:
    assert _terminal_workflow_for_context("GGA_GGA+U_R2SCAN", records) == expected
