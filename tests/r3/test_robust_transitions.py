from __future__ import annotations

import pytest

from phase_evonet.r3.robust_transitions import (
    cohort_memberships,
    exact_flip_amplitude,
    exact_survival_is_monotonic,
    route_r3_1,
    threshold_labels_are_monotonic,
    threshold_transition,
    wilson_interval,
)


THRESHOLDS = (0.001, 0.005, 0.010, 0.025, 0.050)


def test_exact_stable_to_unstable_amplitude_uses_target_distance() -> None:
    result = exact_flip_amplitude(
        source_is_stable=True,
        target_is_stable=False,
        source_energy_above_hull_eV_per_atom=0.0,
        target_energy_above_hull_eV_per_atom=0.012,
        thresholds_eV_per_atom=THRESHOLDS,
    )
    assert result.direction == "stable_to_unstable"
    assert result.amplitude_eV_per_atom == pytest.approx(0.012)
    assert result.survives[0.010] is True
    assert result.survives[0.025] is False
    assert exact_survival_is_monotonic(result.survives)


def test_exact_unstable_to_stable_amplitude_uses_source_distance() -> None:
    result = exact_flip_amplitude(
        source_is_stable=False,
        target_is_stable=True,
        source_energy_above_hull_eV_per_atom=0.030,
        target_energy_above_hull_eV_per_atom=0.0,
        thresholds_eV_per_atom=THRESHOLDS,
    )
    assert result.direction == "unstable_to_stable"
    assert result.amplitude_eV_per_atom == pytest.approx(0.030)
    assert result.survives[0.025] is True
    assert result.survives[0.050] is False


def test_threshold_relabel_is_separate_and_monotonic_per_endpoint() -> None:
    source, target, direction, flip = threshold_transition(0.003, 0.020, 0.010)
    assert (source, target, direction, flip) == (
        True,
        False,
        "stable_to_unstable",
        True,
    )
    assert threshold_labels_are_monotonic(0.020, THRESHOLDS)


def test_non_flip_is_rejected_by_exact_amplitude_function() -> None:
    with pytest.raises(ValueError):
        exact_flip_amplitude(
            source_is_stable=True,
            target_is_stable=True,
            source_energy_above_hull_eV_per_atom=0.0,
            target_energy_above_hull_eV_per_atom=0.0,
            thresholds_eV_per_atom=THRESHOLDS,
        )


def test_strict_cohort_is_subset_of_broad() -> None:
    memberships = cohort_memberships(
        {
            "identity_confidence": "A1",
            "same_workflow": True,
            "same_phase_context": True,
            "candidate_identity_unchanged": True,
            "exact_flip": True,
            "reported_and_unified_agree": True,
        }
    )
    assert memberships["A1_STRICT"] is True
    assert memberships["A1_BROAD"] is True
    assert memberships["A1_A2"] is True
    assert memberships["REPORTED_AND_UNIFIED"] is True
    assert memberships["UNIFIED_ONLY"] is False


def test_route_strong_uses_frozen_two_of_three_rule() -> None:
    result = route_r3_1(
        n10_strict_stu=300,
        n25_strict_stu=99,
        f10_strict_stu=0.15,
        workflow_n10_with_positive_lower_bound=False,
    )
    assert result.route == "ROUTE_STRONG"
    assert result.strong_conditions_met == 2


def test_route_field_and_boundary_are_deterministic() -> None:
    field = route_r3_1(
        n10_strict_stu=50,
        n25_strict_stu=0,
        f10_strict_stu=0.01,
        workflow_n10_with_positive_lower_bound=False,
    )
    boundary = route_r3_1(
        n10_strict_stu=49,
        n25_strict_stu=19,
        f10_strict_stu=0.149,
        workflow_n10_with_positive_lower_bound=False,
    )
    assert field.route == "ROUTE_FIELD"
    assert boundary.route == "ROUTE_BOUNDARY"


def test_wilson_interval_is_bounded_and_positive_for_events() -> None:
    lower, upper = wilson_interval(30, 100)
    assert 0.0 < lower < 0.30 < upper < 1.0
