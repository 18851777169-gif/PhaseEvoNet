"""Pure exact-amplitude, threshold-relabel, cohort, and route functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ExactAmplitudeResult:
    direction: str
    amplitude_eV_per_atom: float
    absolute_delta_eV_per_atom: float
    survives: dict[float, bool]


@dataclass(frozen=True)
class RouteResult:
    route: str
    conditions: dict[str, bool]
    strong_conditions_met: int


def normalized_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in thresholds}))
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("thresholds must be non-empty, finite, and non-negative")
    return values


def exact_flip_amplitude(
    *,
    source_is_stable: bool,
    target_is_stable: bool,
    source_energy_above_hull_eV_per_atom: float,
    target_energy_above_hull_eV_per_atom: float,
    thresholds_eV_per_atom: Iterable[float],
) -> ExactAmplitudeResult:
    if source_is_stable == target_is_stable:
        raise ValueError("exact-flip amplitude requires an exact stability flip")
    source = float(source_energy_above_hull_eV_per_atom)
    target = float(target_energy_above_hull_eV_per_atom)
    if source_is_stable:
        direction = "stable_to_unstable"
        amplitude = target
    else:
        direction = "unstable_to_stable"
        amplitude = source
    thresholds = normalized_thresholds(thresholds_eV_per_atom)
    return ExactAmplitudeResult(
        direction=direction,
        amplitude_eV_per_atom=amplitude,
        absolute_delta_eV_per_atom=abs(target - source),
        survives={threshold: amplitude >= threshold for threshold in thresholds},
    )


def threshold_label(energy_above_hull_eV_per_atom: float, threshold_eV_per_atom: float) -> bool:
    return float(energy_above_hull_eV_per_atom) <= float(threshold_eV_per_atom)


def threshold_transition(
    source_energy_above_hull_eV_per_atom: float,
    target_energy_above_hull_eV_per_atom: float,
    threshold_eV_per_atom: float,
) -> tuple[bool, bool, str, bool]:
    source = threshold_label(source_energy_above_hull_eV_per_atom, threshold_eV_per_atom)
    target = threshold_label(target_energy_above_hull_eV_per_atom, threshold_eV_per_atom)
    direction = (
        ("stable" if source else "unstable")
        + "_to_"
        + ("stable" if target else "unstable")
    )
    return source, target, direction, source != target


def cohort_memberships(record: Mapping[str, object]) -> dict[str, bool]:
    identity = str(record.get("identity_confidence"))
    same_workflow = bool(record.get("same_workflow"))
    same_context = bool(record.get("same_phase_context"))
    identity_unchanged = bool(record.get("candidate_identity_unchanged"))
    exact_flip = bool(record.get("exact_flip", False))
    agreement = bool(record.get("reported_and_unified_agree", False))
    return {
        "A1_STRICT": identity == "A1" and identity_unchanged and same_workflow and same_context,
        "A1_BROAD": identity == "A1" and same_workflow,
        "A1_A2": identity in {"A1", "A2"},
        "REPORTED_AND_UNIFIED": exact_flip and agreement,
        "UNIFIED_ONLY": exact_flip and not agreement,
    }


def exact_survival_is_monotonic(survives: Mapping[float, bool]) -> bool:
    ordered = [bool(survives[key]) for key in sorted(survives)]
    return all(not later or earlier for earlier, later in zip(ordered, ordered[1:]))


def threshold_labels_are_monotonic(
    energy_above_hull_eV_per_atom: float, thresholds_eV_per_atom: Iterable[float]
) -> bool:
    labels = [
        threshold_label(energy_above_hull_eV_per_atom, threshold)
        for threshold in normalized_thresholds(thresholds_eV_per_atom)
    ]
    return all(not earlier or later for earlier, later in zip(labels, labels[1:]))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def route_r3_1(
    *,
    n10_strict_stu: int,
    n25_strict_stu: int,
    f10_strict_stu: float,
    workflow_n10_with_positive_lower_bound: bool,
) -> RouteResult:
    conditions = {
        "N10_strict_STU_gte_300": int(n10_strict_stu) >= 300,
        "N25_strict_STU_gte_100": int(n25_strict_stu) >= 100,
        "F10_strict_STU_gte_0_15": float(f10_strict_stu) >= 0.15,
    }
    strong_count = sum(conditions.values())
    if strong_count >= 2:
        route = "ROUTE_STRONG"
    elif (
        int(n10_strict_stu) >= 50
        or int(n25_strict_stu) >= 20
        or workflow_n10_with_positive_lower_bound
    ):
        route = "ROUTE_FIELD"
    else:
        route = "ROUTE_BOUNDARY"
    return RouteResult(route=route, conditions=conditions, strong_conditions_met=strong_count)
