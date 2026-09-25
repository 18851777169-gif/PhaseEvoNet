from __future__ import annotations

import json

import pandas as pd
import pytest

from phase_evonet.r3.competitor_cascade import (
    _canonical_competitor_id,
    _deduplicate_contextual_mirrors,
    _long_edge_relations,
    apply_competitor_actions,
    search_minimal_actions,
    solve_hull_distance,
)


def row(uid: int, composition: dict[str, float], energy_per_atom: float, *, entry: str) -> dict:
    atoms = sum(composition.values())
    return {
        "snapshot_id": "2025-09-25",
        "thermo_type": "GGA_GGA+U",
        "phase_context_chemsys": "A-B",
        "unified_entry_id": uid.to_bytes(16, "big"),
        "entry_id": entry,
        "task_id": f"task-{entry}",
        "material_id": f"material-{entry}",
        "source_workflow": "GGA_GGA+U",
        "composition_json": json.dumps(composition, sort_keys=True),
        "num_atoms": atoms,
        "uncorrected_energy": energy_per_atom * atoms,
        "correction": 0.0,
        "corrected_energy": energy_per_atom * atoms,
        "corrected_energy_per_atom": energy_per_atom,
    }


def test_irrelevant_and_necessary_single_competitors() -> None:
    candidate = row(1, {"A": 1, "B": 1}, 0.020, entry="candidate")
    a = row(2, {"A": 1}, 0.0, entry="a")
    b = row(3, {"B": 1}, 0.0, entry="b")
    irrelevant = row(4, {"A": 1, "B": 1}, 1.0, entry="irrelevant")
    full, status, weights, _ = solve_hull_distance(candidate, [a, b, irrelevant], ["A", "B"])
    assert status == "feasible"
    assert full == pytest.approx(0.020)
    assert weights is not None and weights[2] == pytest.approx(0.0)
    without_irrelevant, *_ = solve_hull_distance(candidate, [a, b], ["A", "B"])
    assert without_irrelevant == pytest.approx(full)
    without_a, *_ = solve_hull_distance(candidate, [b, irrelevant], ["A", "B"])
    assert without_a == pytest.approx(0.0)


def test_exact_joint_minimum_is_two() -> None:
    candidate = row(1, {"A": 1, "B": 1}, 0.020, entry="candidate")
    first = row(2, {"A": 1, "B": 1}, 0.0, entry="first")
    second = row(3, {"A": 1, "B": 1}, 0.005, entry="second")
    action_ids = [bytes(first["unified_entry_id"]), bytes(second["unified_entry_id"])]
    actions = {uid: {"operation": "remove"} for uid in action_ids}

    def evaluator(selected: frozenset[bytes]) -> float:
        competitors = apply_competitor_actions([first, second], actions, selected)
        value, *_ = solve_hull_distance(candidate, competitors, ["A", "B"])
        return value

    result = search_minimal_actions(
        action_ids,
        evaluator,
        full_value=0.020,
        threshold=0.010,
        exact_pool_cap=20,
        node_budget=100,
        wall_time_limit_seconds=10,
    )
    assert result["status"] == "exact"
    assert result["lower_bound"] == result["upper_bound"] == 2
    assert set(result["members"]) == set(action_ids)


def test_pool_cap_returns_honest_bounds_not_exact() -> None:
    actions = [index.to_bytes(16, "big") for index in range(1, 4)]

    def evaluator(selected: frozenset[bytes]) -> float:
        return 0.0 if len(selected) >= 2 else 0.020

    result = search_minimal_actions(
        actions,
        evaluator,
        full_value=0.020,
        threshold=0.010,
        exact_pool_cap=2,
        node_budget=100,
        wall_time_limit_seconds=10,
    )
    assert result["status"] == "bounded_pool_over_cap"
    assert result["lower_bound"] == 1
    assert result["upper_bound"] == 2


def test_monotone_full_pool_proves_infeasible_without_enumeration() -> None:
    actions = [index.to_bytes(16, "big") for index in range(1, 21)]

    def evaluator(selected: frozenset[bytes]) -> float:
        return 0.020 - 0.0001 * len(selected)

    result = search_minimal_actions(
        actions,
        evaluator,
        full_value=0.020,
        threshold=0.010,
        exact_pool_cap=20,
        node_budget=1_000_000,
        wall_time_limit_seconds=900,
    )
    assert result["status"] == "infeasible_monotone_full_pool"
    assert result["nodes"] == 20


def test_canonical_id_is_row_order_independent_and_workflow_separated() -> None:
    first = row(1, {"B": 1, "A": 2}, -1.0, entry="X")
    reordered = dict(first)
    reordered["composition_json"] = '{"A": 2, "B": 1}'
    assert _canonical_competitor_id(first) == _canonical_competitor_id(reordered)
    other_workflow = dict(first)
    other_workflow["source_workflow"] = "R2SCAN"
    assert _canonical_competitor_id(first)[0] != _canonical_competitor_id(other_workflow)[0]


def test_identical_context_terminal_mirror_is_audited_and_selected_uid_wins() -> None:
    context = row(1, {"A": 1}, -1.0, entry="a")
    context["phase_context_chemsys"] = "A-B"
    terminal = dict(context)
    terminal["unified_entry_id"] = (2).to_bytes(16, "big")
    terminal["phase_context_chemsys"] = "A"
    chosen, audit = _deduplicate_contextual_mirrors(
        [context, terminal],
        candidate_context="A-B",
        preferred_ids={bytes(terminal["unified_entry_id"])},
        tolerance=1e-6,
    )
    assert len(chosen) == 1
    assert bytes(chosen[0]["unified_entry_id"]) == bytes(terminal["unified_entry_id"])
    assert len(audit) == 2
    assert {item["selection_rule"] for item in audit} == {"selected_decomposition_uid"}


def test_separate_edge_definitions_reconcile_counts_and_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "transition_id": (1).to_bytes(16, "big"),
                "competitor_canonical_id": "cmp-a",
                "selected_active": True,
                "necessary_10meV": True,
                "necessary_25meV": False,
                "contributory_5meV": True,
                "minimal_set_member": False,
            },
            {
                "transition_id": (2).to_bytes(16, "big"),
                "competitor_canonical_id": "cmp-b",
                "selected_active": True,
                "necessary_10meV": False,
                "necessary_25meV": False,
                "contributory_5meV": False,
                "minimal_set_member": True,
            },
        ]
    )
    long = _long_edge_relations(frame)
    assert len(long) == 5
    assert not long.duplicated(["transition_id", "competitor_canonical_id", "edge_definition"]).any()
