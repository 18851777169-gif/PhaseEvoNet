import math

from phase_evonet.transition_attribution import (
    PLAYERS,
    attribute_one_transition,
    composition_signature,
    exact_shapley,
    hull_distance_linear_program,
    phase_key,
)


def record(entry_id, composition, uncorrected, correction=0.0, task_id=None):
    return {
        "entry_id": entry_id,
        "task_id": task_id or entry_id,
        "material_id": entry_id.split("-")[0],
        "composition_json": composition,
        "uncorrected_energy": uncorrected,
        "correction": correction,
    }


def transition(source, target, *, source_hull, target_hull):
    return {
        "transition_id": b"t" * 16,
        "identity_edge_id": b"e" * 16,
        "canonical_lineage_id": "lineage-1",
        "identity_confidence": "A1",
        "source_snapshot": "v1",
        "target_snapshot": "v2",
        "thermo_type": "SYNTHETIC",
        "source_material_id": "mp-candidate",
        "target_material_id": "mp-candidate",
        "source_thermo_id": "thermo-v1",
        "target_thermo_id": "thermo-v2",
        "label_flip": True,
        "s_snapshot_id": "v1",
        "q_snapshot_id": "v2",
        "s_phase_context_chemsys": "A-B",
        "q_phase_context_chemsys": "A-B",
        "s_unified_entry_id": b"s" * 16,
        "q_unified_entry_id": b"q" * 16,
        "s_is_stable": True,
        "q_is_stable": False,
        "s_energy_above_hull": source_hull,
        "q_energy_above_hull": target_hull,
        **{f"s_{key}": value for key, value in source.items()},
        **{f"q_{key}": value for key, value in target.items()},
    }


def test_phase_key_is_case_insensitive_and_scale_invariant():
    assert composition_signature('{"A": 1, "B": 1}') == composition_signature(
        '{"A": 2, "B": 2}'
    )
    assert phase_key("MP-1-R2SCAN", '{"A": 1, "B": 1}') == phase_key(
        "mp-1-r2scan", '{"A": 2, "B": 2}'
    )
    assert phase_key("mp-1-r2scan", '{"A": 1, "B": 1}') != phase_key(
        "mp-1-r2scan", '{"A": 1, "B": 2}'
    )


def test_linear_program_treats_unrepresentable_composition_as_stable():
    candidate = {
        "composition_json": '{"A": 1, "B": 1}',
        "uncorrected_energy_per_atom": 0.5,
        "correction_per_atom": 0.0,
    }
    competitors = [
        {
            "composition_json": '{"A": 1}',
            "uncorrected_energy_per_atom": 0.0,
            "correction_per_atom": 0.0,
        }
    ]
    value, status, feasible = hull_distance_linear_program(
        candidate, competitors, ["A", "B"]
    )
    assert value == 0.0
    assert status == "no_feasible_decomposition"
    assert feasible is False


def test_exact_shapley_reconstructs_full_game_difference():
    weights = [0.1, -0.2, 0.3, 0.4]
    values = {
        mask: sum(weight for index, weight in enumerate(weights) if mask & (1 << index))
        for mask in range(16)
    }
    result = exact_shapley(values)
    assert list(result) == list(PLAYERS)
    for player, expected in zip(PLAYERS, weights, strict=True):
        assert math.isclose(result[player], expected, abs_tol=1e-12)
    assert math.isclose(sum(result.values()), values[15] - values[0], abs_tol=1e-12)


def test_four_channel_attribution_reconstructs_endpoints_and_delta():
    source_candidate = record(
        "candidate", '{"A": 1, "B": 1}', 0.0, task_id="task-candidate"
    )
    target_candidate = record(
        "candidate",
        '{"A": 1, "B": 1}',
        0.4,
        -0.1,
        task_id="task-candidate",
    )
    source_rows = [
        record("a", '{"A": 1}', 0.0),
        record("b", '{"B": 1}', 0.0),
        source_candidate,
    ]
    target_rows = [
        record("a", '{"A": 1}', 0.0),
        record("b", '{"B": 1}', 0.0),
        record("new-ab", '{"A": 1, "B": 1}', -0.2),
        target_candidate,
    ]
    row, coalitions = attribute_one_transition(
        transition(source_candidate, target_candidate, source_hull=0.0, target_hull=0.25),
        source_rows,
        target_rows,
        1e-6,
    )
    assert row["attributable"] is True
    assert row["candidate_identity_contribution"] == 0.0
    assert math.isclose(row["attribution_sum"], 0.25, abs_tol=1e-12)
    assert abs(row["reconstruction_residual"]) <= 1e-12
    assert row["source_endpoint_reconstruction_error"] <= 1e-12
    assert row["target_endpoint_reconstruction_error"] <= 1e-12
    assert {item["coalition_mask"] for item in coalitions} == set(range(16))


def test_unmatched_candidate_switch_is_assigned_to_identity_channel():
    source_candidate = record("candidate-old", '{"A": 1, "B": 1}', 0.0)
    target_candidate = record("candidate-new", '{"A": 1, "B": 1}', 0.4)
    competitors = [
        record("a", '{"A": 1}', 0.0),
        record("b", '{"B": 1}', 0.0),
    ]
    row, _ = attribute_one_transition(
        transition(source_candidate, target_candidate, source_hull=0.0, target_hull=0.2),
        competitors + [source_candidate],
        competitors + [target_candidate],
        1e-6,
    )
    assert row["candidate_identity_changed"] is True
    assert math.isclose(row["candidate_identity_contribution"], 0.2, abs_tol=1e-12)
    assert math.isclose(row["competitor_inventory_contribution"], 0.0, abs_tol=1e-12)
    assert math.isclose(row["uncorrected_energy_contribution"], 0.0, abs_tol=1e-12)
    assert math.isclose(row["compatibility_correction_contribution"], 0.0, abs_tol=1e-12)
