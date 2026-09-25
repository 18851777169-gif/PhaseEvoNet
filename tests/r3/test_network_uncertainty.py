from __future__ import annotations

import numpy as np
import pandas as pd

from phase_evonet.r3.network_uncertainty import (
    IMPACT_EDGES,
    candidate_subsampling,
    decide_route,
    encode_network,
    network_metrics,
    synthetic_graph_tests,
)


def test_one_to_one_metrics_are_exact() -> None:
    frame = pd.DataFrame({"node": ["a", "b", "c"], "candidate_lineage_id": ["x", "y", "z"]})
    metrics = network_metrics(frame, "node")
    assert metrics["edge_pairs"] == 3
    assert metrics["competitor_nodes"] == 3
    assert metrics["candidate_nodes"] == 3
    assert metrics["connected_components"] == 3
    assert metrics["largest_cascade"] == 1
    assert metrics["largest_cascade_fraction"] == 1 / 3
    assert metrics["gini"] == 0


def test_duplicate_node_candidate_pairs_are_aggregated_once() -> None:
    frame = pd.DataFrame({"node": ["a", "a", "a"], "candidate_lineage_id": ["x", "x", "y"]})
    metrics = network_metrics(frame, "node")
    assert metrics["edge_pairs"] == 2
    assert metrics["largest_cascade"] == 2


def test_point_estimates_do_not_depend_on_input_order() -> None:
    frame = pd.DataFrame({"node": ["a", "a", "b", "c", "c"], "candidate_lineage_id": ["x", "y", "y", "z", "x"]})
    expected = network_metrics(frame, "node")
    shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    assert network_metrics(shuffled, "node") == expected


def test_candidate_subsampling_is_without_replacement_and_deterministic() -> None:
    frame = pd.DataFrame({"node": [f"n{i % 3}" for i in range(10)], "candidate_lineage_id": [f"c{i}" for i in range(10)]})
    network = encode_network(frame, "node")
    first, first_audit = candidate_subsampling({"id": network}, fraction=0.8, replicates=50, seed=42)
    second, second_audit = candidate_subsampling({"id": network}, fraction=0.8, replicates=50, seed=42)
    assert first_audit == second_audit
    assert first_audit["replacement"] is False
    assert first_audit["sample_size"] == 8
    assert first_audit["duplicate_candidates_across_draws"] == 0
    for metric in first["id"]:
        np.testing.assert_array_equal(first["id"][metric], second["id"][metric])


def test_approved_identity_mapping_changes_alias_aggregation_only() -> None:
    frame = pd.DataFrame({"id0": ["a", "b"], "id2": ["lineage", "lineage"], "candidate_lineage_id": ["x", "y"]})
    assert network_metrics(frame, "id0")["competitor_nodes"] == 2
    assert network_metrics(frame, "id2")["competitor_nodes"] == 1


def test_workflow_partition_prevents_primary_cross_workflow_merge() -> None:
    frame = pd.DataFrame({"id2": ["L|T|w1", "L|T|w2"], "id3": ["L", "L"], "candidate_lineage_id": ["x", "y"]})
    assert network_metrics(frame, "id2")["competitor_nodes"] == 2
    assert network_metrics(frame, "id3")["competitor_nodes"] == 1


def test_all_eight_synthetic_graph_contracts_pass() -> None:
    result = synthetic_graph_tests(seed=42)
    assert len(result) == 8
    assert bool(result["passed"].all())


def test_route_requires_two_full_and_directionally_stable_edges() -> None:
    full = {edge: edge in IMPACT_EDGES[:2] for edge in IMPACT_EDGES}
    directional = pd.DataFrame(
        [
            {"edge_definition": edge, "direction_preserved": True}
            for edge in IMPACT_EDGES
            for _ in range(3)
        ]
    )
    synthetic = pd.DataFrame({"passed": [True] * 8})
    route, gate = decide_route(full, directional, synthetic, minimum_stable=2, mapping_gate_passed=True)
    assert route == "ROUTE_IDENTITY_RESOLVED"
    assert int(gate["final_edge_stable"].sum()) == 2
    directional.loc[directional["edge_definition"].eq(IMPACT_EDGES[0]).idxmax(), "direction_preserved"] = False
    route, _ = decide_route(full, directional, synthetic, minimum_stable=2, mapping_gate_passed=True)
    assert route == "ROUTE_BOUNDED"


def test_bad_mapping_forces_no_network_claim() -> None:
    full = {edge: True for edge in IMPACT_EDGES}
    directional = pd.DataFrame([{"edge_definition": edge, "direction_preserved": True} for edge in IMPACT_EDGES])
    synthetic = pd.DataFrame({"passed": [True] * 8})
    route, _ = decide_route(full, directional, synthetic, minimum_stable=2, mapping_gate_passed=False)
    assert route == "NO_NETWORK_CLAIM"
