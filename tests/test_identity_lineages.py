from __future__ import annotations

from phase_evonet.identity_lineages import (
    MatchEvaluation,
    UnionFind,
    _build_event_rows,
    classify_nonaccepted,
    deterministic_lineage_id,
    select_mutual_unique_matches,
)


def evaluation(edge: int, source: str, target: str, rms: float) -> MatchEvaluation:
    return MatchEvaluation(
        edge_id=edge.to_bytes(16, "big"),
        source_snapshot="2022-10-28",
        target_snapshot="2023-11-01",
        source_material_id=source,
        target_material_id=target,
        source_composition_key='{"Li":1,"O":1}',
        target_composition_key='{"Li":1,"O":1}',
        matched=True,
        rms=rms,
        maximum=rms * 2,
    )


def test_union_find_and_lineage_id_are_deterministic() -> None:
    first = UnionFind()
    first.union("2022|mp-2", "2023|mp-2")
    first.union("2023|mp-2", "2024|mp-2")
    second = UnionFind()
    second.union("2024|mp-2", "2023|mp-2")
    second.union("2022|mp-2", "2023|mp-2")
    assert first.find("2022|mp-2") == first.find("2024|mp-2")
    assert second.find("2022|mp-2") == second.find("2024|mp-2")
    assert deterministic_lineage_id("2022|mp-2") == deterministic_lineage_id(
        "2022|mp-2"
    )


def test_mutual_unique_match_rejects_ties_and_conflicts() -> None:
    clear = [evaluation(1, "old-a", "new-a", 0.01)]
    accepted, ambiguities = select_mutual_unique_matches(clear)
    assert accepted == {clear[0].edge_id}
    assert ambiguities == []

    tied = [
        evaluation(2, "old-b", "new-b", 0.02),
        evaluation(3, "old-b", "new-c", 0.02),
    ]
    accepted, ambiguities = select_mutual_unique_matches(tied)
    assert accepted == set()
    assert any(row["endpoint_type"] == "source" for row in ambiguities)


def test_b_and_c_are_not_silently_accepted() -> None:
    assert classify_nonaccepted(has_anchor=True, evaluated_match=False)[0] == "B"
    assert classify_nonaccepted(has_anchor=False, evaluated_match=True)[0] == "B"
    assert classify_nonaccepted(has_anchor=False, evaluated_match=False)[0] == "C"


def test_anchor_conflicts_are_retained_as_possible_events() -> None:
    events = _build_event_rows(
        [("2022", "2023", "old", "new-a", "A1")],
        [("2022", "2023", "old", "new-a"), ("2022", "2023", "old", "new-b")],
        source_degree={"2022|old": 1},
        target_degree={"2023|new-a": 1},
    )
    possible = [row for row in events if row["event_type"] == "possible_split"]
    assert len(possible) == 1
    assert possible[0]["accepted"] is False
    assert possible[0]["confidence"] == "B"
