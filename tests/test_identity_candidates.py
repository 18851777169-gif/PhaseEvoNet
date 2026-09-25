import csv
import json

from phase_evonet.identity_candidates import (
    ANCHOR_EXACT_STRUCTURE,
    ANCHOR_SAME_ID,
    ANCHOR_SHARED_TASK,
    MaterialDescriptor,
    build_anchor_masks,
    candidate_row,
    structure_fingerprint,
    write_csv_atomic,
)


def descriptor(snapshot: str, material_id: str, composition: str, task_ids=()):
    return MaterialDescriptor(
        snapshot=snapshot,
        material_id=material_id,
        composition_key=composition,
        nsites=2,
        spacegroup=225,
        volume_per_site=10.0,
        task_ids=frozenset(task_ids),
        external_ids=frozenset(),
        structure_fingerprint="fingerprint",
        deprecated=False,
        source_object_sha256="a" * 64,
        source_key="source.jsonl.gz",
        source_row_number=1,
    )


def test_structure_fingerprint_ignores_site_order_properties_and_small_noise():
    first = {
        "lattice": {"matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
        "sites": [
            {"abc": [0, 0, 0], "species": [{"element": "Na", "occu": 1}], "properties": {"x": 1}},
            {"abc": [0.5, 0.5, 0.5], "species": [{"element": "Cl", "occu": 1}]},
        ],
    }
    second = {
        "properties": {},
        "lattice": {"pbc": [True, True, True], "matrix": [[2.00000001, 0, 0], [0, 2, 0], [0, 0, 2]]},
        "sites": list(reversed(first["sites"])),
    }
    assert structure_fingerprint(json.dumps(first), coordinate_decimals=6, lattice_decimals=6) == structure_fingerprint(
        json.dumps(second), coordinate_decimals=6, lattice_decimals=6
    )


def test_anchor_masks_preserve_many_to_many_task_evidence():
    source = [descriptor("v1", "mp-1", "NaCl", ["task-1"])]
    target = [
        descriptor("v2", "mp-1", "NaCl", ["task-1"]),
        descriptor("v2", "mp-2", "NaCl", ["task-1"]),
    ]
    anchors, counts = build_anchor_masks(source, target)
    assert anchors[(0, 0)] & ANCHOR_SAME_ID
    assert anchors[(0, 0)] & ANCHOR_SHARED_TASK
    assert anchors[(0, 0)] & ANCHOR_EXACT_STRUCTURE
    assert anchors[(0, 1)] & ANCHOR_SHARED_TASK
    assert counts["shared_task_id"] == 2


def test_cross_composition_anchor_is_explicit_not_silently_dropped():
    source = descriptor("v1", "mp-1", "NaCl", ["task-1"])
    target = descriptor("v2", "mp-2", "Na2Cl2", ["task-1"])
    row = candidate_row(
        source,
        target,
        composition_block=False,
        anchor_mask=ANCHOR_SHARED_TASK,
    )
    assert row["cross_composition_anchor"] is True
    assert row["shared_task_count"] == 1
    assert row["composition_block"] is False


def test_write_csv_atomic_preserves_workflow_specific_columns(tmp_path):
    path = tmp_path / "reference_audit.csv"
    write_csv_atomic(
        path,
        [
            {"snapshot_id": "v1", "hull_absolute_error": 0.0},
            {
                "snapshot_id": "v1",
                "decomposition_hull_absolute_error": 0.0,
                "serialized_source_matched": True,
            },
        ],
    )

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "snapshot_id",
        "hull_absolute_error",
        "decomposition_hull_absolute_error",
        "serialized_source_matched",
    ]
    assert rows[0]["decomposition_hull_absolute_error"] == ""
    assert rows[1]["hull_absolute_error"] == ""
    assert rows[1]["serialized_source_matched"] == "True"
