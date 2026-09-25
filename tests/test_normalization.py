import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from phase_evonet.normalization import (
    completeness_payload,
    normalize_snapshots,
    selected_objects,
    verify_normalization_outputs,
)


def write_gzip_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return {
        "database_version": "2025-01-01",
        "collection": path.stem.split("-")[0],
        "key": f"collections/2025-01-01/{path.name}",
        "file_format": "jsonl.gz",
        "local_path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(rows),
    }


def synthetic_records():
    entry = {
        "entry_id": "mp-1-GGA",
        "composition": {"Li": 1},
        "energy": -1.0,
        "correction": 0.0,
        "energy_adjustments": [],
        "parameters": {"run_type": "GGA", "hubbards": {}, "potcar_spec": []},
        "structure": {"lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "sites": []},
        "data": {"task_id": "mp-1"},
    }
    builder = {"emmet_version": "1", "pymatgen_version": "2"}
    material = {
        "material_id": "mp-1",
        "composition_reduced": {"Li": 1},
        "structure": entry["structure"],
        "task_ids": ["mp-1"],
        "calc_types": {"mp-1": "GGA Static"},
        "task_types": {"mp-1": "Static"},
        "run_types": {"mp-1": "GGA"},
        "entries": {"GGA": entry},
        "builder_meta": builder,
    }
    thermo = {
        "material_id": "mp-1",
        "thermo_id": "mp-1_GGA_GGA+U",
        "thermo_type": "GGA_GGA+U",
        "composition_reduced": {"Li": 1},
        "formation_energy_per_atom": -1.0,
        "energy_above_hull": 0.0,
        "is_stable": True,
        "entries": {"GGA": entry},
        "builder_meta": builder,
    }
    provenance = {"material_id": "mp-1", "history": [], "builder_meta": builder}
    return material, thermo, provenance


def test_selected_objects_excludes_manifest_indexes():
    config = {
        "source_selection": {
            "include_collections": ["materials"],
            "include_format": "jsonl.gz",
            "exclude_object_basename_contains": "manifest",
        }
    }
    rows = [
        {"collection": "materials", "file_format": "jsonl.gz", "key": "x/data.jsonl.gz"},
        {"collection": "materials", "file_format": "jsonl.gz", "key": "x/manifest.jsonl.gz"},
        {"collection": "materials", "file_format": "parquet", "key": "x/data.parquet"},
    ]
    selected, excluded = selected_objects(rows, config)
    assert [row["key"] for row in selected] == ["x/data.jsonl.gz"]
    assert {row["reason"] for row in excluded} == {
        "collection_manifest_or_index",
        "non_payload_format",
    }


def test_normalization_is_deterministic_and_preserves_task_and_thermo(tmp_path: Path):
    material, thermo, provenance = synthetic_records()
    source_dir = tmp_path / "raw"
    manifest_rows = [
        write_gzip_jsonl(source_dir / "materials-data.jsonl.gz", [material]),
        write_gzip_jsonl(source_dir / "thermo-data.jsonl.gz", [thermo]),
        write_gzip_jsonl(source_dir / "provenance-data.jsonl.gz", [provenance]),
    ]
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    config = {
        "task_id": "P1.2",
        "seed": 42,
        "source_manifest": manifest.as_posix(),
        "output_root": (tmp_path / "out").as_posix(),
        "manifest_dir": (tmp_path / "manifest").as_posix(),
        "report_dir": (tmp_path / "report").as_posix(),
        "compression": "zstd",
        "row_group_size": 5000,
        "critical_completeness_threshold": 0.95,
        "critical_fields": {
            "raw_material": ["snapshot_id", "material_id", "composition_reduced_json", "structure_json", "task_ids_json"],
            "raw_task": ["snapshot_id", "material_id", "task_id", "calc_type", "task_type", "run_type"],
            "raw_thermo": ["snapshot_id", "material_id", "thermo_id", "thermo_type", "composition_reduced_json", "formation_energy_per_atom", "energy_above_hull", "is_stable"],
            "raw_thermo_entry": ["snapshot_id", "material_id", "thermo_id", "entry_label", "energy", "structure_json"],
            "raw_provenance": ["snapshot_id", "material_id"],
        },
        "source_selection": {
            "include_collections": ["materials", "thermo", "provenance"],
            "include_format": "jsonl.gz",
            "exclude_object_basename_contains": "manifest",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = normalize_snapshots(config_path)
    hashes_1 = {item["table"]: item["sha256"] for item in first["artifacts"]}
    second = normalize_snapshots(config_path)
    hashes_2 = {item["table"]: item["sha256"] for item in second["artifacts"]}
    assert hashes_1 == hashes_2
    assert first["gate_passed"] is True
    verified = verify_normalization_outputs(config_path, update_manifest=True)
    assert verified["status"] == "PASS"
    assert verified["artifact_count"] == 5
    task_path = tmp_path / "out" / "snapshot=2025-01-01" / "raw_task.parquet"
    assert pq.read_table(task_path).to_pylist()[0]["calc_type"] == "GGA Static"
    thermo_entry_path = tmp_path / "out" / "snapshot=2025-01-01" / "raw_thermo_entry.parquet"
    assert pq.read_table(thermo_entry_path).to_pylist()[0]["task_id"] == "mp-1"


def test_completeness_gate_fails_without_lowering_threshold():
    rows, passed = completeness_payload(
        {("v", "raw_material"): 100},
        {("v", "raw_material", "material_id"): 94},
        {"raw_material": ["material_id"]},
        0.95,
    )
    assert passed is False
    assert rows[0]["status"] == "FAIL"
    assert rows[0]["completeness"] == pytest.approx(0.94)
