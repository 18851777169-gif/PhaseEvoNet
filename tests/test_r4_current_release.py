from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import yaml

from phase_evonet.manifest import sha256_file
from phase_evonet.normalization import NORMALIZERS
from phase_evonet.r4_current_release import (
    json_safe,
    normalize_current_release,
    select_benchmark_phase_targets,
    thermo_type_from_delta_path,
)


def test_json_safe_converts_arrow_maps_and_datetimes() -> None:
    value = [
        ("mp-1", {"date": datetime(2026, 4, 13, tzinfo=UTC)}),
        ("mp-2", [("run_type", "GGA")]),
    ]
    assert json_safe(value) == {
        "mp-1": {"date": "2026-04-13T00:00:00+00:00"},
        "mp-2": {"run_type": "GGA"},
    }


def test_thermo_type_is_decoded_from_delta_partition() -> None:
    assert (
        thermo_type_from_delta_path(
            "version=2026-04-13/thermo_type=GGA_GGA%252BU_R2SCAN/part.parquet"
        )
        == "GGA_GGA+U_R2SCAN"
    )
    assert (
        thermo_type_from_delta_path(
            "version=2026-04-13/thermo_type=GGA_GGA%252BU/part.parquet"
        )
        == "GGA_GGA+U"
    )
    assert (
        thermo_type_from_delta_path(
            "version=2026-04-13/thermo_type=r2SCAN/part.parquet"
        )
        == "R2SCAN"
    )


def test_current_release_thermo_id_is_reconstructed_when_official_field_absent() -> None:
    source = {"database_version": "2026-04-13", "sha256": "a" * 64, "key": "x"}
    record = {
        "material_id": "mp-42",
        "thermo_type": "GGA_GGA+U",
        "composition_reduced": {"Li": 1},
        "formation_energy_per_atom": -1.0,
        "energy_above_hull": 0.0,
        "is_stable": True,
        "entries": {},
    }
    rows = NORMALIZERS["thermo"](record, source, 1)
    assert rows["raw_thermo"][0]["thermo_id"] == "mp-42_GGA_GGA+U"


def test_benchmark_phase_target_uses_explicit_thermo_selection() -> None:
    phase = pd.DataFrame(
        [
            {
                "snapshot_id": "2026-04-13",
                "material_id": "mp-1",
                "thermo_id": "mp-1_GGA_GGA+U",
                "thermo_type": "GGA_GGA+U",
                "phase_context_chemsys": "Nb-O-Sr",
                "entry_id": "mp-1-GGA",
                "energy_above_hull": 0.0010,
            },
            {
                "snapshot_id": "2026-04-13",
                "material_id": "mp-1",
                "thermo_id": "mp-1_GGA_GGA+U",
                "thermo_type": "GGA_GGA+U",
                "phase_context_chemsys": "Nb-O-Sr",
                "entry_id": "mp-1-GGA+U",
                "energy_above_hull": 0.0011,
            },
        ]
    )
    selection = pd.DataFrame(
        [
            {
                "thermo_id": "mp-1_GGA_GGA+U",
                "thermo_type": "GGA_GGA+U",
                "chemsys": "Nb-O-Sr",
                "selected_entry_id": "mp-1-GGA+U",
            }
        ]
    )
    selected = select_benchmark_phase_targets(phase, selection)
    assert selected["entry_id"].tolist() == ["mp-1-GGA+U"]
    assert selected["energy_above_hull"].tolist() == [0.0011]


def test_delta_normalization_is_scoped_and_deterministic(tmp_path: Path) -> None:
    repo = tmp_path
    raw = repo / "data/raw"
    raw.mkdir(parents=True)
    records = {
        "materials": {
            "material_id": "mp-a",
            "composition_reduced": {"Li": 1.0},
            "structure": {"lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "sites": []},
            "task_ids": ["mp-b"],
            "calc_types": {"mp-b": "GGA Static"},
            "task_types": {"mp-b": "Static"},
            "run_types": {"mp-b": "GGA"},
        },
        "thermo": {
            "material_id": "mp-a",
            "thermo_id": "mp-a_GGA_GGA+U",
            "composition_reduced": {"Li": 1.0},
            "formation_energy_per_atom": 0.0,
            "energy_above_hull": 0.0,
            "is_stable": True,
            "entries": {
                "GGA": {
                    "entry_id": "mp-b",
                    "energy": -1.0,
                    "composition": {"Li": 1.0},
                    "structure": {
                        "lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                        "sites": [],
                    },
                    "data": {"task_id": "mp-b"},
                }
            },
        },
        "provenance": {"material_id": "mp-a"},
    }
    manifest_rows = []
    for collection, record in records.items():
        suffix = (
            "version=2026-04-13/thermo_type=GGA_GGA%252BU/part.parquet"
            if collection == "thermo"
            else "version=2026-04-13/part.parquet"
        )
        path = raw / collection / "part.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([record]), path)
        manifest_rows.append(
            {
                "collection": collection,
                "database_version": "2026.04.13",
                "source_delta_path": suffix,
                "local_path": path.relative_to(repo).as_posix(),
                "sha256": sha256_file(path),
                "row_count": 1,
            }
        )
    manifest = repo / "data/manifests/R4_0/source_objects.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    config = {
        "task_id": "R4.0",
        "source_manifest": "data/manifests/R4_0/source_objects.jsonl",
        "output_root": "data/interim/R4_0/normalized",
        "combined_output_root": "data/interim/R4_0/combined",
        "historical_normalized_root": "data/interim/P1_2",
        "combined_historical_snapshots": [],
        "manifest_dir": "data/manifests/R4_0",
        "report_dir": "reports/R4_0",
        "compression": "zstd",
        "row_group_size": 2,
        "input_batch_size": 1,
        "critical_completeness_threshold": 1.0,
        "critical_fields": {
            "raw_material": ["snapshot_id", "material_id"],
            "raw_task": ["snapshot_id", "task_id"],
            "raw_thermo": ["snapshot_id", "thermo_type"],
            "raw_thermo_entry": ["snapshot_id", "entry_id"],
            "raw_provenance": ["snapshot_id", "material_id"],
        },
    }
    config_path = repo / "configs/data/r4_current_release.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = normalize_current_release(config_path)
    assert result["status"] == "PASS"
    thermo = pq.read_table(
        repo / "data/interim/R4_0/normalized/snapshot=2026-04-13/raw_thermo.parquet"
    ).to_pandas()
    assert thermo.loc[0, "thermo_type"] == "GGA_GGA+U"
    assert result["old_frozen_outputs_modified"] is False
