from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import yaml
from emmet.core.mpid import AlphaID
from emmet.core.types.typing import format_identifier
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .manifest import sha256_file
from .normalization import (
    DatasetWriter,
    NORMALIZERS,
    SCHEMAS,
    canonical_json,
    completeness_payload,
    nonempty,
    structured_identifier_text,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
)
from .identity_lineages import MatchEvaluation, select_mutual_unique_matches
from .context_phase_diagrams import _thermo_target_selection
from .r3.versioned_benchmark import (
    FROZEN_HASHES,
    MODELS,
    THRESHOLDS,
    bootstrap_metrics,
    error_transition_rows,
    point_metrics,
    prediction_vector_hash,
)


SNAPSHOT_ID = "2026-04-13"
BUILDER_DATABASE_VERSION = "2026.04.13"


def select_benchmark_phase_targets(
    phase: pd.DataFrame, target_selection: pd.DataFrame
) -> pd.DataFrame:
    """Select the entry explicitly named by each frozen thermo source state.

    A thermo document may embed more than one workflow-specific entry for the
    same material and chemical context. The document's frozen ``energy_type``
    resolves that multiplicity through ``selected_entry_id``; benchmark joins
    must never choose by row order or minimum energy.
    """
    phase_keys = [
        "thermo_id",
        "thermo_type",
        "phase_context_chemsys",
        "entry_id",
    ]
    selection_keys = [
        "thermo_id",
        "thermo_type",
        "chemsys",
        "selected_entry_id",
    ]
    missing_phase = sorted(set(phase_keys) - set(phase.columns))
    missing_selection = sorted(set(selection_keys) - set(target_selection.columns))
    if missing_phase or missing_selection:
        raise KeyError(
            "Missing explicit phase-target selection columns: "
            f"phase={missing_phase}, selection={missing_selection}"
        )
    selected = phase.merge(
        target_selection[selection_keys],
        left_on=phase_keys,
        right_on=selection_keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    join_key = ["material_id", "thermo_type", "phase_context_chemsys"]
    duplicates = selected[selected.duplicated(join_key, keep=False)]
    if not duplicates.empty:
        evidence = duplicates[
            join_key + ["thermo_id", "entry_id"]
        ].sort_values(join_key + ["entry_id"], kind="mergesort")
        raise RuntimeError(
            "Explicit thermo target selection is not unique for benchmark join: "
            f"{evidence.head(10).to_dict(orient='records')}"
        )
    return selected[phase.columns].copy()


def json_safe(value: Any) -> Any:
    """Convert Arrow values to deterministic JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [json_safe(item) for item in value]
        if converted and all(
            isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)
            for item in converted
        ):
            return {item[0]: item[1] for item in converted}
        return converted
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def thermo_type_from_delta_path(path: str) -> str | None:
    marker = "thermo_type="
    if marker not in path:
        return None
    value = path.split(marker, 1)[1].split("/", 1)[0]
    # Delta paths escape '+' as %252B. Decode until stable, with a hard bound.
    for _ in range(3):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    if value == "r2SCAN":
        return "R2SCAN"
    return value


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size or sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Refusing to replace non-identical combined input: {target}")
        return "existing_identical"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def normalize_current_release(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "R4.0":
        raise ValueError("Current-release normalization config task_id must be R4.0")

    repo = config_file.resolve().parents[2]
    source_manifest_path = repo / config["source_manifest"]
    source_rows = [
        json.loads(line)
        for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_rows.sort(key=lambda row: (row["collection"], row["source_delta_path"]))
    if not source_rows or any(row["database_version"] != BUILDER_DATABASE_VERSION for row in source_rows):
        raise RuntimeError("Source manifest is not the frozen v2026.04.13 release")

    output_root = repo / config["output_root"]
    snapshot_root = output_root / f"snapshot={SNAPSHOT_ID}"
    if snapshot_root.exists() and any(snapshot_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite an existing normalized release: {snapshot_root}")

    row_group_size = int(config["row_group_size"])
    compression = str(config["compression"])
    batch_size = int(config.get("input_batch_size", row_group_size))
    critical_fields = config["critical_fields"]
    threshold = float(config["critical_completeness_threshold"])
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    writers: dict[str, DatasetWriter] = {}
    totals: dict[tuple[str, str], int] = defaultdict(int)
    present: dict[tuple[str, str, str], int] = defaultdict(int)
    observed_source_rows = 0

    try:
        for object_index, source_row in enumerate(source_rows, start=1):
            collection = source_row["collection"]
            if collection not in NORMALIZERS:
                raise RuntimeError(f"Unsupported source collection: {collection}")
            local_path = repo / source_row["local_path"]
            if sha256_file(local_path) != source_row["sha256"]:
                raise RuntimeError(f"Source SHA-256 mismatch: {source_row['local_path']}")
            source = {
                "database_version": SNAPSHOT_ID,
                "sha256": source_row["sha256"],
                "key": source_row["source_delta_path"],
            }
            thermo_type = thermo_type_from_delta_path(source_row["source_delta_path"])
            parquet_file = pq.ParquetFile(local_path)
            source_position = 0
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                for arrow_record in batch.to_pylist():
                    source_position += 1
                    record = json_safe(arrow_record)
                    if collection == "thermo":
                        if not thermo_type:
                            raise RuntimeError(f"Missing thermo_type partition: {source_row['source_delta_path']}")
                        observed_type = record.get("thermo_type")
                        if observed_type not in (None, thermo_type):
                            raise RuntimeError(
                                f"Thermo partition/value mismatch: {observed_type!r} != {thermo_type!r}"
                            )
                        record["thermo_type"] = thermo_type
                    normalized = NORMALIZERS[collection](record, source, source_position)
                    for table_name, rows in normalized.items():
                        buffers[table_name].extend(rows)
                        totals[(SNAPSHOT_ID, table_name)] += len(rows)
                        for field in critical_fields[table_name]:
                            present[(SNAPSHOT_ID, table_name, field)] += sum(
                                nonempty(row.get(field)) for row in rows
                            )
                        if len(buffers[table_name]) >= row_group_size:
                            writer = writers.get(table_name)
                            if writer is None:
                                writer = DatasetWriter(
                                    snapshot_root / f"{table_name}.parquet",
                                    table_name,
                                    compression,
                                    row_group_size,
                                )
                                writers[table_name] = writer
                            writer.write(buffers[table_name])
                            buffers[table_name].clear()
                    if source_position % 25_000 == 0:
                        print(
                            f"normalizing_object={object_index}/{len(source_rows)} "
                            f"collection={collection} object_rows={source_position}",
                            flush=True,
                        )
            if source_position != int(source_row["row_count"]):
                raise RuntimeError(
                    f"Source row mismatch for {source_row['source_delta_path']}: "
                    f"expected={source_row['row_count']} observed={source_position}"
                )
            observed_source_rows += source_position
            print(
                f"normalized_objects={object_index}/{len(source_rows)} source_rows={observed_source_rows}",
                flush=True,
            )

        for table_name in sorted(buffers):
            if buffers[table_name]:
                writer = writers.get(table_name)
                if writer is None:
                    writer = DatasetWriter(
                        snapshot_root / f"{table_name}.parquet",
                        table_name,
                        compression,
                        row_group_size,
                    )
                    writers[table_name] = writer
                writer.write(buffers[table_name])
        for table_name in sorted(writers):
            writers[table_name].close()
    except Exception:
        for writer in writers.values():
            try:
                writer.writer.close()
            except Exception:
                pass
            writer.temporary.unlink(missing_ok=True)
        raise

    missing_tables = sorted(set(critical_fields) - set(writers))
    if missing_tables:
        raise RuntimeError(f"Missing normalized tables: {missing_tables}")
    completeness, passed = completeness_payload(totals, present, critical_fields, threshold)
    if not passed:
        raise RuntimeError("Current-release normalization failed critical completeness")

    artifacts: list[dict[str, Any]] = []
    for table_name, writer in sorted(writers.items()):
        path = writer.path
        actual_schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
        if actual_schema != SCHEMAS[table_name].remove_metadata():
            raise RuntimeError(f"Normalized schema mismatch: {table_name}")
        artifacts.append(
            {
                "snapshot_id": SNAPSHOT_ID,
                "table": table_name,
                "path": path.relative_to(repo).as_posix(),
                "rows": int(pq.ParquetFile(path).metadata.num_rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": hashlib.sha256(str(actual_schema).encode()).hexdigest(),
            }
        )

    combined_root = repo / config["combined_output_root"]
    link_rows: list[dict[str, Any]] = []
    for old_snapshot in config["combined_historical_snapshots"]:
        old_snapshot_text = str(old_snapshot)
        old_root = repo / config["historical_normalized_root"] / f"snapshot={old_snapshot_text}"
        for source_path in sorted(old_root.glob("*.parquet")):
            target = combined_root / f"snapshot={old_snapshot_text}" / source_path.name
            mode = _link_or_copy(source_path, target)
            link_rows.append(
                {
                    "snapshot_id": old_snapshot_text,
                    "table": source_path.stem,
                    "source_path": source_path.relative_to(repo).as_posix(),
                    "combined_path": target.relative_to(repo).as_posix(),
                    "mode": mode,
                    "sha256": sha256_file(target),
                }
            )
    for artifact in artifacts:
        source_path = repo / artifact["path"]
        target = combined_root / f"snapshot={SNAPSHOT_ID}" / source_path.name
        mode = _link_or_copy(source_path, target)
        link_rows.append(
            {
                "snapshot_id": SNAPSHOT_ID,
                "table": source_path.stem,
                "source_path": source_path.relative_to(repo).as_posix(),
                "combined_path": target.relative_to(repo).as_posix(),
                "mode": mode,
                "sha256": sha256_file(target),
            }
        )

    report_dir = repo / config["report_dir"]
    manifest_dir = repo / config["manifest_dir"]
    write_csv_atomic(report_dir / "normalization_completeness.csv", completeness)
    write_csv_atomic(report_dir / "combined_input_inventory.csv", link_rows)
    result = {
        "task_id": "R4.0",
        "stage": "normalize_current_release",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "snapshot_id": SNAPSHOT_ID,
        "builder_database_version": BUILDER_DATABASE_VERSION,
        "source_manifest": source_manifest_path.relative_to(repo).as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_objects": len(source_rows),
        "source_rows": observed_source_rows,
        "artifacts": artifacts,
        "critical_completeness": completeness,
        "combined_inputs": link_rows,
        "network_access": False,
        "old_frozen_outputs_modified": False,
        "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
    }
    write_json_atomic(manifest_dir / "normalization_manifest.json", result)
    result["normalization_manifest_sha256"] = sha256_file(
        manifest_dir / "normalization_manifest.json"
    )
    write_json_atomic(report_dir / "normalization_result.json", result)
    return result


def finalize_existing_current_release_normalization(
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate and finalize an atomically closed normalization after a late interruption.

    This recovery path is intentionally stricter than trusting file presence: it
    revalidates every frozen source hash and row count, every normalized schema,
    base-table row conservation, Parquet readability, and all configured critical
    field completeness before emitting the normal manifest.
    """
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "R4.0":
        raise ValueError("Current-release normalization config task_id must be R4.0")

    repo = config_file.resolve().parents[2]
    source_manifest_path = repo / config["source_manifest"]
    source_rows = [
        json.loads(line)
        for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_rows.sort(key=lambda row: (row["collection"], row["source_delta_path"]))
    if not source_rows or any(
        row["database_version"] != BUILDER_DATABASE_VERSION for row in source_rows
    ):
        raise RuntimeError("Source manifest is not the frozen v2026.04.13 release")

    expected_source_rows = 0
    collection_rows: dict[str, int] = defaultdict(int)
    for source_row in source_rows:
        local_path = repo / source_row["local_path"]
        if sha256_file(local_path) != source_row["sha256"]:
            raise RuntimeError(f"Source SHA-256 mismatch: {source_row['local_path']}")
        actual_rows = int(pq.ParquetFile(local_path).metadata.num_rows)
        if actual_rows != int(source_row["row_count"]):
            raise RuntimeError(
                f"Source row mismatch for {source_row['local_path']}: "
                f"expected={source_row['row_count']} observed={actual_rows}"
            )
        expected_source_rows += actual_rows
        collection_rows[str(source_row["collection"])] += actual_rows

    snapshot_root = repo / config["output_root"] / f"snapshot={SNAPSHOT_ID}"
    critical_fields = config["critical_fields"]
    threshold = float(config["critical_completeness_threshold"])
    unexpected = sorted(
        path.name
        for path in snapshot_root.iterdir()
        if path.is_file() and path.suffix != ".parquet"
    )
    if unexpected:
        raise RuntimeError(f"Unexpected files in normalized release: {unexpected}")

    totals: dict[tuple[str, str], int] = defaultdict(int)
    present: dict[tuple[str, str, str], int] = defaultdict(int)
    artifacts: list[dict[str, Any]] = []
    for table_name in sorted(critical_fields):
        path = snapshot_root / f"{table_name}.parquet"
        if not path.is_file():
            raise RuntimeError(f"Missing normalized table: {path}")
        parquet_file = pq.ParquetFile(path)
        actual_schema = parquet_file.schema_arrow.remove_metadata()
        if actual_schema != SCHEMAS[table_name].remove_metadata():
            raise RuntimeError(f"Normalized schema mismatch: {table_name}")
        row_count = int(parquet_file.metadata.num_rows)
        totals[(SNAPSHOT_ID, table_name)] = row_count
        for batch in parquet_file.iter_batches(
            batch_size=10_000, columns=list(critical_fields[table_name])
        ):
            for row in batch.to_pylist():
                for field in critical_fields[table_name]:
                    present[(SNAPSHOT_ID, table_name, field)] += nonempty(row.get(field))
        artifacts.append(
            {
                "snapshot_id": SNAPSHOT_ID,
                "table": table_name,
                "path": path.relative_to(repo).as_posix(),
                "rows": row_count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": hashlib.sha256(str(actual_schema).encode()).hexdigest(),
            }
        )

    conserved = {
        "raw_material": collection_rows.get("materials", 0),
        "raw_thermo": collection_rows.get("thermo", 0),
        "raw_provenance": collection_rows.get("provenance", 0),
    }
    artifact_rows = {item["table"]: item["rows"] for item in artifacts}
    for table_name, expected_rows in conserved.items():
        if artifact_rows.get(table_name) != expected_rows:
            raise RuntimeError(
                f"Normalized base-table row conservation failed for {table_name}: "
                f"expected={expected_rows} observed={artifact_rows.get(table_name)}"
            )

    completeness, passed = completeness_payload(
        totals, present, critical_fields, threshold
    )
    if not passed:
        raise RuntimeError("Current-release normalization failed critical completeness")

    combined_root = repo / config["combined_output_root"]
    link_rows: list[dict[str, Any]] = []
    for old_snapshot in config["combined_historical_snapshots"]:
        old_snapshot_text = str(old_snapshot)
        old_root = repo / config["historical_normalized_root"] / f"snapshot={old_snapshot_text}"
        for source_path in sorted(old_root.glob("*.parquet")):
            target = combined_root / f"snapshot={old_snapshot_text}" / source_path.name
            mode = _link_or_copy(source_path, target)
            link_rows.append(
                {
                    "snapshot_id": old_snapshot_text,
                    "table": source_path.stem,
                    "source_path": source_path.relative_to(repo).as_posix(),
                    "combined_path": target.relative_to(repo).as_posix(),
                    "mode": mode,
                    "sha256": sha256_file(target),
                }
            )
    for artifact in artifacts:
        source_path = repo / artifact["path"]
        target = combined_root / f"snapshot={SNAPSHOT_ID}" / source_path.name
        mode = _link_or_copy(source_path, target)
        link_rows.append(
            {
                "snapshot_id": SNAPSHOT_ID,
                "table": source_path.stem,
                "source_path": source_path.relative_to(repo).as_posix(),
                "combined_path": target.relative_to(repo).as_posix(),
                "mode": mode,
                "sha256": sha256_file(target),
            }
        )

    report_dir = repo / config["report_dir"]
    manifest_dir = repo / config["manifest_dir"]
    write_csv_atomic(report_dir / "normalization_completeness.csv", completeness)
    write_csv_atomic(report_dir / "combined_input_inventory.csv", link_rows)
    result = {
        "task_id": "R4.0",
        "stage": "normalize_current_release",
        "status": "PASS",
        "completion_mode": "validated_finalize_after_late_interruption",
        "completed_at_utc": utc_now(),
        "snapshot_id": SNAPSHOT_ID,
        "builder_database_version": BUILDER_DATABASE_VERSION,
        "source_manifest": source_manifest_path.relative_to(repo).as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_objects": len(source_rows),
        "source_rows": expected_source_rows,
        "source_collection_rows": dict(sorted(collection_rows.items())),
        "base_table_row_conservation": conserved,
        "artifacts": artifacts,
        "critical_completeness": completeness,
        "combined_inputs": link_rows,
        "network_access": False,
        "old_frozen_outputs_modified": False,
        "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
    }
    write_json_atomic(manifest_dir / "normalization_manifest.json", result)
    result["normalization_manifest_sha256"] = sha256_file(
        manifest_dir / "normalization_manifest.json"
    )
    write_json_atomic(report_dir / "normalization_result.json", result)
    return result


def repair_missing_current_release_thermo_ids(config_path: str | Path) -> dict[str, Any]:
    """Atomically restore the historical MP thermo identifier in R4 outputs."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "R4.0":
        raise ValueError("Thermo ID repair is restricted to R4.0")
    repo = config_file.resolve().parents[2]
    snapshot_root = repo / config["output_root"] / f"snapshot={SNAPSHOT_ID}"
    table_names = ("raw_thermo", "raw_thermo_entry")
    before: list[dict[str, Any]] = []
    temporary_paths: dict[str, Path] = {}
    base_keys: set[tuple[str, str]] = set()

    try:
        for table_name in table_names:
            path = snapshot_root / f"{table_name}.parquet"
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
            if schema.remove_metadata() != SCHEMAS[table_name].remove_metadata():
                raise RuntimeError(f"Schema mismatch before thermo ID repair: {table_name}")
            null_count = 0
            nonnull_count = 0
            for batch in parquet_file.iter_batches(
                batch_size=20_000,
                columns=["material_id", "thermo_id", "thermo_type"],
            ):
                for row in batch.to_pylist():
                    if row["thermo_id"] is None:
                        null_count += 1
                    else:
                        nonnull_count += 1
                        expected_id = f"{row['material_id']}_{row['thermo_type']}"
                        if row["thermo_id"] != expected_id:
                            raise RuntimeError(
                                f"Existing thermo ID violates reconstruction rule: {table_name}"
                            )
                    if table_name == "raw_thermo":
                        key = (str(row["material_id"]), str(row["thermo_type"]))
                        if key in base_keys:
                            raise RuntimeError(f"Non-unique reconstructed thermo key: {key}")
                        base_keys.add(key)
            parquet_file.close()
            if nonnull_count and null_count:
                raise RuntimeError(
                    f"Refusing mixed null/non-null thermo ID repair for {table_name}"
                )
            before.append(
                {
                    "table": table_name,
                    "path": path.relative_to(repo).as_posix(),
                    "rows": int(parquet_file.metadata.num_rows),
                    "missing_thermo_ids": null_count,
                    "sha256": sha256_file(path),
                }
            )

        tables_to_repair = {
            item["table"] for item in before if item["missing_thermo_ids"] > 0
        }
        for table_name in table_names:
            if table_name not in tables_to_repair:
                continue
            path = snapshot_root / f"{table_name}.parquet"
            temporary = path.with_name(path.name + ".thermo-id-repair.tmp")
            temporary.unlink(missing_ok=True)
            temporary_paths[table_name] = temporary
            parquet_file = pq.ParquetFile(path)
            writer = pq.ParquetWriter(
                temporary,
                SCHEMAS[table_name],
                compression=str(config["compression"]),
            )
            try:
                for batch in parquet_file.iter_batches(batch_size=5_000):
                    rows = batch.to_pylist()
                    for row in rows:
                        material_id = str(row["material_id"] or "")
                        thermo_type = str(row["thermo_type"] or "")
                        if not material_id or not thermo_type:
                            raise RuntimeError(
                                f"Cannot reconstruct thermo ID in {table_name}"
                            )
                        row["thermo_id"] = f"{material_id}_{thermo_type}"
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[table_name]))
            finally:
                writer.close()
                parquet_file.close()

        for table_name, temporary in temporary_paths.items():
            expected = next(item["rows"] for item in before if item["table"] == table_name)
            parquet_file = pq.ParquetFile(temporary)
            if int(parquet_file.metadata.num_rows) != expected:
                raise RuntimeError(f"Row loss during thermo ID repair: {table_name}")
            for batch in parquet_file.iter_batches(
                batch_size=20_000,
                columns=["material_id", "thermo_id", "thermo_type"],
            ):
                for row in batch.to_pylist():
                    expected_id = f"{row['material_id']}_{row['thermo_type']}"
                    if row["thermo_id"] != expected_id:
                        raise RuntimeError(f"Thermo ID repair mismatch: {table_name}")
            parquet_file.close()

        for table_name, temporary in temporary_paths.items():
            os.replace(temporary, snapshot_root / f"{table_name}.parquet")
    except Exception:
        for temporary in temporary_paths.values():
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass
        raise

    after = []
    for table_name in table_names:
        path = snapshot_root / f"{table_name}.parquet"
        parquet_file = pq.ParquetFile(path)
        after.append(
            {
                "table": table_name,
                "path": path.relative_to(repo).as_posix(),
                "rows": int(parquet_file.metadata.num_rows),
                "sha256": sha256_file(path),
                "thermo_id_rule": "material_id + '_' + thermo_type",
            }
        )
        parquet_file.close()
    result = {
        "task_id": "R4.0",
        "stage": "repair_current_release_thermo_ids",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "reason": "Official v2026.04.13 Delta thermo schema omits the redundant thermo_id column",
        "historical_contract_preserved": True,
        "base_key_uniqueness_verified": True,
        "before": before,
        "after": after,
    }
    write_json_atomic(repo / config["report_dir"] / "thermo_id_schema_repair.json", result)
    return result


def repair_structured_current_release_entry_ids(config_path: str | Path) -> dict[str, Any]:
    """Atomically replace repr(dict) entry IDs with official composite strings."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    repo = config_file.resolve().parents[2]
    path = (
        repo
        / config["output_root"]
        / f"snapshot={SNAPSHOT_ID}"
        / "raw_thermo_entry.parquet"
    )
    before_hash = sha256_file(path)
    temporary = path.with_name(path.name + ".entry-id-repair.tmp")
    temporary.unlink(missing_ok=True)
    parquet_file = pq.ParquetFile(path)
    writer = pq.ParquetWriter(
        temporary, SCHEMAS["raw_thermo_entry"], compression=str(config["compression"])
    )
    rows_seen = 0
    rows_repaired = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=5_000):
            rows = batch.to_pylist()
            for row in rows:
                rows_seen += 1
                value = row.get("entry_id")
                if isinstance(value, str) and value.startswith("{'identifier':"):
                    parsed = ast.literal_eval(value)
                    repaired = structured_identifier_text(parsed)
                    if repaired is None:
                        raise RuntimeError("Structured entry ID could not be serialized")
                    row["entry_id"] = repaired
                    rows_repaired += 1
                elif not value:
                    raise RuntimeError("Missing entry ID during structured-ID repair")
            writer.write_table(
                pa.Table.from_pylist(rows, schema=SCHEMAS["raw_thermo_entry"])
            )
    finally:
        writer.close()
        parquet_file.close()
    if rows_repaired not in (0, rows_seen):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Refusing partially structured entry-ID repair")
    if rows_repaired == 0:
        temporary.unlink(missing_ok=True)
        status = "already_canonical"
    else:
        check = pq.ParquetFile(temporary)
        if int(check.metadata.num_rows) != rows_seen:
            check.close()
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Row loss during structured entry-ID repair")
        for batch in check.iter_batches(batch_size=20_000, columns=["entry_id"]):
            if any(str(value).startswith("{'identifier':") for value in batch.column(0).to_pylist()):
                check.close()
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Structured entry-ID repair left repr(dict) values")
        check.close()
        os.replace(temporary, path)
        status = "repaired"

    combined = (
        repo
        / config["combined_output_root"]
        / f"snapshot={SNAPSHOT_ID}"
        / "raw_thermo_entry.parquet"
    )
    if rows_repaired:
        if not combined.is_file() or sha256_file(combined) != before_hash:
            raise RuntimeError("Combined input is not the expected pre-repair hardlink")
        combined.unlink()
        try:
            os.link(path, combined)
            combined_mode = "hardlink_refreshed"
        except OSError:
            shutil.copy2(path, combined)
            combined_mode = "copy_refreshed"
    else:
        combined_mode = "unchanged"
    after_hash = sha256_file(path)
    result = {
        "task_id": "R4.0",
        "stage": "repair_structured_entry_ids",
        "status": "PASS",
        "repair_status": status,
        "completed_at_utc": utc_now(),
        "rows": rows_seen,
        "rows_repaired": rows_repaired,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "rule": "identifier + separator + suffix",
        "combined_input_mode": combined_mode,
        "combined_input_sha256": sha256_file(combined),
    }
    write_json_atomic(
        repo / config["report_dir"] / "structured_entry_id_repair.json", result
    )
    return result


def build_current_release_terminal_supplement(config_path: str | Path) -> dict[str, Any]:
    """Recover missing elemental terminals from official same-release material entries."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    repo = config_file.resolve().parents[2]
    snapshot_root = repo / config["output_root"] / f"snapshot={SNAPSHOT_ID}"
    thermo_path = snapshot_root / "raw_thermo.parquet"
    entry_path = snapshot_root / "raw_thermo_entry.parquet"
    material_path = snapshot_root / "raw_material.parquet"
    workflows = {"GGA_GGA+U": ("GGA", "GGA_U"), "R2SCAN": ("R2SCAN",)}

    needed: dict[str, set[str]] = {workflow: set() for workflow in workflows}
    referenced_materials: dict[str, set[str]] = {
        workflow: set() for workflow in workflows
    }
    for batch in pq.ParquetFile(thermo_path).iter_batches(
        batch_size=20_000,
        columns=["thermo_type", "composition_reduced_json", "decomposes_to_json"],
    ):
        for row in batch.to_pylist():
            workflow = str(row["thermo_type"])
            if workflow in needed:
                needed[workflow].update(json.loads(row["composition_reduced_json"]))
                for component in json.loads(row["decomposes_to_json"] or "[]"):
                    material_id = component.get("material_id")
                    if material_id:
                        referenced_materials[workflow].add(str(material_id))
    present: dict[str, set[str]] = {workflow: set() for workflow in workflows}
    present_materials: dict[str, set[str]] = {
        workflow: set() for workflow in workflows
    }
    for batch in pq.ParquetFile(entry_path).iter_batches(
        batch_size=20_000,
        columns=["thermo_type", "material_id", "composition_json"],
    ):
        for row in batch.to_pylist():
            workflow = str(row["thermo_type"])
            if workflow not in present:
                continue
            if row["material_id"]:
                present_materials[workflow].add(str(row["material_id"]))
            composition = json.loads(row["composition_json"])
            if len(composition) == 1:
                present[workflow].update(composition)
    missing = {
        workflow: sorted(needed[workflow] - present[workflow])
        for workflow in workflows
    }
    missing_references = {
        workflow: sorted(
            referenced_materials[workflow] - present_materials[workflow]
        )
        for workflow in workflows
    }

    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(material_path).iter_batches(
        batch_size=10_000,
        columns=[
            "material_id",
            "composition_reduced_json",
            "entries_json",
            "source_object_sha256",
            "source_key",
            "source_row_number",
        ],
        ):
        for material in batch.to_pylist():
            composition = json.loads(material["composition_reduced_json"])
            element = next(iter(composition)) if len(composition) == 1 else None
            entries = json.loads(material["entries_json"] or "{}")
            for workflow, labels in workflows.items():
                needs_elemental_terminal = element in missing[workflow]
                needs_referenced_competitor = str(material["material_id"]) in set(
                    missing_references[workflow]
                )
                if not (needs_elemental_terminal or needs_referenced_competitor):
                    continue
                for label in labels:
                    entry = entries.get(label)
                    if not isinstance(entry, dict):
                        continue
                    parameters = entry.get("parameters") or {}
                    data = entry.get("data") or {}
                    display_label = label.replace("_U", "+U")
                    rows.append(
                        {
                            "snapshot_id": SNAPSHOT_ID,
                            "material_id": str(material["material_id"]),
                            "thermo_id": f"{material['material_id']}_{workflow}_TERMINAL_SUPPLEMENT",
                            "thermo_type": workflow,
                            "entry_label": display_label,
                            "entry_id": structured_identifier_text(entry.get("entry_id")),
                            "task_id": str(data.get("task_id")) if data.get("task_id") else None,
                            "energy": float(entry["energy"]),
                            "correction": float(entry.get("correction") or 0.0),
                            "composition_json": canonical_json(entry.get("composition")),
                            "energy_adjustments_json": canonical_json(entry.get("energy_adjustments")),
                            "parameters_json": canonical_json(parameters),
                            "run_type": str(parameters.get("run_type") or data.get("run_type") or ""),
                            "hubbards_json": canonical_json(parameters.get("hubbards")),
                            "potcar_spec_json": canonical_json(parameters.get("potcar_spec")),
                            "structure_json": canonical_json(entry.get("structure")),
                            "entry_data_json": canonical_json(data),
                            "source_object_sha256": str(material["source_object_sha256"]),
                            "source_key": str(material["source_key"]),
                            "source_row_number": int(material["source_row_number"]),
                        }
                    )
    deduplicated = {
        (str(row["thermo_type"]), str(row["entry_id"])): row for row in rows
    }
    rows = list(deduplicated.values())
    covered = {
        workflow: sorted(
            {
                element
                for row in rows
                if row["thermo_type"] == workflow
                for composition in [json.loads(row["composition_json"])]
                if len(composition) == 1
                for element in composition
            }
        )
        for workflow in workflows
    }
    covered_references = {
        workflow: sorted(
            {
                str(row["material_id"])
                for row in rows
                if row["thermo_type"] == workflow
                and str(row["material_id"]) in set(missing_references[workflow])
            }
        )
        for workflow in workflows
    }
    unresolved = {
        workflow: sorted(set(missing[workflow]) - set(covered[workflow]))
        for workflow in workflows
    }
    unresolved_references = {
        workflow: sorted(
            set(missing_references[workflow]) - set(covered_references[workflow])
        )
        for workflow in workflows
    }
    if any(unresolved.values()) or any(unresolved_references.values()):
        raise RuntimeError(
            "Missing official phase-context supplements: "
            f"terminals={unresolved} references={unresolved_references}"
        )
    rows.sort(key=lambda row: (row["thermo_type"], row["entry_id"], row["material_id"]))
    output = snapshot_root / "raw_terminal_entry_supplement.parquet"
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["raw_thermo_entry"]),
        temporary,
        compression=str(config["compression"]),
    )
    os.replace(temporary, output)
    report_rows = []
    for workflow in workflows:
        report_rows.append(
            {
                "thermo_type": workflow,
                "needed_elements": len(needed[workflow]),
                "native_terminal_elements": len(present[workflow]),
                "missing_terminal_elements_json": json.dumps(missing[workflow]),
                "supplemented_terminal_elements_json": json.dumps(covered[workflow]),
                "unresolved_terminal_elements_json": json.dumps(unresolved[workflow]),
                "referenced_competitor_materials": len(referenced_materials[workflow]),
                "missing_referenced_materials_json": json.dumps(missing_references[workflow]),
                "supplemented_referenced_materials_json": json.dumps(covered_references[workflow]),
                "unresolved_referenced_materials_json": json.dumps(unresolved_references[workflow]),
                "supplement_rows": sum(row["thermo_type"] == workflow for row in rows),
            }
        )
    write_csv_atomic(
        repo / config["report_dir"] / "terminal_supplement_coverage.csv", report_rows
    )
    result = {
        "task_id": "R4.0",
        "stage": "build_official_terminal_supplement",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "output": output.relative_to(repo).as_posix(),
        "rows": len(rows),
        "sha256": sha256_file(output),
        "coverage": report_rows,
        "source_scope": "official same-release materials collection only",
        "target_states_created": 0,
        "intended_use": "context-only official terminal and decomposition competitors",
    }
    write_json_atomic(
        repo / config["manifest_dir"] / "terminal_supplement_manifest.json", result
    )
    write_json_atomic(
        repo / config["report_dir"] / "terminal_supplement_result.json", result
    )
    return result


def _numeric_identifier(value: Any) -> int | None:
    try:
        return int(AlphaID(str(value)))
    except (TypeError, ValueError):
        return None


def _task_key_set(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()
    result: set[int] = set()
    for item in json.loads(value):
        numeric = _numeric_identifier(item)
        if numeric is not None:
            result.add(numeric)
    return frozenset(result)


def _load_panel_materials(
    path: Path, required_ids: set[str] | None, required_compositions: set[str] | None
) -> dict[str, dict[str, Any]]:
    columns = [
        "material_id",
        "composition_reduced_json",
        "task_ids_json",
        "structure_json",
        "source_object_sha256",
        "source_key",
        "source_row_number",
    ]
    result: dict[str, dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=10_000, columns=columns):
        for row in batch.to_pylist():
            material_id = str(row["material_id"])
            composition = str(row["composition_reduced_json"])
            if required_ids is not None and material_id in required_ids:
                pass
            elif required_compositions is not None and composition in required_compositions:
                pass
            else:
                continue
            if material_id in result:
                raise RuntimeError(f"Duplicate release-level material_id: {material_id}")
            result[material_id] = {
                **row,
                "material_id": material_id,
                "composition_reduced_json": composition,
                "task_keys": _task_key_set(row["task_ids_json"]),
                "numeric_material_id": _numeric_identifier(material_id),
            }
    return result


def freeze_panel_identity_mapping(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "R4.0" or config.get("seed") != 42:
        raise ValueError("R4.0 identity mapping requires the frozen seed 42 config")
    repo = config_file.resolve().parents[2]
    common_path = repo / config["identity"]["common_panel_index"]
    common = pd.read_parquet(common_path)
    source_snapshot = str(config["identity"]["source_snapshot"])
    target_snapshot = str(config["identity"]["target_snapshot"])
    source_panel = common[common["snapshot_id"].astype(str).eq(source_snapshot)].copy()
    if source_panel.empty or source_panel["panel_unit_id"].duplicated().any():
        raise RuntimeError("Frozen common panel does not have one 2025 state per panel unit")
    source_ids = set(source_panel["material_id"].astype(str))
    normalized_root = repo / config["combined_output_root"]
    source_materials = _load_panel_materials(
        normalized_root / f"snapshot={source_snapshot}" / "raw_material.parquet",
        source_ids,
        None,
    )
    missing_source = sorted(source_ids - set(source_materials))
    if missing_source:
        raise RuntimeError(f"Missing frozen 2025 panel materials: {missing_source[:5]}")
    relevant_compositions = {
        row["composition_reduced_json"] for row in source_materials.values()
    }
    target_materials = _load_panel_materials(
        normalized_root / f"snapshot={target_snapshot}" / "raw_material.parquet",
        None,
        relevant_compositions,
    )

    target_by_numeric: dict[int, str] = {}
    target_by_composition: dict[str, list[str]] = defaultdict(list)
    for material_id, row in target_materials.items():
        numeric = row["numeric_material_id"]
        if numeric is not None:
            prior = target_by_numeric.get(numeric)
            if prior is not None and prior != material_id:
                raise RuntimeError(f"Current release has duplicate canonical ID {numeric}")
            target_by_numeric[numeric] = material_id
        target_by_composition[row["composition_reduced_json"]].append(material_id)
    for values in target_by_composition.values():
        values.sort()

    accepted: dict[str, dict[str, Any]] = {}
    used_targets: set[str] = set()
    format_rows: list[dict[str, Any]] = []
    for source_id in sorted(source_ids):
        source = source_materials[source_id]
        numeric = source["numeric_material_id"]
        canonical_alpha = (
            str(format_identifier(AlphaID(source_id), legacy=False))
            if numeric is not None
            else None
        )
        target_id = target_by_numeric.get(numeric) if numeric is not None else None
        shared_tasks = (
            len(source["task_keys"] & target_materials[target_id]["task_keys"])
            if target_id is not None
            else 0
        )
        format_rows.append(
            {
                "source_material_id": source_id,
                "canonical_numeric_id": numeric,
                "canonical_alpha_id": canonical_alpha,
                "observed_target_material_id": target_id,
                "canonical_target_present": target_id is not None,
                "shared_task_count": shared_tasks,
                "format_only_event": False,
            }
        )
        if target_id is not None and shared_tasks >= 1:
            if target_id in used_targets:
                raise RuntimeError(f"A1 target conflict: {target_id}")
            accepted[source_id] = {
                "target_material_id": target_id,
                "confidence": "A1",
                "method": "canonical_material_id_plus_shared_task",
                "shared_task_count": shared_tasks,
                "rms": None,
                "maximum": None,
            }
            used_targets.add(target_id)

    matcher_config = config["identity"]["structure_matcher"]
    matcher = StructureMatcher(**matcher_config)
    unresolved_ids = sorted(source_ids - set(accepted))
    source_structures: dict[str, Structure] = {}
    target_structures: dict[str, Structure] = {}
    evaluations: list[MatchEvaluation] = []
    evaluation_errors: list[dict[str, Any]] = []
    for source_id in unresolved_ids:
        source = source_materials[source_id]
        candidates = [
            item
            for item in target_by_composition[source["composition_reduced_json"]]
            if item not in used_targets
        ]
        if source_id not in source_structures:
            source_structures[source_id] = Structure.from_dict(
                json.loads(source["structure_json"])
            )
        for target_id in candidates:
            if target_id not in target_structures:
                target_structures[target_id] = Structure.from_dict(
                    json.loads(target_materials[target_id]["structure_json"])
                )
            body = f"{source_snapshot}|{source_id}|{target_snapshot}|{target_id}".encode()
            edge_id = hashlib.blake2b(body, digest_size=16).digest()
            matched = False
            rms = None
            maximum = None
            error = None
            try:
                matched = bool(matcher.fit(source_structures[source_id], target_structures[target_id]))
                if matched:
                    distances = matcher.get_rms_dist(
                        source_structures[source_id], target_structures[target_id]
                    )
                    if distances is not None:
                        rms, maximum = float(distances[0]), float(distances[1])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                evaluation_errors.append(
                    {
                        "record_type": "structure_evaluation_error",
                        "source_material_id": source_id,
                        "target_material_id": target_id,
                        "error": error,
                    }
                )
            evaluations.append(
                MatchEvaluation(
                    edge_id=edge_id,
                    source_snapshot=source_snapshot,
                    target_snapshot=target_snapshot,
                    source_material_id=source_id,
                    target_material_id=target_id,
                    source_composition_key=source["composition_reduced_json"],
                    target_composition_key=target_materials[target_id]["composition_reduced_json"],
                    matched=matched,
                    rms=rms,
                    maximum=maximum,
                    error=error,
                )
            )
    accepted_edges, ambiguity_rows = select_mutual_unique_matches(evaluations)
    for item in evaluations:
        if item.edge_id not in accepted_edges:
            continue
        if item.target_material_id in used_targets or item.source_material_id in accepted:
            raise RuntimeError("A2 endpoint conflicts with a frozen A1 endpoint")
        accepted[item.source_material_id] = {
            "target_material_id": item.target_material_id,
            "confidence": "A2",
            "method": "mutual_unique_periodic_structure_match",
            "shared_task_count": len(
                source_materials[item.source_material_id]["task_keys"]
                & target_materials[item.target_material_id]["task_keys"]
            ),
            "rms": item.rms,
            "maximum": item.maximum,
        }
        used_targets.add(item.target_material_id)

    strict_matcher = StructureMatcher(**config["identity"]["independent_audit_matcher"])
    a2_evaluations = [item for item in evaluations if item.edge_id in accepted_edges]
    a2_evaluations.sort(
        key=lambda item: hashlib.sha256(
            f"{config['seed']}|{item.edge_id.hex()}".encode()
        ).hexdigest()
    )
    audit_sample = a2_evaluations[: int(config["identity"]["audit_maximum_rows"])]
    audit_rows = []
    for item in audit_sample:
        strict_match = bool(
            strict_matcher.fit(
                source_structures[item.source_material_id],
                target_structures[item.target_material_id],
            )
        )
        audit_rows.append(
            {
                "source_material_id": item.source_material_id,
                "target_material_id": item.target_material_id,
                "primary_rms": item.rms,
                "primary_maximum": item.maximum,
                "independent_strict_match": strict_match,
            }
        )
    audit_precision = (
        sum(row["independent_strict_match"] for row in audit_rows) / len(audit_rows)
        if audit_rows
        else 1.0
    )
    if audit_precision < float(config["identity"]["minimum_a2_precision"]):
        raise RuntimeError(f"R4 A2 precision gate failed: {audit_precision}")

    mapping_rows: list[dict[str, Any]] = []
    for row in source_panel.sort_values("panel_unit_id", kind="mergesort").to_dict(orient="records"):
        source_id = str(row["material_id"])
        decision = accepted.get(source_id)
        mapping_rows.append(
            {
                "panel_unit_id": str(row["panel_unit_id"]),
                "canonical_lineage_id": str(row["canonical_lineage_id"]),
                "source_snapshot": source_snapshot,
                "target_snapshot": target_snapshot,
                "source_material_id": source_id,
                "target_material_id": decision["target_material_id"] if decision else None,
                "thermo_type": str(row["thermo_type"]),
                "phase_context_chemsys": str(row["phase_context_chemsys"]),
                "mapping_confidence": decision["confidence"] if decision else "UNMAPPED",
                "mapping_method": decision["method"] if decision else "no_accepted_label_blind_identity_edge",
                "shared_task_count": decision["shared_task_count"] if decision else 0,
                "structure_rms": decision["rms"] if decision else None,
                "structure_maximum": decision["maximum"] if decision else None,
            }
        )
    mapping_frame = pd.DataFrame(mapping_rows)
    output_path = repo / config["identity"]["mapping_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_frame.to_parquet(output_path, index=False, compression="zstd")
    report_dir = repo / config["report_dir"]
    manifest_dir = repo / config["manifest_dir"]
    write_csv_atomic(report_dir / "identifier_format_audit.csv", format_rows)
    if audit_rows:
        write_csv_atomic(report_dir / "identity_a2_precision_audit.csv", audit_rows)
    else:
        (report_dir / "identity_a2_precision_audit.csv").write_text(
            "source_material_id,target_material_id,primary_rms,primary_maximum,independent_strict_match\n",
            encoding="utf-8",
        )
    ledger_rows = evaluation_errors + ambiguity_rows
    for source_id in sorted(source_ids - set(accepted)):
        ledger_rows.append(
            {
                "record_type": "unmapped_panel_material",
                "source_material_id": source_id,
                "composition_reduced_json": source_materials[source_id]["composition_reduced_json"],
                "candidate_count": len(
                    target_by_composition[source_materials[source_id]["composition_reduced_json"]]
                ),
                "resolution": "excluded_from_current_release_comparable_panel",
            }
        )
    ledger_path = manifest_dir / "identity_exclusion_ambiguity_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in ledger_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    mapped_units = int(mapping_frame["target_material_id"].notna().sum())
    result = {
        "task_id": "R4.0",
        "stage": "freeze_identity_mapping",
        "status": "PASS",
        "frozen_at_utc": utc_now(),
        "mapping_frozen_before_current_outcome_access": True,
        "current_outcome_files_read": [],
        "identity_input_files": [
            common_path.relative_to(repo).as_posix(),
            (normalized_root / f"snapshot={source_snapshot}" / "raw_material.parquet").relative_to(repo).as_posix(),
            (normalized_root / f"snapshot={target_snapshot}" / "raw_material.parquet").relative_to(repo).as_posix(),
        ],
        "panel_units": len(mapping_frame),
        "source_materials": len(source_ids),
        "target_composition_candidates": len(target_materials),
        "mapped_panel_units": mapped_units,
        "panel_mapping_coverage": mapped_units / len(mapping_frame),
        "mapped_source_materials": len(accepted),
        "a1_source_materials": sum(item["confidence"] == "A1" for item in accepted.values()),
        "a2_source_materials": sum(item["confidence"] == "A2" for item in accepted.values()),
        "unmapped_source_materials": len(source_ids - set(accepted)),
        "structure_candidate_pairs_evaluated": len(evaluations),
        "a2_audit_rows": len(audit_rows),
        "a2_audit_precision": audit_precision,
        "format_only_events": 0,
        "format_conversion_rows": len(format_rows),
        "unledgered_errors": 0,
        "mapping_output": output_path.relative_to(repo).as_posix(),
        "mapping_sha256": sha256_file(output_path),
        "mapping_rows": len(mapping_frame),
        "ambiguity_ledger": ledger_path.relative_to(repo).as_posix(),
        "ambiguity_ledger_sha256": sha256_file(ledger_path),
        "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
    }
    write_json_atomic(manifest_dir / "identity_mapping_manifest.json", result)
    result["manifest_sha256"] = sha256_file(manifest_dir / "identity_mapping_manifest.json")
    write_json_atomic(report_dir / "identity_mapping_result.json", result)
    return result


def build_current_release_benchmark(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    repo = config_file.resolve().parents[2]
    report_dir = repo / config["report_dir"]
    manifest_dir = repo / config["manifest_dir"]
    output_root = repo / "data/processed/R4_0"
    output_root.mkdir(parents=True, exist_ok=True)

    identity_manifest_path = manifest_dir / "identity_mapping_manifest.json"
    identity_manifest = json.loads(identity_manifest_path.read_text(encoding="utf-8"))
    if not identity_manifest.get("mapping_frozen_before_current_outcome_access"):
        raise RuntimeError("Identity mapping was not frozen before current outcome access")
    mapping_path = repo / identity_manifest["mapping_output"]
    if sha256_file(mapping_path) != identity_manifest["mapping_sha256"]:
        raise RuntimeError("Frozen R4 identity mapping hash mismatch")
    phase_manifest_path = repo / config["benchmark"]["phase_manifest"]
    phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
    if not (
        phase_manifest.get("task_id") == "R4.0"
        and phase_manifest.get("status") == "PASS"
        and phase_manifest.get("gate_status") == "GO"
    ):
        raise RuntimeError("R4 current-release phase reconstruction has not passed")

    frozen_inputs = {
        "data/processed/R3_5/candidate_panel.parquet": FROZEN_HASHES[
            "data/processed/R3_5/candidate_panel.parquet"
        ],
        "data/processed/R3_5/frozen_predictions.parquet": FROZEN_HASHES[
            "data/processed/R3_5/frozen_predictions.parquet"
        ],
    }
    old_labels_path = repo / "data/processed/R3_5/versioned_labels.parquet"
    old_labels_expected = json.loads(
        (repo / "data/manifests/R3_5/benchmark_manifest.json").read_text(encoding="utf-8")
    )["outputs"]["versioned_labels"]["sha256"]
    frozen_inputs["data/processed/R3_5/versioned_labels.parquet"] = old_labels_expected
    for relative, expected in frozen_inputs.items():
        if sha256_file(repo / relative) != expected:
            raise RuntimeError(f"Frozen R3.5 input hash mismatch: {relative}")

    mapping = pd.read_parquet(mapping_path)
    panel = pd.read_parquet(repo / "data/processed/R3_5/candidate_panel.parquet")
    predictions = pd.read_parquet(repo / "data/processed/R3_5/frozen_predictions.parquet")
    old_labels = pd.read_parquet(old_labels_path)
    phase_path = repo / config["benchmark"]["phase_entries"]
    phase = pq.read_table(
        phase_path,
        columns=[
            "snapshot_id",
            "thermo_type",
            "phase_context_chemsys",
            "unified_entry_id",
            "material_id",
            "thermo_id",
            "entry_id",
            "is_target",
            "phase_diagram_status",
            "energy_above_hull",
        ],
    ).to_pandas()
    target_version = str(config["identity"]["target_snapshot"])
    source_version = str(config["identity"]["source_snapshot"])
    phase = phase[
        phase["snapshot_id"].astype(str).eq(target_version) & phase["is_target"]
    ].copy()
    normalized_root = repo / config["output_root"]
    target_selection = pd.concat(
        [
            _thermo_target_selection(
                normalized_root, target_version, str(thermo_type)
            )
            for thermo_type in sorted(phase["thermo_type"].astype(str).unique())
        ],
        ignore_index=True,
    )
    phase = select_benchmark_phase_targets(phase, target_selection)
    join_keys_left = [
        "target_material_id",
        "thermo_type",
        "phase_context_chemsys",
    ]
    join_keys_right = ["material_id", "thermo_type", "phase_context_chemsys"]
    current = mapping.merge(
        phase,
        left_on=join_keys_left,
        right_on=join_keys_right,
        how="left",
        validate="one_to_one",
        suffixes=("", "_phase"),
    )
    current["availability_reason"] = np.select(
        [
            current["target_material_id"].isna(),
            current["unified_entry_id"].isna(),
            current["phase_diagram_status"].ne("computed"),
            ~np.isfinite(current["energy_above_hull"]),
        ],
        [
            "identity_unmapped",
            "phase_entry_missing",
            "phase_diagram_not_computed",
            "nonfinite_energy_above_hull",
        ],
        default="estimable",
    )
    current["current_release_common_panel"] = current["availability_reason"].eq(
        "estimable"
    )
    current_panel_path = output_root / "current_release_panel_index.parquet"
    current.sort_values("panel_unit_id", kind="mergesort").to_parquet(
        current_panel_path, index=False, compression="zstd"
    )
    current_ok = current[current["current_release_common_panel"]].copy()
    accepted_panel_ids = set(current_ok["panel_unit_id"].astype(str))
    if not accepted_panel_ids:
        raise RuntimeError("No estimable R4 current-release panel units")

    old_comparable = old_labels[
        old_labels["panel_unit_id"].astype(str).isin(accepted_panel_ids)
    ].copy()
    old_2025 = old_comparable[
        old_comparable["label_version"].astype(str).eq(source_version)
    ].copy()
    if len(old_2025) != len(accepted_panel_ids) * len(THRESHOLDS):
        raise RuntimeError("Frozen 2025 labels do not cover the R4 comparable panel")
    source_energy = (
        old_2025[
            ["panel_unit_id", "source_energy_above_hull", "target_energy_above_hull"]
        ]
        .drop_duplicates("panel_unit_id")
        .rename(columns={"target_energy_above_hull": "energy_above_hull_2025"})
    )
    target_energy = current_ok[
        [
            "panel_unit_id",
            "canonical_lineage_id",
            "thermo_type",
            "phase_context_chemsys",
            "energy_above_hull",
        ]
    ].rename(columns={"energy_above_hull": "target_energy_above_hull"})
    joined = target_energy.merge(
        source_energy, on="panel_unit_id", how="inner", validate="one_to_one"
    )
    joined["label_version"] = target_version
    joined["energy_change_eV_per_atom"] = (
        joined["target_energy_above_hull"] - joined["source_energy_above_hull"]
    )
    joined["target_signed_margin"] = np.nan
    joined["source_signed_margin"] = np.nan
    joined["signed_margin_change_eV_per_atom"] = np.nan
    new_label_rows = []
    for name, threshold in THRESHOLDS:
        item = joined.copy()
        item["label_definition"] = name
        item["threshold_eV_per_atom"] = threshold
        item["source_at_risk"] = item["source_energy_above_hull"].le(threshold)
        item["target_stable"] = item["target_energy_above_hull"].le(threshold)
        item["event"] = item["source_at_risk"] & ~item["target_stable"]
        new_label_rows.append(item.drop(columns=["energy_above_hull_2025"]))
    new_labels = pd.concat(new_label_rows, ignore_index=True)
    column_order = list(old_2025.columns)
    new_labels = new_labels[column_order]
    labels = pd.concat([old_comparable, new_labels], ignore_index=True).sort_values(
        ["label_definition", "label_version", "panel_unit_id"], kind="mergesort"
    )
    benchmark_versions = ("2024-12-18", source_version, target_version)
    labels_path = output_root / "versioned_labels_2024_2026.parquet"
    labels.to_parquet(labels_path, index=False, compression="zstd")

    predictions_subset = predictions[
        predictions["panel_unit_id"].astype(str).isin(accepted_panel_ids)
    ].copy()
    benchmark = predictions_subset.merge(
        labels,
        on=["panel_unit_id", "canonical_lineage_id", "thermo_type"],
        how="inner",
        validate="many_to_many",
    )
    expected_benchmark_rows = (
        len(accepted_panel_ids) * len(MODELS) * len(benchmark_versions) * len(THRESHOLDS)
    )
    if len(benchmark) != expected_benchmark_rows:
        raise RuntimeError(
            f"R4 benchmark cross-product mismatch: {len(benchmark)} != {expected_benchmark_rows}"
        )
    benchmark_path = output_root / "fixed_score_benchmark_2024_2026.parquet"
    benchmark.to_parquet(benchmark_path, index=False, compression="zstd")

    byte_rows = []
    for model in MODELS:
        frozen_subset = predictions_subset[
            predictions_subset["model_id"].eq(model)
        ][["panel_unit_id", "probability"]]
        expected_vector = prediction_vector_hash(frozen_subset)
        for label, _ in THRESHOLDS:
            for version in benchmark_versions:
                observed_subset = benchmark[
                    benchmark["model_id"].eq(model)
                    & benchmark["label_definition"].eq(label)
                    & benchmark["label_version"].eq(version)
                ][["panel_unit_id", "probability"]]
                observed_vector = prediction_vector_hash(observed_subset)
                byte_rows.append(
                    {
                        "model_id": model,
                        "label_definition": label,
                        "label_version": version,
                        "expected_subset_vector_sha256": expected_vector,
                        "observed_vector_sha256": observed_vector,
                        "byte_identical": expected_vector == observed_vector,
                    }
                )
    byte_frame = pd.DataFrame(byte_rows)
    byte_frame.to_csv(report_dir / "prediction_byte_audit_2026.csv", index=False)
    if not byte_frame["byte_identical"].all():
        raise RuntimeError("Frozen prediction scores changed in the R4 extension")

    metric_rows = []
    for _, group in benchmark.groupby(
        ["model_id", "label_definition", "label_version"], sort=True
    ):
        metric_rows.extend(point_metrics(group))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(report_dir / "fixed_score_metrics_2024_2026.csv", index=False)
    index_cols = ["model_id", "label_definition", "metric", "budget"]
    earlier = metrics[metrics["label_version"].eq(source_version)].set_index(index_cols)
    later = metrics[metrics["label_version"].eq(target_version)].set_index(index_cols)
    deltas = later[["value"]].join(
        earlier[["value"]], lsuffix="_later", rsuffix="_earlier"
    ).reset_index()
    deltas["earlier_version"] = source_version
    deltas["later_version"] = target_version
    deltas["delta_later_minus_earlier"] = (
        deltas["value_later"] - deltas["value_earlier"]
    )
    deltas.to_csv(report_dir / "fixed_score_metric_deltas_2025_2026.csv", index=False)

    label_drift_rows = []
    state_drift_rows = []
    for label, threshold in THRESHOLDS:
        subset = labels[labels["label_definition"].eq(label)]
        wide = subset.pivot(index="panel_unit_id", columns="label_version", values="event")
        label_drift_rows.append(
            {
                "label_definition": label,
                "threshold_eV_per_atom": threshold,
                "panel_rows": len(wide),
                "events_2025": int(wide[source_version].sum()),
                "events_2026": int(wide[target_version].sum()),
                "changed_2025_to_2026": int(
                    (wide[source_version] != wide[target_version]).sum()
                ),
                "zero_to_one": int((~wide[source_version] & wide[target_version]).sum()),
                "one_to_zero": int((wide[source_version] & ~wide[target_version]).sum()),
            }
        )
        direct = joined.copy()
        stable_2025 = direct["energy_above_hull_2025"].le(threshold)
        stable_2026 = direct["target_energy_above_hull"].le(threshold)
        state_drift_rows.append(
            {
                "label_definition": label,
                "threshold_eV_per_atom": threshold,
                "panel_rows": len(direct),
                "stable_2025": int(stable_2025.sum()),
                "stable_2026": int(stable_2026.sum()),
                "state_label_changes": int((stable_2025 != stable_2026).sum()),
                "stable_to_unstable": int((stable_2025 & ~stable_2026).sum()),
                "unstable_to_stable": int((~stable_2025 & stable_2026).sum()),
            }
        )
    write_csv_atomic(report_dir / "benchmark_label_drift_2025_2026.csv", label_drift_rows)
    write_csv_atomic(report_dir / "state_label_evolution_2025_2026.csv", state_drift_rows)
    current.groupby("availability_reason", dropna=False).size().rename("rows").reset_index().to_csv(
        report_dir / "current_release_availability.csv", index=False
    )
    errors = pd.DataFrame(
        error_transition_rows(benchmark, (source_version, target_version))
    )
    errors.to_csv(report_dir / "error_transitions_2025_2026.csv", index=False)
    ranking_rows = []
    for (label, metric, budget, version), group in metrics[metrics["estimable"]].groupby(
        ["label_definition", "metric", "budget", "label_version"],
        dropna=False,
        sort=True,
    ):
        ascending = metric in {"brier_score", "expected_calibration_error"}
        ranked = group.copy()
        ranked["rank"] = ranked["value"].rank(method="min", ascending=ascending).astype(int)
        for row in ranked.itertuples():
            ranking_rows.append(
                {
                    "label_definition": label,
                    "metric": metric,
                    "budget": budget,
                    "label_version": version,
                    "model_id": row.model_id,
                    "value": row.value,
                    "rank": row.rank,
                    "same_panel_rows": row.panel_rows,
                }
            )
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(report_dir / "model_rankings_2024_2026.csv", index=False)
    ranking_changes = 0
    if not rankings.empty:
        wide_rank = rankings.pivot_table(
            index=["label_definition", "metric", "budget", "model_id"],
            columns="label_version",
            values="rank",
            aggfunc="first",
        ).dropna()
        ranking_changes = int(
            (wide_rank[source_version] != wide_rank[target_version]).sum()
        )

    bootstrap, bootstrap_info = bootstrap_metrics(
        benchmark,
        report_dir / "bootstrap_replicate_audit_2025_2026.csv",
        (source_version, target_version),
    )
    bootstrap.to_csv(report_dir / "bootstrap_intervals_2025_2026.csv", index=False)

    artifacts = []
    for path, rows in (
        (current_panel_path, len(current)),
        (labels_path, len(labels)),
        (benchmark_path, len(benchmark)),
    ):
        artifacts.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    result = {
        "task_id": "R4.0",
        "stage": "current_release_fixed_score_benchmark",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "interpretation": "post-amendment descriptive database-version sensitivity extension",
        "external_confirmation": False,
        "source_label_version": source_version,
        "target_label_version": target_version,
        "benchmark_label_versions": list(benchmark_versions),
        "base_panel_rows": len(panel),
        "comparable_panel_rows": len(accepted_panel_ids),
        "comparable_panel_fraction": len(accepted_panel_ids) / len(panel),
        "label_drift": label_drift_rows,
        "state_label_evolution": state_drift_rows,
        "metric_rows": len(metrics),
        "metric_delta_rows": len(deltas),
        "ranking_records_changed": ranking_changes,
        "bootstrap": bootstrap_info,
        "prediction_byte_checks": len(byte_frame),
        "prediction_byte_checks_passed": int(byte_frame["byte_identical"].sum()),
        "frozen_inputs": frozen_inputs,
        "identity_manifest_sha256": sha256_file(identity_manifest_path),
        "phase_manifest_sha256": sha256_file(phase_manifest_path),
        "artifacts": artifacts,
        "prohibited_actions": {
            "models_loaded": 0,
            "models_trained": 0,
            "predictions_regenerated": 0,
            "predictions_recalibrated": 0,
            "external_confirmation_claimed": 0,
        },
        "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
    }
    write_json_atomic(manifest_dir / "current_release_benchmark_manifest.json", result)
    result["manifest_sha256"] = sha256_file(
        manifest_dir / "current_release_benchmark_manifest.json"
    )
    write_json_atomic(report_dir / "current_release_benchmark_result.json", result)
    return result
