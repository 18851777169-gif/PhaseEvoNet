from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .manifest import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def date_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "$date" in value:
            return str(value["$date"])
        if "string" in value:
            return str(value["string"])
    return str(value)


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def structured_identifier_text(value: Any) -> str | None:
    """Serialize an emmet structured identifier without using ``repr(dict)``."""
    if isinstance(value, dict) and value.get("identifier") is not None:
        identifier = str(value["identifier"])
        suffix = value.get("suffix")
        if suffix in (None, ""):
            return identifier
        separator = str(value.get("separator") or "-")
        return f"{identifier}{separator}{suffix}"
    return text(value)


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value not in ("[]", "{}", "null")
    return True


COMMON_SOURCE_FIELDS = [
    pa.field("source_object_sha256", pa.string(), nullable=False),
    pa.field("source_key", pa.string(), nullable=False),
    pa.field("source_row_number", pa.int64(), nullable=False),
]

SCHEMAS: dict[str, pa.Schema] = {
    "raw_material": pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string()),
            pa.field("formula_pretty", pa.string()),
            pa.field("formula_anonymous", pa.string()),
            pa.field("chemsys", pa.string()),
            pa.field("nelements", pa.int32()),
            pa.field("nsites", pa.int32()),
            pa.field("elements_json", pa.large_string()),
            pa.field("composition_json", pa.large_string()),
            pa.field("composition_reduced_json", pa.large_string()),
            pa.field("structure_json", pa.large_string()),
            pa.field("initial_structures_json", pa.large_string()),
            pa.field("symmetry_json", pa.large_string()),
            pa.field("task_ids_json", pa.large_string()),
            pa.field("calc_types_json", pa.large_string()),
            pa.field("task_types_json", pa.large_string()),
            pa.field("run_types_json", pa.large_string()),
            pa.field("entries_json", pa.large_string()),
            pa.field("origins_json", pa.large_string()),
            pa.field("deprecated", pa.bool_()),
            pa.field("deprecated_tasks_json", pa.large_string()),
            pa.field("deprecation_reasons_json", pa.large_string()),
            pa.field("warnings_json", pa.large_string()),
            pa.field("created_at", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("density", pa.float64()),
            pa.field("density_atomic", pa.float64()),
            pa.field("volume", pa.float64()),
            pa.field("builder_build_date", pa.string()),
            pa.field("builder_database_version", pa.string()),
            pa.field("builder_run_id", pa.string()),
            pa.field("builder_emmet_version", pa.string()),
            pa.field("builder_pymatgen_version", pa.string()),
            pa.field("builder_license", pa.string()),
            *COMMON_SOURCE_FIELDS,
        ]
    ),
    "raw_task": pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string()),
            pa.field("task_id", pa.string()),
            pa.field("calc_type", pa.string()),
            pa.field("task_type", pa.string()),
            pa.field("run_type", pa.string()),
            pa.field("is_deprecated_task", pa.bool_()),
            pa.field("is_origin_task", pa.bool_()),
            pa.field("origin_names_json", pa.large_string()),
            pa.field("material_entries_json", pa.large_string()),
            *COMMON_SOURCE_FIELDS,
        ]
    ),
    "raw_thermo": pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string()),
            pa.field("thermo_id", pa.string()),
            pa.field("thermo_type", pa.string()),
            pa.field("energy_type", pa.string()),
            pa.field("entry_types_json", pa.large_string()),
            pa.field("formula_pretty", pa.string()),
            pa.field("formula_anonymous", pa.string()),
            pa.field("chemsys", pa.string()),
            pa.field("nelements", pa.int32()),
            pa.field("nsites", pa.int32()),
            pa.field("elements_json", pa.large_string()),
            pa.field("composition_json", pa.large_string()),
            pa.field("composition_reduced_json", pa.large_string()),
            pa.field("formation_energy_per_atom", pa.float64()),
            pa.field("energy_above_hull", pa.float64()),
            pa.field("energy_per_atom", pa.float64()),
            pa.field("uncorrected_energy_per_atom", pa.float64()),
            pa.field("is_stable", pa.bool_()),
            pa.field("decomposes_to_json", pa.large_string()),
            pa.field("entries_json", pa.large_string()),
            pa.field("origins_json", pa.large_string()),
            pa.field("deprecated", pa.bool_()),
            pa.field("deprecation_reasons_json", pa.large_string()),
            pa.field("warnings_json", pa.large_string()),
            pa.field("last_updated", pa.string()),
            pa.field("builder_build_date", pa.string()),
            pa.field("builder_database_version", pa.string()),
            pa.field("builder_run_id", pa.string()),
            pa.field("builder_emmet_version", pa.string()),
            pa.field("builder_pymatgen_version", pa.string()),
            pa.field("builder_license", pa.string()),
            *COMMON_SOURCE_FIELDS,
        ]
    ),
    "raw_thermo_entry": pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string()),
            pa.field("thermo_id", pa.string()),
            pa.field("thermo_type", pa.string()),
            pa.field("entry_label", pa.string()),
            pa.field("entry_id", pa.string()),
            pa.field("task_id", pa.string()),
            pa.field("energy", pa.float64()),
            pa.field("correction", pa.float64()),
            pa.field("composition_json", pa.large_string()),
            pa.field("energy_adjustments_json", pa.large_string()),
            pa.field("parameters_json", pa.large_string()),
            pa.field("run_type", pa.string()),
            pa.field("hubbards_json", pa.large_string()),
            pa.field("potcar_spec_json", pa.large_string()),
            pa.field("structure_json", pa.large_string()),
            pa.field("entry_data_json", pa.large_string()),
            *COMMON_SOURCE_FIELDS,
        ]
    ),
    "raw_provenance": pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string()),
            pa.field("theoretical", pa.bool_()),
            pa.field("authors_json", pa.large_string()),
            pa.field("history_json", pa.large_string()),
            pa.field("references_json", pa.large_string()),
            pa.field("database_ids_json", pa.large_string()),
            pa.field("remarks_json", pa.large_string()),
            pa.field("tags_json", pa.large_string()),
            pa.field("origins_json", pa.large_string()),
            pa.field("deprecated", pa.bool_()),
            pa.field("deprecation_reasons_json", pa.large_string()),
            pa.field("last_updated", pa.string()),
            pa.field("builder_build_date", pa.string()),
            pa.field("builder_database_version", pa.string()),
            pa.field("builder_run_id", pa.string()),
            pa.field("builder_emmet_version", pa.string()),
            pa.field("builder_pymatgen_version", pa.string()),
            pa.field("builder_license", pa.string()),
            *COMMON_SOURCE_FIELDS,
        ]
    ),
}


def builder_fields(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("builder_meta") or {}
    return {
        "builder_build_date": date_text(meta.get("build_date")),
        "builder_database_version": text(meta.get("database_version")),
        "builder_run_id": text(meta.get("run_id")),
        "builder_emmet_version": text(meta.get("emmet_version")),
        "builder_pymatgen_version": text(meta.get("pymatgen_version")),
        "builder_license": text(meta.get("license")),
    }


def source_fields(source: dict[str, Any], row_number: int) -> dict[str, Any]:
    return {
        "source_object_sha256": source["sha256"],
        "source_key": source["key"],
        "source_row_number": row_number,
    }


def material_rows(
    record: dict[str, Any], source: dict[str, Any], row_number: int
) -> dict[str, list[dict[str, Any]]]:
    snapshot = source["database_version"]
    common = source_fields(source, row_number)
    material_id = text(record.get("material_id"))
    material = {
        "snapshot_id": snapshot,
        "material_id": material_id,
        "formula_pretty": text(record.get("formula_pretty")),
        "formula_anonymous": text(record.get("formula_anonymous")),
        "chemsys": text(record.get("chemsys")),
        "nelements": integer(record.get("nelements")),
        "nsites": integer(record.get("nsites")),
        "elements_json": canonical_json(record.get("elements")),
        "composition_json": canonical_json(record.get("composition")),
        "composition_reduced_json": canonical_json(record.get("composition_reduced")),
        "structure_json": canonical_json(record.get("structure")),
        "initial_structures_json": canonical_json(record.get("initial_structures")),
        "symmetry_json": canonical_json(record.get("symmetry")),
        "task_ids_json": canonical_json(record.get("task_ids")),
        "calc_types_json": canonical_json(record.get("calc_types")),
        "task_types_json": canonical_json(record.get("task_types")),
        "run_types_json": canonical_json(record.get("run_types")),
        "entries_json": canonical_json(record.get("entries")),
        "origins_json": canonical_json(record.get("origins")),
        "deprecated": record.get("deprecated"),
        "deprecated_tasks_json": canonical_json(record.get("deprecated_tasks")),
        "deprecation_reasons_json": canonical_json(record.get("deprecation_reasons")),
        "warnings_json": canonical_json(record.get("warnings")),
        "created_at": date_text(record.get("created_at")),
        "last_updated": date_text(record.get("last_updated")),
        "density": numeric(record.get("density")),
        "density_atomic": numeric(record.get("density_atomic")),
        "volume": numeric(record.get("volume")),
        **builder_fields(record),
        **common,
    }

    task_ids = [str(item) for item in (record.get("task_ids") or [])]
    calc_types = record.get("calc_types") or {}
    task_types = record.get("task_types") or {}
    run_types = record.get("run_types") or {}
    deprecated_tasks = {str(item) for item in (record.get("deprecated_tasks") or [])}
    origins_by_task: dict[str, list[str]] = defaultdict(list)
    for origin in record.get("origins") or []:
        if isinstance(origin, dict) and origin.get("task_id") is not None:
            origins_by_task[str(origin["task_id"])].append(str(origin.get("name", "")))
    entries_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries = record.get("entries") or {}
    entry_items = entries.items() if isinstance(entries, dict) else enumerate(entries)
    for label, entry in entry_items:
        if not isinstance(entry, dict):
            continue
        entry_data = entry.get("data") or {}
        task_id = entry_data.get("task_id")
        if task_id is not None:
            entries_by_task[str(task_id)].append({"entry_label": str(label), **entry})
    tasks = []
    for task_id in sorted(set(task_ids)):
        origin_names = sorted(set(origins_by_task.get(task_id, [])))
        tasks.append(
            {
                "snapshot_id": snapshot,
                "material_id": material_id,
                "task_id": task_id,
                "calc_type": text(calc_types.get(task_id)),
                "task_type": text(task_types.get(task_id)),
                "run_type": text(run_types.get(task_id)),
                "is_deprecated_task": task_id in deprecated_tasks,
                "is_origin_task": bool(origin_names),
                "origin_names_json": canonical_json(origin_names),
                "material_entries_json": canonical_json(entries_by_task.get(task_id, [])),
                **common,
            }
        )
    return {"raw_material": [material], "raw_task": tasks}


def thermo_rows(
    record: dict[str, Any], source: dict[str, Any], row_number: int
) -> dict[str, list[dict[str, Any]]]:
    snapshot = source["database_version"]
    common = source_fields(source, row_number)
    material_id = text(record.get("material_id"))
    thermo_type = text(record.get("thermo_type"))
    # Newer official MP Delta releases omit the redundant document identifier.
    # Historical frozen snapshots encode it deterministically as
    # ``<material_id>_<thermo_type>``; preserve that cross-version contract.
    thermo_id = text(record.get("thermo_id"))
    if thermo_id is None and material_id is not None and thermo_type is not None:
        thermo_id = f"{material_id}_{thermo_type}"
    thermo = {
        "snapshot_id": snapshot,
        "material_id": material_id,
        "thermo_id": thermo_id,
        "thermo_type": thermo_type,
        "energy_type": text(record.get("energy_type")),
        "entry_types_json": canonical_json(record.get("entry_types")),
        "formula_pretty": text(record.get("formula_pretty")),
        "formula_anonymous": text(record.get("formula_anonymous")),
        "chemsys": text(record.get("chemsys")),
        "nelements": integer(record.get("nelements")),
        "nsites": integer(record.get("nsites")),
        "elements_json": canonical_json(record.get("elements")),
        "composition_json": canonical_json(record.get("composition")),
        "composition_reduced_json": canonical_json(record.get("composition_reduced")),
        "formation_energy_per_atom": numeric(record.get("formation_energy_per_atom")),
        "energy_above_hull": numeric(record.get("energy_above_hull")),
        "energy_per_atom": numeric(record.get("energy_per_atom")),
        "uncorrected_energy_per_atom": numeric(record.get("uncorrected_energy_per_atom")),
        "is_stable": record.get("is_stable"),
        "decomposes_to_json": canonical_json(record.get("decomposes_to")),
        "entries_json": canonical_json(record.get("entries")),
        "origins_json": canonical_json(record.get("origins")),
        "deprecated": record.get("deprecated"),
        "deprecation_reasons_json": canonical_json(record.get("deprecation_reasons")),
        "warnings_json": canonical_json(record.get("warnings")),
        "last_updated": date_text(record.get("last_updated")),
        **builder_fields(record),
        **common,
    }
    normalized_entries = []
    entries = record.get("entries") or {}
    entry_items = entries.items() if isinstance(entries, dict) else enumerate(entries)
    for label, entry in entry_items:
        if not isinstance(entry, dict):
            continue
        data = entry.get("data") or {}
        parameters = entry.get("parameters") or {}
        normalized_entries.append(
            {
                "snapshot_id": snapshot,
                "material_id": material_id,
                "thermo_id": thermo_id,
                "thermo_type": thermo_type,
                "entry_label": str(label),
                "entry_id": structured_identifier_text(entry.get("entry_id")),
                "task_id": text(data.get("task_id")),
                "energy": numeric(entry.get("energy")),
                "correction": numeric(entry.get("correction")),
                "composition_json": canonical_json(entry.get("composition")),
                "energy_adjustments_json": canonical_json(entry.get("energy_adjustments")),
                "parameters_json": canonical_json(entry.get("parameters")),
                "run_type": text(parameters.get("run_type") or data.get("run_type")),
                "hubbards_json": canonical_json(parameters.get("hubbards")),
                "potcar_spec_json": canonical_json(parameters.get("potcar_spec")),
                "structure_json": canonical_json(entry.get("structure")),
                "entry_data_json": canonical_json(entry.get("data")),
                **common,
            }
        )
    return {"raw_thermo": [thermo], "raw_thermo_entry": normalized_entries}


def provenance_rows(
    record: dict[str, Any], source: dict[str, Any], row_number: int
) -> dict[str, list[dict[str, Any]]]:
    row = {
        "snapshot_id": source["database_version"],
        "material_id": text(record.get("material_id")),
        "theoretical": record.get("theoretical"),
        "authors_json": canonical_json(record.get("authors")),
        "history_json": canonical_json(record.get("history")),
        "references_json": canonical_json(record.get("references")),
        "database_ids_json": canonical_json(record.get("database_IDs")),
        "remarks_json": canonical_json(record.get("remarks")),
        "tags_json": canonical_json(record.get("tags")),
        "origins_json": canonical_json(record.get("origins")),
        "deprecated": record.get("deprecated"),
        "deprecation_reasons_json": canonical_json(record.get("deprecation_reasons")),
        "last_updated": date_text(record.get("last_updated")),
        **builder_fields(record),
        **source_fields(source, row_number),
    }
    return {"raw_provenance": [row]}


NORMALIZERS = {
    "materials": material_rows,
    "thermo": thermo_rows,
    "provenance": provenance_rows,
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selected_objects(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = config["source_selection"]
    collections = set(map(str, selection["include_collections"]))
    file_format = str(selection["include_format"])
    marker = str(selection["exclude_object_basename_contains"]).lower()
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        reason = None
        if row.get("collection") not in collections:
            reason = "collection_not_selected"
        elif row.get("file_format") != file_format:
            reason = "non_payload_format"
        elif marker in Path(row["key"]).name.lower():
            reason = "collection_manifest_or_index"
        if reason:
            excluded.append(
                {
                    "record_type": "source_object_exclusion",
                    "database_version": row.get("database_version"),
                    "collection": row.get("collection"),
                    "source_key": row.get("key"),
                    "source_row_count": row.get("row_count"),
                    "reason": reason,
                    "action": "excluded_from_normalization_only; raw object retained",
                }
            )
        else:
            selected.append(row)
    selected.sort(key=lambda item: item["key"])
    return selected, excluded


class DatasetWriter:
    def __init__(
        self, path: Path, table_name: str, compression: str, row_group_size: int
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.writer = pq.ParquetWriter(
            self.temporary,
            SCHEMAS[table_name],
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        self.table_name = table_name
        self.row_group_size = row_group_size
        self.rows = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMAS[self.table_name])
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows += len(rows)

    def close(self) -> None:
        self.writer.close()
        os.replace(self.temporary, self.path)


def field_dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "snapshot_id": "Frozen dated Materials Project collection namespace.",
        "material_id": "Release-level Materials Project aggregate ID; not a permanent identity.",
        "task_id": "Calculation/task identifier associated with a release-level material.",
        "thermo_id": "Thermodynamic document identifier, normally material_id plus thermo_type.",
        "structure_json": "Canonical compact JSON preserving the Monty-serialized structure.",
        "composition_reduced_json": "Canonical compact JSON of reduced composition.",
        "formation_energy_per_atom": "Reported formation energy in eV/atom; no cross-version recomputation in P1.2.",
        "energy_above_hull": "Reported energy above hull in eV/atom; no unified hull recomputation in P1.2.",
        "is_stable": "Reported snapshot stability flag.",
        "calc_type": "Snapshot-reported calculation type for a task association.",
        "task_type": "Snapshot-reported task purpose/type.",
        "run_type": "Snapshot-reported DFT run type or entry parameter run type.",
        "entries_json": "Canonical JSON preserving nested computed-entry metadata.",
        "source_object_sha256": "SHA256 of the immutable P1.1 source object.",
        "source_row_number": "One-based nonblank JSONL record position within the source object.",
    }
    rows: list[dict[str, Any]] = []
    for table_name, schema in SCHEMAS.items():
        critical_default = set()
        for field in schema:
            rows.append(
                {
                    "table": table_name,
                    "field": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "unit": "eV/atom"
                    if field.name
                    in ("formation_energy_per_atom", "energy_above_hull", "energy_per_atom", "uncorrected_energy_per_atom")
                    else "eV"
                    if field.name in ("energy", "correction")
                    else None,
                    "description": descriptions.get(
                        field.name,
                        "Source-preserving normalized field; JSON suffix denotes canonical compact JSON.",
                    ),
                }
            )
    return rows


def completeness_payload(
    total: dict[tuple[str, str], int],
    present: dict[tuple[str, str, str], int],
    critical_fields: dict[str, list[str]],
    threshold: float,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    passed = True
    for (snapshot, table_name), count in sorted(total.items()):
        for field in critical_fields[table_name]:
            have = present.get((snapshot, table_name, field), 0)
            fraction = have / count if count else 0.0
            status = "PASS" if count > 0 and fraction >= threshold else "FAIL"
            passed = passed and status == "PASS"
            rows.append(
                {
                    "snapshot_id": snapshot,
                    "table": table_name,
                    "field": field,
                    "present_rows": have,
                    "total_rows": count,
                    "completeness": fraction,
                    "threshold": threshold,
                    "status": status,
                }
            )
    return rows, passed


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalize_snapshots(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "P1.2":
        raise ValueError("Normalization config task_id must be P1.2")
    source_manifest = Path(config["source_manifest"])
    source_rows = load_manifest(source_manifest)
    selected, exclusions = selected_objects(source_rows, config)
    if not selected:
        raise RuntimeError("No payload objects selected from P1.1 manifest")

    output_root = Path(config["output_root"])
    manifest_dir = Path(config["manifest_dir"])
    report_dir = Path(config["report_dir"])
    compression = str(config["compression"])
    row_group_size = int(config["row_group_size"])
    if row_group_size < 1:
        raise ValueError("row_group_size must be positive")
    critical_fields = {
        str(table): list(map(str, fields))
        for table, fields in config["critical_fields"].items()
    }
    if set(critical_fields) != set(SCHEMAS):
        raise ValueError("Critical field config must cover every normalized table")
    for table_name, fields in critical_fields.items():
        unknown = set(fields) - set(SCHEMAS[table_name].names)
        if unknown:
            raise ValueError(f"Unknown critical fields for {table_name}: {sorted(unknown)}")

    started_at = utc_now()
    writers: dict[tuple[str, str], DatasetWriter] = {}
    buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    present: dict[tuple[str, str, str], int] = defaultdict(int)
    input_rows_by_collection: dict[tuple[str, str], int] = defaultdict(int)
    processed_objects = 0
    for source in selected:
        snapshot = str(source["database_version"])
        collection = str(source["collection"])
        normalizer = NORMALIZERS[collection]
        seen_rows = 0
        with gzip.open(source["local_path"], "rt", encoding="utf-8") as stream:
            for row_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                seen_rows += 1
                normalized = normalizer(record, source, row_number)
                for table_name, rows in normalized.items():
                    key = (snapshot, table_name)
                    buffers[key].extend(rows)
                    totals[key] += len(rows)
                    for field in critical_fields[table_name]:
                        present[(snapshot, table_name, field)] += sum(
                            nonempty(row.get(field)) for row in rows
                        )
                    if len(buffers[key]) >= row_group_size:
                        writer = writers.get(key)
                        if writer is None:
                            writer = DatasetWriter(
                                output_root
                                / f"snapshot={snapshot}"
                                / f"{table_name}.parquet",
                                table_name,
                                compression,
                                row_group_size,
                            )
                            writers[key] = writer
                        writer.write(buffers[key])
                        buffers[key].clear()
        if seen_rows != int(source["row_count"]):
            raise RuntimeError(
                f"Source row count mismatch for {source['key']}: "
                f"manifest={source['row_count']} observed={seen_rows}"
            )
        input_rows_by_collection[(snapshot, collection)] += seen_rows
        processed_objects += 1
        if processed_objects % 250 == 0 or processed_objects == len(selected):
            print(
                f"normalized_objects={processed_objects}/{len(selected)} "
                f"source_rows={sum(input_rows_by_collection.values())}",
                flush=True,
            )

    for key in sorted(buffers):
        if buffers[key]:
            writer = writers.get(key)
            if writer is None:
                snapshot, table_name = key
                writer = DatasetWriter(
                    output_root / f"snapshot={snapshot}" / f"{table_name}.parquet",
                    table_name,
                    compression,
                    row_group_size,
                )
                writers[key] = writer
            writer.write(buffers[key])
            buffers[key].clear()
    for key in sorted(writers):
        writers[key].close()

    threshold = float(config["critical_completeness_threshold"])
    completeness, gate_passed = completeness_payload(
        totals, present, critical_fields, threshold
    )
    dictionary = field_dictionary_rows()
    for row in dictionary:
        row["critical"] = row["field"] in critical_fields[row["table"]]
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(report_dir / "data_dictionary.csv", dictionary)
    write_csv_atomic(report_dir / "schema_completeness.csv", completeness)

    ambiguity_rows = list(exclusions)
    ambiguity_rows.extend(
        [
            {
                "record_type": "semantic_ambiguity",
                "database_version": None,
                "collection": "materials",
                "source_key": None,
                "source_row_count": None,
                "reason": "material_id_is_release_level_not_permanent_identity",
                "action": "preserved verbatim; identity resolution deferred to P2",
            },
            {
                "record_type": "semantic_ambiguity",
                "database_version": None,
                "collection": "thermo",
                "source_key": None,
                "source_row_count": None,
                "reason": "reported energies and hull labels use snapshot-specific compatibility workflows",
                "action": "preserved without recomputation; unified hull work deferred to P3",
            },
            {
                "record_type": "semantic_ambiguity",
                "database_version": None,
                "collection": "provenance",
                "source_key": None,
                "source_row_count": None,
                "reason": "provenance collection does not necessarily cover every material row",
                "action": "no join, imputation, or row drop performed",
            },
        ]
    )
    write_jsonl_atomic(manifest_dir / "exclusion_ambiguity_ledger.jsonl", ambiguity_rows)

    artifacts = []
    for (snapshot, table_name), writer in sorted(writers.items()):
        path = writer.path
        parquet_file = pq.ParquetFile(path)
        artifacts.append(
            {
                "snapshot_id": snapshot,
                "table": table_name,
                "path": path.as_posix(),
                "rows": parquet_file.metadata.num_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": hashlib.sha256(
                    str(parquet_file.schema_arrow.remove_metadata()).encode("utf-8")
                ).hexdigest(),
            }
        )
    manifest = {
        "task_id": "P1.2",
        "status": "PASS" if gate_passed else "FAIL",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "source_manifest": source_manifest.as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest),
        "selected_source_objects": len(selected),
        "excluded_source_objects": len(exclusions),
        "input_rows_by_snapshot_collection": [
            {"snapshot_id": key[0], "collection": key[1], "rows": value}
            for key, value in sorted(input_rows_by_collection.items())
        ],
        "output_rows_by_snapshot_table": [
            {"snapshot_id": key[0], "table": key[1], "rows": value}
            for key, value in sorted(totals.items())
        ],
        "critical_completeness_threshold": threshold,
        "critical_completeness": completeness,
        "gate_passed": gate_passed,
        "artifacts": artifacts,
        "ledger_path": (manifest_dir / "exclusion_ambiguity_ledger.jsonl").as_posix(),
        "network_access": False,
        "seed": int(config["seed"]),
    }
    write_json_atomic(manifest_dir / "normalization_manifest.json", manifest)
    if not gate_passed:
        raise RuntimeError(
            "P1.2 critical-field completeness gate failed; inspect schema_completeness.csv"
        )
    return manifest


def normalize_snapshot_collections(
    config_path: str | Path,
    snapshot_id: str,
    collections: Iterable[str],
) -> dict[str, Any]:
    """Deterministically rebuild selected snapshot collections after interruption.

    This uses the same adapters and schemas as the full P1.2 run, but intentionally
    does not replace the task-level normalization manifest. The caller must perform
    the full manifest reconciliation after the scoped rebuild.
    """
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    requested = set(map(str, collections))
    unknown = requested - set(NORMALIZERS)
    if unknown or not requested:
        raise ValueError(f"Unsupported or empty collection selection: {sorted(unknown)}")
    all_selected, _ = selected_objects(
        load_manifest(Path(config["source_manifest"])), config
    )
    selected = [
        row
        for row in all_selected
        if row["database_version"] == snapshot_id and row["collection"] in requested
    ]
    if not selected:
        raise RuntimeError("No source objects matched the snapshot/collection selection")

    output_root = Path(config["output_root"])
    compression = str(config["compression"])
    row_group_size = int(config["row_group_size"])
    critical_fields = {
        str(table): list(map(str, fields))
        for table, fields in config["critical_fields"].items()
    }
    writers: dict[str, DatasetWriter] = {}
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    present: dict[tuple[str, str, str], int] = defaultdict(int)
    input_rows = 0
    for object_index, source in enumerate(selected, start=1):
        observed = 0
        with gzip.open(source["local_path"], "rt", encoding="utf-8") as stream:
            for row_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                observed += 1
                record = json.loads(line)
                normalized = NORMALIZERS[source["collection"]](
                    record, source, row_number
                )
                for table_name, rows in normalized.items():
                    buffers[table_name].extend(rows)
                    totals[(snapshot_id, table_name)] += len(rows)
                    for field in critical_fields[table_name]:
                        present[(snapshot_id, table_name, field)] += sum(
                            nonempty(row.get(field)) for row in rows
                        )
                    if len(buffers[table_name]) >= row_group_size:
                        writer = writers.get(table_name)
                        if writer is None:
                            writer = DatasetWriter(
                                output_root
                                / f"snapshot={snapshot_id}"
                                / f"{table_name}.parquet",
                                table_name,
                                compression,
                                row_group_size,
                            )
                            writers[table_name] = writer
                        writer.write(buffers[table_name])
                        buffers[table_name].clear()
        if observed != int(source["row_count"]):
            raise RuntimeError(f"Source row count mismatch for {source['key']}")
        input_rows += observed
        if object_index % 250 == 0 or object_index == len(selected):
            print(
                f"rebuilt_objects={object_index}/{len(selected)} source_rows={input_rows}",
                flush=True,
            )
    for table_name in sorted(buffers):
        if buffers[table_name]:
            writer = writers.get(table_name)
            if writer is None:
                writer = DatasetWriter(
                    output_root
                    / f"snapshot={snapshot_id}"
                    / f"{table_name}.parquet",
                    table_name,
                    compression,
                    row_group_size,
                )
                writers[table_name] = writer
            writer.write(buffers[table_name])
    for table_name in sorted(writers):
        writers[table_name].close()

    threshold = float(config["critical_completeness_threshold"])
    completeness, passed = completeness_payload(
        totals,
        present,
        {name: critical_fields[name] for name in writers},
        threshold,
    )
    artifacts = []
    for table_name, writer in sorted(writers.items()):
        actual_rows = pq.ParquetFile(writer.path).metadata.num_rows
        expected_rows = totals[(snapshot_id, table_name)]
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Scoped rebuild row mismatch for {table_name}: "
                f"expected={expected_rows} actual={actual_rows}"
            )
        artifacts.append(
            {
                "snapshot_id": snapshot_id,
                "table": table_name,
                "path": writer.path.as_posix(),
                "rows": actual_rows,
                "bytes": writer.path.stat().st_size,
                "sha256": sha256_file(writer.path),
            }
        )
    if not passed:
        raise RuntimeError("Scoped rebuild failed the configured completeness gate")
    return {
        "status": "PASS",
        "snapshot_id": snapshot_id,
        "collections": sorted(requested),
        "selected_source_objects": len(selected),
        "input_rows": input_rows,
        "artifacts": artifacts,
        "critical_completeness": completeness,
        "network_access": False,
    }


def verify_normalization_outputs(
    config_path: str | Path, *, update_manifest: bool = False
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest_dir"]) / "normalization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (row["snapshot_id"], row["table"]): int(row["rows"])
        for row in manifest["output_rows_by_snapshot_table"]
    }
    artifacts = []
    failures = []
    for (snapshot_id, table_name), expected_rows in sorted(expected.items()):
        path = (
            Path(config["output_root"])
            / f"snapshot={snapshot_id}"
            / f"{table_name}.parquet"
        )
        if not path.is_file():
            failures.append(f"missing:{path.as_posix()}")
            continue
        parquet_file = pq.ParquetFile(path)
        actual_rows = int(parquet_file.metadata.num_rows)
        actual_schema = parquet_file.schema_arrow.remove_metadata()
        if actual_rows != expected_rows:
            failures.append(
                f"row_count:{snapshot_id}/{table_name}:"
                f"expected={expected_rows}:actual={actual_rows}"
            )
        if actual_schema != SCHEMAS[table_name].remove_metadata():
            failures.append(f"schema:{snapshot_id}/{table_name}")
        artifacts.append(
            {
                "snapshot_id": snapshot_id,
                "table": table_name,
                "path": path.as_posix(),
                "rows": actual_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": hashlib.sha256(
                    str(actual_schema).encode("utf-8")
                ).hexdigest(),
            }
        )
    completeness_path = Path(config["report_dir"]) / "schema_completeness.csv"
    dictionary_path = Path(config["report_dir"]) / "data_dictionary.csv"
    ledger_path = Path(config["manifest_dir"]) / "exclusion_ambiguity_ledger.jsonl"
    for required in (completeness_path, dictionary_path, ledger_path):
        if not required.is_file():
            failures.append(f"missing:{required.as_posix()}")
    if completeness_path.is_file():
        import csv

        with completeness_path.open(encoding="utf-8", newline="") as stream:
            completeness_rows = list(csv.DictReader(stream))
        if not completeness_rows or any(row["status"] != "PASS" for row in completeness_rows):
            failures.append("critical_completeness_not_all_pass")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "verified_at_utc": utc_now(),
        "artifact_count": len(artifacts),
        "expected_artifact_count": len(expected),
        "artifacts": artifacts,
        "failures": failures,
        "config_sha256": sha256_file(config_file),
        "source_manifest_sha256": sha256_file(Path(config["source_manifest"])),
        "network_access": False,
    }
    if failures:
        raise RuntimeError(f"P1.2 output verification failed: {failures[:3]}")
    if update_manifest:
        manifest["artifacts"] = artifacts
        manifest["last_verified_at_utc"] = result["verified_at_utc"]
        manifest["verification"] = {
            "status": "PASS",
            "artifact_count": len(artifacts),
            "all_row_counts_match": True,
            "all_schemas_match": True,
            "all_critical_completeness_checks_pass": True,
            "network_access": False,
        }
        write_json_atomic(manifest_path, manifest)
        result["manifest_updated"] = True
        result["manifest_sha256"] = sha256_file(manifest_path)
    else:
        result["manifest_updated"] = False
    return result
