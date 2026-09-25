from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from .identity_candidates import write_csv_atomic, write_json_atomic, write_jsonl_atomic
from .manifest import sha256_file


STATE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("material_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("thermo_id", pa.string()),
        pa.field("is_stable", pa.bool_()),
        pa.field("energy_above_hull", pa.float64()),
        pa.field("formation_energy_per_atom", pa.float64()),
        pa.field("usable", pa.bool_(), nullable=False),
        pa.field("reported_value_conflict", pa.bool_(), nullable=False),
        pa.field("source_record_count", pa.int32(), nullable=False),
        pa.field("builder_database_version", pa.string()),
        pa.field("builder_run_id", pa.string()),
        pa.field("builder_emmet_version", pa.string()),
        pa.field("builder_pymatgen_version", pa.string()),
        pa.field("source_object_sha256", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
    ]
)

DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("material_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("selected_source_object_sha256", pa.string(), nullable=False),
        pa.field("selected_source_key", pa.string(), nullable=False),
        pa.field("selected_source_row_number", pa.int64(), nullable=False),
        pa.field("excluded_source_object_sha256", pa.string(), nullable=False),
        pa.field("excluded_source_key", pa.string(), nullable=False),
        pa.field("excluded_source_row_number", pa.int64(), nullable=False),
        pa.field("reported_values_equal", pa.bool_(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)

TRANSITION_SCHEMA = pa.schema(
    [
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("identity_edge_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string()),
        pa.field("observation_status", pa.string(), nullable=False),
        pa.field("source_state_present", pa.bool_(), nullable=False),
        pa.field("target_state_present", pa.bool_(), nullable=False),
        pa.field("source_state_usable", pa.bool_(), nullable=False),
        pa.field("target_state_usable", pa.bool_(), nullable=False),
        pa.field("source_thermo_id", pa.string()),
        pa.field("target_thermo_id", pa.string()),
        pa.field("source_is_stable", pa.bool_()),
        pa.field("target_is_stable", pa.bool_()),
        pa.field("reported_label_transition", pa.string()),
        pa.field("label_flip", pa.bool_()),
        pa.field("source_energy_above_hull", pa.float64()),
        pa.field("target_energy_above_hull", pa.float64()),
        pa.field("delta_reported_energy_above_hull", pa.float64()),
        pa.field("source_formation_energy_per_atom", pa.float64()),
        pa.field("target_formation_energy_per_atom", pa.float64()),
        pa.field("delta_reported_formation_energy_per_atom", pa.float64()),
        pa.field("source_builder_database_version", pa.string()),
        pa.field("target_builder_database_version", pa.string()),
        pa.field("source_builder_run_id", pa.string()),
        pa.field("target_builder_run_id", pa.string()),
        pa.field("source_builder_emmet_version", pa.string()),
        pa.field("target_builder_emmet_version", pa.string()),
        pa.field("source_builder_pymatgen_version", pa.string()),
        pa.field("target_builder_pymatgen_version", pa.string()),
        pa.field("source_source_record_count", pa.int32()),
        pa.field("target_source_record_count", pa.int32()),
        pa.field("source_object_sha256", pa.string()),
        pa.field("target_object_sha256", pa.string()),
        pa.field("source_key", pa.string()),
        pa.field("target_key", pa.string()),
        pa.field("source_row_number", pa.int64()),
        pa.field("target_row_number", pa.int64()),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transition_id(edge_id: bytes, thermo_type: str | None) -> bytes:
    label = thermo_type if thermo_type is not None else "<NO_REPORTED_THERMO>"
    return hashlib.blake2b(edge_id + b"|" + label.encode(), digest_size=16).digest()


def classify_transition(
    source_present: bool,
    target_present: bool,
    source_usable: bool,
    target_usable: bool,
    source_stable: bool | None,
    target_stable: bool | None,
) -> tuple[str, str | None, bool | None]:
    if not source_present and not target_present:
        return "no_reported_thermo", None, None
    if source_present and not source_usable:
        return "source_ambiguous", None, None
    if target_present and not target_usable:
        return "target_ambiguous", None, None
    if source_present and not target_present:
        return "source_only", None, None
    if target_present and not source_present:
        return "target_only", None, None
    if source_stable is None or target_stable is None:
        return "missing_reported_label", None, None
    source_label = "stable" if source_stable else "unstable"
    target_label = "stable" if target_stable else "unstable"
    return "observed", f"{source_label}_to_{target_label}", source_stable != target_stable


class DataFrameParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, config: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.schema = schema
        self.row_group_size = int(config["row_group_size"])
        self.writer = pq.ParquetWriter(
            self.temporary,
            schema,
            compression=str(config["compression"]),
            use_dictionary=True,
            write_statistics=True,
        )
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(
            frame[self.schema.names], schema=self.schema, preserve_index=False
        )
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows += len(frame)

    def close(self) -> None:
        self.writer.close()
        os.replace(self.temporary, self.path)


def _thermo_columns() -> list[str]:
    return [
        "snapshot_id",
        "material_id",
        "thermo_id",
        "thermo_type",
        "is_stable",
        "energy_above_hull",
        "formation_energy_per_atom",
        "builder_database_version",
        "builder_run_id",
        "builder_emmet_version",
        "builder_pymatgen_version",
        "source_object_sha256",
        "source_key",
        "source_row_number",
    ]


def deduplicate_reported_states(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    key = ["material_id", "thermo_type"]
    required = key + ["source_object_sha256", "source_key", "source_row_number"]
    if frame[required].isna().any().any():
        raise RuntimeError("Reported thermo state has missing key or provenance")
    sorted_frame = frame.sort_values(
        key + ["source_object_sha256", "source_key", "source_row_number"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = sorted_frame.groupby(key, sort=False, dropna=False)
    reported_fields = [
        "thermo_id",
        "is_stable",
        "energy_above_hull",
        "formation_energy_per_atom",
    ]
    conflicts = grouped[reported_fields].nunique(dropna=False).gt(1).any(axis=1)
    counts = grouped.size().rename("source_record_count")
    selected = sorted_frame.drop_duplicates(key, keep="first").copy()
    selected = selected.merge(counts.reset_index(), on=key, how="left", validate="one_to_one")
    selected = selected.merge(
        conflicts.rename("reported_value_conflict").reset_index(),
        on=key,
        how="left",
        validate="one_to_one",
    )
    selected["usable"] = ~selected["reported_value_conflict"]

    selected_trace = selected[
        key + ["source_object_sha256", "source_key", "source_row_number"]
    ].rename(
        columns={
            "source_object_sha256": "selected_source_object_sha256",
            "source_key": "selected_source_key",
            "source_row_number": "selected_source_row_number",
        }
    )
    duplicate_mask = sorted_frame.duplicated(key, keep="first")
    duplicates = sorted_frame.loc[
        duplicate_mask,
        key + ["source_object_sha256", "source_key", "source_row_number"],
    ].rename(
        columns={
            "source_object_sha256": "excluded_source_object_sha256",
            "source_key": "excluded_source_key",
            "source_row_number": "excluded_source_row_number",
        }
    )
    duplicates = duplicates.merge(selected_trace, on=key, how="left", validate="many_to_one")
    conflict_lookup = conflicts.to_dict()
    duplicates["reported_values_equal"] = [
        not bool(conflict_lookup[(row.material_id, row.thermo_type)])
        for row in duplicates.itertuples()
    ]
    duplicates["reason"] = "duplicate_reported_state_key"

    ambiguity_rows: list[dict[str, Any]] = []
    for (material_id, thermo_type), is_conflict in conflicts.items():
        if not is_conflict:
            continue
        rows = sorted_frame[
            (sorted_frame["material_id"] == material_id)
            & (sorted_frame["thermo_type"] == thermo_type)
        ]
        ambiguity_rows.append(
            {
                "record_type": "reported_value_conflict",
                "snapshot_id": str(rows.iloc[0]["snapshot_id"]),
                "material_id": str(material_id),
                "thermo_type": str(thermo_type),
                "source_record_count": len(rows),
                "source_records": json.loads(
                    rows[
                        reported_fields
                        + ["source_object_sha256", "source_key", "source_row_number"]
                    ].to_json(orient="records")
                ),
                "resolution": "mark state unusable; do not create an observed transition",
            }
        )
    return selected, duplicates, ambiguity_rows


def _state_frame_for_merge(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    rename = {
        "material_id": f"{side}_state_material_id",
        "thermo_id": f"{side}_thermo_id",
        "is_stable": f"{side}_is_stable",
        "energy_above_hull": f"{side}_energy_above_hull",
        "formation_energy_per_atom": f"{side}_formation_energy_per_atom",
        "usable": f"{side}_state_usable",
        "source_record_count": f"{side}_source_record_count",
        "builder_database_version": f"{side}_builder_database_version",
        "builder_run_id": f"{side}_builder_run_id",
        "builder_emmet_version": f"{side}_builder_emmet_version",
        "builder_pymatgen_version": f"{side}_builder_pymatgen_version",
        "source_object_sha256": f"{side}_object_sha256",
        "source_key": f"{side}_key",
        "source_row_number": f"{side}_row_number",
    }
    columns = [
        "material_id",
        "thermo_type",
        "thermo_id",
        "is_stable",
        "energy_above_hull",
        "formation_energy_per_atom",
        "usable",
        "source_record_count",
        "builder_database_version",
        "builder_run_id",
        "builder_emmet_version",
        "builder_pymatgen_version",
        "source_object_sha256",
        "source_key",
        "source_row_number",
    ]
    return frame[columns].rename(columns=rename)


def build_pair_transitions(
    edges: pd.DataFrame,
    source_states: pd.DataFrame,
    target_states: pd.DataFrame,
) -> pd.DataFrame:
    edge_columns = [
        "edge_id",
        "canonical_lineage_id",
        "confidence",
        "source_snapshot",
        "target_snapshot",
        "source_material_id",
        "target_material_id",
    ]
    base = edges[edge_columns].copy()
    source = base[["edge_id", "source_material_id"]].merge(
        _state_frame_for_merge(source_states, "source"),
        left_on="source_material_id",
        right_on="source_state_material_id",
        how="inner",
    )
    target = base[["edge_id", "target_material_id"]].merge(
        _state_frame_for_merge(target_states, "target"),
        left_on="target_material_id",
        right_on="target_state_material_id",
        how="inner",
    )
    types = pd.concat(
        [source[["edge_id", "thermo_type"]], target[["edge_id", "thermo_type"]]],
        ignore_index=True,
    ).drop_duplicates()
    covered = set(types["edge_id"])
    uncovered = base.loc[~base["edge_id"].isin(covered), ["edge_id"]].copy()
    uncovered["thermo_type"] = None
    types = pd.concat([types, uncovered], ignore_index=True)
    result = base.merge(types, on="edge_id", how="left", validate="one_to_many")
    source_columns = [item for item in source.columns if item not in {"source_material_id"}]
    target_columns = [item for item in target.columns if item not in {"target_material_id"}]
    result = result.merge(
        source[source_columns], on=["edge_id", "thermo_type"], how="left", validate="many_to_one"
    )
    result = result.merge(
        target[target_columns], on=["edge_id", "thermo_type"], how="left", validate="many_to_one"
    )
    result["source_state_present"] = result["source_state_material_id"].notna()
    result["target_state_present"] = result["target_state_material_id"].notna()
    result["source_state_usable"] = (
        result["source_state_usable"].astype("boolean").fillna(False).astype(bool)
    )
    result["target_state_usable"] = (
        result["target_state_usable"].astype("boolean").fillna(False).astype(bool)
    )

    statuses: list[str] = []
    transitions: list[str | None] = []
    flips: list[bool | None] = []
    for row in result.itertuples():
        status, label_transition, flip = classify_transition(
            bool(row.source_state_present),
            bool(row.target_state_present),
            bool(row.source_state_usable),
            bool(row.target_state_usable),
            None if pd.isna(row.source_is_stable) else bool(row.source_is_stable),
            None if pd.isna(row.target_is_stable) else bool(row.target_is_stable),
        )
        statuses.append(status)
        transitions.append(label_transition)
        flips.append(flip)
    result["observation_status"] = statuses
    result["reported_label_transition"] = transitions
    result["label_flip"] = flips
    observed = result["observation_status"] == "observed"
    result["delta_reported_energy_above_hull"] = (
        result["target_energy_above_hull"] - result["source_energy_above_hull"]
    ).where(observed)
    result["delta_reported_formation_energy_per_atom"] = (
        result["target_formation_energy_per_atom"]
        - result["source_formation_energy_per_atom"]
    ).where(observed)
    result["transition_id"] = [
        transition_id(bytes(edge_id), thermo_type if isinstance(thermo_type, str) else None)
        for edge_id, thermo_type in zip(result["edge_id"], result["thermo_type"])
    ]
    result = result.rename(
        columns={
            "edge_id": "identity_edge_id",
            "confidence": "identity_confidence",
            "source_energy_above_hull": "source_energy_above_hull",
            "target_energy_above_hull": "target_energy_above_hull",
        }
    )
    return result


def _transition_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    direct = [
        "transition_id", "identity_edge_id", "canonical_lineage_id",
        "identity_confidence", "source_snapshot", "target_snapshot",
        "source_material_id", "target_material_id", "thermo_type",
        "observation_status", "source_state_present", "target_state_present",
        "source_state_usable", "target_state_usable", "source_thermo_id",
        "target_thermo_id", "source_is_stable", "target_is_stable",
        "reported_label_transition", "label_flip", "source_energy_above_hull",
        "target_energy_above_hull", "delta_reported_energy_above_hull",
        "source_formation_energy_per_atom", "target_formation_energy_per_atom",
        "delta_reported_formation_energy_per_atom", "source_builder_database_version",
        "target_builder_database_version", "source_builder_run_id", "target_builder_run_id",
        "source_builder_emmet_version", "target_builder_emmet_version",
        "source_builder_pymatgen_version", "target_builder_pymatgen_version",
        "source_source_record_count", "target_source_record_count",
        "source_object_sha256", "target_object_sha256", "source_key", "target_key",
        "source_row_number", "target_row_number",
    ]
    source_map = {
        "source_object_sha256": "source_object_sha256",
        "source_key": "source_key",
        "source_row_number": "source_row_number",
    }
    target_map = {
        "target_object_sha256": "target_object_sha256",
        "target_key": "target_key",
        "target_row_number": "target_row_number",
    }
    for name in direct:
        if name in frame:
            output[name] = frame[name]
        elif name in source_map:
            output[name] = frame[source_map[name]]
        elif name in target_map:
            output[name] = frame[target_map[name]]
        else:
            output[name] = None
    return output


def _raw_manifest_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("collection") == "thermo" and row.get("validation_status") == "PASS":
                result[str(row["sha256"])] = row
    return result


def _equal_float(expected: Any, observed: Any, tolerance: float) -> bool:
    if expected is None and observed is None:
        return True
    if expected is None or observed is None:
        return False
    if pd.isna(expected) and observed is None:
        return True
    try:
        return math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def audit_transition_sources(
    transition_path: Path,
    raw_manifest_path: Path,
    audit_config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns = [
        "transition_id", "canonical_lineage_id", "identity_confidence",
        "source_snapshot", "target_snapshot", "thermo_type", "observation_status",
        "source_material_id", "target_material_id", "source_thermo_id", "target_thermo_id",
        "source_is_stable", "target_is_stable", "source_energy_above_hull",
        "target_energy_above_hull", "source_formation_energy_per_atom",
        "target_formation_energy_per_atom", "source_object_sha256", "target_object_sha256",
        "source_key", "target_key", "source_row_number", "target_row_number",
    ]
    frame = pq.read_table(transition_path, columns=columns).to_pandas()
    observed = frame[frame["observation_status"] == "observed"].copy()
    observed["audit_rank"] = [
        hashlib.sha256(f"{seed}|{bytes(item).hex()}".encode()).hexdigest()
        for item in observed["transition_id"]
    ]
    per_stratum = int(audit_config["sample_per_snapshot_pair_thermo_type"])
    base_sample = (
        observed.sort_values("audit_rank", kind="mergesort")
        .groupby(["source_snapshot", "target_snapshot", "thermo_type"], sort=True)
        .head(per_stratum)
    )
    if bool(audit_config["include_all_a2_observed"]):
        base_sample = pd.concat(
            [base_sample, observed[observed["identity_confidence"] == "A2"]],
            ignore_index=True,
        ).drop_duplicates("transition_id")
    sample = base_sample.sort_values(
        ["source_snapshot", "target_snapshot", "thermo_type", "audit_rank"],
        kind="mergesort",
    )
    manifest = _raw_manifest_map(raw_manifest_path)
    requests: dict[str, set[int]] = defaultdict(set)
    for row in sample.itertuples():
        requests[str(row.source_object_sha256)].add(int(row.source_row_number))
        requests[str(row.target_object_sha256)].add(int(row.target_row_number))
    raw_records: dict[tuple[str, int], dict[str, Any]] = {}
    object_hash_matches: dict[str, bool] = {}
    for object_sha, row_numbers in sorted(requests.items()):
        source = manifest.get(object_sha)
        if source is None:
            object_hash_matches[object_sha] = False
            continue
        local_path = Path(source["local_path"])
        object_hash_matches[object_sha] = (
            local_path.exists() and sha256_file(local_path) == object_sha
        )
        if not local_path.exists():
            continue
        with gzip.open(local_path, "rt", encoding="utf-8") as stream:
            for row_number, line in enumerate(stream, start=1):
                if row_number in row_numbers:
                    raw_records[(object_sha, row_number)] = json.loads(line)
                if len(
                    [number for number in row_numbers if (object_sha, number) in raw_records]
                ) == len(row_numbers):
                    break

    tolerance = float(audit_config["float_absolute_tolerance"])
    audit_rows: list[dict[str, Any]] = []
    for row in sample.itertuples():
        for side in ("source", "target"):
            object_sha = str(getattr(row, f"{side}_object_sha256"))
            row_number = int(getattr(row, f"{side}_row_number"))
            raw = raw_records.get((object_sha, row_number))
            source_manifest_row = manifest.get(object_sha)
            expected_key = str(getattr(row, f"{side}_key"))
            expected_material = str(getattr(row, f"{side}_material_id"))
            expected_thermo_id = getattr(row, f"{side}_thermo_id")
            expected_stable = bool(getattr(row, f"{side}_is_stable"))
            expected_hull = getattr(row, f"{side}_energy_above_hull")
            expected_formation = getattr(row, f"{side}_formation_energy_per_atom")
            material_match = raw is not None and str(raw.get("material_id")) == expected_material
            thermo_type_match = raw is not None and str(raw.get("thermo_type")) == str(row.thermo_type)
            thermo_id_match = raw is not None and str(raw.get("thermo_id")) == str(expected_thermo_id)
            stable_match = raw is not None and raw.get("is_stable") is expected_stable
            hull_match = raw is not None and _equal_float(expected_hull, raw.get("energy_above_hull"), tolerance)
            formation_match = raw is not None and _equal_float(
                expected_formation, raw.get("formation_energy_per_atom"), tolerance
            )
            hash_match = bool(object_hash_matches.get(object_sha, False))
            source_key_match = (
                source_manifest_row is not None
                and str(source_manifest_row.get("key")) == expected_key
            )
            all_match = all(
                [material_match, thermo_type_match, thermo_id_match, stable_match,
                 hull_match, formation_match, hash_match, source_key_match]
            )
            audit_rows.append(
                {
                    "transition_id": bytes(row.transition_id).hex(),
                    "canonical_lineage_id": row.canonical_lineage_id,
                    "identity_confidence": row.identity_confidence,
                    "source_snapshot": row.source_snapshot,
                    "target_snapshot": row.target_snapshot,
                    "thermo_type": row.thermo_type,
                    "side": side,
                    "material_id": expected_material,
                    "thermo_id": expected_thermo_id,
                    "source_object_sha256": object_sha,
                    "source_key": expected_key,
                    "source_row_number": row_number,
                    "raw_object_hash_match": hash_match,
                    "source_key_match": source_key_match,
                    "material_id_match": material_match,
                    "thermo_type_match": thermo_type_match,
                    "thermo_id_match": thermo_id_match,
                    "is_stable_match": stable_match,
                    "energy_above_hull_match": hull_match,
                    "formation_energy_per_atom_match": formation_match,
                    "all_fields_match": all_match,
                }
            )
    matches = sum(bool(row["all_fields_match"]) for row in audit_rows)
    match_rate = matches / len(audit_rows) if audit_rows else 0.0
    required_rate = float(audit_config["required_match_rate"])
    return audit_rows, {
        "sampled_transitions": len(sample),
        "sampled_source_records": len(audit_rows),
        "matched_source_records": matches,
        "match_rate": match_rate,
        "required_match_rate": required_rate,
        "raw_objects_checked": len(requests),
        "raw_object_hashes_matched": sum(object_hash_matches.values()),
        "a2_observed_transitions_in_population": int(
            ((observed["identity_confidence"] == "A2")).sum()
        ),
        "a2_observed_transitions_audited": int(
            ((sample["identity_confidence"] == "A2")).sum()
        ),
        "status": "PASS" if audit_rows and match_rate >= required_rate else "FAIL",
        "seed": seed,
        "sampling": "deterministic hash sample per snapshot-pair/thermo-type plus all observed A2 transitions",
    }


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()


def _dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "thermo_type": "Snapshot-reported compatibility workflow; transitions never compare different thermo_type values.",
        "is_stable": "Source-reported stability flag; not recomputed in P3.1.",
        "label_flip": "True only when both same-thermo_type source labels are present/usable and differ.",
        "observation_status": "observed, source_only, target_only, ambiguous, missing label, or no reported thermo.",
        "delta_reported_energy_above_hull": "Target minus source reported hull energy for an observed same-thermo_type transition; not a unified-hull delta.",
        "reported_value_conflict": "True when duplicate state keys disagree on a preregistered reported field.",
        "source_record_count": "Number of normalized raw thermo rows sharing the selected snapshot/material/thermo_type state key.",
    }
    rows: list[dict[str, Any]] = []
    for table_name, schema in (
        ("reported_state", STATE_SCHEMA),
        ("transition_label", TRANSITION_SCHEMA),
        ("thermo_duplicate_ledger", DUPLICATE_SCHEMA),
    ):
        for field in schema:
            rows.append(
                {
                    "table": table_name,
                    "field": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "unit": "eV/atom" if "energy" in field.name else "",
                    "description": descriptions.get(field.name, "Traceable P3.1 reported-state or transition field."),
                }
            )
    return rows


def build_reported_transitions(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "P3.1":
        raise RuntimeError("P3.1 config must declare task_id: P3.1")
    started = utc_now()
    seed = int(config["seed"])
    snapshots = [str(item) for item in config["snapshots"]]
    inputs = config["input"]
    outputs = config["output"]
    p2_report = json.loads(Path(inputs["p2_2_report"]).read_text(encoding="utf-8"))
    if not (
        p2_report.get("task_status") == "DONE"
        and p2_report.get("status") == "PASS"
        and p2_report.get("gate_status", {}).get("status") == "GO"
    ):
        raise RuntimeError("P2.2 DONE/PASS/GO is required before P3.1")
    identity_path = Path(inputs["identity_edges"])
    expected_identity_hash = next(
        item["sha256"] for item in p2_report["outputs"] if item["path"].endswith("identity_edge.parquet")
    )
    if sha256_file(identity_path) != expected_identity_hash:
        raise RuntimeError("P2.2 identity edge hash does not match its passing report")

    normalized_root = Path(inputs["normalized_root"])
    state_path = Path(outputs["reported_states"])
    duplicate_path = Path(outputs["duplicate_ledger"])
    state_writer = DataFrameParquetWriter(state_path, STATE_SCHEMA, config["parquet"])
    duplicate_writer = DataFrameParquetWriter(duplicate_path, DUPLICATE_SCHEMA, config["parquet"])
    states: dict[str, pd.DataFrame] = {}
    ambiguity_rows: list[dict[str, Any]] = []
    raw_rows_by_snapshot: dict[str, int] = {}
    state_rows_by_snapshot: dict[str, int] = {}
    duplicate_rows_by_snapshot: dict[str, int] = {}
    for snapshot in snapshots:
        thermo_path = normalized_root / f"snapshot={snapshot}" / "raw_thermo.parquet"
        raw = pq.read_table(thermo_path, columns=_thermo_columns()).to_pandas()
        raw_rows_by_snapshot[snapshot] = len(raw)
        selected, duplicates, ambiguities = deduplicate_reported_states(raw)
        selected["snapshot_id"] = snapshot
        duplicates["snapshot_id"] = snapshot
        state_writer.write(selected)
        duplicate_writer.write(duplicates)
        states[snapshot] = selected
        ambiguity_rows.extend(ambiguities)
        state_rows_by_snapshot[snapshot] = len(selected)
        duplicate_rows_by_snapshot[snapshot] = len(duplicates)
    state_writer.close()
    duplicate_writer.close()
    ambiguity_path = Path(outputs["ambiguity_ledger"])
    write_jsonl_atomic(ambiguity_path, ambiguity_rows)

    edge_columns = [
        "edge_id", "canonical_lineage_id", "confidence", "accepted",
        "source_snapshot", "target_snapshot", "source_material_id", "target_material_id",
    ]
    edges = pq.read_table(identity_path, columns=edge_columns).to_pandas()
    edges = edges[
        edges["accepted"] & edges["confidence"].isin(config["transition_policy"]["accepted_identity_confidences"])
    ].copy()
    transition_path = Path(outputs["transition_labels"])
    transition_writer = DataFrameParquetWriter(
        transition_path, TRANSITION_SCHEMA, config["parquet"]
    )
    counts_by_pair: dict[str, dict[str, Any]] = {}
    transition_counts: Counter[str] = Counter()
    for (source_snapshot, target_snapshot), pair_edges in edges.groupby(
        ["source_snapshot", "target_snapshot"], sort=True
    ):
        pair = build_pair_transitions(
            pair_edges, states[str(source_snapshot)], states[str(target_snapshot)]
        )
        output_frame = _transition_output_frame(pair)
        transition_writer.write(output_frame)
        status_counts = pair["observation_status"].value_counts().to_dict()
        observed = pair[pair["observation_status"] == "observed"]
        flip_count = int(observed["label_flip"].astype("boolean").fillna(False).sum())
        class_counts = observed["reported_label_transition"].value_counts().to_dict()
        pair_name = f"{source_snapshot}_to_{target_snapshot}"
        counts_by_pair[pair_name] = {
            "identity_edges": len(pair_edges),
            "transition_rows": len(pair),
            "observation_status": {str(k): int(v) for k, v in status_counts.items()},
            "observed_label_classes": {str(k): int(v) for k, v in class_counts.items()},
            "label_flips": flip_count,
        }
        transition_counts.update({str(k): int(v) for k, v in status_counts.items()})
    transition_writer.close()

    audit_rows, audit_summary = audit_transition_sources(
        transition_path,
        Path(inputs["raw_snapshot_manifest"]),
        config["audit"],
        seed,
    )
    audit_csv_path = Path(outputs["source_audit_csv"])
    write_csv_atomic(audit_csv_path, audit_rows)
    audit_json_path = Path(outputs["source_audit_json"])
    audit_summary.update(
        {
            "task_id": "P3.1",
            "created_at_utc": utc_now(),
            "audit_csv": str(audit_csv_path),
            "audit_csv_sha256": sha256_file(audit_csv_path),
        }
    )
    write_json_atomic(audit_json_path, audit_summary)
    dictionary_path = Path(outputs["data_dictionary"])
    write_csv_atomic(dictionary_path, _dictionary_rows())

    transition_table = pq.read_table(
        transition_path,
        columns=["identity_edge_id", "observation_status", "label_flip", "identity_confidence"],
    )
    distinct_identity_edges = pc.count_distinct(transition_table["identity_edge_id"]).as_py()
    total_flips = pc.sum(
        pc.cast(pc.fill_null(transition_table["label_flip"], False), pa.int64())
    ).as_py()
    a2_rows = transition_table.filter(pc.equal(transition_table["identity_confidence"], "A2"))
    output_paths = [state_path, transition_path, duplicate_path]
    gate_passed = audit_summary["status"] == "PASS"
    manifest = {
        "task_id": "P3.1",
        "status": "PASS" if gate_passed else "FAIL",
        "gate_status": "GO" if gate_passed else "NO-GO",
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "seed": seed,
        "network_access": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_identity_edge_sha256": sha256_file(identity_path),
        "p2_2_report_sha256": sha256_file(inputs["p2_2_report"]),
        "raw_rows_by_snapshot": raw_rows_by_snapshot,
        "reported_state_rows_by_snapshot": state_rows_by_snapshot,
        "duplicate_rows_by_snapshot": duplicate_rows_by_snapshot,
        "reported_value_conflicts": len(ambiguity_rows),
        "accepted_identity_edges": len(edges),
        "distinct_identity_edges_in_transition_table": distinct_identity_edges,
        "transition_rows": transition_writer.rows,
        "transition_status_counts": dict(sorted(transition_counts.items())),
        "transition_counts_by_pair": counts_by_pair,
        "reported_label_flips": int(total_flips),
        "a2_transition_rows": a2_rows.num_rows,
        "source_value_audit": audit_summary,
        "outputs": [
            {
                "path": str(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": _schema_hash(path),
            }
            for path in output_paths
        ],
        "supporting_files": {
            str(ambiguity_path): sha256_file(ambiguity_path),
            str(audit_csv_path): sha256_file(audit_csv_path),
            str(audit_json_path): sha256_file(audit_json_path),
            str(dictionary_path): sha256_file(dictionary_path),
        },
        "gate": {
            "sampled_values_match_source": {
                "observed": audit_summary["match_rate"],
                "minimum": float(config["audit"]["required_match_rate"]),
                "passed": gate_passed,
            },
            "passed": gate_passed,
        },
    }
    manifest_path = Path(outputs["manifest"])
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_reported_transitions(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest_path = Path(config["output"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for output in manifest["outputs"]:
        path = Path(output["path"])
        if not path.exists():
            failures.append(f"missing:{path}")
            continue
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(output["rows"]):
            failures.append(f"rows:{path}")
        if sha256_file(path) != output["sha256"]:
            failures.append(f"sha256:{path}")
        if _schema_hash(path) != output["schema_sha256"]:
            failures.append(f"schema:{path}")
    for path_text, expected in manifest["supporting_files"].items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected:
            failures.append(f"supporting:{path}")
    transition_path = Path(config["output"]["transition_labels"])
    table = pq.read_table(
        transition_path,
        columns=[
            "transition_id", "identity_edge_id", "observation_status",
            "source_is_stable", "target_is_stable", "label_flip",
        ],
    )
    distinct_transition_ids = pc.count_distinct(table["transition_id"]).as_py()
    if distinct_transition_ids != table.num_rows:
        failures.append("duplicate_transition_id")
    distinct_edges = pc.count_distinct(table["identity_edge_id"]).as_py()
    if distinct_edges != int(manifest["accepted_identity_edges"]):
        failures.append("accepted_identity_edge_coverage")
    observed = table.filter(pc.equal(table["observation_status"], "observed"))
    expected_flip = pc.not_equal(observed["source_is_stable"], observed["target_is_stable"])
    if not pc.all(pc.equal(expected_flip, observed["label_flip"])).as_py():
        failures.append("label_flip_inconsistent")
    if not manifest["gate"]["passed"]:
        failures.append("gate_not_passed")
    temporary_files = list(Path(config["output"]["root"]).rglob("*.tmp"))
    temporary_files += list(Path("reports/P3_1").rglob("*.tmp"))
    if temporary_files:
        failures.append("temporary_files")
    return {
        "task_id": "P3.1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "reported_states": pq.ParquetFile(config["output"]["reported_states"]).metadata.num_rows,
        "transition_rows": table.num_rows,
        "distinct_transition_ids": distinct_transition_ids,
        "accepted_identity_edges_covered": distinct_edges,
        "observed_transition_rows": observed.num_rows,
        "reported_label_flips": manifest["reported_label_flips"],
        "source_audit_match_rate": manifest["source_value_audit"]["match_rate"],
        "source_audit_records": manifest["source_value_audit"]["sampled_source_records"],
        "gate_status": manifest["gate_status"],
        "temporary_files": len(temporary_files),
        "network_access": False,
        "verified_at_utc": utc_now(),
    }
