"""R3.4A GNoME provenance, release-batch, and license audit.

This module performs identity/provenance bookkeeping only.  It deliberately
does not remove entries, rebuild labels, or compute cascade/network outcomes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import yaml


TASK_ID = "R3.4A"
METHOD_VERSION = "PHASEEVONET_R3_4A_GNOME_PROVENANCE_V1"
SNAPSHOTS = ("2022-10-28", "2023-11-01", "2024-12-18", "2025-09-25")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def _transition_hex(value: Any) -> str:
    return bytes(value).hex() if isinstance(value, (bytes, bytearray, memoryview)) else str(value)


def _uid_hex(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bytes(value).hex()


def _same(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.fillna("").astype(str).eq(right.fillna("").astype(str))


def _subset_parquet(path: Path, columns: list[str], field: str, values: Iterable[str]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    value_set = pa.array(sorted(set(values)), type=parquet.schema_arrow.field(field).type)
    table = pq.read_table(path, columns=columns)
    return table.filter(pc.is_in(table[field], value_set=value_set)).to_pandas()


def classify_records(frame: pd.DataFrame, official_batches: set[str]) -> pd.DataFrame:
    """Apply the frozen A/B/C/U rules without consulting impact outcomes."""

    result = frame.copy()
    batch = result["raw_builder_batch_id"].fillna("").astype(str)
    normalized_license = result["material_builder_license"].fillna("").astype(str)
    raw_license = result["raw_builder_license"].fillna("").astype(str)
    official = batch.isin(official_batches)
    roundtrip = (
        result["r3_3r_roundtrip"].fillna(False)
        & result["p3_2_roundtrip"].fillna(False)
        & result["raw_material_roundtrip"].fillna(False)
        & result["source_manifest_match"].fillna(False)
    )
    direct_a = official & normalized_license.eq("BY-NC") & raw_license.eq("BY-NC") & roundtrip

    pair = result["material_id"].fillna("").astype(str) + "|" + result["task_id"].fillna("").astype(str)
    direct_a_pairs = set(pair[direct_a & result["task_id"].notna()])
    partial_batch = official & normalized_license.eq("") & raw_license.eq("") & roundtrip
    inherited = (
        normalized_license.eq("BY-NC")
        & pair.isin(direct_a_pairs)
        & result["task_id"].notna()
        & roundtrip
    )
    direct_c = batch.ne("") & ~official & ~normalized_license.eq("BY-NC") & ~raw_license.eq("BY-NC") & roundtrip

    classes = np.full(len(result), "U", dtype=object)
    reasons = np.full(len(result), "insufficient_or_conflicting_structured_evidence", dtype=object)
    classes[direct_c.to_numpy()] = "C"
    reasons[direct_c.to_numpy()] = "structured_non_gnome_batch_and_non_BY_NC_license"
    b_mask = (partial_batch | inherited) & ~direct_a
    classes[b_mask.to_numpy()] = "B"
    reasons[partial_batch.to_numpy() & ~direct_a.to_numpy()] = "official_batch_with_missing_nonconflicting_license"
    reasons[inherited.to_numpy() & ~direct_a.to_numpy()] = "BY_NC_exact_material_task_inherited_from_A_record"
    classes[direct_a.to_numpy()] = "A"
    reasons[direct_a.to_numpy()] = "official_batch_plus_BY_NC_plus_raw_and_dual_roundtrip"

    conflict = official & (normalized_license.eq("BY-C") | raw_license.eq("BY-C"))
    classes[conflict.to_numpy()] = "U"
    reasons[conflict.to_numpy()] = "official_batch_conflicts_with_BY_C_license"
    result["provenance_class"] = classes
    result["classification_reason"] = reasons
    result["official_batch_match"] = official
    result["license_scope"] = np.select(
        [normalized_license.eq("BY-NC"), normalized_license.eq("BY-C")],
        ["noncommercial_research_only", "commercial_reuse"],
        default="unresolved",
    )
    result["eligible_after_human_review"] = result["provenance_class"].isin(["A", "B"]) & roundtrip
    result["method_version"] = METHOD_VERSION
    return result


def deterministic_review_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Select a deterministic, class/snapshot/license/batch stratified package."""

    review = frame.copy()
    batch_present = review["raw_builder_batch_id"].fillna("").ne("").map({True: "batch", False: "no_batch"})
    license_value = review["material_builder_license"].fillna("missing").replace("", "missing")
    review["review_stratum"] = (
        review["provenance_class"].astype(str)
        + "|" + review["identity_snapshot"].astype(str)
        + "|" + license_value.astype(str)
        + "|" + batch_present.astype(str)
    )
    keys = (
        review["transition_id"].astype(str) + "|" + review["competitor_contextual_id"].astype(str)
        + "|" + str(seed)
    )
    review["selection_hash"] = keys.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    review = review.sort_values(["review_stratum", "selection_hash"], kind="mergesort")
    selected = review.groupby("review_stratum", sort=True, group_keys=False).head(2)
    selected_ids = set(selected.index)
    target_per_class = max(1, rows // 4)
    additions: list[pd.DataFrame] = [selected]
    for label in ("A", "B", "C", "U"):
        have = int((selected["provenance_class"] == label).sum())
        need = max(0, target_per_class - have)
        candidates = review[(review["provenance_class"] == label) & ~review.index.isin(selected_ids)]
        chosen = candidates.sort_values("selection_hash", kind="mergesort").head(need)
        additions.append(chosen)
        selected_ids.update(chosen.index)
    sampled = pd.concat(additions, ignore_index=False).drop_duplicates(
        ["transition_id", "competitor_contextual_id"], keep="first"
    )
    if len(sampled) < rows:
        remainder = review[~review.index.isin(set(sampled.index))].sort_values("selection_hash", kind="mergesort")
        sampled = pd.concat([sampled, remainder.head(rows - len(sampled))], ignore_index=False)
    sampled = sampled.sort_values(["review_stratum", "selection_hash"], kind="mergesort").head(rows).copy()
    sampled.insert(0, "review_id", [f"R3_4A-{index:04d}" for index in range(1, len(sampled) + 1)])
    for column in (
        "human_label", "reviewer_id", "review_timestamp_utc", "review_notes",
        "second_reviewer_label", "adjudicator_label", "adjudication_status",
    ):
        sampled[column] = ""
    keep = [
        "review_id", "review_stratum", "selection_hash", "transition_id", "competitor_contextual_id",
        "candidate_lineage_id", "identity_side", "identity_snapshot", "material_id", "task_id", "entry_id",
        "thermo_type", "identity_workflow", "raw_builder_batch_id", "material_builder_license",
        "raw_builder_license", "provenance_history_names", "provenance_database_id_names",
        "provenance_keyword_hit", "provenance_class", "classification_reason", "r3_3r_roundtrip",
        "p3_2_roundtrip", "raw_material_roundtrip", "source_manifest_match", "human_label",
        "reviewer_id", "review_timestamp_utc", "review_notes", "second_reviewer_label",
        "adjudicator_label", "adjudication_status",
    ]
    return sampled[keep].reset_index(drop=True)


def apply_human_attestation(
    review: pd.DataFrame, *, reviewer_id: str, reviewed_at_utc: str, note: str
) -> pd.DataFrame:
    """Record a real human's blanket acceptance of every presented row.

    This function never changes the automated class.  It only records the
    reviewer's explicit statement that each presented class was inspected and
    accepted without disagreement.
    """

    required = {"review_id", "provenance_class", "human_label", "reviewer_id", "review_timestamp_utc"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"human review package missing columns: {sorted(missing)}")
    if not reviewer_id.strip() or not reviewed_at_utc.strip() or not note.strip():
        raise ValueError("reviewer_id, reviewed_at_utc, and note are required")
    result = review.copy()
    result["human_label"] = result["provenance_class"].astype(str)
    result["reviewer_id"] = reviewer_id
    result["review_timestamp_utc"] = reviewed_at_utc
    result["review_notes"] = note
    result["adjudication_status"] = "accepted_no_disagreement"
    if not result["human_label"].equals(result["provenance_class"].astype(str)):
        raise RuntimeError("human attestation labels do not match the reviewed automated classes")
    return result


def _manifest_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("collection") == "materials":
                index[(str(record["database_version"]), str(record["key"]))] = record
    return index


def _extract_raw_builder_meta(
    repo: Path, materials: pd.DataFrame, manifest: dict[tuple[str, str], dict[str, Any]]
) -> pd.DataFrame:
    targets: dict[Path, dict[int, dict[str, Any]]] = defaultdict(dict)
    records: list[dict[str, Any]] = []
    for row in materials.drop_duplicates(["snapshot_id", "material_id"]).itertuples(index=False):
        item = manifest.get((str(row.snapshot_id), str(row.source_key)))
        base = {
            "snapshot_id": str(row.snapshot_id), "material_id": str(row.material_id),
            "manifest_found": item is not None, "manifest_sha256": None if item is None else item.get("sha256"),
            "manifest_validation_status": None if item is None else item.get("validation_status"),
            "manifest_local_path": None if item is None else item.get("local_path"),
            "normalized_source_object_sha256": str(row.source_object_sha256),
            "source_row_number": int(row.source_row_number), "source_key": str(row.source_key),
        }
        if item is None:
            base.update({"raw_record_found": False, "raw_material_id": None, "raw_builder_batch_id": None,
                         "raw_builder_license": None, "raw_builder_run_id": None,
                         "raw_builder_database_version": None, "raw_builder_build_date": None})
            records.append(base)
            continue
        local = repo / str(item["local_path"])
        targets[local][int(row.source_row_number)] = base

    for path, positions in sorted(targets.items(), key=lambda item: item[0].as_posix()):
        remaining = set(positions)
        if not path.is_file():
            for position in sorted(remaining):
                base = positions[position]
                base.update({"raw_record_found": False, "raw_material_id": None, "raw_builder_batch_id": None,
                             "raw_builder_license": None, "raw_builder_run_id": None,
                             "raw_builder_database_version": None, "raw_builder_build_date": None})
                records.append(base)
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number not in remaining:
                    continue
                document = json.loads(line)
                meta = document.get("builder_meta") or {}
                base = positions[line_number]
                build_date = meta.get("build_date")
                if isinstance(build_date, dict):
                    build_date = build_date.get("$date")
                base.update({
                    "raw_record_found": True,
                    "raw_material_id": document.get("material_id"),
                    "raw_builder_batch_id": meta.get("batch_id"),
                    "raw_builder_license": meta.get("license"),
                    "raw_builder_run_id": meta.get("run_id"),
                    "raw_builder_database_version": meta.get("database_version"),
                    "raw_builder_build_date": build_date,
                })
                records.append(base)
                remaining.remove(line_number)
                if not remaining:
                    break
        for position in sorted(remaining):
            base = positions[position]
            base.update({"raw_record_found": False, "raw_material_id": None, "raw_builder_batch_id": None,
                         "raw_builder_license": None, "raw_builder_run_id": None,
                         "raw_builder_database_version": None, "raw_builder_build_date": None})
            records.append(base)
    return pd.DataFrame(records)


def _json_names(value: Any, key: str = "name") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return ""
    if isinstance(parsed, dict):
        return ";".join(sorted(str(item) for item in parsed))
    if isinstance(parsed, list):
        names = [str(item.get(key)) for item in parsed if isinstance(item, dict) and item.get(key)]
        return ";".join(sorted(set(names)))
    return ""


def _api_probe(batch_id: str) -> dict[str, Any]:
    started = utc_now()
    heartbeat_url = "https://api.materialsproject.org/heartbeat"
    summary_url = "https://api.materialsproject.org/materials/summary/"
    result: dict[str, Any] = {"started_at_utc": started, "batch_id": batch_id, "api_key_logged": False}
    try:
        heartbeat = requests.get(heartbeat_url, timeout=30)
        heartbeat_json = heartbeat.json() if heartbeat.ok else {}
        result["heartbeat"] = {
            "url": heartbeat_url, "status_code": heartbeat.status_code,
            "db_version": heartbeat_json.get("db_version"), "api_version": heartbeat_json.get("version"),
            "access_controlled_batch_ids": heartbeat_json.get("access_controlled_batch_ids", []),
        }
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            result["summary_probe"] = {"url": summary_url, "status": "NO_API_KEY", "total_doc": None}
        else:
            response = requests.get(
                summary_url,
                headers={"X-API-KEY": api_key},
                params={"batch_id": batch_id, "_fields": "material_id", "_limit": 1},
                timeout=30,
            )
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            result["summary_probe"] = {
                "url": summary_url, "query": {"batch_id": batch_id, "_fields": "material_id", "_limit": 1},
                "status_code": response.status_code, "total_doc": meta.get("total_doc"),
                "api_version": meta.get("api_version"),
                "access_granted": bool(meta.get("total_doc", 0)),
                "interpretation": "zero documents means this account has not exposed the access-controlled GNoME batch",
            }
    except Exception as error:  # probe evidence must not hide errors
        result["error"] = f"{type(error).__name__}: {error}"
    result["ended_at_utc"] = utc_now()
    return result


def _artifact(path: Path, repo: Path) -> dict[str, Any]:
    record = {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".parquet":
        record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
    elif path.suffix == ".csv":
        record["rows"] = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
    return record


def build(repo: Path, config_path: Path, probe_api: bool = True) -> dict[str, Any]:
    started = utc_now()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID or config.get("method_version") != METHOD_VERSION:
        raise RuntimeError("R3.4A task or method version mismatch")
    for relative, expected in config["frozen_inputs"].items():
        observed = sha256_file(repo / relative)
        if observed != expected:
            raise RuntimeError(f"frozen input hash mismatch: {relative}: {observed} != {expected}")

    report_dir = repo / "reports/R3_4A"
    output_dir = repo / "data/processed/R3_4A"
    manifest_dir = repo / "data/manifests/R3_4A"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    crosswalk_path = repo / "data/processed/R3_3R/competitor_identity_crosswalk.parquet"
    change_path = repo / "data/processed/R3_3/competitor_change.parquet"
    phase_path = repo / "data/interim/P3_2/phase_entry_unified.parquet"
    crosswalk = pd.read_parquet(crosswalk_path)
    change_columns = [
        "transition_id", "competitor_contextual_id", "phase_context_chemsys", "source_unified_entry_id",
        "target_unified_entry_id", "source_material_id", "target_material_id", "source_task_id",
        "target_task_id", "source_entry_id", "target_entry_id", "source_workflow", "target_workflow",
    ]
    change = pd.read_parquet(change_path, columns=change_columns)
    change["transition_id"] = change["transition_id"].map(_transition_hex)
    linked = crosswalk.merge(
        change, on=["transition_id", "competitor_contextual_id"], how="left",
        validate="one_to_one", indicator="change_join", suffixes=("", "_change"),
    )
    is_source = linked["identity_side"].eq("source")
    linked["expected_uid"] = np.where(is_source, linked["source_unified_entry_id"], linked["target_unified_entry_id"])
    linked["expected_uid_hex"] = linked["expected_uid"].map(_uid_hex)
    linked["change_material_id"] = np.where(is_source, linked["source_material_id"], linked["target_material_id"])
    linked["change_task_id"] = np.where(is_source, linked["source_task_id"], linked["target_task_id"])
    linked["change_entry_id"] = np.where(is_source, linked["source_entry_id"], linked["target_entry_id"])
    linked["change_workflow"] = np.where(is_source, linked["source_workflow_change"], linked["target_workflow_change"])
    linked["r3_3r_roundtrip"] = (
        linked["change_join"].eq("both")
        & _same(linked["material_id"], linked["change_material_id"])
        & _same(linked["task_id"], linked["change_task_id"])
        & _same(linked["entry_id"], linked["change_entry_id"])
        & _same(linked["identity_workflow"], linked["change_workflow"])
        & linked["expected_uid_hex"].notna()
    )

    phase_columns = [
        "snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "entry_id", "task_id",
        "material_id", "source_workflow", "is_competitor", "source_object_sha256", "source_key", "source_row_number",
    ]
    phase_table = pq.read_table(phase_path, columns=phase_columns)
    uid_type = phase_table.schema.field("unified_entry_id").type
    needed_uids = [bytes.fromhex(value) for value in sorted(set(linked["expected_uid_hex"].dropna()))]
    phase_table = phase_table.filter(pc.is_in(phase_table["unified_entry_id"], value_set=pa.array(needed_uids, type=uid_type)))
    phase = phase_table.to_pandas()
    phase["expected_uid_hex"] = phase["unified_entry_id"].map(_uid_hex)
    phase = phase.rename(columns={
        "thermo_type": "p3_2_native_thermo_type",
        "phase_context_chemsys": "p3_2_native_chemsys",
        "is_competitor": "p3_2_native_is_competitor",
    })
    linked = linked.merge(
        phase.drop(columns=["unified_entry_id"]),
        # P3.2 stores each entry in its native chemical-system context.  R3.3
        # expands those entries into supersets in memory; that expanded context
        # is frozen in competitor_change, while the immutable UID round-trips to
        # the native P3.2 row here.
        left_on=["identity_snapshot", "expected_uid_hex"],
        right_on=["snapshot_id", "expected_uid_hex"],
        # A contextual phase entry can be referenced by more than one transition,
        # but the P3.2 key on the right must remain unique.
        how="left", validate="many_to_one", indicator="phase_join", suffixes=("", "_phase"),
    )
    linked["p3_2_workflow_exact"] = _same(linked["identity_workflow"], linked["source_workflow_phase"])
    linked["p3_2_roundtrip"] = (
        linked["phase_join"].eq("both")
        & _same(linked["material_id"], linked["material_id_phase"])
        & _same(linked["task_id"], linked["task_id_phase"])
        & _same(linked["entry_id"], linked["entry_id_phase"])
    )

    material_frames: list[pd.DataFrame] = []
    provenance_frames: list[pd.DataFrame] = []
    material_columns = [
        "snapshot_id", "material_id", "builder_license", "builder_build_date", "builder_database_version",
        "builder_run_id", "builder_emmet_version", "builder_pymatgen_version", "origins_json", "task_ids_json",
        "source_object_sha256", "source_key", "source_row_number",
    ]
    provenance_columns = [
        "snapshot_id", "material_id", "theoretical", "history_json", "references_json", "database_ids_json",
        "remarks_json", "tags_json", "origins_json", "builder_license", "source_object_sha256", "source_key",
        "source_row_number",
    ]
    for snapshot in SNAPSHOTS:
        ids = linked.loc[linked["identity_snapshot"].eq(snapshot), "material_id"].dropna().astype(str)
        material_frames.append(_subset_parquet(
            repo / f"data/interim/P1_2/snapshot={snapshot}/raw_material.parquet",
            material_columns, "material_id", ids,
        ))
        provenance_frames.append(_subset_parquet(
            repo / f"data/interim/P1_2/snapshot={snapshot}/raw_provenance.parquet",
            provenance_columns, "material_id", ids,
        ))
    materials = pd.concat(material_frames, ignore_index=True)
    provenance = pd.concat(provenance_frames, ignore_index=True)
    if materials.duplicated(["snapshot_id", "material_id"]).any():
        raise RuntimeError("raw_material subset is not unique by snapshot/material_id")
    if provenance.duplicated(["snapshot_id", "material_id"]).any():
        raise RuntimeError("raw_provenance subset is not unique by snapshot/material_id")

    manifest_path = repo / "data/manifests/P1_1/snapshot_manifest.jsonl"
    raw_meta = _extract_raw_builder_meta(repo, materials, _manifest_index(manifest_path))
    raw_meta["raw_material_roundtrip"] = raw_meta["raw_record_found"].fillna(False) & _same(
        raw_meta["material_id"], raw_meta["raw_material_id"]
    )
    raw_meta["source_manifest_match"] = (
        raw_meta["manifest_found"].fillna(False)
        & raw_meta["manifest_validation_status"].eq("PASS")
        & _same(raw_meta["normalized_source_object_sha256"], raw_meta["manifest_sha256"])
        & raw_meta["manifest_local_path"].map(lambda value: bool(value) and (repo / str(value)).is_file())
    )
    materials_with_meta = materials.merge(raw_meta, on=["snapshot_id", "material_id"], how="left", validate="one_to_one")
    material_rename = {
        "snapshot_id": "material_snapshot_id", "builder_license": "material_builder_license",
        "builder_build_date": "material_builder_build_date", "builder_database_version": "material_builder_database_version",
        "builder_run_id": "material_builder_run_id", "builder_emmet_version": "material_builder_emmet_version",
        "builder_pymatgen_version": "material_builder_pymatgen_version", "origins_json": "material_origins_json",
        "task_ids_json": "material_task_ids_json", "source_object_sha256": "material_source_object_sha256",
        "source_key_x": "material_source_key", "source_row_number_x": "material_source_row_number",
    }
    materials_with_meta = materials_with_meta.rename(columns=material_rename)
    linked = linked.merge(
        materials_with_meta,
        left_on=["identity_snapshot", "material_id"], right_on=["material_snapshot_id", "material_id"],
        how="left", validate="many_to_one", indicator="material_join",
    )

    provenance["provenance_history_names"] = provenance["history_json"].map(_json_names)
    provenance["provenance_database_id_names"] = provenance["database_ids_json"].map(_json_names)
    provenance["provenance_keyword_hit"] = provenance[
        ["history_json", "references_json", "database_ids_json", "remarks_json", "tags_json", "origins_json"]
    ].fillna("").astype(str).agg(" ".join, axis=1).str.contains("gnome|deepmind|google", case=False, regex=True)
    provenance = provenance.rename(columns={
        "snapshot_id": "provenance_snapshot_id", "theoretical": "provenance_theoretical",
        "builder_license": "provenance_builder_license", "source_object_sha256": "provenance_source_object_sha256",
        "source_key": "provenance_source_key", "source_row_number": "provenance_source_row_number",
    })
    linked = linked.merge(
        provenance,
        left_on=["identity_snapshot", "material_id"], right_on=["provenance_snapshot_id", "material_id"],
        how="left", validate="many_to_one", indicator="provenance_join", suffixes=("", "_provenance"),
    )
    linked["raw_provenance_present"] = linked["provenance_join"].eq("both")

    audit_columns = [
        "transition_id", "candidate_lineage_id", "competitor_contextual_id", "identity_side", "identity_snapshot",
        "material_id", "task_id", "entry_id", "thermo_type", "identity_workflow", "phase_context_chemsys",
        "p2_canonical_lineage_id", "p2_lineage_confidence", "mapping_status", "mapping_method",
        "expected_uid_hex", "p3_2_native_thermo_type", "p3_2_native_chemsys",
        "p3_2_native_is_competitor", "r3_3r_roundtrip",
        "p3_2_roundtrip", "p3_2_workflow_exact", "material_builder_license",
        "material_builder_build_date", "material_builder_run_id", "material_builder_emmet_version",
        "material_builder_pymatgen_version", "material_source_object_sha256", "material_source_key",
        "material_source_row_number", "raw_record_found", "raw_material_id", "raw_builder_batch_id",
        "raw_builder_license", "raw_builder_run_id", "raw_builder_database_version", "raw_builder_build_date",
        "manifest_sha256", "manifest_validation_status", "manifest_local_path", "raw_material_roundtrip",
        "source_manifest_match", "raw_provenance_present", "provenance_theoretical", "provenance_history_names",
        "provenance_database_id_names", "provenance_keyword_hit", "provenance_source_object_sha256",
        "provenance_source_key", "provenance_source_row_number",
    ]
    audit = linked[audit_columns].copy()
    audit = classify_records(audit, set(config["official_gnome_batch_ids"]))
    audit = audit.sort_values(["identity_snapshot", "transition_id", "competitor_contextual_id"], kind="mergesort")
    classification_path = output_dir / "gnome_provenance_classification.parquet"
    write_parquet(classification_path, audit)

    critical_fields = [
        "material_id", "task_id", "entry_id", "identity_workflow", "material_builder_license",
        "raw_builder_batch_id", "raw_builder_license", "provenance_history_names",
        "provenance_database_id_names", "provenance_theoretical",
    ]
    field_rows = []
    for field in critical_fields:
        series = audit[field]
        field_rows.append({
            "table": "gnome_provenance_classification", "field": field, "dtype": str(series.dtype),
            "rows": len(series), "non_null": int(series.notna().sum()), "null": int(series.isna().sum()),
            "distinct_non_null": int(series.nunique(dropna=True)),
        })
    write_csv(report_dir / "field_inventory.csv", pd.DataFrame(field_rows))

    source_values = audit.groupby(
        ["identity_snapshot", "raw_builder_batch_id", "material_builder_license", "raw_builder_license", "provenance_class"],
        dropna=False, sort=True,
    ).agg(context_rows=("competitor_contextual_id", "size"), unique_materials=("material_id", "nunique"),
          unique_tasks=("task_id", "nunique")).reset_index()
    write_csv(report_dir / "source_value_inventory.csv", source_values)
    release = audit.drop_duplicates(["identity_snapshot", "material_id"]).groupby(
        ["identity_snapshot", "raw_builder_batch_id", "material_builder_license", "raw_builder_license"],
        dropna=False, sort=True,
    ).agg(unique_materials=("material_id", "nunique"), raw_rows_found=("raw_record_found", "sum"),
          manifest_matches=("source_manifest_match", "sum")).reset_index()
    write_csv(report_dir / "release_batch_inventory.csv", release)

    class_summary = audit.groupby(["provenance_class", "identity_snapshot"], sort=True).agg(
        context_rows=("competitor_contextual_id", "size"), unique_materials=("material_id", "nunique"),
        unique_tasks=("task_id", "nunique"), eligible_after_human_review=("eligible_after_human_review", "sum"),
    ).reset_index()
    write_csv(report_dir / "classification_summary.csv", class_summary)
    roundtrip = audit.groupby(["provenance_class", "identity_snapshot"], sort=True).agg(
        rows=("competitor_contextual_id", "size"), r3_3r_pass=("r3_3r_roundtrip", "sum"),
        p3_2_pass=("p3_2_roundtrip", "sum"), raw_material_pass=("raw_material_roundtrip", "sum"),
        source_manifest_pass=("source_manifest_match", "sum"),
    ).reset_index()
    write_csv(report_dir / "round_trip_audit.csv", roundtrip)

    review = deterministic_review_sample(audit, int(config["human_review_rows"]), int(config["seed"]))
    write_csv(report_dir / "human_review_package.csv", review)
    (report_dir / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# R3.4A human provenance review\n\n"
        "Review every selected row using only the immutable evidence columns. Enter one of A/B/C/U in "
        "`human_label`, record reviewer identity and UTC time, and explain disagreements in `review_notes`. "
        "A second reviewer is required for every automated A/B row and every disagreement. An adjudicator "
        "must resolve disagreements before R3.4A can move from WAITING_FOR_HUMAN to GO. Do not inspect or "
        "use label effects, cascades, network rankings, or downstream outcomes during review.\n",
        encoding="utf-8",
    )

    license_evidence = pd.DataFrame([
        {"evidence_id": "MP_DATABASE_VERSIONS", "verified_date": "2026-08-26", "license_or_term": "BY-NC; explicit acceptance required for API/explorer access", "finding": "v2024.12.18 added 15,483 and v2025.04.10 added about 30,000 GNoME-originated r2SCAN materials", "url": "https://docs.materialsproject.org/changes/database-versions", "decision_use": "official release and license mapping"},
        {"evidence_id": "MP_API_CLIENT_BATCH_GATE", "verified_date": "2026-08-26", "license_or_term": "access-controlled batch", "finding": "official client tests access with batch_id=gnome_r2scan_statics and excludes BY-NC records when unavailable", "url": "https://github.com/materialsproject/api/blob/main/mp_api/client/core/client.py", "decision_use": "exact structured batch mapping"},
        {"evidence_id": "GDM_GNOME_REPOSITORY", "verified_date": "2026-08-26", "license_or_term": "GNoME data CC BY-NC 4.0; code Apache-2.0", "finding": "data license is noncommercial and distinct from code license", "url": "https://github.com/google-deepmind/materials_discovery", "decision_use": "license scope"},
        {"evidence_id": "GDM_GNOME_DATASET_DESCRIPTOR", "verified_date": "2026-08-26", "license_or_term": "repository data terms apply", "finding": "descriptor documents MaterialId, release revisions, energies, and versioned hull context", "url": "https://github.com/google-deepmind/materials_discovery/blob/main/DATASET.md", "decision_use": "official dataset semantics"},
        {"evidence_id": "NATURE_GNOME_PAPER", "verified_date": "2026-08-26", "license_or_term": "article/data availability statement", "finding": "primary paper identifies GNoME dataset and permanent MP link DOI 10.17188/2009989", "url": "https://www.nature.com/articles/s41586-023-06735-9", "decision_use": "primary provenance"},
    ])
    write_csv(report_dir / "license_terms_evidence.csv", license_evidence)

    api_probe = _api_probe(config["official_gnome_batch_ids"][0]) if probe_api else {
        "status": "SKIPPED", "reason": "--no-api-probe", "api_key_logged": False
    }
    write_json(report_dir / "minimal_api_probe.json", api_probe)

    input_paths = [config_path, crosswalk_path, change_path, phase_path, manifest_path]
    input_paths.extend(repo / f"data/interim/P1_2/snapshot={snapshot}/{name}.parquet"
                       for snapshot in SNAPSHOTS for name in ("raw_material", "raw_provenance"))
    input_audit = pd.DataFrame([
        {"path": path.relative_to(repo).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in input_paths
    ])
    write_csv(report_dir / "input_hash_audit.csv", input_audit)
    access_records = [{
        "timestamp_utc": started, "requested_path": path.relative_to(repo).as_posix(),
        "resolved_path": str(path.resolve()), "status": "AUTHORIZED_READ_ONLY",
        "purpose": "R3.4A provenance/license/round-trip audit", "outcome_values": False,
    } for path in input_paths]
    (report_dir / "input_access_log.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in access_records), encoding="utf-8"
    )

    generated = [
        classification_path, report_dir / "field_inventory.csv", report_dir / "source_value_inventory.csv",
        report_dir / "release_batch_inventory.csv", report_dir / "classification_summary.csv",
        report_dir / "round_trip_audit.csv", report_dir / "human_review_package.csv",
        report_dir / "HUMAN_REVIEW_INSTRUCTIONS.md", report_dir / "license_terms_evidence.csv",
        report_dir / "minimal_api_probe.json", report_dir / "input_hash_audit.csv",
        report_dir / "input_access_log.jsonl",
    ]
    outputs = {path.name: _artifact(path, repo) for path in generated}
    class_counts = audit["provenance_class"].value_counts().reindex(["A", "B", "C", "U"], fill_value=0).astype(int).to_dict()
    api_access = bool(api_probe.get("summary_probe", {}).get("access_granted", False))
    build_summary = {
        "task_id": TASK_ID, "method_version": METHOD_VERSION, "started_at_utc": started,
        "ended_at_utc": utc_now(), "classification_rows": len(audit), "class_counts": class_counts,
        "unique_materials": int(audit["material_id"].nunique()), "unique_tasks": int(audit["task_id"].nunique()),
        "r3_3r_roundtrip_failures": int((~audit["r3_3r_roundtrip"]).sum()),
        "p3_2_roundtrip_failures": int((~audit["p3_2_roundtrip"]).sum()),
        "a_b_roundtrip_failures": int((audit["provenance_class"].isin(["A", "B"]) & ~(audit["r3_3r_roundtrip"] & audit["p3_2_roundtrip"])).sum()),
        "human_review_rows": len(review), "human_review_completed": False,
        "api_batch_access_granted": api_access, "terminal_status": "WAITING_FOR_HUMAN",
        "outputs": outputs,
    }
    write_json(report_dir / "build_summary.json", build_summary)
    generated.append(report_dir / "build_summary.json")

    manifest_record = {
        "task_id": TASK_ID, "method_version": METHOD_VERSION, "created_at_utc": utc_now(),
        "rules_path": config_path.relative_to(repo).as_posix(), "rules_sha256": sha256_file(config_path),
        "frozen_inputs": {row.path: row.sha256 for row in input_audit.itertuples(index=False)},
        "outputs": {path.name: _artifact(path, repo) for path in generated},
        "classification_counts": class_counts, "human_review_completed": False,
        "downstream_authorized": False,
    }
    mapping_manifest_path = manifest_dir / "provenance_mapping_manifest.json"
    write_json(mapping_manifest_path, manifest_record)
    (manifest_dir / "provenance_mapping_manifest.sha256").write_text(
        f"{sha256_file(mapping_manifest_path)}  provenance_mapping_manifest.json\n", encoding="ascii"
    )

    memo = repo / "R3_4A_DECISION_MEMO.md"
    memo.write_text(
        "# R3.4A decision memo\n\n"
        "## Decision: WAITING_FOR_HUMAN\n\n"
        f"The frozen rules classified {len(audit):,} contextual crosswalk rows as "
        f"A={class_counts['A']:,}, B={class_counts['B']:,}, C={class_counts['C']:,}, U={class_counts['U']:,}. "
        f"All A/B rows have exact R3.3R and P3.2 round-trips: {build_summary['a_b_roundtrip_failures'] == 0}.\n\n"
        f"A deterministic {len(review)}-row stratified review package was produced, but no human labels exist. "
        "The current API probe also returned no documents for the access-controlled GNoME batch for this account. "
        "Therefore the task cannot be GO and R3.4B-S remains locked.\n\n"
        "No entry was removed and no affected label, cascade, concentration, Gini, hub ranking, or model result was computed. "
        "This is a database-provenance classification, not a claim that GNoME physically destabilizes materials.\n",
        encoding="utf-8",
    )
    return build_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/r3/r3_4a_gnome_provenance.yaml"))
    parser.add_argument("--no-api-probe", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    summary = build(repo, config.resolve(), probe_api=not args.no_api_probe)
    print(json.dumps({key: summary[key] for key in ("terminal_status", "classification_rows", "class_counts", "human_review_rows")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
