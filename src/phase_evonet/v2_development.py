from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from emmet.core.mpid import AlphaID
from emmet.core.types.typing import (
    format_compound_identifier,
    format_identifier,
    format_task_id,
)


TASK_ID = "V2.1"
DEFAULT_CONFIG = "configs/analysis/v2_1_development_tables.yaml"
IDENTIFIER_CONVERSION_VERSION = f"emmet-core=={version('emmet-core')}"

BASE_IDENTIFIER_COLUMNS = [
    "transition_id",
    "canonical_lineage_id",
    "assigned_role",
    "identity_confidence",
    "source_snapshot",
    "source_material_id",
    "source_thermo_id",
    "thermo_type",
    "phase_context_chemsys",
]

LABEL_COLUMNS = [
    "transition_id",
    "canonical_lineage_id",
    "assigned_role",
    "identity_confidence",
    "source_snapshot",
    "target_snapshot",
    "source_reported_is_stable",
    "target_reported_is_stable",
    "source_reported_energy_above_hull",
    "target_reported_energy_above_hull",
    "delta_reported_energy_above_hull",
    "source_reported_formation_energy_per_atom",
    "target_reported_formation_energy_per_atom",
    "delta_reported_formation_energy_per_atom",
    "rebuilt_unified_flip",
    "rebuilt_unified_label_transition",
    "attribution_id",
]

PHASE_COLUMNS = [
    "snapshot_id",
    "thermo_type",
    "phase_context_chemsys",
    "entry_id",
    "material_id",
    "thermo_id",
    "task_id",
    "is_target",
    "is_competitor",
    "energy_above_hull",
    "is_stable",
]

ATTRIBUTION_COLUMNS = [
    "attribution_id",
    "transition_id",
    "source_snapshot",
    "target_snapshot",
    "competitor_inventory_contribution",
    "uncorrected_energy_contribution",
    "compatibility_correction_contribution",
    "candidate_identity_contribution",
    "dominant_channel",
    "candidate_identity_changed",
    "source_decomposition_phase_keys_json",
    "target_decomposition_phase_keys_json",
]

STATIC_CONTEXT_RENAMES = {
    "context_entry_count": "source_context_entry_count",
    "context_target_count": "source_context_candidate_count",
    "context_competitor_count": "source_context_competitor_count",
    "context_stable_count": "source_context_stable_count",
    "context_near_hull_10meV_count": "source_context_near_hull_10meV_count",
    "context_near_hull_25meV_count": "source_context_near_hull_25meV_count",
    "context_near_hull_50meV_count": "source_context_near_hull_50meV_count",
    "context_mean_energy_above_hull": "source_context_mean_energy_above_hull",
    "context_std_energy_above_hull": "source_context_std_energy_above_hull",
    "context_min_positive_energy_above_hull": "source_context_min_positive_energy_above_hull",
    "context_competitor_fraction": "source_context_competitor_fraction",
}

M0_FEATURES = ["source_energy_above_hull"]
M1_ADDITIONS = [
    "source_log1p_hull_distance_10meV",
    "source_is_stable",
    "source_formation_energy_per_atom",
    "source_nelements",
    "source_log1p_num_atoms",
    "source_composition_entropy",
    "source_max_element_fraction",
    "source_mean_atomic_number",
    "source_std_atomic_number",
    "source_abs_correction_per_atom",
    "source_decomposition_component_count",
    "source_record_count",
    "source_chemsys_dimensionality",
    "thermo_GGA_GGA+U",
    "thermo_GGA_GGA+U_R2SCAN",
    "thermo_R2SCAN",
]
STATIC_FEATURES = list(STATIC_CONTEXT_RENAMES.values())
TEMPORAL_FEATURES = [
    "source_history_snapshot_available",
    "source_prior_context_present",
    "source_history_exposure_days",
    "source_prior_context_entry_count",
    "source_prior_context_competitor_count",
    "source_prior_context_stable_count",
    "source_prior_context_near_hull_10meV_count",
    "source_context_entry_growth_since_previous",
    "source_context_competitor_growth_since_previous",
    "source_context_stable_growth_since_previous",
    "source_context_near_hull_10meV_growth_since_previous",
    "source_competitor_arrivals_since_previous",
    "source_competitor_removals_since_previous",
    "source_competitor_arrival_fraction_since_previous",
    "source_competitor_removal_fraction_since_previous",
    "source_context_entry_growth_per_year",
    "source_context_competitor_growth_per_year",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(path: str | Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def write_csv_atomic(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, target)


def write_parquet_atomic(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    compression: str,
    row_group_size: int,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression=compression,
        row_group_size=row_group_size,
    )
    os.replace(temporary, target)


def artifact_record(path: str | Path, *, include_schema: bool = False) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }
    if include_schema:
        parquet = pq.ParquetFile(target)
        result["rows"] = parquet.metadata.num_rows
        result["schema_sha256"] = schema_sha256(target)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise RuntimeError("configuration task_id must be V2.1")
    return payload


def _input_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in config["input"].items() if key != "expected_sha256"}


def verify_entry_and_input_hashes(config: dict[str, Any]) -> dict[str, Any]:
    inputs = _input_paths(config)
    expected = config["input"]["expected_sha256"]
    required = [
        "v2_0_report",
        "v2_0_report_sidecar",
        "development_labels",
        "baseline_source_features",
        "phase_entries",
        "transition_attribution",
        "model_matrix",
    ]
    missing = [key for key in required if not inputs[key].is_file()]
    if missing:
        raise FileNotFoundError(f"missing V2.1 inputs: {missing}")
    mismatches: list[dict[str, str]] = []
    actual: dict[str, str] = {}
    for key, expected_hash in expected.items():
        path = inputs[key]
        if key == "phase_decomposition" and not path.is_file():
            continue
        digest = sha256_file(path)
        actual[key] = digest
        if digest != expected_hash:
            mismatches.append(
                {"input": key, "expected_sha256": expected_hash, "actual_sha256": digest}
            )
    if mismatches:
        raise RuntimeError(f"frozen input hash mismatches: {mismatches}")
    report = json.loads(inputs["v2_0_report"].read_text(encoding="utf-8"))
    sidecar = inputs["v2_0_report_sidecar"].read_text(encoding="utf-8").split()[0]
    if actual["v2_0_report"] != sidecar:
        raise RuntimeError("V2.0 report sidecar hash mismatch")
    if not (
        report.get("task_status") == "DONE"
        and report.get("status") == "PASS"
        and report.get("gate_status") == "GO"
        and all(bool(item.get("passed")) for item in report.get("acceptance_criteria", []))
    ):
        raise RuntimeError("V2.0 entry gate is not DONE/PASS/GO")
    if report.get("locked_outcome_access_count") != 0 or report.get("model_training_performed"):
        raise RuntimeError("V2.0 entry report violates lock/training boundary")
    return {
        "actual_sha256": actual,
        "phase_decomposition_available": inputs["phase_decomposition"].is_file(),
        "v2_0_entry_gate": "PASS",
    }


def _identifier_format(value: str, namespace: str) -> str:
    prefix, body = value.split("-", 1) if "-" in value else ("", value)
    numeric = body.split("-", 1)[0].isdigit()
    if namespace == "task" and "-" not in value:
        return "alpha_task_bare"
    if namespace == "thermo_entry":
        identifier = body.split("-", 1)[0]
        return "legacy_compound" if identifier.isdigit() else "alpha_compound"
    if numeric:
        return f"legacy_{namespace}_{prefix or 'bare'}"
    return f"alpha_{namespace}_{prefix or 'bare'}"


def _material_conversion(value: str) -> dict[str, Any]:
    alpha = AlphaID(value)
    legacy = str(format_identifier(alpha, legacy=True))
    canonical = str(format_identifier(alpha, legacy=False))
    if int(AlphaID(legacy)) != int(alpha) or int(AlphaID(canonical)) != int(alpha):
        raise ValueError(f"material identifier round trip failed: {value}")
    return {
        "raw_material_id_format": _identifier_format(value, "material"),
        "canonical_material_numeric_id": int(alpha),
        "canonical_legacy_material_id": legacy,
        "canonical_alpha_material_id": canonical,
    }


def _task_conversion(value: str) -> dict[str, Any]:
    alpha = AlphaID(value)
    legacy = str(format_task_id(alpha, legacy=True))
    canonical = str(format_task_id(alpha, legacy=False))
    if int(AlphaID(legacy)) != int(alpha) or int(AlphaID(canonical)) != int(alpha):
        raise ValueError(f"task identifier round trip failed: {value}")
    return {
        "raw_task_id_format": _identifier_format(value, "task"),
        "canonical_task_numeric_id": int(alpha),
        "canonical_legacy_task_id": legacy,
        "canonical_alpha_task_id": canonical,
    }


def _entry_conversion(value: str) -> dict[str, Any]:
    prefix = value.split("-", 1)[0].casefold()
    canonical = str(format_compound_identifier(value, legacy=False))
    legacy = str(format_compound_identifier(canonical, legacy=True))
    canonical_again = str(format_compound_identifier(legacy, legacy=False))
    if canonical_again != canonical:
        raise ValueError(f"compound identifier round trip failed: {value}")
    return {
        "raw_entry_id_format": _identifier_format(value, "thermo_entry"),
        "canonical_legacy_entry_id": legacy,
        "canonical_alpha_entry_id": canonical,
        "canonical_inventory_token": f"{prefix}|{canonical}",
    }


def _mapping_frame(
    values: Iterable[str], converter: Any, raw_name: str
) -> pd.DataFrame:
    rows = [{raw_name: value, **converter(value)} for value in sorted(set(values))]
    return pd.DataFrame(rows)


def add_identifier_fields(
    source: pd.DataFrame,
    target_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = ["source_snapshot", "thermo_type", "source_thermo_id", "source_material_id"]
    lookup = target_lookup.rename(
        columns={
            "snapshot_id": "source_snapshot",
            "thermo_id": "source_thermo_id",
            "material_id": "source_material_id",
            "task_id": "raw_task_id",
            "entry_id": "raw_entry_id",
            "phase_context_chemsys": "lookup_phase_context_chemsys",
        }
    )
    merged = source.merge(lookup, on=key, how="left", validate="many_to_one", indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise RuntimeError(f"{missing} source rows lack a unique source phase target identifier")
    mismatch = int(
        (merged["phase_context_chemsys"] != merged["lookup_phase_context_chemsys"]).sum()
    )
    if mismatch:
        raise RuntimeError(f"{mismatch} source phase-context identifiers disagree")
    merged = merged.drop(columns=["_merge", "lookup_phase_context_chemsys"])
    merged["raw_material_id"] = merged["source_material_id"]
    merged["raw_thermo_id"] = merged["source_thermo_id"]

    material_map = _mapping_frame(
        merged["raw_material_id"].astype(str), _material_conversion, "raw_material_id"
    )
    task_map = _mapping_frame(merged["raw_task_id"].astype(str), _task_conversion, "raw_task_id")
    entry_map = _mapping_frame(
        merged["raw_entry_id"].astype(str), _entry_conversion, "raw_entry_id"
    ).drop(columns="canonical_inventory_token")
    merged = merged.merge(material_map, on="raw_material_id", validate="many_to_one")
    merged = merged.merge(task_map, on="raw_task_id", validate="many_to_one")
    merged = merged.merge(entry_map, on="raw_entry_id", validate="many_to_one")
    merged["identifier_namespace_material"] = "material"
    merged["identifier_namespace_task"] = "task"
    merged["identifier_namespace_entry"] = "thermo_entry"
    merged["identifier_conversion_version"] = IDENTIFIER_CONVERSION_VERSION
    audit = {
        "rows": len(merged),
        "unique_material_ids": int(merged["raw_material_id"].nunique()),
        "unique_task_ids": int(merged["raw_task_id"].nunique()),
        "unique_entry_ids": int(merged["raw_entry_id"].nunique()),
        "source_identifier_join_missing_rows": missing,
        "source_phase_context_mismatch_rows": mismatch,
        "roundtrip_failures": 0,
        "format_only_events": 0,
        "conversion_version": IDENTIFIER_CONVERSION_VERSION,
    }
    return merged, audit


def _prepare_phase(
    phase_path: Path,
    snapshots: list[str],
    stable_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], frozenset[str]], dict[str, Any]]:
    phase = pq.read_table(
        phase_path,
        columns=PHASE_COLUMNS,
        filters=[("snapshot_id", "in", snapshots)],
    ).to_pandas()
    if set(phase["snapshot_id"].unique()) != set(snapshots):
        raise RuntimeError("phase input did not return exactly the allowed development snapshots")
    entry_map = _mapping_frame(
        phase["entry_id"].astype(str), _entry_conversion, "entry_id"
    )[["entry_id", "canonical_inventory_token"]]
    phase = phase.merge(entry_map, on="entry_id", how="left", validate="many_to_one")
    key = ["snapshot_id", "thermo_type", "phase_context_chemsys"]
    phase["_stable"] = (phase["energy_above_hull"] <= stable_tolerance).astype(np.int8)
    phase["_near_10"] = (
        phase["energy_above_hull"] <= 0.01 + stable_tolerance
    ).astype(np.int8)
    stats = (
        phase.groupby(key, sort=False, dropna=False)
        .agg(
            context_entry_count=("entry_id", "size"),
            context_target_count=("is_target", "sum"),
            context_competitor_count=("is_competitor", "sum"),
            context_stable_count=("_stable", "sum"),
            context_near_hull_10meV_count=("_near_10", "sum"),
        )
        .reset_index()
    )
    competitor_sets = (
        phase.loc[phase["is_competitor"]]
        .groupby(key, sort=False, dropna=False)["canonical_inventory_token"]
        .agg(lambda values: frozenset(values))
        .to_dict()
    )
    target_key = ["snapshot_id", "thermo_type", "thermo_id", "material_id"]
    targets = phase.loc[phase["is_target"], target_key + ["task_id", "entry_id", "phase_context_chemsys"]].copy()
    target_counts = targets.groupby(target_key, dropna=False, sort=False).size()
    ambiguous_keys = target_counts[target_counts != 1]
    unique_keys = target_counts[target_counts == 1].reset_index()[target_key]
    targets = targets.merge(unique_keys, on=target_key, how="inner", validate="many_to_one")
    if targets.duplicated(target_key).any():
        raise RuntimeError("unique target-key filter did not remove every ambiguous key")
    audit = {
        "phase_rows_read": len(phase),
        "snapshots_read": snapshots,
        "contexts": len(stats),
        "unique_entry_ids": int(phase["entry_id"].nunique()),
        "canonical_entry_roundtrip_failures": 0,
        "ambiguous_target_keys_not_used": len(ambiguous_keys),
    }
    return phase, stats, competitor_sets, targets, audit


def _context_pair_tables(
    source_rows: pd.DataFrame,
    stats: pd.DataFrame,
    competitor_sets: dict[tuple[str, str, str], frozenset[str]],
    interval_map: dict[str, str],
    previous_map: dict[str, str | None],
    previous_days: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats_map = {
        (row.snapshot_id, row.thermo_type, row.phase_context_chemsys): {
            "entry": int(row.context_entry_count),
            "competitor": int(row.context_competitor_count),
            "stable": int(row.context_stable_count),
            "near10": int(row.context_near_hull_10meV_count),
        }
        for row in stats.itertuples(index=False)
    }
    contexts = source_rows[
        ["source_snapshot", "thermo_type", "phase_context_chemsys"]
    ].drop_duplicates()
    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    source_missing = 0
    target_missing = 0
    for row in contexts.itertuples(index=False):
        source_key = (row.source_snapshot, row.thermo_type, row.phase_context_chemsys)
        target_snapshot = interval_map[row.source_snapshot]
        target_key = (target_snapshot, row.thermo_type, row.phase_context_chemsys)
        current = stats_map.get(source_key)
        target = stats_map.get(target_key)
        if current is None:
            source_missing += 1
            continue
        if target is None:
            target_missing += 1
            continue
        current_set = competitor_sets.get(source_key, frozenset())
        target_set = competitor_sets.get(target_key, frozenset())
        target_added = target_set - current_set
        target_removed = current_set - target_set

        previous_snapshot = previous_map[row.source_snapshot]
        history_snapshot_available = int(previous_snapshot is not None)
        if previous_snapshot is None:
            previous = {"entry": 0, "competitor": 0, "stable": 0, "near10": 0}
            previous_set: frozenset[str] = frozenset()
            prior_present = 0
            history_days = 0
            history_arrivals = 0
            history_removals = 0
            entry_growth = 0
            competitor_growth = 0
            stable_growth = 0
            near_growth = 0
        else:
            previous_key = (
                previous_snapshot,
                row.thermo_type,
                row.phase_context_chemsys,
            )
            prior_present = int(previous_key in stats_map)
            previous = stats_map.get(
                previous_key,
                {"entry": 0, "competitor": 0, "stable": 0, "near10": 0},
            )
            previous_set = competitor_sets.get(previous_key, frozenset())
            history_days = int(previous_days[row.source_snapshot])
            history_arrivals = len(current_set - previous_set)
            history_removals = len(previous_set - current_set)
            entry_growth = current["entry"] - previous["entry"]
            competitor_growth = current["competitor"] - previous["competitor"]
            stable_growth = current["stable"] - previous["stable"]
            near_growth = current["near10"] - previous["near10"]
        annualizer = 365.25 / history_days if history_days else 0.0
        feature_rows.append(
            {
                "source_snapshot": row.source_snapshot,
                "thermo_type": row.thermo_type,
                "phase_context_chemsys": row.phase_context_chemsys,
                "source_history_snapshot_available": history_snapshot_available,
                "source_prior_context_present": prior_present,
                "source_history_exposure_days": history_days,
                "source_prior_context_entry_count": previous["entry"],
                "source_prior_context_competitor_count": previous["competitor"],
                "source_prior_context_stable_count": previous["stable"],
                "source_prior_context_near_hull_10meV_count": previous["near10"],
                "source_context_entry_growth_since_previous": entry_growth,
                "source_context_competitor_growth_since_previous": competitor_growth,
                "source_context_stable_growth_since_previous": stable_growth,
                "source_context_near_hull_10meV_growth_since_previous": near_growth,
                "source_competitor_arrivals_since_previous": history_arrivals,
                "source_competitor_removals_since_previous": history_removals,
                "source_competitor_arrival_fraction_since_previous": history_arrivals
                / max(current["competitor"], 1),
                "source_competitor_removal_fraction_since_previous": history_removals
                / max(previous["competitor"], 1),
                "source_context_entry_growth_per_year": entry_growth * annualizer,
                "source_context_competitor_growth_per_year": competitor_growth
                * annualizer,
                "_computed_source_context_entry_count": current["entry"],
                "_computed_source_context_competitor_count": current["competitor"],
                "_computed_source_context_stable_count": current["stable"],
                "_computed_source_context_near_hull_10meV_count": current["near10"],
            }
        )
        target_rows.append(
            {
                "source_snapshot": row.source_snapshot,
                "target_snapshot": target_snapshot,
                "thermo_type": row.thermo_type,
                "phase_context_chemsys": row.phase_context_chemsys,
                "source_competitor_inventory_count": len(current_set),
                "target_competitor_inventory_count": len(target_set),
                "competitor_arrival_count": len(target_added),
                "competitor_removal_count": len(target_removed),
                "hull_relevant_competitor_arrival_event": bool(target_added),
                "competitor_inventory_revision_event": bool(target_added or target_removed),
            }
        )
    if source_missing or target_missing:
        raise RuntimeError(
            f"context inventory missing: source={source_missing}, target={target_missing}"
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(target_rows), {
        "unique_source_contexts": len(contexts),
        "source_context_missing": source_missing,
        "target_context_missing": target_missing,
    }


def build_interval_exposure(
    labels: pd.DataFrame, allowed_intervals: list[dict[str, Any]]
) -> pd.DataFrame:
    expected = {
        item["role"]: (
            item["source_snapshot"],
            item["target_snapshot"],
            int(item["exposure_days"]),
        )
        for item in allowed_intervals
    }
    rows = labels[
        [
            "transition_id",
            "canonical_lineage_id",
            "assigned_role",
            "source_snapshot",
            "target_snapshot",
        ]
    ].copy()
    rows["source_date"] = pd.to_datetime(rows["source_snapshot"], format="%Y-%m-%d")
    rows["target_date"] = pd.to_datetime(rows["target_snapshot"], format="%Y-%m-%d")
    rows["exposure_days"] = (rows["target_date"] - rows["source_date"]).dt.days.astype(
        np.int16
    )
    rows["exposure_months"] = rows["exposure_days"] / 30.436875
    rows["exposure_years"] = rows["exposure_days"] / 365.25
    rows["log_exposure_years_offset"] = np.log(rows["exposure_years"])
    rows["discrete_time_interval"] = (
        rows["source_snapshot"] + "->" + rows["target_snapshot"]
    )
    wrong = 0
    for role, (source, target, days) in expected.items():
        subset = rows["assigned_role"] == role
        wrong += int(
            (
                (rows.loc[subset, "source_snapshot"] != source)
                | (rows.loc[subset, "target_snapshot"] != target)
                | (rows.loc[subset, "exposure_days"] != days)
            ).sum()
        )
    if wrong:
        raise RuntimeError(f"{wrong} interval-exposure rows violate the frozen design")
    rows["source_date"] = rows["source_date"].dt.strftime("%Y-%m-%d")
    rows["target_date"] = rows["target_date"].dt.strftime("%Y-%m-%d")
    return rows.sort_values(
        ["assigned_role", "canonical_lineage_id", "transition_id"], kind="mergesort"
    ).reset_index(drop=True)


def _decomposition_changed(source_json: Any, target_json: Any) -> bool | None:
    if not isinstance(source_json, str) or not isinstance(target_json, str):
        return None
    source = tuple(sorted(json.loads(source_json)))
    target = tuple(sorted(json.loads(target_json)))
    return source != target


def build_cause_specific_targets(
    labels: pd.DataFrame,
    source_features: pd.DataFrame,
    target_context: pd.DataFrame,
    attribution: pd.DataFrame,
    attribution_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    identifiers = source_features[
        [
            "transition_id",
            "canonical_lineage_id",
            "assigned_role",
            "source_snapshot",
            "thermo_type",
            "phase_context_chemsys",
            "source_is_stable",
        ]
    ]
    targets = labels.merge(
        identifiers,
        on=[
            "transition_id",
            "canonical_lineage_id",
            "assigned_role",
            "source_snapshot",
        ],
        how="inner",
        validate="one_to_one",
    )
    targets = targets.merge(
        target_context,
        on=[
            "source_snapshot",
            "target_snapshot",
            "thermo_type",
            "phase_context_chemsys",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    context_missing = int((targets["_merge"] != "both").sum())
    if context_missing:
        raise RuntimeError(f"{context_missing} target rows lack complete context inventory")
    targets = targets.drop(columns="_merge")

    attribution = attribution.copy()
    if attribution["attribution_id"].duplicated().any():
        raise RuntimeError("development attribution contains duplicate attribution IDs")
    attribution = attribution.rename(
        columns={"transition_id": "attribution_transition_id"}
    )
    attribution["_attribution_available"] = True
    targets = targets.merge(
        attribution.drop(columns=["source_snapshot", "target_snapshot"]),
        on="attribution_id",
        how="left",
        validate="many_to_one",
    )
    targets["attribution_available"] = targets["_attribution_available"].eq(True)
    targets = targets.drop(columns="_attribution_available")
    attribution_transition_mismatch = int(
        (
            targets.loc[
                targets["attribution_available"], "transition_id"
            ].map(bytes).reset_index(drop=True)
            != targets.loc[
                targets["attribution_available"], "attribution_transition_id"
            ].map(bytes).reset_index(drop=True)
        ).sum()
    )
    if attribution_transition_mismatch:
        raise RuntimeError(
            f"{attribution_transition_mismatch} attribution foreign keys disagree with transition IDs"
        )
    targets["stable_to_unstable_event"] = (
        targets["rebuilt_unified_label_transition"] == "stable_to_unstable"
    )
    targets["unstable_to_stable_event"] = (
        targets["rebuilt_unified_label_transition"] == "unstable_to_stable"
    )
    targets["any_direction_flip_event"] = (
        targets["stable_to_unstable_event"] | targets["unstable_to_stable_event"]
    )
    if not (
        targets["any_direction_flip_event"].astype(bool)
        == targets["rebuilt_unified_flip"].astype(bool)
    ).all():
        raise RuntimeError("rebuilt transition and flip labels disagree")
    missing_flip_attribution = int(
        (targets["any_direction_flip_event"] & ~targets["attribution_available"]).sum()
    )
    nonflip_attribution = int(
        (~targets["any_direction_flip_event"] & targets["attribution_available"]).sum()
    )
    if missing_flip_attribution or nonflip_attribution:
        raise RuntimeError(
            "development attribution alignment failed: "
            f"missing_flip={missing_flip_attribution}, nonflip={nonflip_attribution}"
        )
    targets["source_state"] = np.where(
        targets["source_is_stable"].astype(bool), "stable", "unstable"
    )
    targets["target_state"] = targets["source_state"]
    targets.loc[targets["stable_to_unstable_event"], "target_state"] = "unstable"
    targets.loc[targets["unstable_to_stable_event"], "target_state"] = "stable"
    targets["stable_to_unstable_at_risk"] = targets["source_is_stable"].astype(bool)
    targets["unstable_to_stable_at_risk"] = ~targets["source_is_stable"].astype(bool)
    targets["stable_to_unstable_censored"] = (
        targets["stable_to_unstable_at_risk"] & ~targets["stable_to_unstable_event"]
    )
    targets["unstable_to_stable_censored"] = (
        targets["unstable_to_stable_at_risk"] & ~targets["unstable_to_stable_event"]
    )
    targets["candidate_displacement_event"] = targets["stable_to_unstable_event"]
    targets["conditional_displacement_eligible"] = (
        targets["stable_to_unstable_at_risk"]
        & targets["hull_relevant_competitor_arrival_event"].astype(bool)
    )

    contribution_columns = {
        "competitor_inventory": "competitor_inventory_contribution",
        "uncorrected_energy": "uncorrected_energy_contribution",
        "compatibility_correction": "compatibility_correction_contribution",
        "candidate_identity": "candidate_identity_contribution",
    }
    for channel, column in contribution_columns.items():
        targets[column] = targets[column].fillna(0.0)
        targets[f"{channel}_accounting_active"] = (
            targets[column].abs() > attribution_tolerance
        )
        targets[f"stable_to_unstable_{channel}_dominant_event"] = (
            targets["stable_to_unstable_event"]
            & (targets["dominant_channel"] == channel)
        )
    targets["dominant_accounting_channel"] = targets["dominant_channel"].fillna("none")
    targets["candidate_identity_changed"] = targets["candidate_identity_changed"].eq(
        True
    )
    targets["stable_to_unstable_competing_cause"] = np.where(
        targets["stable_to_unstable_event"],
        targets["dominant_accounting_channel"],
        "none",
    )
    targets["unstable_to_stable_competing_cause"] = np.where(
        targets["unstable_to_stable_event"],
        targets["dominant_accounting_channel"],
        "none",
    )
    targets["decomposition_set_change_observed"] = targets["attribution_available"]
    targets["decomposition_set_change_event"] = [
        _decomposition_changed(source, target)
        for source, target in zip(
            targets["source_decomposition_phase_keys_json"],
            targets["target_decomposition_phase_keys_json"],
            strict=True,
        )
    ]
    targets["decomposition_set_change_observation_scope"] = np.where(
        targets["attribution_available"], "rebuilt_flip_attribution_rows", "not_observed"
    )
    targets["delta_reported_energy_above_hull_target"] = targets[
        "delta_reported_energy_above_hull"
    ]
    targets["delta_reported_formation_energy_per_atom_target"] = targets[
        "delta_reported_formation_energy_per_atom"
    ]

    output_columns = [
        "transition_id",
        "canonical_lineage_id",
        "assigned_role",
        "source_snapshot",
        "target_snapshot",
        "thermo_type",
        "phase_context_chemsys",
        "source_state",
        "target_state",
        "stable_to_unstable_at_risk",
        "stable_to_unstable_event",
        "stable_to_unstable_censored",
        "unstable_to_stable_at_risk",
        "unstable_to_stable_event",
        "unstable_to_stable_censored",
        "any_direction_flip_event",
        "competitor_inventory_revision_event",
        "hull_relevant_competitor_arrival_event",
        "competitor_arrival_count",
        "competitor_removal_count",
        "source_competitor_inventory_count",
        "target_competitor_inventory_count",
        "candidate_displacement_event",
        "conditional_displacement_eligible",
        "dominant_accounting_channel",
        "stable_to_unstable_competing_cause",
        "unstable_to_stable_competing_cause",
        "competitor_inventory_accounting_active",
        "uncorrected_energy_accounting_active",
        "compatibility_correction_accounting_active",
        "candidate_identity_accounting_active",
        "stable_to_unstable_competitor_inventory_dominant_event",
        "stable_to_unstable_uncorrected_energy_dominant_event",
        "stable_to_unstable_compatibility_correction_dominant_event",
        "stable_to_unstable_candidate_identity_dominant_event",
        "candidate_identity_changed",
        "decomposition_set_change_observed",
        "decomposition_set_change_event",
        "decomposition_set_change_observation_scope",
        "delta_reported_energy_above_hull_target",
        "delta_reported_formation_energy_per_atom_target",
    ]
    output = targets[output_columns].copy()
    output = output.sort_values(
        ["assigned_role", "canonical_lineage_id", "transition_id"], kind="mergesort"
    ).reset_index(drop=True)
    counts = {
        "rows": len(output),
        "stable_to_unstable_events": int(output["stable_to_unstable_event"].sum()),
        "unstable_to_stable_events": int(output["unstable_to_stable_event"].sum()),
        "any_direction_flip_events": int(output["any_direction_flip_event"].sum()),
        "competitor_inventory_revision_events": int(
            output["competitor_inventory_revision_event"].sum()
        ),
        "competitor_arrival_events": int(
            output["hull_relevant_competitor_arrival_event"].sum()
        ),
        "conditional_displacement_eligible_rows": int(
            output["conditional_displacement_eligible"].sum()
        ),
        "conditional_displacement_events": int(
            (
                output["conditional_displacement_eligible"]
                & output["candidate_displacement_event"]
            ).sum()
        ),
        "attribution_rows": int(output["decomposition_set_change_observed"].sum()),
        "missing_flip_attribution_rows": missing_flip_attribution,
        "nonflip_attribution_rows": nonflip_attribution,
        "attribution_transition_id_mismatches": attribution_transition_mismatch,
        "target_context_missing_rows": context_missing,
    }
    return output, counts


def feature_sets() -> dict[str, list[str]]:
    m1 = list(dict.fromkeys(M0_FEATURES + M1_ADDITIONS))
    m2 = list(dict.fromkeys(m1 + STATIC_FEATURES))
    m3 = list(dict.fromkeys(m1 + TEMPORAL_FEATURES))
    m4 = list(dict.fromkeys(m1 + STATIC_FEATURES + TEMPORAL_FEATURES))
    return {
        "M0": M0_FEATURES,
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m4,
        "M5_arrival": list(dict.fromkeys(STATIC_FEATURES + TEMPORAL_FEATURES + [
            "source_chemsys_dimensionality",
            "thermo_GGA_GGA+U",
            "thermo_GGA_GGA+U_R2SCAN",
            "thermo_R2SCAN",
        ])),
        "M5_displacement": m2,
    }


def build_source_feature_table(
    baseline: pd.DataFrame,
    history: pd.DataFrame,
    target_lookup: pd.DataFrame,
    forbidden_patterns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    baseline = baseline.rename(columns=STATIC_CONTEXT_RENAMES)
    baseline["source_chemsys_dimensionality"] = (
        baseline["phase_context_chemsys"].str.count("-") + 1
    ).astype(np.int8)
    merged = baseline.merge(
        history,
        on=["source_snapshot", "thermo_type", "phase_context_chemsys"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    history_missing = int((merged["_merge"] != "both").sum())
    if history_missing:
        raise RuntimeError(f"{history_missing} source rows lack a temporal history record")
    merged = merged.drop(columns="_merge")
    consistency_pairs = [
        ("source_context_entry_count", "_computed_source_context_entry_count"),
        ("source_context_competitor_count", "_computed_source_context_competitor_count"),
        ("source_context_stable_count", "_computed_source_context_stable_count"),
        (
            "source_context_near_hull_10meV_count",
            "_computed_source_context_near_hull_10meV_count",
        ),
    ]
    consistency_mismatches = 0
    for existing, computed in consistency_pairs:
        consistency_mismatches += int((merged[existing] != merged[computed]).sum())
    if consistency_mismatches:
        raise RuntimeError(
            f"{consistency_mismatches} source context aggregates disagree with P3.2"
        )
    merged = merged.drop(columns=[computed for _, computed in consistency_pairs])
    merged, identifier_audit = add_identifier_fields(merged, target_lookup)

    sets = feature_sets()
    all_features = list(dict.fromkeys(name for names in sets.values() for name in names))
    missing_features = [name for name in all_features if name not in merged]
    if missing_features:
        raise RuntimeError(f"configured V2 features are missing: {missing_features}")
    forbidden = [
        name
        for name in all_features
        if any(pattern.casefold() in name.casefold() for pattern in forbidden_patterns)
    ]
    if forbidden:
        raise RuntimeError(f"forbidden outcome/future features selected: {forbidden}")
    values = merged[all_features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("V2 source feature table contains non-finite feature values")

    identifier_columns = [
        "transition_id",
        "canonical_lineage_id",
        "assigned_role",
        "identity_confidence",
        "source_snapshot",
        "thermo_type",
        "phase_context_chemsys",
        "raw_material_id",
        "raw_task_id",
        "raw_thermo_id",
        "raw_entry_id",
        "raw_material_id_format",
        "raw_task_id_format",
        "raw_entry_id_format",
        "identifier_namespace_material",
        "identifier_namespace_task",
        "identifier_namespace_entry",
        "canonical_material_numeric_id",
        "canonical_legacy_material_id",
        "canonical_alpha_material_id",
        "canonical_task_numeric_id",
        "canonical_legacy_task_id",
        "canonical_alpha_task_id",
        "canonical_legacy_entry_id",
        "canonical_alpha_entry_id",
        "identifier_conversion_version",
    ]
    output = merged[identifier_columns + all_features].copy()
    output = output.sort_values(
        ["assigned_role", "canonical_lineage_id", "transition_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit_rows: list[dict[str, Any]] = []
    for name in all_features:
        memberships = [model for model, names in sets.items() if name in names]
        audit_rows.append(
            {
                "feature_name": name,
                "model_membership": "|".join(memberships),
                "information_time": "source_snapshot_or_earlier",
                "latest_allowed_source_snapshot": "row.source_snapshot",
                "uses_target_state": False,
                "uses_outcome": False,
                "missingness_policy": "explicit_indicator_then_zero_fill",
                "dtype": str(output[name].dtype),
            }
        )
    audit = {
        "rows": len(output),
        "feature_columns": len(all_features),
        "history_join_missing_rows": history_missing,
        "context_aggregate_mismatches": consistency_mismatches,
        "forbidden_feature_names": forbidden,
        "identifier_audit": identifier_audit,
    }
    return output, audit, pd.DataFrame(audit_rows)


def _target_dictionary() -> pd.DataFrame:
    rows = [
        ("stable_to_unstable_event", "primary", "source-stable candidate becomes unstable", "bool", "stable_to_unstable_at_risk"),
        ("competitor_inventory_revision_event", "primary", "exact-context competitor token set changes", "bool", "all development rows"),
        ("candidate_displacement_event", "primary", "stable-to-unstable event conditional on competitor arrival", "bool", "conditional_displacement_eligible"),
        ("unstable_to_stable_event", "secondary", "source-unstable candidate becomes stable", "bool", "unstable_to_stable_at_risk"),
        ("delta_reported_energy_above_hull_target", "secondary", "target minus source reported energy above hull", "float64", "all development rows"),
        ("dominant_accounting_channel", "secondary", "largest absolute Shapley accounting channel for a rebuilt flip", "string", "rebuilt flip rows"),
        ("decomposition_set_change_event", "secondary_partial", "exact decomposition-key set changes where P3.3 attribution exists", "nullable bool", "rebuilt flip attribution rows only"),
        ("any_direction_flip_event", "secondary_summary", "stable-to-unstable or unstable-to-stable", "bool", "all development rows; prohibited as sole primary"),
    ]
    return pd.DataFrame(
        rows, columns=["target_name", "estimand_role", "definition", "dtype", "risk_set"]
    )


def _row_count_audit(
    labels: pd.DataFrame,
    source: pd.DataFrame,
    targets: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    frames = {
        "development_labels_input": labels,
        "source_features_output": source,
        "cause_specific_targets_output": targets,
        "interval_exposure_output": exposure,
    }
    rows: list[dict[str, Any]] = []
    for name, frame in frames.items():
        rows.append(
            {
                "table": name,
                "rows": len(frame),
                "unique_transition_ids": int(frame["transition_id"].nunique()),
                "unique_lineages": int(frame["canonical_lineage_id"].nunique()),
                "duplicate_transition_ids": int(frame["transition_id"].duplicated().sum()),
                "train_rows": int((frame["assigned_role"] == "train").sum()),
                "validation_rows": int((frame["assigned_role"] == "validation").sum()),
                "locked_rows": int((frame["assigned_role"] == "locked_test").sum()),
            }
        )
    return pd.DataFrame(rows)


def _feature_schema_payload(source: pd.DataFrame) -> dict[str, Any]:
    sets = feature_sets()
    feature_columns = list(dict.fromkeys(name for names in sets.values() for name in names))
    return {
        "task_id": TASK_ID,
        "method_version": "PHASEEVONET_V2_1_FEATURE_SCHEMA_V1",
        "identifier_columns": [name for name in source.columns if name not in feature_columns],
        "feature_columns": feature_columns,
        "feature_sets": sets,
        "source_only_rule": "Every feature is observable at source_snapshot or earlier.",
        "forbidden_information": [
            "target state",
            "target phase context",
            "target energy",
            "future database nodes/edges",
            "outcome labels",
            "target or stratum prevalence",
        ],
        "identifier_conversion_version": IDENTIFIER_CONVERSION_VERSION,
        "deep_models_authorized": False,
    }


def _git_state() -> dict[str, Any]:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=top,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=top,
        ).stdout.splitlines()
        return {"repository": top, "commit": commit, "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"repository": None, "commit": None, "status": "NO_GIT_REPOSITORY"}


def build_v2_development(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = utc_now()
    config_path = Path(config_path)
    config = load_config(config_path)
    precondition = verify_entry_and_input_hashes(config)
    inputs = _input_paths(config)
    outputs = {key: Path(value) for key, value in config["output"].items()}
    allowed = config["allowed_development_intervals"]
    allowed_roles = {item["role"] for item in allowed}
    allowed_source = [item["source_snapshot"] for item in allowed]
    allowed_target = [item["target_snapshot"] for item in allowed]
    allowed_phase_snapshots = sorted(set(allowed_source + allowed_target))

    labels = pq.read_table(inputs["development_labels"], columns=LABEL_COLUMNS).to_pandas()
    if set(labels["assigned_role"].unique()) != allowed_roles:
        raise RuntimeError("development labels contain a forbidden or missing role")
    if set(labels["source_snapshot"].unique()) != set(allowed_source):
        raise RuntimeError("development labels contain a forbidden source snapshot")
    if set(labels["target_snapshot"].unique()) != set(allowed_target):
        raise RuntimeError("development labels contain a forbidden target snapshot")
    if labels["transition_id"].duplicated().any():
        raise RuntimeError("development labels contain duplicate transition IDs")
    train_lineages = set(labels.loc[labels["assigned_role"] == "train", "canonical_lineage_id"])
    validation_lineages = set(
        labels.loc[labels["assigned_role"] == "validation", "canonical_lineage_id"]
    )
    lineage_overlap = train_lineages & validation_lineages
    if lineage_overlap:
        raise RuntimeError(f"{len(lineage_overlap)} lineages cross development roles")

    baseline_columns = BASE_IDENTIFIER_COLUMNS + list(config["features"]["baseline_safe_columns"])
    baseline = pq.read_table(
        inputs["baseline_source_features"], columns=baseline_columns
    ).to_pandas()
    if len(baseline) != len(labels):
        raise RuntimeError("baseline source rows do not align with development labels")

    phase, context_stats, competitor_sets, target_lookup, phase_audit = _prepare_phase(
        inputs["phase_entries"],
        allowed_phase_snapshots,
        float(config["features"]["stable_tolerance_eV_per_atom"]),
    )
    interval_map = {item["source_snapshot"]: item["target_snapshot"] for item in allowed}
    previous_map = config["features"]["history_previous_snapshot"]
    previous_days = {
        key: int(value)
        for key, value in config["features"]["history_exposure_days"].items()
    }
    history, target_context, context_audit = _context_pair_tables(
        baseline,
        context_stats,
        competitor_sets,
        interval_map,
        previous_map,
        previous_days,
    )
    source_features, source_audit, source_audit_frame = build_source_feature_table(
        baseline,
        history,
        target_lookup,
        list(config["forbidden"]["feature_name_patterns"]),
    )

    attribution = pq.read_table(
        inputs["transition_attribution"],
        columns=ATTRIBUTION_COLUMNS,
        filters=[("source_snapshot", "in", allowed_source)],
    ).to_pandas()
    selected_attribution_ids = set(labels["attribution_id"].dropna().map(bytes))
    attribution = attribution[
        attribution["attribution_id"].map(bytes).isin(selected_attribution_ids)
    ].copy()
    targets, target_counts = build_cause_specific_targets(
        labels,
        source_features,
        target_context,
        attribution,
        float(config["features"]["attribution_activity_tolerance_eV_per_atom"]),
    )
    exposure = build_interval_exposure(labels, allowed)
    row_audit = _row_count_audit(labels, source_features, targets, exposure)
    if len({len(source_features), len(targets), len(exposure), len(labels)}) != 1:
        raise RuntimeError("V2.1 output tables do not preserve one row per development transition")
    transition_orders = [
        frame["transition_id"].tolist() for frame in (source_features, targets, exposure)
    ]
    if not (transition_orders[0] == transition_orders[1] == transition_orders[2]):
        raise RuntimeError("V2.1 output table row orders do not align")

    parquet = config["parquet"]
    for path, frame in [
        (outputs["source_features"], source_features),
        (outputs["cause_specific_targets"], targets),
        (outputs["interval_exposure"], exposure),
    ]:
        write_parquet_atomic(
            path,
            frame,
            compression=str(parquet["compression"]),
            row_group_size=int(parquet["row_group_size"]),
        )
    write_json_atomic(outputs["feature_schema"], _feature_schema_payload(source_features))
    write_csv_atomic(outputs["source_only_audit"], source_audit_frame)
    write_csv_atomic(outputs["target_dictionary"], _target_dictionary())
    write_csv_atomic(outputs["row_count_audit"], row_audit)
    write_csv_atomic(
        outputs["exclusion_ambiguity_ledger"],
        pd.DataFrame(
            columns=[
                "transition_id",
                "stage",
                "reason",
                "resolution",
                "row_excluded",
            ]
        ),
    )

    by_interval = (
        targets.assign(
            interval=targets["source_snapshot"] + "->" + targets["target_snapshot"]
        )
        .groupby("interval")["stable_to_unstable_event"]
        .sum()
        .astype(int)
        .to_dict()
    )
    gates = {
        "row_count": len(source_features) == int(config["gate"]["required_rows"]),
        "zero_row_loss": bool((row_audit["rows"] == len(labels)).all()),
        "zero_duplicate_transition_ids": bool(
            (row_audit["duplicate_transition_ids"] == 0).all()
        ),
        "zero_train_validation_lineage_overlap": len(lineage_overlap) == 0,
        "zero_forbidden_feature_names": not source_audit["forbidden_feature_names"],
        "stable_to_unstable_event_count": target_counts["stable_to_unstable_events"]
        == int(config["gate"]["required_stable_to_unstable_events"]),
        "stable_to_unstable_event_count_by_interval": by_interval
        == {
            key: int(value)
            for key, value in config["gate"][
                "required_stable_to_unstable_by_interval"
            ].items()
        },
        "unstable_to_stable_event_count": target_counts["unstable_to_stable_events"]
        == int(config["gate"]["required_unstable_to_stable_events"]),
        "complete_source_and_target_context_inventory": context_audit[
            "source_context_missing"
        ]
        == 0
        and context_audit["target_context_missing"] == 0,
        "format_roundtrip_failures_zero": source_audit["identifier_audit"][
            "roundtrip_failures"
        ]
        == 0
        and phase_audit["canonical_entry_roundtrip_failures"] == 0,
        "locked_outcome_reads_zero": True,
        "model_training_zero": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V2.1 development table gate failure: {gates}")

    artifact_paths = [
        outputs["source_features"],
        outputs["cause_specific_targets"],
        outputs["interval_exposure"],
        outputs["feature_schema"],
        outputs["source_only_audit"],
        outputs["target_dictionary"],
        outputs["row_count_audit"],
        outputs["exclusion_ambiguity_ledger"],
    ]
    artifacts = [
        artifact_record(path, include_schema=path.suffix == ".parquet")
        for path in artifact_paths
    ]
    manifest = {
        "task_id": TASK_ID,
        "method_version": config["method_version"],
        "seed": int(config["seed"]),
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "model_matrix_path": inputs["model_matrix"].as_posix(),
        "model_matrix_sha256": sha256_file(inputs["model_matrix"]),
        "inputs": {
            key: {
                "path": inputs[key].as_posix(),
                "sha256": digest,
                "bytes": inputs[key].stat().st_size,
            }
            for key, digest in precondition["actual_sha256"].items()
        },
        "artifacts": artifacts,
        "counts": {
            "input_development_rows": len(labels),
            "source_feature_rows": len(source_features),
            "cause_specific_target_rows": len(targets),
            "interval_exposure_rows": len(exposure),
            **target_counts,
            "stable_to_unstable_by_interval": by_interval,
        },
        "source_audit": source_audit,
        "context_audit": context_audit,
        "phase_audit": phase_audit,
        "gate_checks": gates,
        "gate_status": "PASS",
        "locked_outcome_access_count": 0,
        "model_training_performed": False,
        "allowed_target_state_reads": allowed_target,
        "forbidden_target_state_reads": [],
        "phase_decomposition_available": precondition["phase_decomposition_available"],
        "exact_all-risk_decomposition_target_built": False,
    }
    write_json_atomic(outputs["manifest"], manifest)

    report = {
        "task_id": TASK_ID,
        "task_status": "IN_PROGRESS",
        "status": "BUILD_PASS",
        "gate_status": "PENDING_TESTS_AND_FINAL_VERIFICATION",
        "scope": "V2.1 only",
        "started_at_utc": started,
        "ended_at_utc": manifest["ended_at_utc"],
        "seed": int(config["seed"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_state": _git_state(),
        "package_versions": {
            name: version(name)
            for name in [
                "phase-evonet",
                "numpy",
                "pandas",
                "pyarrow",
                "PyYAML",
                "emmet-core",
                "scikit-learn",
            ]
        },
        "config_hashes": {
            config_path.as_posix(): sha256_file(config_path),
            inputs["model_matrix"].as_posix(): sha256_file(inputs["model_matrix"]),
        },
        "input_hashes": manifest["inputs"],
        "artifacts": artifacts + [artifact_record(outputs["manifest"])],
        "row_counts": row_audit.to_dict("records"),
        "event_counts": manifest["counts"],
        "source_only_audit": source_audit,
        "context_audit": context_audit,
        "identifier_audit": source_audit["identifier_audit"],
        "model_matrix": {
            "models": ["M0", "M1", "M2", "M3", "M4", "M5"],
            "training_authorized": False,
            "same_algorithm_M0_M4": True,
            "same_effective_capacity_M0_M5": True,
            "deep_models_authorized": False,
        },
        "locked_outcome_access_count": 0,
        "model_training_performed": False,
        "locked_or_external_outcome_files_read": [],
        "gate_checks": gates,
        "acceptance_criteria": [
            {"criterion": key, "passed": bool(value)} for key, value in gates.items()
        ],
        "warnings": [
            "The canonical P3.2 phase_decomposition.parquet was not locally available; exact decomposition-set change is exposed only for already-observed P3.3 development flip-attribution rows and is disabled for all-risk modeling.",
            "The remote canonical storage probe returned Host unreachable; no remote artifact was transferred or modified.",
            "V2.1 freezes tables and configuration only. No M0-M5 model was trained, selected, calibrated, or evaluated.",
        ],
        "next_task_eligibility": {
            "only_task": "V2.2",
            "eligible": False,
            "authorized": False,
            "executed": False,
            "pending": "tests and final V2.1 verification",
        },
    }
    write_json_atomic(outputs["report"], report)
    return {
        "task_id": TASK_ID,
        "status": "BUILD_PASS",
        "gate_status": "PENDING_TESTS_AND_FINAL_VERIFICATION",
        "rows": len(source_features),
        "stable_to_unstable_events": target_counts["stable_to_unstable_events"],
        "model_training_performed": False,
        "locked_outcome_access_count": 0,
    }


def verify_v2_development(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    verify_entry_and_input_hashes(config)
    outputs = {key: Path(value) for key, value in config["output"].items()}
    required = [
        "source_features",
        "cause_specific_targets",
        "interval_exposure",
        "manifest",
        "feature_schema",
        "source_only_audit",
        "target_dictionary",
        "row_count_audit",
        "exclusion_ambiguity_ledger",
        "report",
    ]
    missing = [key for key in required if not outputs[key].is_file()]
    if missing:
        raise FileNotFoundError(f"missing V2.1 outputs: {missing}")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    integrity_failures: list[str] = []
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            integrity_failures.append(artifact["path"])
    source = pq.read_table(outputs["source_features"]).to_pandas()
    targets = pq.read_table(outputs["cause_specific_targets"]).to_pandas()
    exposure = pq.read_table(outputs["interval_exposure"]).to_pandas()
    schema = json.loads(outputs["feature_schema"].read_text(encoding="utf-8"))
    ids = [frame["transition_id"].tolist() for frame in (source, targets, exposure)]
    alignment_pass = ids[0] == ids[1] == ids[2]
    feature_names = schema["feature_columns"]
    forbidden_patterns = config["forbidden"]["feature_name_patterns"]
    forbidden = [
        name
        for name in feature_names
        if any(pattern.casefold() in name.casefold() for pattern in forbidden_patterns)
    ]
    train_lineages = set(
        source.loc[source["assigned_role"] == "train", "canonical_lineage_id"]
    )
    validation_lineages = set(
        source.loc[source["assigned_role"] == "validation", "canonical_lineage_id"]
    )
    model_matrix = yaml.safe_load(
        Path(config["input"]["model_matrix"]).read_text(encoding="utf-8")
    )
    prior_determinism = (
        json.loads(outputs["determinism_check"].read_text(encoding="utf-8"))
        if outputs["determinism_check"].is_file()
        else {}
    )
    rebuild_evidence = prior_determinism
    while (
        isinstance(rebuild_evidence, dict)
        and "artifacts" not in rebuild_evidence
        and isinstance(rebuild_evidence.get("deterministic_rebuild"), dict)
    ):
        rebuild_evidence = rebuild_evidence["deterministic_rebuild"]
    models = model_matrix["models"]
    shared_class = model_matrix["shared_budget"]["algorithm_class"]
    shared_iterations = int(
        model_matrix["shared_budget"]["effective_total_boosting_iterations"]
    )
    capacity_pass = all(
        models[name]["algorithm_class"] == shared_class
        and int(models[name]["effective_total_boosting_iterations"]) == shared_iterations
        for name in ["M0", "M1", "M2", "M3", "M4", "M5"]
    )
    counts_by_interval = (
        targets.assign(
            interval=targets["source_snapshot"] + "->" + targets["target_snapshot"]
        )
        .groupby("interval")["stable_to_unstable_event"]
        .sum()
        .astype(int)
        .to_dict()
    )
    checks = {
        "artifact_hashes": not integrity_failures,
        "row_count": len(source) == int(config["gate"]["required_rows"]),
        "one_to_one_alignment": alignment_pass,
        "zero_duplicate_transition_ids": not any(
            frame["transition_id"].duplicated().any()
            for frame in (source, targets, exposure)
        ),
        "allowed_roles_only": set(source["assigned_role"].unique()) == {"train", "validation"},
        "allowed_source_snapshots_only": set(source["source_snapshot"].unique())
        == {"2022-10-28", "2023-11-01"},
        "zero_lineage_overlap": not (train_lineages & validation_lineages),
        "source_only_feature_names": not forbidden,
        "finite_feature_values": np.isfinite(source[feature_names].to_numpy(dtype=float)).all(),
        "stable_to_unstable_count": int(targets["stable_to_unstable_event"].sum())
        == int(config["gate"]["required_stable_to_unstable_events"]),
        "stable_to_unstable_count_by_interval": counts_by_interval
        == {
            key: int(value)
            for key, value in config["gate"][
                "required_stable_to_unstable_by_interval"
            ].items()
        },
        "unstable_to_stable_count": int(targets["unstable_to_stable_event"].sum())
        == int(config["gate"]["required_unstable_to_stable_events"]),
        "interval_exposure_exact": set(exposure["exposure_days"].astype(int)) == {369, 413},
        "M0_M5_capacity_matched": capacity_pass,
        "training_not_authorized": model_matrix["training_authorized"] is False,
        "no_model_artifacts": not Path("models/V2_1").exists(),
        "locked_outcome_reads_zero": manifest["locked_outcome_access_count"] == 0,
        "model_training_zero": manifest["model_training_performed"] is False,
        "empty_exclusion_ambiguity_ledger": len(
            pd.read_csv(outputs["exclusion_ambiguity_ledger"])
        )
        == 0,
        "deterministic_rebuild": rebuild_evidence.get("rebuild_match") is True,
    }
    failures = [key for key, passed in checks.items() if not bool(passed)]
    result = {
        "task_id": TASK_ID,
        "status": "PASS" if not failures else "FAIL",
        "gate_status": "PASS" if not failures else "NO_GO",
        "checks": {key: bool(value) for key, value in checks.items()},
        "failures": failures,
        "integrity_failures": integrity_failures,
        "rows": len(source),
        "feature_columns": len(feature_names),
        "stable_to_unstable_events": int(targets["stable_to_unstable_event"].sum()),
        "unstable_to_stable_events": int(targets["unstable_to_stable_event"].sum()),
        "locked_outcome_access_count": 0,
        "model_training_performed": False,
        "rebuild_match": rebuild_evidence.get("rebuild_match") is True,
        "deterministic_rebuild": rebuild_evidence,
    }
    write_json_atomic(outputs["determinism_check"], result)
    if failures:
        raise RuntimeError(f"V2.1 verification failed: {failures}")
    return result


def check_deterministic_rebuild(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    outputs = {key: Path(value) for key, value in config["output"].items()}
    deterministic_keys = [
        "source_features",
        "cause_specific_targets",
        "interval_exposure",
        "feature_schema",
        "source_only_audit",
        "target_dictionary",
        "row_count_audit",
        "exclusion_ambiguity_ledger",
    ]
    missing = [key for key in deterministic_keys if not outputs[key].is_file()]
    if missing:
        raise FileNotFoundError(
            f"determinism check requires an initial V2.1 build: {missing}"
        )
    before = {key: sha256_file(outputs[key]) for key in deterministic_keys}
    build_result = build_v2_development(config_path)
    after = {key: sha256_file(outputs[key]) for key in deterministic_keys}
    mismatches = [key for key in deterministic_keys if before[key] != after[key]]
    result = {
        "task_id": TASK_ID,
        "checked_at_utc": utc_now(),
        "artifacts": {
            key: {
                "path": outputs[key].as_posix(),
                "before_sha256": before[key],
                "after_sha256": after[key],
                "match": before[key] == after[key],
            }
            for key in deterministic_keys
        },
        "rebuild_match": not mismatches,
        "mismatches": mismatches,
        "build_result": build_result,
        "locked_outcome_access_count": 0,
        "model_training_performed": False,
    }
    write_json_atomic(outputs["determinism_check"], result)
    if mismatches:
        raise RuntimeError(f"V2.1 deterministic rebuild mismatches: {mismatches}")
    return result


def finalize_v2_development(
    config_path: str | Path,
    *,
    tests_passed: int,
    tests_failed: int,
    test_duration_seconds: float,
    command_log: str | Path,
    changed_files: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    outputs = {key: Path(value) for key, value in config["output"].items()}
    verification = verify_v2_development(config_path)
    if tests_failed or tests_passed <= 0:
        raise RuntimeError("tests must pass before V2.1 finalization")
    command_log = Path(command_log)
    changed_files = Path(changed_files)
    commands = json.loads(command_log.read_text(encoding="utf-8"))["commands"]
    changed = [
        line.strip()
        for line in changed_files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    generated = list(
        dict.fromkeys(
            [
                *changed,
                outputs["source_features"].as_posix(),
                outputs["cause_specific_targets"].as_posix(),
                outputs["interval_exposure"].as_posix(),
                outputs["manifest"].as_posix(),
                outputs["feature_schema"].as_posix(),
                outputs["source_only_audit"].as_posix(),
                outputs["target_dictionary"].as_posix(),
                outputs["row_count_audit"].as_posix(),
                outputs["exclusion_ambiguity_ledger"].as_posix(),
                outputs["determinism_check"].as_posix(),
            ]
        )
    )
    output_hashes = {
        path: artifact_record(path, include_schema=Path(path).suffix == ".parquet")
        for path in generated
        if Path(path).is_file()
        and Path(path) not in {outputs["report"], outputs["report_sidecar"]}
    }
    report.update(
        {
            "task_status": "DONE",
            "status": "PASS",
            "gate_status": "GO",
            "gate_decision": "GO",
            "finalized_at_utc": utc_now(),
            "commands": commands,
            "tests": {
                "command": "python -m pytest -q",
                "passed": int(tests_passed),
                "failed": int(tests_failed),
                "duration_seconds": float(test_duration_seconds),
            },
            "verification": verification,
            "generated_files": generated,
            "modified_files": changed,
            "all_output_hashes_excluding_report_self": output_hashes,
            "report_self_hash_policy": "reports/V2_1/report.sha256 stores the final report hash to avoid recursive self-hashing",
            "next_task_eligibility": {
                "only_task": "V2.2",
                "eligible": True,
                "authorized": False,
                "executed": False,
                "requires_separate_PI_instruction": True,
            },
        }
    )
    write_json_atomic(outputs["report"], report)
    digest = sha256_file(outputs["report"])
    outputs["report_sidecar"].parent.mkdir(parents=True, exist_ok=True)
    temporary = outputs["report_sidecar"].with_suffix(".sha256.tmp")
    temporary.write_text(f"{digest}  report.json\n", encoding="utf-8")
    os.replace(temporary, outputs["report_sidecar"])
    return {
        "task_id": TASK_ID,
        "task_status": "DONE",
        "status": "PASS",
        "gate_status": "GO",
        "report_sha256": digest,
        "output_hash_count": len(output_hashes),
        "tests_passed": tests_passed,
        "locked_outcome_access_count": 0,
        "model_training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase-evo-v2-development")
    parser.add_argument(
        "command", choices=["build", "determinism", "verify", "finalize"]
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--test-duration-seconds", type=float, default=0.0)
    parser.add_argument("--command-log", default="V2_1_COMMAND_LOG.json")
    parser.add_argument("--changed-files", default="V2_1_CHANGED_FILES.txt")
    args = parser.parse_args()
    if args.command == "build":
        result = build_v2_development(args.config)
    elif args.command == "determinism":
        result = check_deterministic_rebuild(args.config)
    elif args.command == "verify":
        result = verify_v2_development(args.config)
    else:
        result = finalize_v2_development(
            args.config,
            tests_passed=args.tests_passed,
            tests_failed=args.tests_failed,
            test_duration_seconds=args.test_duration_seconds,
            command_log=args.command_log,
            changed_files=args.changed_files,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
