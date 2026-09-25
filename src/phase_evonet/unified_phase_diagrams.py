from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from .identity_candidates import write_csv_atomic, write_json_atomic, write_jsonl_atomic
from .manifest import sha256_file
from .reported_transitions import DataFrameParquetWriter, utc_now


ENTRY_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("unified_entry_id", pa.binary(16), nullable=False),
        pa.field("entry_id", pa.string(), nullable=False),
        pa.field("task_id", pa.string()),
        pa.field("material_id", pa.string()),
        pa.field("thermo_id", pa.string()),
        pa.field("entry_label", pa.string()),
        pa.field("run_type", pa.string()),
        pa.field("composition_json", pa.large_string(), nullable=False),
        pa.field("reduced_formula", pa.string()),
        pa.field("chemsys", pa.string()),
        pa.field("nelements", pa.int16()),
        pa.field("num_atoms", pa.float64()),
        pa.field("uncorrected_energy", pa.float64()),
        pa.field("correction", pa.float64()),
        pa.field("corrected_energy", pa.float64()),
        pa.field("corrected_energy_per_atom", pa.float64()),
        pa.field("formation_energy_per_atom", pa.float64()),
        pa.field("energy_above_hull", pa.float64()),
        pa.field("is_stable", pa.bool_()),
        pa.field("decomposition_component_count", pa.int16()),
        pa.field("phase_diagram_status", pa.string(), nullable=False),
        pa.field("source_record_count", pa.int32(), nullable=False),
        pa.field("duplicate_conflict", pa.bool_(), nullable=False),
        pa.field("energy_adjustments_json", pa.large_string()),
        pa.field("parameters_json", pa.large_string()),
        pa.field("hubbards_json", pa.large_string()),
        pa.field("potcar_spec_json", pa.large_string()),
        pa.field("entry_data_json", pa.large_string()),
        pa.field("source_object_sha256", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
    ]
)

DECOMPOSITION_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("unified_entry_id", pa.binary(16), nullable=False),
        pa.field("component_unified_entry_id", pa.binary(16), nullable=False),
        pa.field("component_entry_id", pa.string(), nullable=False),
        pa.field("component_formula", pa.string(), nullable=False),
        pa.field("amount", pa.float64(), nullable=False),
        pa.field("component_energy_per_atom", pa.float64(), nullable=False),
    ]
)

DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("entry_id", pa.string(), nullable=False),
        pa.field("selected_source_object_sha256", pa.string(), nullable=False),
        pa.field("selected_source_key", pa.string(), nullable=False),
        pa.field("selected_source_row_number", pa.int64(), nullable=False),
        pa.field("excluded_source_object_sha256", pa.string(), nullable=False),
        pa.field("excluded_source_key", pa.string(), nullable=False),
        pa.field("excluded_source_row_number", pa.int64(), nullable=False),
        pa.field("composition_equal", pa.bool_(), nullable=False),
        pa.field("corrected_energy_equal", pa.bool_(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)

ENTRY_COLUMNS = [
    "snapshot_id", "material_id", "thermo_id", "thermo_type", "entry_label",
    "entry_id", "task_id", "energy", "correction", "composition_json",
    "energy_adjustments_json", "parameters_json", "run_type", "hubbards_json",
    "potcar_spec_json", "entry_data_json", "source_object_sha256", "source_key",
    "source_row_number",
]


def unified_entry_id(snapshot: str, thermo_type: str, entry_id: str) -> bytes:
    body = f"{snapshot}|{thermo_type}|{entry_id}".encode()
    return hashlib.blake2b(body, digest_size=16).digest()


def _float_equal(left: Any, right: Any, tolerance: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def deduplicate_entries(
    frame: pd.DataFrame, *, tolerance: float = 1e-12
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Select one immutable-provenance representative and ledger every duplicate."""
    required = [
        "snapshot_id", "thermo_type", "entry_id", "energy", "correction",
        "composition_json", "source_object_sha256", "source_key", "source_row_number",
    ]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Missing entry columns: {missing_columns}")
    key_missing = frame[["snapshot_id", "thermo_type", "entry_id"]].isna().any(axis=1)
    if key_missing.any():
        raise RuntimeError(f"Raw thermo entries have {int(key_missing.sum())} missing keys")
    work = frame.copy()
    work["corrected_energy"] = work["energy"].astype(float) + work["correction"].fillna(0.0).astype(float)
    key = ["snapshot_id", "thermo_type", "entry_id"]
    provenance = ["source_object_sha256", "source_key", "source_row_number"]
    work = work.sort_values(key + provenance, kind="mergesort").reset_index(drop=True)
    counts = work.groupby(key, sort=False).size().rename("source_record_count")
    selected = work.drop_duplicates(key, keep="first").copy()
    selected = selected.merge(counts.reset_index(), on=key, how="left", validate="one_to_one")
    selected["duplicate_conflict"] = False
    selected_index = selected.set_index(key)
    duplicate_rows: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for group_key, group in work.groupby(key, sort=False):
        if len(group) == 1:
            continue
        first = group.iloc[0]
        conflict = False
        for _, excluded in group.iloc[1:].iterrows():
            composition_equal = str(first["composition_json"]) == str(excluded["composition_json"])
            energy_equal = _float_equal(first["corrected_energy"], excluded["corrected_energy"], tolerance)
            conflict = conflict or not (composition_equal and energy_equal)
            duplicate_rows.append(
                {
                    "snapshot_id": str(first["snapshot_id"]),
                    "thermo_type": str(first["thermo_type"]),
                    "entry_id": str(first["entry_id"]),
                    "selected_source_object_sha256": str(first["source_object_sha256"]),
                    "selected_source_key": str(first["source_key"]),
                    "selected_source_row_number": int(first["source_row_number"]),
                    "excluded_source_object_sha256": str(excluded["source_object_sha256"]),
                    "excluded_source_key": str(excluded["source_key"]),
                    "excluded_source_row_number": int(excluded["source_row_number"]),
                    "composition_equal": composition_equal,
                    "corrected_energy_equal": energy_equal,
                    "reason": "exact_duplicate" if composition_equal and energy_equal else "conflicting_duplicate",
                }
            )
        if conflict:
            selected_index.loc[group_key, "duplicate_conflict"] = True
            ambiguities.append(
                {
                    "snapshot_id": str(first["snapshot_id"]),
                    "thermo_type": str(first["thermo_type"]),
                    "entry_id": str(first["entry_id"]),
                    "reason": "duplicate entries disagree in composition or corrected energy",
                    "source_record_count": len(group),
                }
            )
    selected = selected_index.reset_index()
    duplicate_frame = pd.DataFrame(duplicate_rows, columns=DUPLICATE_SCHEMA.names)
    return selected, duplicate_frame, ambiguities


def solve_phase_entries(
    selected: pd.DataFrame, *, stable_tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Build one compatibility partition and return entry/decomposition tables."""
    if selected.empty:
        raise RuntimeError("Cannot build an empty phase-diagram partition")
    snapshot = str(selected.iloc[0]["snapshot_id"])
    thermo_type = str(selected.iloc[0]["thermo_type"])
    computed: list[ComputedEntry] = []
    valid_rows: list[pd.Series] = []
    errors: list[dict[str, Any]] = []
    warning_text: set[str] = set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _, row in selected.iterrows():
            if bool(row["duplicate_conflict"]):
                errors.append({"snapshot_id": snapshot, "thermo_type": thermo_type, "entry_id": str(row["entry_id"]), "reason": "conflicting_duplicate_excluded"})
                continue
            try:
                composition = Composition(json.loads(str(row["composition_json"])))
                energy = float(row["corrected_energy"])
                if not math.isfinite(energy) or composition.num_atoms <= 0:
                    raise ValueError("non-finite energy or empty composition")
                entry = ComputedEntry(composition, energy, entry_id=str(row["entry_id"]))
            except Exception as exc:
                errors.append({"snapshot_id": snapshot, "thermo_type": thermo_type, "entry_id": str(row["entry_id"]), "reason": "invalid_phase_entry", "error": f"{type(exc).__name__}: {exc}"})
                continue
            computed.append(entry)
            valid_rows.append(row)
        if not computed:
            raise RuntimeError(f"No valid entries for {snapshot}/{thermo_type}")
        phase_diagram = PatchedPhaseDiagram(computed, keep_all_spaces=False, verbose=False)
        warning_text.update(str(item.message) for item in caught)

    phase_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    by_entry_id = {str(entry.entry_id): entry for entry in computed}
    row_by_entry_id = {str(row["entry_id"]): row for row in valid_rows}
    with warnings.catch_warnings(record=True) as solve_warnings:
        warnings.simplefilter("always")
        for entry_id in sorted(by_entry_id):
            entry = by_entry_id[entry_id]
            row = row_by_entry_id[entry_id]
            uid = unified_entry_id(snapshot, thermo_type, entry_id)
            try:
                decomposition, hull = phase_diagram.get_decomp_and_e_above_hull(
                    entry, check_stable=True, on_error="raise"
                )
                formation = float(phase_diagram.get_form_energy_per_atom(entry))
                hull = float(hull)
                if hull < 0 and abs(hull) <= stable_tolerance:
                    hull = 0.0
                if hull < -stable_tolerance:
                    raise RuntimeError(f"negative hull energy {hull}")
                components = sorted(decomposition.items(), key=lambda item: str(item[0].entry_id))
                for component, amount in components:
                    component_id = str(component.entry_id)
                    decomp_rows.append(
                        {
                            "snapshot_id": snapshot,
                            "thermo_type": thermo_type,
                            "unified_entry_id": uid,
                            "component_unified_entry_id": unified_entry_id(snapshot, thermo_type, component_id),
                            "component_entry_id": component_id,
                            "component_formula": component.composition.reduced_formula,
                            "amount": float(amount),
                            "component_energy_per_atom": float(component.energy_per_atom),
                        }
                    )
                status = "computed"
            except Exception as exc:
                formation = hull = None
                components = []
                status = "phase_diagram_error"
                errors.append({"snapshot_id": snapshot, "thermo_type": thermo_type, "entry_id": entry_id, "reason": status, "error": f"{type(exc).__name__}: {exc}"})
            composition = entry.composition
            phase_rows.append(
            {
                "snapshot_id": snapshot,
                "thermo_type": thermo_type,
                "unified_entry_id": uid,
                "entry_id": entry_id,
                "task_id": row.get("task_id"),
                "material_id": row.get("material_id"),
                "thermo_id": row.get("thermo_id"),
                "entry_label": row.get("entry_label"),
                "run_type": row.get("run_type"),
                "composition_json": str(row["composition_json"]),
                "reduced_formula": composition.reduced_formula,
                "chemsys": "-".join(sorted(str(element) for element in composition.elements)),
                "nelements": len(composition.elements),
                "num_atoms": float(composition.num_atoms),
                "uncorrected_energy": float(row["energy"]),
                "correction": float(row["correction"] or 0.0),
                "corrected_energy": float(row["corrected_energy"]),
                "corrected_energy_per_atom": float(row["corrected_energy"]) / float(composition.num_atoms),
                "formation_energy_per_atom": formation,
                "energy_above_hull": hull,
                "is_stable": None if hull is None else hull <= stable_tolerance,
                "decomposition_component_count": len(components),
                "phase_diagram_status": status,
                "source_record_count": int(row["source_record_count"]),
                "duplicate_conflict": bool(row["duplicate_conflict"]),
                "energy_adjustments_json": row.get("energy_adjustments_json"),
                "parameters_json": row.get("parameters_json"),
                "hubbards_json": row.get("hubbards_json"),
                "potcar_spec_json": row.get("potcar_spec_json"),
                "entry_data_json": row.get("entry_data_json"),
                "source_object_sha256": str(row["source_object_sha256"]),
                "source_key": str(row["source_key"]),
                "source_row_number": int(row["source_row_number"]),
                }
            )
        fallback_count = 0
        for item in solve_warnings:
            message = str(item.message)
            if message.startswith("No suitable PhaseDiagrams found for"):
                fallback_count += 1
            else:
                warning_text.add(message)
        if fallback_count:
            warning_text.add(
                "PatchedPhaseDiagram used its documented SLSQP decomposition "
                f"fallback for uncovered composition subspaces: {fallback_count} entries"
            )
    return (
        pd.DataFrame(phase_rows, columns=ENTRY_SCHEMA.names),
        pd.DataFrame(decomp_rows, columns=DECOMPOSITION_SCHEMA.names),
        errors,
        sorted(warning_text),
    )


def _write_table(path: Path, frame: pd.DataFrame, schema: pa.Schema, parquet_config: dict[str, Any]) -> None:
    writer = DataFrameParquetWriter(path, schema, parquet_config)
    writer.write(frame)
    writer.close()


def _deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    keys = frame.apply(
        lambda row: hashlib.sha256(f"{seed}|{row['thermo_id']}|{row['entry_id']}".encode()).hexdigest(), axis=1
    )
    return frame.loc[keys.sort_values(kind="mergesort").index[:n]].copy()


def _reference_audit(
    snapshot: str,
    thermo_type: str,
    normalized_root: Path,
    phase: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thermo_path = normalized_root / f"snapshot={snapshot}" / "raw_thermo.parquet"
    source = pq.read_table(
        thermo_path,
        columns=["thermo_id", "thermo_type", "energy_type", "energy_above_hull", "source_object_sha256", "source_key", "source_row_number"],
        filters=[("thermo_type", "=", thermo_type)],
    ).to_pandas()
    source = source.sort_values(
        ["thermo_id", "source_object_sha256", "source_key", "source_row_number"], kind="mergesort"
    ).drop_duplicates(["thermo_id", "thermo_type"], keep="first")
    chosen = phase[phase["phase_diagram_status"] == "computed"][
        ["thermo_id", "entry_label", "entry_id", "energy_above_hull"]
    ].rename(columns={"energy_above_hull": "reconstructed_energy_above_hull"})
    audit = source.merge(
        chosen,
        left_on=["thermo_id", "energy_type"],
        right_on=["thermo_id", "entry_label"],
        how="left",
        validate="one_to_one",
    )
    audit["absolute_error_eV_per_atom"] = (
        audit["reconstructed_energy_above_hull"] - audit["energy_above_hull"]
    ).abs()
    tolerance = float(config["reference_test"]["absolute_tolerance_eV_per_atom"])
    audit["matched"] = audit["absolute_error_eV_per_atom"].le(tolerance).fillna(False)
    sample = _deterministic_sample(
        audit, int(config["reference_test"]["sample_per_snapshot_thermo_type"]), int(config["seed"])
    )
    sample.insert(0, "snapshot_id", snapshot)
    sample.insert(1, "thermo_type_audit", thermo_type)
    summary = {
        "snapshot_id": snapshot,
        "thermo_type": thermo_type,
        "source_states": len(audit),
        "comparable_states": int(audit["reconstructed_energy_above_hull"].notna().sum()),
        "matched_states": int(audit["matched"].sum()),
        "match_rate": float(audit["matched"].mean()) if len(audit) else 0.0,
        "max_absolute_error_eV_per_atom": float(audit["absolute_error_eV_per_atom"].max()) if audit["absolute_error_eV_per_atom"].notna().any() else None,
        "tolerance_eV_per_atom": tolerance,
    }
    return summary, sample.to_dict(orient="records")


def _group_token(snapshot: str, thermo_type: str) -> str:
    return hashlib.sha256(f"{snapshot}|{thermo_type}".encode()).hexdigest()[:16]


def _build_group(job: dict[str, Any]) -> dict[str, Any]:
    snapshot = job["snapshot"]
    thermo_type = job["thermo_type"]
    config = job["config"]
    normalized_root = Path(config["input"]["normalized_root"])
    source_path = normalized_root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet"
    parts = Path(job["parts_root"])
    token = _group_token(snapshot, thermo_type)
    phase_path = parts / f"{token}.phase.parquet"
    decomp_path = parts / f"{token}.decomposition.parquet"
    duplicate_path = parts / f"{token}.duplicates.parquet"
    ambiguity_path = parts / f"{token}.ambiguities.jsonl"
    stats_path = parts / f"{token}.stats.json"
    source_hash = job["source_hash"]
    config_hash = job["config_hash"]
    if bool(config["execution"].get("resume_group_parts")) and stats_path.exists():
        prior = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            prior.get("complete") is True
            and prior.get("source_sha256") == source_hash
            and prior.get("config_sha256") == config_hash
            and all(Path(item["path"]).exists() and sha256_file(item["path"]) == item["sha256"] for item in prior["outputs"])
        ):
            return prior
    raw = pq.read_table(source_path, columns=ENTRY_COLUMNS, filters=[("thermo_type", "=", thermo_type)]).to_pandas()
    selected, duplicates, ambiguities = deduplicate_entries(
        raw, tolerance=float(config["energy_policy"]["duplicate_float_absolute_tolerance"])
    )
    phase, decompositions, solve_errors, warning_rows = solve_phase_entries(
        selected,
        stable_tolerance=float(config["energy_policy"]["stable_energy_tolerance_eV_per_atom"]),
    )
    ambiguities.extend(solve_errors)
    _write_table(phase_path, phase, ENTRY_SCHEMA, config["parquet"])
    _write_table(decomp_path, decompositions, DECOMPOSITION_SCHEMA, config["parquet"])
    _write_table(duplicate_path, duplicates, DUPLICATE_SCHEMA, config["parquet"])
    write_jsonl_atomic(ambiguity_path, ambiguities)
    reference, reference_rows = _reference_audit(snapshot, thermo_type, normalized_root, phase, config)
    outputs = [
        {"path": str(path), "rows": pq.ParquetFile(path).metadata.num_rows, "sha256": sha256_file(path)}
        for path in (phase_path, decomp_path, duplicate_path)
    ]
    outputs.append({"path": str(ambiguity_path), "rows": len(ambiguities), "sha256": sha256_file(ambiguity_path)})
    stats = {
        "complete": True,
        "snapshot_id": snapshot,
        "thermo_type": thermo_type,
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "raw_rows": len(raw),
        "unique_entries": len(selected),
        "computed_entries": int((phase["phase_diagram_status"] == "computed").sum()),
        "stable_entries": int(phase["is_stable"].fillna(False).sum()),
        "decomposition_rows": len(decompositions),
        "duplicate_rows": len(duplicates),
        "ambiguity_rows": len(ambiguities),
        "warnings": warning_rows,
        "reference_test": reference,
        "reference_sample": reference_rows,
        "outputs": outputs,
    }
    write_json_atomic(stats_path, stats)
    return stats


def _combine_parquet(paths: Iterable[Path], target: Path, schema: pa.Schema, config: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression=str(config["compression"]), use_dictionary=True, write_statistics=True)
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow != schema:
                raise RuntimeError(f"Unexpected group schema: {path}")
            for batch in parquet.iter_batches(batch_size=int(config["row_group_size"])):
                writer.write_batch(batch, row_group_size=int(config["row_group_size"]))
    finally:
        writer.close()
    os.replace(temporary, target)


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()


def _dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "corrected_energy": "Total entry energy after applying the source correction exactly once (energy + correction).",
        "energy_above_hull": "Reconstructed value within one snapshot and one thermo_type compatibility partition.",
        "amount": "Atomic-fraction coefficient returned by pymatgen for the reconstructed decomposition.",
        "duplicate_conflict": "True only if records sharing the entry key disagree in composition or corrected energy.",
        "phase_diagram_status": "computed or an explicit exclusion/error state; errors are also written to the ambiguity ledger.",
    }
    rows: list[dict[str, Any]] = []
    for table, schema in (("phase_entry_unified", ENTRY_SCHEMA), ("phase_decomposition", DECOMPOSITION_SCHEMA), ("entry_duplicate_ledger", DUPLICATE_SCHEMA)):
        for field in schema:
            energy_unit = "eV/atom" if "per_atom" in field.name or field.name == "energy_above_hull" else ("eV" if "energy" in field.name or field.name == "correction" else "")
            rows.append({"table": table, "field": field.name, "arrow_type": str(field.type), "nullable": field.nullable, "unit": energy_unit, "description": descriptions.get(field.name, "Traceable P3.2 unified-entry, decomposition, or duplicate-ledger field.")})
    return rows


def build_unified_phase_diagrams(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "P3.2":
        raise RuntimeError("P3.2 config must declare task_id: P3.2")
    context_dependent = list(
        config.get("energy_policy", {}).get("context_dependent_workflows", [])
    )
    if context_dependent:
        raise RuntimeError(
            "P3.2 is blocked: context-dependent DFT mixing workflows require a "
            "contracted phase_context key before entries from different chemical "
            f"systems can be rebuilt safely: {context_dependent}"
        )
    p3_1_report = json.loads(Path(config["input"]["p3_1_report"]).read_text(encoding="utf-8"))
    if not (p3_1_report.get("task_status") == "DONE" and p3_1_report.get("status") == "PASS" and p3_1_report.get("gate_decision") == "GO"):
        raise RuntimeError("P3.1 DONE/PASS/GO is required before P3.2")
    started = utc_now()
    config_hash = sha256_file(config_path)
    normalized_root = Path(config["input"]["normalized_root"])
    snapshots = [str(item) for item in config["snapshots"]]
    source_hashes = {
        snapshot: sha256_file(normalized_root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet")
        for snapshot in snapshots
    }
    thermo_types: dict[str, list[str]] = {}
    group_sizes: dict[tuple[str, str], int] = {}
    for snapshot in snapshots:
        table = pq.read_table(normalized_root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet", columns=["thermo_type"])
        thermo_types[snapshot] = sorted(str(item) for item in pc.unique(table["thermo_type"]).to_pylist() if item is not None)
        counts = table.to_pandas()["thermo_type"].value_counts()
        group_sizes.update({(snapshot, str(name)): int(count) for name, count in counts.items()})
    output = config["output"]
    parts_root = Path(output["root"]) / "_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"snapshot": snapshot, "thermo_type": thermo_type, "config": config, "parts_root": str(parts_root), "source_hash": source_hashes[snapshot], "config_hash": config_hash}
        for snapshot in snapshots for thermo_type in thermo_types[snapshot]
    ]
    jobs.sort(key=lambda job: (-group_sizes[(job["snapshot"], job["thermo_type"])], job["snapshot"], job["thermo_type"]))
    results: list[dict[str, Any]] = []
    workers = int(config["execution"]["workers"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_build_group, job): (job["snapshot"], job["thermo_type"]) for job in jobs}
        for future in as_completed(future_map):
            snapshot, thermo_type = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Phase-diagram group failed: {snapshot}/{thermo_type}: {exc}") from exc
    results.sort(key=lambda item: (item["snapshot_id"], item["thermo_type"]))
    phase_parts = [Path(next(item for item in result["outputs"] if item["path"].endswith(".phase.parquet"))["path"]) for result in results]
    decomp_parts = [Path(next(item for item in result["outputs"] if item["path"].endswith(".decomposition.parquet"))["path"]) for result in results]
    duplicate_parts = [Path(next(item for item in result["outputs"] if item["path"].endswith(".duplicates.parquet"))["path"]) for result in results]
    phase_path = Path(output["phase_entries"])
    decomp_path = Path(output["decompositions"])
    duplicate_path = Path(output["duplicate_ledger"])
    _combine_parquet(phase_parts, phase_path, ENTRY_SCHEMA, config["parquet"])
    _combine_parquet(decomp_parts, decomp_path, DECOMPOSITION_SCHEMA, config["parquet"])
    _combine_parquet(duplicate_parts, duplicate_path, DUPLICATE_SCHEMA, config["parquet"])
    ambiguity_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for result in results:
        ambiguity_file = Path(next(item for item in result["outputs"] if item["path"].endswith(".ambiguities.jsonl"))["path"])
        ambiguity_rows.extend(json.loads(line) for line in ambiguity_file.read_text(encoding="utf-8").splitlines() if line)
        reference_rows.extend(result["reference_sample"])
    ambiguity_path = Path(output["ambiguity_ledger"])
    write_jsonl_atomic(ambiguity_path, ambiguity_rows)
    audit_csv = Path(output["reference_audit_csv"])
    write_csv_atomic(audit_csv, reference_rows)
    reference_states = sum(item["reference_test"]["source_states"] for item in results)
    matched_states = sum(item["reference_test"]["matched_states"] for item in results)
    match_rate = matched_states / reference_states if reference_states else 0.0
    reference_summary = {
        "task_id": "P3.2", "created_at_utc": utc_now(), "selector": config["reference_test"]["source_entry_selector"],
        "source_states": reference_states, "matched_states": matched_states, "match_rate": match_rate,
        "required_match_rate": float(config["reference_test"]["required_match_rate"]),
        "absolute_tolerance_eV_per_atom": float(config["reference_test"]["absolute_tolerance_eV_per_atom"]),
        "groups": [item["reference_test"] for item in results],
    }
    reference_summary["status"] = "PASS" if match_rate >= reference_summary["required_match_rate"] else "FAIL"
    audit_json = Path(output["reference_audit_json"])
    write_json_atomic(audit_json, reference_summary)
    dictionary_path = Path(output["data_dictionary"])
    write_csv_atomic(dictionary_path, _dictionary_rows())
    gate_passed = not ambiguity_rows and reference_summary["status"] == "PASS"
    artifact_paths = [phase_path, decomp_path, duplicate_path]
    manifest = {
        "task_id": "P3.2", "status": "PASS" if gate_passed else "FAIL", "gate_status": "GO" if gate_passed else "NO-GO",
        "started_at_utc": started, "ended_at_utc": utc_now(), "seed": int(config["seed"]), "network_access": False,
        "config_path": str(config_path), "config_sha256": config_hash, "p3_1_report_sha256": sha256_file(config["input"]["p3_1_report"]),
        "source_hashes": source_hashes, "partition_policy": config["energy_policy"], "groups": [{k: v for k, v in item.items() if k not in {"reference_sample", "outputs"}} for item in results],
        "raw_rows": sum(item["raw_rows"] for item in results), "unique_entries": sum(item["unique_entries"] for item in results),
        "computed_entries": sum(item["computed_entries"] for item in results), "stable_entries": sum(item["stable_entries"] for item in results),
        "decomposition_rows": sum(item["decomposition_rows"] for item in results), "duplicate_rows": sum(item["duplicate_rows"] for item in results),
        "ambiguity_rows": len(ambiguity_rows), "warnings": sorted({warning for item in results for warning in item["warnings"]}),
        "reference_test": reference_summary,
        "outputs": [{"path": str(path), "rows": pq.ParquetFile(path).metadata.num_rows, "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema_sha256": _schema_hash(path)} for path in artifact_paths],
        "supporting_files": {str(path): sha256_file(path) for path in (ambiguity_path, audit_csv, audit_json, dictionary_path)},
        "gate": {"unit_tests_required": True, "reference_tests_passed": reference_summary["status"] == "PASS", "no_unledgered_phase_errors": not ambiguity_rows, "passed": gate_passed},
    }
    manifest_path = Path(output["manifest"])
    write_json_atomic(manifest_path, manifest)
    shutil.rmtree(parts_root)
    return manifest


def verify_unified_phase_diagrams(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest_path = Path(config["output"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for artifact in manifest["outputs"]:
        path = Path(artifact["path"])
        if not path.exists():
            failures.append(f"missing:{path}")
            continue
        if pq.ParquetFile(path).metadata.num_rows != artifact["rows"]:
            failures.append(f"rows:{path}")
        if sha256_file(path) != artifact["sha256"]:
            failures.append(f"sha256:{path}")
        if _schema_hash(path) != artifact["schema_sha256"]:
            failures.append(f"schema:{path}")
    for path_text, expected_hash in manifest["supporting_files"].items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"supporting:{path}")
    phase_path = Path(config["output"]["phase_entries"])
    phase = pq.read_table(phase_path, columns=["unified_entry_id", "energy_above_hull", "phase_diagram_status", "decomposition_component_count"])
    if pc.count_distinct(phase["unified_entry_id"]).as_py() != phase.num_rows:
        failures.append("duplicate_unified_entry_id")
    computed = phase.filter(pc.equal(phase["phase_diagram_status"], "computed"))
    if pc.any(pc.less(computed["energy_above_hull"], -float(config["energy_policy"]["stable_energy_tolerance_eV_per_atom"]))).as_py():
        failures.append("negative_hull_energy")
    if pc.any(pc.less_equal(computed["decomposition_component_count"], 0)).as_py():
        failures.append("missing_decomposition")
    decomposition = pq.read_table(config["output"]["decompositions"], columns=["unified_entry_id", "component_unified_entry_id", "amount"])
    entry_ids = set(phase["unified_entry_id"].to_pylist())
    if any(item not in entry_ids for item in pc.unique(decomposition["component_unified_entry_id"]).to_pylist()):
        failures.append("unknown_decomposition_component")
    sums = decomposition.group_by("unified_entry_id").aggregate([("amount", "sum")])
    maximum_sum_error = max((abs(float(value) - 1.0) for value in sums["amount_sum"].to_pylist()), default=0.0)
    if maximum_sum_error > 1e-8:
        failures.append("decomposition_amount_sum")
    if not manifest["gate"]["passed"]:
        failures.append("gate_not_passed")
    temporary = list(Path(config["output"]["root"]).rglob("*.tmp")) + list(Path("reports/P3_2").rglob("*.tmp"))
    if temporary:
        failures.append("temporary_files")
    return {
        "task_id": "P3.2", "status": "PASS" if not failures else "FAIL", "failures": failures,
        "phase_entries": phase.num_rows, "computed_entries": computed.num_rows, "decomposition_rows": decomposition.num_rows,
        "distinct_unified_entry_ids": pc.count_distinct(phase["unified_entry_id"]).as_py(),
        "maximum_decomposition_sum_error": maximum_sum_error, "reference_match_rate": manifest["reference_test"]["match_rate"],
        "gate_status": manifest["gate_status"], "temporary_files": len(temporary), "network_access": False, "verified_at_utc": utc_now(),
    }
