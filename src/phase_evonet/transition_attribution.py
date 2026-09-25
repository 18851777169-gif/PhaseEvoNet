from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from scipy.optimize import linprog

from .identity_candidates import write_csv_atomic, write_json_atomic
from .manifest import sha256_file


PLAYERS = (
    "competitor_inventory",
    "uncorrected_energy",
    "compatibility_correction",
    "candidate_identity",
)
METHOD_VERSION = "EXACT_CONTEXT_SHAPLEY_V1"


ATTRIBUTION_SCHEMA = pa.schema(
    [
        pa.field("attribution_id", pa.binary(16), nullable=False),
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("identity_edge_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("phase_context_chemsys", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("source_thermo_id", pa.string(), nullable=False),
        pa.field("target_thermo_id", pa.string(), nullable=False),
        pa.field("source_unified_entry_id", pa.binary(16), nullable=False),
        pa.field("target_unified_entry_id", pa.binary(16), nullable=False),
        pa.field("source_entry_id", pa.string(), nullable=False),
        pa.field("target_entry_id", pa.string(), nullable=False),
        pa.field("source_task_id", pa.string()),
        pa.field("target_task_id", pa.string()),
        pa.field("source_is_stable", pa.bool_(), nullable=False),
        pa.field("target_is_stable", pa.bool_(), nullable=False),
        pa.field("unified_label_transition", pa.string(), nullable=False),
        pa.field("reported_label_flip", pa.bool_(), nullable=False),
        pa.field("important_high_confidence_flip", pa.bool_(), nullable=False),
        pa.field("source_energy_above_hull", pa.float64(), nullable=False),
        pa.field("target_energy_above_hull", pa.float64(), nullable=False),
        pa.field("delta_energy_above_hull", pa.float64(), nullable=False),
        pa.field("competitor_inventory_contribution", pa.float64(), nullable=False),
        pa.field("uncorrected_energy_contribution", pa.float64(), nullable=False),
        pa.field("compatibility_correction_contribution", pa.float64(), nullable=False),
        pa.field("candidate_identity_contribution", pa.float64(), nullable=False),
        pa.field("attribution_sum", pa.float64(), nullable=False),
        pa.field("reconstruction_residual", pa.float64(), nullable=False),
        pa.field("source_endpoint_reconstruction_error", pa.float64(), nullable=False),
        pa.field("target_endpoint_reconstruction_error", pa.float64(), nullable=False),
        pa.field("competitor_inventory_signed_share", pa.float64()),
        pa.field("uncorrected_energy_signed_share", pa.float64()),
        pa.field("compatibility_correction_signed_share", pa.float64()),
        pa.field("candidate_identity_signed_share", pa.float64()),
        pa.field("dominant_channel", pa.string(), nullable=False),
        pa.field("dominant_absolute_contribution", pa.float64(), nullable=False),
        pa.field("candidate_identity_changed", pa.bool_(), nullable=False),
        pa.field("candidate_material_id_changed", pa.bool_(), nullable=False),
        pa.field("candidate_task_id_changed", pa.bool_(), nullable=False),
        pa.field("candidate_composition_changed", pa.bool_(), nullable=False),
        pa.field("source_decomposition_component_count", pa.int16(), nullable=False),
        pa.field("target_decomposition_component_count", pa.int16(), nullable=False),
        pa.field("common_decomposition_phase_keys", pa.int16(), nullable=False),
        pa.field("added_decomposition_phase_keys", pa.int16(), nullable=False),
        pa.field("removed_decomposition_phase_keys", pa.int16(), nullable=False),
        pa.field("source_decomposition_phase_keys_json", pa.large_string(), nullable=False),
        pa.field("target_decomposition_phase_keys_json", pa.large_string(), nullable=False),
        pa.field("source_phase_entries", pa.int32(), nullable=False),
        pa.field("target_phase_entries", pa.int32(), nullable=False),
        pa.field("common_phase_entries", pa.int32(), nullable=False),
        pa.field("added_phase_entries", pa.int32(), nullable=False),
        pa.field("removed_phase_entries", pa.int32(), nullable=False),
        pa.field("coalition_count", pa.int8(), nullable=False),
        pa.field("attributable", pa.bool_(), nullable=False),
        pa.field("attribution_status", pa.string(), nullable=False),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


COUNTERFACTUAL_SCHEMA = pa.schema(
    [
        pa.field("attribution_id", pa.binary(16), nullable=False),
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("coalition_mask", pa.uint8(), nullable=False),
        pa.field("use_target_competitor_inventory", pa.bool_(), nullable=False),
        pa.field("use_target_uncorrected_energy", pa.bool_(), nullable=False),
        pa.field("use_target_compatibility_correction", pa.bool_(), nullable=False),
        pa.field("use_target_candidate_identity", pa.bool_(), nullable=False),
        pa.field("counterfactual_energy_above_hull", pa.float64(), nullable=False),
        pa.field("solver_status", pa.string(), nullable=False),
        pa.field("competitor_count", pa.int32(), nullable=False),
        pa.field("feasible_decomposition", pa.bool_(), nullable=False),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


INELIGIBLE_COLUMNS = [
    "transition_id",
    "identity_edge_id",
    "canonical_lineage_id",
    "identity_confidence",
    "source_snapshot",
    "target_snapshot",
    "source_material_id",
    "target_material_id",
    "thermo_type",
    "observation_status",
    "reported_label_flip",
    "source_thermo_id",
    "target_thermo_id",
    "source_endpoint_status",
    "target_endpoint_status",
    "reasons_json",
    "compatibility_rule_versions_json",
]


TRANSITION_COLUMNS = [
    "transition_id",
    "identity_edge_id",
    "canonical_lineage_id",
    "identity_confidence",
    "source_snapshot",
    "target_snapshot",
    "source_material_id",
    "target_material_id",
    "thermo_type",
    "observation_status",
    "source_state_present",
    "target_state_present",
    "source_thermo_id",
    "target_thermo_id",
    "source_is_stable",
    "target_is_stable",
    "label_flip",
]


PHASE_COLUMNS = [
    "snapshot_id",
    "thermo_type",
    "phase_context_chemsys",
    "unified_entry_id",
    "entry_id",
    "task_id",
    "material_id",
    "thermo_id",
    "composition_json",
    "uncorrected_energy",
    "correction",
    "energy_above_hull",
    "is_stable",
    "is_target",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def attribution_id(transition: bytes) -> bytes:
    return hashlib.blake2b(
        bytes(transition) + b"|" + METHOD_VERSION.encode(), digest_size=16
    ).digest()


def _json_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _hex(value: Any) -> str | None:
    value = _json_or_none(value)
    return bytes(value).hex() if value is not None else None


def _composition_parts(text: str) -> tuple[dict[str, float], float]:
    values = {str(key): float(value) for key, value in json.loads(text).items()}
    total = float(sum(values.values()))
    if not values or not math.isfinite(total) or total <= 0:
        raise ValueError("Composition must contain a positive finite atom count")
    return values, total


def composition_signature(text: str) -> str:
    values, total = _composition_parts(text)
    fractional = {
        element: round(amount / total, 12)
        for element, amount in sorted(values.items())
        if amount != 0
    }
    return json.dumps(fractional, sort_keys=True, separators=(",", ":"))


def phase_key(entry_id: str, composition_json: str) -> str:
    return f"{str(entry_id).casefold()}|{composition_signature(composition_json)}"


def _phase_record(row: dict[str, Any]) -> dict[str, Any]:
    composition_json = str(row["composition_json"])
    _, atoms = _composition_parts(composition_json)
    return {
        "phase_key": phase_key(str(row["entry_id"]), composition_json),
        "entry_id": str(row["entry_id"]),
        "task_id": _json_or_none(row.get("task_id")),
        "material_id": _json_or_none(row.get("material_id")),
        "unified_entry_id": _json_or_none(row.get("unified_entry_id")),
        "composition_json": composition_json,
        "uncorrected_energy_per_atom": float(row["uncorrected_energy"]) / atoms,
        "correction_per_atom": float(row["correction"]) / atoms,
    }


def _phase_records(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = _phase_record(row)
        key = str(record["phase_key"])
        if key in result:
            raise RuntimeError(f"Duplicate phase key inside exact context: {key}")
        result[key] = record
    return result


def _composition_vector(text: str, elements: list[str]) -> np.ndarray:
    values, total = _composition_parts(text)
    return np.asarray([values.get(element, 0.0) / total for element in elements])


def hull_distance_linear_program(
    candidate: dict[str, Any],
    competitors: list[dict[str, Any]],
    elements: list[str],
) -> tuple[float, str, bool]:
    """Return hull distance without requiring artificial elemental terminals.

    The P3.2 contextual artifact can contain only the phases needed by its
    stored target/decomposition relations.  A non-negative mixture LP is
    therefore the exact primitive: if no retained competitor combination can
    express the candidate composition, the candidate is stable and its hull
    distance is zero.
    """

    if not competitors:
        return 0.0, "no_competitors", False
    target = _composition_vector(str(candidate["composition_json"]), elements)
    energies = np.asarray(
        [
            float(item["uncorrected_energy_per_atom"])
            + float(item["correction_per_atom"])
            for item in competitors
        ]
    )
    matrix = np.asarray(
        [
            _composition_vector(str(item["composition_json"]), elements)
            for item in competitors
        ]
    ).T
    result = linprog(
        energies,
        A_eq=matrix,
        b_eq=target,
        bounds=(0.0, None),
        method="highs",
    )
    if result.status == 2:
        return 0.0, "no_feasible_decomposition", False
    if not result.success or result.fun is None:
        raise RuntimeError(
            f"Counterfactual decomposition solver failed: status={result.status}; "
            f"message={result.message}"
        )
    candidate_energy = float(candidate["uncorrected_energy_per_atom"]) + float(
        candidate["correction_per_atom"]
    )
    return max(0.0, candidate_energy - float(result.fun)), "feasible", True


def _hybrid_common_record(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    use_target_energy: bool,
    use_target_compatibility: bool,
) -> dict[str, Any]:
    return {
        **source,
        "uncorrected_energy_per_atom": float(
            target["uncorrected_energy_per_atom"]
            if use_target_energy
            else source["uncorrected_energy_per_atom"]
        ),
        "correction_per_atom": float(
            target["correction_per_atom"]
            if use_target_compatibility
            else source["correction_per_atom"]
        ),
    }


def counterfactual_value(
    source_phases: dict[str, dict[str, Any]],
    target_phases: dict[str, dict[str, Any]],
    source_candidate: dict[str, Any],
    target_candidate: dict[str, Any],
    phase_context_chemsys: str,
    coalition_mask: int,
) -> tuple[float, str, int, bool]:
    use_target_inventory = bool(coalition_mask & 1)
    use_target_energy = bool(coalition_mask & 2)
    use_target_compatibility = bool(coalition_mask & 4)
    use_target_identity = bool(coalition_mask & 8)
    common = source_phases.keys() & target_phases.keys()
    inventory = target_phases if use_target_inventory else source_phases
    candidate = target_candidate if use_target_identity else source_candidate
    candidate_key = str(candidate["phase_key"])
    if str(source_candidate["phase_key"]) == str(target_candidate["phase_key"]):
        candidate = _hybrid_common_record(
            source_candidate,
            target_candidate,
            use_target_energy=use_target_energy,
            use_target_compatibility=use_target_compatibility,
        )
    competitors: list[dict[str, Any]] = []
    for key in sorted(inventory):
        if key == candidate_key:
            continue
        if key in common:
            competitors.append(
                _hybrid_common_record(
                    source_phases[key],
                    target_phases[key],
                    use_target_energy=use_target_energy,
                    use_target_compatibility=use_target_compatibility,
                )
            )
        else:
            competitors.append(inventory[key])
    value, status, feasible = hull_distance_linear_program(
        candidate, competitors, phase_context_chemsys.split("-")
    )
    return value, status, len(competitors), feasible


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    if set(values) != set(range(1 << len(PLAYERS))):
        raise ValueError("Exact Shapley requires every coalition value")
    n = len(PLAYERS)
    denominator = math.factorial(n)
    result: dict[str, float] = {}
    for player_index, player in enumerate(PLAYERS):
        bit = 1 << player_index
        contribution = 0.0
        for coalition in range(1 << n):
            if coalition & bit:
                continue
            size = coalition.bit_count()
            weight = (
                math.factorial(size) * math.factorial(n - size - 1) / denominator
            )
            contribution += weight * (values[coalition | bit] - values[coalition])
        result[player] = float(contribution)
    return result


def _candidate_from_prefixed(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return _phase_record(
        {
            "entry_id": row[f"{prefix}entry_id"],
            "task_id": row.get(f"{prefix}task_id"),
            "material_id": row[f"{prefix}material_id"],
            "composition_json": row[f"{prefix}composition_json"],
            "uncorrected_energy": row[f"{prefix}uncorrected_energy"],
            "correction": row[f"{prefix}correction"],
        }
    )


def attribute_one_transition(
    transition: dict[str, Any],
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_phases = _phase_records(source_rows)
    target_phases = _phase_records(target_rows)
    source_id_to_key = {
        bytes(record["unified_entry_id"]): key
        for key, record in source_phases.items()
        if record["unified_entry_id"] is not None
    }
    target_id_to_key = {
        bytes(record["unified_entry_id"]): key
        for key, record in target_phases.items()
        if record["unified_entry_id"] is not None
    }
    source_decomposition_keys = sorted(
        {
            source_id_to_key[bytes(component)]
            for component in transition.get("s_decomposition_component_ids", [])
        }
    )
    target_decomposition_keys = sorted(
        {
            target_id_to_key[bytes(component)]
            for component in transition.get("q_decomposition_component_ids", [])
        }
    )
    source_decomposition_set = set(source_decomposition_keys)
    target_decomposition_set = set(target_decomposition_keys)
    source_candidate = _candidate_from_prefixed(transition, "s_")
    target_candidate = _candidate_from_prefixed(transition, "q_")
    context = str(transition["s_phase_context_chemsys"])
    if context != str(transition["q_phase_context_chemsys"]):
        raise RuntimeError("Important transition crosses phase contexts")
    values: dict[int, float] = {}
    counterfactual_rows: list[dict[str, Any]] = []
    attr_id = attribution_id(bytes(transition["transition_id"]))
    for mask in range(1 << len(PLAYERS)):
        value, solver_status, competitor_count, feasible = counterfactual_value(
            source_phases,
            target_phases,
            source_candidate,
            target_candidate,
            context,
            mask,
        )
        values[mask] = value
        counterfactual_rows.append(
            {
                "attribution_id": attr_id,
                "transition_id": bytes(transition["transition_id"]),
                "coalition_mask": mask,
                "use_target_competitor_inventory": bool(mask & 1),
                "use_target_uncorrected_energy": bool(mask & 2),
                "use_target_compatibility_correction": bool(mask & 4),
                "use_target_candidate_identity": bool(mask & 8),
                "counterfactual_energy_above_hull": value,
                "solver_status": solver_status,
                "competitor_count": competitor_count,
                "feasible_decomposition": feasible,
                "method_version": METHOD_VERSION,
            }
        )
    contributions = exact_shapley(values)
    source_value = float(transition["s_energy_above_hull"])
    target_value = float(transition["q_energy_above_hull"])
    delta = target_value - source_value
    contribution_sum = float(sum(contributions.values()))
    residual = contribution_sum - delta
    source_error = abs(values[0] - source_value)
    target_error = abs(values[15] - target_value)
    attributable = (
        abs(residual) <= tolerance
        and source_error <= tolerance
        and target_error <= tolerance
    )
    dominant = max(PLAYERS, key=lambda item: (abs(contributions[item]), -PLAYERS.index(item)))
    shares = {
        player: (contributions[player] / delta if abs(delta) > tolerance else None)
        for player in PLAYERS
    }
    source_key = str(source_candidate["phase_key"])
    target_key = str(target_candidate["phase_key"])
    source_task = _json_or_none(transition.get("s_task_id"))
    target_task = _json_or_none(transition.get("q_task_id"))
    source_stable = bool(transition["s_is_stable"])
    target_stable = bool(transition["q_is_stable"])
    row = {
        "attribution_id": attr_id,
        "transition_id": bytes(transition["transition_id"]),
        "identity_edge_id": bytes(transition["identity_edge_id"]),
        "canonical_lineage_id": str(transition["canonical_lineage_id"]),
        "identity_confidence": str(transition["identity_confidence"]),
        "source_snapshot": str(transition["source_snapshot"]),
        "target_snapshot": str(transition["target_snapshot"]),
        "thermo_type": str(transition["thermo_type"]),
        "phase_context_chemsys": context,
        "source_material_id": str(transition["source_material_id"]),
        "target_material_id": str(transition["target_material_id"]),
        "source_thermo_id": str(transition["source_thermo_id"]),
        "target_thermo_id": str(transition["target_thermo_id"]),
        "source_unified_entry_id": bytes(transition["s_unified_entry_id"]),
        "target_unified_entry_id": bytes(transition["q_unified_entry_id"]),
        "source_entry_id": str(transition["s_entry_id"]),
        "target_entry_id": str(transition["q_entry_id"]),
        "source_task_id": None if source_task is None else str(source_task),
        "target_task_id": None if target_task is None else str(target_task),
        "source_is_stable": source_stable,
        "target_is_stable": target_stable,
        "unified_label_transition": (
            ("stable" if source_stable else "unstable")
            + "_to_"
            + ("stable" if target_stable else "unstable")
        ),
        "reported_label_flip": bool(transition["label_flip"]),
        "important_high_confidence_flip": True,
        "source_energy_above_hull": source_value,
        "target_energy_above_hull": target_value,
        "delta_energy_above_hull": delta,
        "competitor_inventory_contribution": contributions["competitor_inventory"],
        "uncorrected_energy_contribution": contributions["uncorrected_energy"],
        "compatibility_correction_contribution": contributions[
            "compatibility_correction"
        ],
        "candidate_identity_contribution": contributions["candidate_identity"],
        "attribution_sum": contribution_sum,
        "reconstruction_residual": residual,
        "source_endpoint_reconstruction_error": source_error,
        "target_endpoint_reconstruction_error": target_error,
        "competitor_inventory_signed_share": shares["competitor_inventory"],
        "uncorrected_energy_signed_share": shares["uncorrected_energy"],
        "compatibility_correction_signed_share": shares["compatibility_correction"],
        "candidate_identity_signed_share": shares["candidate_identity"],
        "dominant_channel": dominant,
        "dominant_absolute_contribution": abs(contributions[dominant]),
        "candidate_identity_changed": source_key != target_key,
        "candidate_material_id_changed": (
            str(transition["source_material_id"])
            != str(transition["target_material_id"])
        ),
        "candidate_task_id_changed": source_task != target_task,
        "candidate_composition_changed": (
            composition_signature(str(transition["s_composition_json"]))
            != composition_signature(str(transition["q_composition_json"]))
        ),
        "source_decomposition_component_count": len(source_decomposition_keys),
        "target_decomposition_component_count": len(target_decomposition_keys),
        "common_decomposition_phase_keys": len(
            source_decomposition_set & target_decomposition_set
        ),
        "added_decomposition_phase_keys": len(
            target_decomposition_set - source_decomposition_set
        ),
        "removed_decomposition_phase_keys": len(
            source_decomposition_set - target_decomposition_set
        ),
        "source_decomposition_phase_keys_json": json.dumps(
            source_decomposition_keys, separators=(",", ":")
        ),
        "target_decomposition_phase_keys_json": json.dumps(
            target_decomposition_keys, separators=(",", ":")
        ),
        "source_phase_entries": len(source_phases),
        "target_phase_entries": len(target_phases),
        "common_phase_entries": len(source_phases.keys() & target_phases.keys()),
        "added_phase_entries": len(target_phases.keys() - source_phases.keys()),
        "removed_phase_entries": len(source_phases.keys() - target_phases.keys()),
        "coalition_count": 16,
        "attributable": attributable,
        "attribution_status": "PASS" if attributable else "RECONSTRUCTION_MISMATCH",
        "method_version": METHOD_VERSION,
    }
    return row, counterfactual_rows


def _attribute_context_job(
    job: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attribution_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    for transition in job["transitions"]:
        attribution, counterfactuals = attribute_one_transition(
            transition,
            job["source_rows"],
            job["target_rows"],
            float(job["tolerance"]),
        )
        attribution_rows.append(attribution)
        counterfactual_rows.extend(counterfactuals)
    return attribution_rows, counterfactual_rows


def _write_parquet_atomic(
    path: Path, rows: list[dict[str, Any]], schema: pa.Schema, parquet: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        temporary,
        compression=str(parquet["compression"]),
        row_group_size=int(parquet["row_group_size"]),
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def _schema_hash(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
    return hashlib.sha256(str(schema).encode()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.as_posix(),
        "rows": parquet.metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema_sha256": _schema_hash(path),
    }


def _endpoint_status(
    count: Any,
    snapshot: Any,
    thermo_type: Any,
    thermo_id: Any,
    exclusions: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    count = 0 if pd.isna(count) else int(count)
    if count == 1:
        return "eligible"
    key = (str(snapshot), str(thermo_type), str(thermo_id))
    if key in exclusions:
        return f"compatibility_excluded:{exclusions[key]['reason']}"
    if count > 1:
        return "multiple_contextual_targets"
    return "missing_contextual_target"


def _ineligible_rows(
    transitions: pd.DataFrame,
    observed: pd.DataFrame,
    exclusions: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_transition: dict[str, dict[str, Any]] = {}

    def add(row: Any, source_status: str, target_status: str, reasons: set[str]) -> None:
        key = _hex(row.transition_id)
        assert key is not None
        prior = rows_by_transition.get(key)
        versions = {
            exclusions[item]["rule_version"]
            for item in (
                (
                    str(row.source_snapshot),
                    str(row.thermo_type),
                    str(row.source_thermo_id),
                ),
                (
                    str(row.target_snapshot),
                    str(row.thermo_type),
                    str(row.target_thermo_id),
                ),
            )
            if item in exclusions
        }
        if prior is None:
            rows_by_transition[key] = {
                "transition_id": key,
                "identity_edge_id": _hex(row.identity_edge_id),
                "canonical_lineage_id": str(row.canonical_lineage_id),
                "identity_confidence": str(row.identity_confidence),
                "source_snapshot": str(row.source_snapshot),
                "target_snapshot": str(row.target_snapshot),
                "source_material_id": str(row.source_material_id),
                "target_material_id": str(row.target_material_id),
                "thermo_type": _json_or_none(row.thermo_type),
                "observation_status": str(row.observation_status),
                "reported_label_flip": _json_or_none(row.label_flip),
                "source_thermo_id": _json_or_none(row.source_thermo_id),
                "target_thermo_id": _json_or_none(row.target_thermo_id),
                "source_endpoint_status": source_status,
                "target_endpoint_status": target_status,
                "_reasons": set(reasons),
                "_versions": set(versions),
            }
        else:
            prior["_reasons"].update(reasons)
            prior["_versions"].update(versions)

    for row in observed.itertuples(index=False):
        source_status = _endpoint_status(
            row.s_target_count,
            row.source_snapshot,
            row.thermo_type,
            row.source_thermo_id,
            exclusions,
        )
        target_status = _endpoint_status(
            row.q_target_count,
            row.target_snapshot,
            row.thermo_type,
            row.target_thermo_id,
            exclusions,
        )
        if source_status != "eligible" or target_status != "eligible":
            add(row, source_status, target_status, {source_status, target_status} - {"eligible"})

    exclusion_keys = set(exclusions)
    for row in transitions.itertuples(index=False):
        source_key = (
            str(row.source_snapshot),
            str(row.thermo_type),
            str(row.source_thermo_id),
        )
        target_key = (
            str(row.target_snapshot),
            str(row.thermo_type),
            str(row.target_thermo_id),
        )
        source_excluded = bool(row.source_state_present) and source_key in exclusion_keys
        target_excluded = bool(row.target_state_present) and target_key in exclusion_keys
        if not source_excluded and not target_excluded:
            continue
        add(
            row,
            (
                f"compatibility_excluded:{exclusions[source_key]['reason']}"
                if source_excluded
                else "not_applicable_or_eligible"
            ),
            (
                f"compatibility_excluded:{exclusions[target_key]['reason']}"
                if target_excluded
                else "not_applicable_or_eligible"
            ),
            {
                "compatibility_excluded_source" if source_excluded else "",
                "compatibility_excluded_target" if target_excluded else "",
            }
            - {""},
        )

    result = []
    for key in sorted(rows_by_transition):
        row = rows_by_transition[key]
        row["reasons_json"] = json.dumps(sorted(row.pop("_reasons")))
        row["compatibility_rule_versions_json"] = json.dumps(
            sorted(row.pop("_versions"))
        )
        result.append(row)
    return result


def data_dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "unified_entry_id": "P3.2 contextual foreign key; never replaced by entry_id-only joins.",
        "competitor_inventory_contribution": "Exact Shapley contribution from switching the retained competitor key set.",
        "uncorrected_energy_contribution": "Exact Shapley contribution from switching uncorrected energies for cross-snapshot common phase keys.",
        "compatibility_correction_contribution": "Exact Shapley contribution from switching compatibility corrections for common phase keys.",
        "candidate_identity_contribution": "Exact Shapley contribution from switching the selected candidate calculation/composition identity.",
        "reconstruction_residual": "Sum of four Shapley contributions minus target-minus-source rebuilt hull distance.",
        "source_decomposition_phase_keys_json": "P3.2 source decomposition components resolved through component_unified_entry_id within the exact phase context.",
        "target_decomposition_phase_keys_json": "P3.2 target decomposition components resolved through component_unified_entry_id within the exact phase context.",
        "coalition_mask": "Four-bit mask in PLAYERS order; all 16 coalitions are stored.",
    }
    rows = []
    for table_name, schema in (
        ("attribution", ATTRIBUTION_SCHEMA),
        ("counterfactual_value", COUNTERFACTUAL_SCHEMA),
    ):
        for field in schema:
            rows.append(
                {
                    "table": table_name,
                    "field": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "description": descriptions.get(
                        field.name,
                        "Deterministic transition attribution, provenance, gate, or audit field.",
                    ),
                }
            )
    return rows


def build_transition_attribution(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "P3.3":
        raise ValueError("Attribution config task_id must be P3.3")
    if tuple(config["counterfactual"]["players"]) != PLAYERS:
        raise ValueError("P3.3 player order is locked by the config and schema")
    started_at = utc_now()
    inputs = config["input"]
    output = config["output"]
    p3_1_report = json.loads(Path(inputs["p3_1_report"]).read_text(encoding="utf-8"))
    p3_2_report = json.loads(Path(inputs["p3_2_report"]).read_text(encoding="utf-8"))
    p3_1_manifest = json.loads(Path(inputs["p3_1_manifest"]).read_text(encoding="utf-8"))
    p3_2_manifest = json.loads(Path(inputs["p3_2_manifest"]).read_text(encoding="utf-8"))
    if not (
        p3_1_report.get("task_status") == "DONE"
        and p3_1_report.get("status") == "PASS"
        and p3_1_report.get("gate_decision") == "GO"
        and p3_2_report.get("task_status") == "DONE"
        and p3_2_report.get("status") == "PASS"
        and p3_2_report.get("gate_status") == "GO"
        and p3_1_manifest.get("gate_status") == "GO"
        and p3_2_manifest.get("gate_status") == "GO"
    ):
        raise RuntimeError("P3.1 and P3.2 DONE/PASS/GO prerequisites are required")

    transitions = pq.read_table(
        inputs["transition_labels"], columns=TRANSITION_COLUMNS
    ).to_pandas()
    phase_path = Path(inputs["phase_entries"])
    targets = pq.read_table(
        phase_path, filters=[("is_target", "=", True)], columns=PHASE_COLUMNS
    ).to_pandas()
    target_key = ["snapshot_id", "thermo_type", "thermo_id", "material_id"]
    counts = (
        targets.groupby(target_key, dropna=False, sort=False)
        .size()
        .rename("target_count")
        .reset_index()
    )
    unique_targets = targets.merge(counts, on=target_key, how="left")
    unique_targets = unique_targets[unique_targets["target_count"] == 1].drop(
        columns=["target_count"]
    )
    observed = transitions[transitions["observation_status"] == "observed"].copy()
    source_counts = counts.rename(
        columns={
            "snapshot_id": "source_snapshot",
            "thermo_id": "source_thermo_id",
            "material_id": "source_material_id",
            "target_count": "s_target_count",
        }
    )
    target_counts = counts.rename(
        columns={
            "snapshot_id": "target_snapshot",
            "thermo_id": "target_thermo_id",
            "material_id": "target_material_id",
            "target_count": "q_target_count",
        }
    )
    observed = observed.merge(
        source_counts,
        on=[
            "source_snapshot",
            "thermo_type",
            "source_thermo_id",
            "source_material_id",
        ],
        how="left",
        validate="many_to_one",
    )
    observed = observed.merge(
        target_counts,
        on=[
            "target_snapshot",
            "thermo_type",
            "target_thermo_id",
            "target_material_id",
        ],
        how="left",
        validate="many_to_one",
    )
    source_targets = unique_targets.add_prefix("s_")
    target_targets = unique_targets.add_prefix("q_")
    observed = observed.merge(
        source_targets,
        left_on=[
            "source_snapshot",
            "thermo_type",
            "source_thermo_id",
            "source_material_id",
        ],
        right_on=[
            "s_snapshot_id",
            "s_thermo_type",
            "s_thermo_id",
            "s_material_id",
        ],
        how="left",
        validate="many_to_one",
    )
    observed = observed.merge(
        target_targets,
        left_on=[
            "target_snapshot",
            "thermo_type",
            "target_thermo_id",
            "target_material_id",
        ],
        right_on=[
            "q_snapshot_id",
            "q_thermo_type",
            "q_thermo_id",
            "q_material_id",
        ],
        how="left",
        validate="many_to_one",
    )
    exclusions_frame = pd.read_csv(inputs["compatibility_exclusion_ledger"])
    exclusions = {
        (str(row.snapshot_id), str(row.thermo_type), str(row.thermo_id)): {
            "reason": str(row.reason),
            "rule_version": str(row.rule_version),
        }
        for row in exclusions_frame.itertuples(index=False)
    }
    ineligible_rows = _ineligible_rows(transitions, observed, exclusions)
    ineligible_path = Path(output["ineligible_transition_ledger"])
    if ineligible_rows:
        write_csv_atomic(ineligible_path, ineligible_rows)
    else:
        ineligible_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ineligible_path.with_suffix(ineligible_path.suffix + ".tmp")
        pd.DataFrame(columns=INELIGIBLE_COLUMNS).to_csv(temporary, index=False)
        os.replace(temporary, ineligible_path)

    eligible = observed[
        observed["s_unified_entry_id"].notna()
        & observed["q_unified_entry_id"].notna()
    ].copy()
    accepted_confidence = set(config["population"]["identity_confidences"])
    important = eligible[
        eligible["identity_confidence"].isin(accepted_confidence)
        & (eligible["s_is_stable"] != eligible["q_is_stable"])
    ].copy()
    cross_context = important[
        important["s_phase_context_chemsys"]
        != important["q_phase_context_chemsys"]
    ]
    if not cross_context.empty:
        raise RuntimeError(
            f"{len(cross_context)} important transitions cross exact phase contexts"
        )
    if important.empty:
        raise RuntimeError("No important high-confidence unified-hull flips found")

    decomposition_table = pq.read_table(
        inputs["phase_decompositions"],
        columns=["unified_entry_id", "component_unified_entry_id"],
    )
    important_endpoint_ids = pa.array(
        sorted(
            {
                bytes(value)
                for value in important["s_unified_entry_id"]
            }
            | {
                bytes(value)
                for value in important["q_unified_entry_id"]
            }
        ),
        type=pa.binary(16),
    )
    decomposition = decomposition_table.filter(
        pc.is_in(
            decomposition_table["unified_entry_id"],
            value_set=important_endpoint_ids,
        )
    ).to_pandas()
    decomposition_map = {
        bytes(unified_id): [bytes(value) for value in group["component_unified_entry_id"]]
        for unified_id, group in decomposition.groupby("unified_entry_id", sort=False)
    }
    important["s_decomposition_component_ids"] = important[
        "s_unified_entry_id"
    ].map(lambda value: decomposition_map.get(bytes(value), []))
    important["q_decomposition_component_ids"] = important[
        "q_unified_entry_id"
    ].map(lambda value: decomposition_map.get(bytes(value), []))
    missing_decomposition = important[
        important["s_decomposition_component_ids"].map(len).eq(0)
        | important["q_decomposition_component_ids"].map(len).eq(0)
    ]
    if not missing_decomposition.empty:
        raise RuntimeError(
            f"{len(missing_decomposition)} important transitions lack P3.2 decomposition rows"
        )

    group_columns = ["snapshot_id", "thermo_type", "phase_context_chemsys"]
    needed = set(
        zip(
            important["s_snapshot_id"],
            important["thermo_type"],
            important["s_phase_context_chemsys"],
            strict=True,
        )
    ) | set(
        zip(
            important["q_snapshot_id"],
            important["thermo_type"],
            important["q_phase_context_chemsys"],
            strict=True,
        )
    )
    phases = pq.read_table(phase_path, columns=PHASE_COLUMNS).to_pandas()
    phase_index = pd.MultiIndex.from_frame(phases[group_columns])
    needed_index = pd.MultiIndex.from_tuples(needed, names=group_columns)
    phases = phases[phase_index.isin(needed_index)].copy()
    phase_groups = {
        key: group[PHASE_COLUMNS].to_dict(orient="records")
        for key, group in phases.groupby(group_columns, sort=False)
    }

    jobs = []
    context_columns = [
        "source_snapshot",
        "target_snapshot",
        "thermo_type",
        "s_phase_context_chemsys",
    ]
    tolerance = float(config["counterfactual"]["absolute_tolerance_eV_per_atom"])
    for key, group in important.groupby(context_columns, sort=False):
        source_snapshot, target_snapshot, thermo_type, context = map(str, key)
        jobs.append(
            {
                "transitions": group.to_dict(orient="records"),
                "source_rows": phase_groups[
                    (source_snapshot, thermo_type, context)
                ],
                "target_rows": phase_groups[
                    (target_snapshot, thermo_type, context)
                ],
                "tolerance": tolerance,
            }
        )
    attribution_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    workers = min(int(config["execution"]["workers"]), len(jobs))
    if workers == 1:
        job_results = map(_attribute_context_job, jobs)
        for rows, counterfactuals in job_results:
            attribution_rows.extend(rows)
            counterfactual_rows.extend(counterfactuals)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_attribute_context_job, job) for job in jobs]
            for future in as_completed(futures):
                rows, counterfactuals = future.result()
                attribution_rows.extend(rows)
                counterfactual_rows.extend(counterfactuals)
    attribution_rows.sort(key=lambda row: bytes(row["transition_id"]))
    counterfactual_rows.sort(
        key=lambda row: (bytes(row["transition_id"]), int(row["coalition_mask"]))
    )
    attribution_path = Path(output["attribution"])
    counterfactual_path = Path(output["counterfactual_values"])
    _write_parquet_atomic(
        attribution_path, attribution_rows, ATTRIBUTION_SCHEMA, config["parquet"]
    )
    _write_parquet_atomic(
        counterfactual_path,
        counterfactual_rows,
        COUNTERFACTUAL_SCHEMA,
        config["parquet"],
    )

    attribution_frame = pd.DataFrame(attribution_rows)
    attributable_count = int(attribution_frame["attributable"].sum())
    coverage = attributable_count / len(attribution_frame)
    max_residual = float(attribution_frame["reconstruction_residual"].abs().max())
    max_endpoint_error = float(
        attribution_frame[
            [
                "source_endpoint_reconstruction_error",
                "target_endpoint_reconstruction_error",
            ]
        ].max().max()
    )
    sample_limit = int(config["audit"]["sample_per_snapshot_pair_thermo_type"])
    audit_frame = attribution_frame.copy()
    audit_frame["_rank"] = audit_frame.apply(
        lambda row: hashlib.sha256(
            f"{config['seed']}|{bytes(row['transition_id']).hex()}".encode()
        ).hexdigest(),
        axis=1,
    )
    audit_frame = (
        audit_frame.sort_values("_rank", kind="mergesort")
        .groupby(["source_snapshot", "target_snapshot", "thermo_type"], sort=True)
        .head(sample_limit)
        .drop(columns=["_rank"])
    )
    for column in ("attribution_id", "transition_id", "identity_edge_id", "source_unified_entry_id", "target_unified_entry_id"):
        audit_frame[column] = audit_frame[column].map(lambda value: bytes(value).hex())
    audit_columns = [
        "attribution_id",
        "transition_id",
        "source_snapshot",
        "target_snapshot",
        "thermo_type",
        "phase_context_chemsys",
        "source_unified_entry_id",
        "target_unified_entry_id",
        "delta_energy_above_hull",
        "competitor_inventory_contribution",
        "uncorrected_energy_contribution",
        "compatibility_correction_contribution",
        "candidate_identity_contribution",
        "reconstruction_residual",
        "source_endpoint_reconstruction_error",
        "target_endpoint_reconstruction_error",
        "dominant_channel",
        "attributable",
    ]
    audit_csv = Path(output["reconstruction_audit_csv"])
    write_csv_atomic(audit_csv, audit_frame[audit_columns].to_dict(orient="records"))
    dominant_counts = Counter(attribution_frame["dominant_channel"])
    audit = {
        "task_id": "P3.3",
        "status": "PASS" if coverage == 1.0 and max_residual <= tolerance and max_endpoint_error <= tolerance else "FAIL",
        "method_version": METHOD_VERSION,
        "players": list(PLAYERS),
        "important_high_confidence_flips": len(attribution_rows),
        "attributable_transitions": attributable_count,
        "attribution_coverage": coverage,
        "minimum_attribution_coverage": float(config["gate"]["minimum_attribution_fraction"]),
        "max_absolute_reconstruction_residual_eV_per_atom": max_residual,
        "max_endpoint_reconstruction_error_eV_per_atom": max_endpoint_error,
        "absolute_tolerance_eV_per_atom": tolerance,
        "dominant_channel_counts": dict(sorted(dominant_counts.items())),
        "sample_rows": len(audit_frame),
        "interpretation": (
            "Exact Shapley values are order-averaged accounting contributions "
            "under the stored phase-context counterfactual game; they are not "
            "causal effects on physical stability."
        ),
    }
    audit_json = Path(output["reconstruction_audit_json"])
    write_json_atomic(audit_json, audit)
    dictionary_path = Path(output["data_dictionary"])
    write_csv_atomic(dictionary_path, data_dictionary_rows())

    minimum_flips = int(config["gate"]["minimum_important_high_confidence_flips"])
    minimum_coverage = float(config["gate"]["minimum_attribution_fraction"])
    gate_checks = {
        "important_high_confidence_flips": len(attribution_rows) >= minimum_flips,
        "attribution_fraction": coverage >= minimum_coverage,
        "all_shapley_sums_reconstruct_delta": max_residual <= tolerance,
        "all_counterfactual_endpoints_match_p3_2": max_endpoint_error <= tolerance,
        "all_important_transitions_attributable": coverage == 1.0,
        "all_coalitions_present": len(counterfactual_rows) == 16 * len(attribution_rows),
        "no_cross_context_attribution": cross_context.empty,
    }
    gate_passed = all(gate_checks.values())
    manifest = {
        "task_id": "P3.3",
        "status": "PASS" if gate_passed else "FAIL",
        "gate_status": "GO" if gate_passed else "NO-GO",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "input_hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                inputs["transition_labels"],
                inputs["p3_1_manifest"],
                inputs["p3_1_report"],
                inputs["phase_entries"],
                inputs["phase_decompositions"],
                inputs["p3_2_manifest"],
                inputs["p3_2_report"],
                inputs["compatibility_exclusion_ledger"],
            )
        },
        "population": {
            "transition_rows": len(transitions),
            "observed_transition_rows": len(observed),
            "eligible_endpoint_transition_rows": len(eligible),
            "ineligible_ledger_rows": len(ineligible_rows),
            "important_high_confidence_flips": len(attribution_rows),
            "reported_and_unified_flips": int(
                attribution_frame["reported_label_flip"].sum()
            ),
            "unified_only_flips": int(
                (~attribution_frame["reported_label_flip"]).sum()
            ),
        },
        "attribution": {
            "method_version": METHOD_VERSION,
            "players": list(PLAYERS),
            "rows": len(attribution_rows),
            "counterfactual_rows": len(counterfactual_rows),
            "attributable_rows": attributable_count,
            "coverage": coverage,
            "max_absolute_reconstruction_residual_eV_per_atom": max_residual,
            "max_endpoint_reconstruction_error_eV_per_atom": max_endpoint_error,
            "dominant_channel_counts": dict(sorted(dominant_counts.items())),
            "candidate_identity_changed_rows": int(
                attribution_frame["candidate_identity_changed"].sum()
            ),
            "candidate_composition_changed_rows": int(
                attribution_frame["candidate_composition_changed"].sum()
            ),
        },
        "outputs": [_artifact(attribution_path), _artifact(counterfactual_path)],
        "supporting_files": {
            str(path): sha256_file(path)
            for path in (
                ineligible_path,
                audit_csv,
                audit_json,
                dictionary_path,
            )
        },
        "gate": {
            "minimum_important_high_confidence_flips": minimum_flips,
            "minimum_attribution_fraction": minimum_coverage,
            "absolute_tolerance_eV_per_atom": tolerance,
            "checks": gate_checks,
            "passed": gate_passed,
        },
        "seed": int(config["seed"]),
        "network_access": False,
        "warnings": [
            "Shapley attribution is a deterministic accounting decomposition, not a causal physical claim.",
            "Transitions with compatibility-excluded or non-unique endpoints remain in the ineligible ledger.",
        ],
    }
    manifest_path = Path(output["manifest"])
    write_json_atomic(manifest_path, manifest)
    if not gate_passed:
        raise RuntimeError(f"P3.3 attribution gate failed: {gate_checks}")
    return manifest


def verify_transition_attribution(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = config["output"]
    manifest = json.loads(Path(output["manifest"]).read_text(encoding="utf-8"))
    failures: list[str] = []
    schemas = {
        str(Path(output["attribution"])): ATTRIBUTION_SCHEMA,
        str(Path(output["counterfactual_values"])): COUNTERFACTUAL_SCHEMA,
    }
    for artifact in manifest["outputs"]:
        path = Path(artifact["path"])
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"hash_or_missing:{path}")
            continue
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != artifact["rows"]:
            failures.append(f"rows:{path}")
        if parquet.schema_arrow.remove_metadata() != schemas[str(path)].remove_metadata():
            failures.append(f"schema:{path}")
    for path_text, expected in manifest["supporting_files"].items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected:
            failures.append(f"supporting:{path}")

    attribution = pq.read_table(output["attribution"])
    counterfactuals = pq.read_table(output["counterfactual_values"])
    if pc.count_distinct(attribution["attribution_id"]).as_py() != attribution.num_rows:
        failures.append("duplicate_attribution_id")
    if pc.count_distinct(attribution["transition_id"]).as_py() != attribution.num_rows:
        failures.append("duplicate_transition_id")
    frame = attribution.to_pandas()
    tolerance = float(config["counterfactual"]["absolute_tolerance_eV_per_atom"])
    if frame["reconstruction_residual"].abs().max() > tolerance:
        failures.append("shapley_reconstruction")
    if frame[
        [
            "source_endpoint_reconstruction_error",
            "target_endpoint_reconstruction_error",
        ]
    ].max().max() > tolerance:
        failures.append("endpoint_reconstruction")
    if not frame["attributable"].all():
        failures.append("unattributable_rows")
    endpoint_ids = pa.array(
        sorted(
            {bytes(value) for value in frame["source_unified_entry_id"]}
            | {bytes(value) for value in frame["target_unified_entry_id"]}
        ),
        type=pa.binary(16),
    )
    decomposition_table = pq.read_table(
        config["input"]["phase_decompositions"], columns=["unified_entry_id"]
    )
    decomposition = decomposition_table.filter(
        pc.is_in(decomposition_table["unified_entry_id"], value_set=endpoint_ids)
    ).to_pandas()
    decomposition_counts = decomposition["unified_entry_id"].map(bytes).value_counts()
    observed_source_counts = frame["source_unified_entry_id"].map(
        lambda value: int(decomposition_counts.get(bytes(value), 0))
    )
    observed_target_counts = frame["target_unified_entry_id"].map(
        lambda value: int(decomposition_counts.get(bytes(value), 0))
    )
    if not (
        observed_source_counts.eq(frame["source_decomposition_component_count"]).all()
        and observed_target_counts.eq(frame["target_decomposition_component_count"]).all()
    ):
        failures.append("decomposition_context_join")
    counter = counterfactuals.to_pandas()
    counts = counter.groupby("transition_id", sort=False)["coalition_mask"].agg(
        ["size", "nunique", "min", "max"]
    )
    if not (
        (counts["size"] == 16).all()
        and (counts["nunique"] == 16).all()
        and (counts["min"] == 0).all()
        and (counts["max"] == 15).all()
    ):
        failures.append("coalition_coverage")
    if counterfactuals.num_rows != 16 * attribution.num_rows:
        failures.append("counterfactual_row_count")
    if not manifest.get("gate", {}).get("passed"):
        failures.append("manifest_gate")
    temporary_files = [
        path
        for root in (Path(output["root"]), Path(output["manifest"]).parent, Path(output["reconstruction_audit_json"]).parent)
        for path in root.rglob("*.tmp")
    ]
    if temporary_files:
        failures.append("temporary_files")
    if failures:
        raise RuntimeError(f"P3.3 verification failed: {failures}")
    return {
        "task_id": "P3.3",
        "status": "PASS",
        "gate_status": "GO",
        "failures": [],
        "attribution_rows": attribution.num_rows,
        "counterfactual_rows": counterfactuals.num_rows,
        "attribution_coverage": float(frame["attributable"].mean()),
        "max_absolute_reconstruction_residual_eV_per_atom": float(
            frame["reconstruction_residual"].abs().max()
        ),
        "max_endpoint_reconstruction_error_eV_per_atom": float(
            frame[
                [
                    "source_endpoint_reconstruction_error",
                    "target_endpoint_reconstruction_error",
                ]
            ].max().max()
        ),
        "temporary_files": 0,
        "network_access": False,
        "verified_at_utc": utc_now(),
    }
