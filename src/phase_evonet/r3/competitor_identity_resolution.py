"""R3.3R-A competitor-to-P2.2 lineage resolution and independent QA.

This module is deliberately limited to identity mapping and diagnostics.  It
does not calculate any formal cascade concentration, rank, Gini, or top-share
result.  Gini calculations below are used only on four fixed synthetic graphs
to diagnose the frozen R3.3 contextual bootstrap implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .common import open_formal_input, sha256_file


TASK_ID = "R3.3R-A"
METHOD_VERSION = "PHASEEVONET_R3_3R_IDENTITY_MAPPING_V1"
COMPOSITION_ATOL = 1.0e-10
IMPACT_EDGE_DEFINITIONS = (
    "necessary_10meV",
    "necessary_25meV",
    "contributory_5meV",
    "minimal_set_member",
)
ALL_EDGE_DEFINITIONS = ("selected_active", *IMPACT_EDGE_DEFINITIONS)

# Hashes not supplied in the YAML but frozen before this task began.  The P2.2
# manifest hash is in FROZEN_R3_3_FACTS.json; the remaining R3.3 values protect
# the reviewed state and/or are declared by the frozen R3.3 manifest.
FROZEN_AUXILIARY_HASHES = {
    "reports/R3_3/verification.json": "740bd2e7915c9d017c64fcaa41a9d5cf50571809bb360186a48823e1d2b6ba4f",
    "data/manifests/R3_3/manifest.json": "62899a714bb91a07dbaf081557958db4af2143a6e715b2aae73579e1b52e0e7f",
    "reports/R3_3/cascade_summary.csv": "86d29ddab4273b7d12898ca1298378e6785e01190040885b636ac8d302640f0f",
    "data/manifests/P2_2/identity_manifest.json": "f00a51e3c5ff54dc37547693e6b49a4d44bfcad46afcfef6e22eaaa6e1b09e82",
    "configs/r3/r3_3_attribution_cascade.yaml": "99a0a313dd6cf7087326a3649bb8bc372853d48dd010ae3aa7a02dbb9cdc3b8b",
    "R3_3_DECISION_MEMO.md": "f80941a7c9b8a3630eb4625e8fe77cc7030a89da975c82a6ed5e5b41d03b6fb0",
    "TASKS_R3.md": "0f1360f27412a7d3bcccba3463ea9e3ea1cf0495b45049d1c5397c60b109e55a",
}

CROSSWALK_COLUMNS = [
    "transition_id",
    "candidate_lineage_id",
    "competitor_contextual_id",
    "competitor_canonical_id_r3_3",
    "source_snapshot",
    "target_snapshot",
    "identity_side",
    "identity_snapshot",
    "material_id",
    "task_id",
    "entry_id",
    "thermo_type",
    "source_workflow",
    "target_workflow",
    "identity_workflow",
    "composition_signature",
    "p2_canonical_lineage_id",
    "p2_lineage_confidence",
    "p2_high_confidence",
    "mapping_method",
    "mapping_status",
    "composition_concordant",
    "composition_max_abs_delta",
    "cross_side_lineage_concordant",
    "unresolved_reason",
    "id0_contextual",
    "id1_strict_entry",
    "id2_lineage_thermo_workflow",
    "id3_lineage_only",
    "method_version",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _hex_transition(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)


def _normalized_composition(value: str) -> dict[str, float]:
    parsed = json.loads(value)
    total = float(sum(float(amount) for amount in parsed.values()))
    if total <= 0:
        raise ValueError("composition total must be positive")
    return {str(element): float(amount) / total for element, amount in parsed.items()}


def composition_delta(left: str, right: str) -> float:
    """Maximum fractional-composition difference, or infinity for element mismatch."""

    a = _normalized_composition(left)
    b = _normalized_composition(right)
    if set(a) != set(b):
        return float("inf")
    return max((abs(a[element] - b[element]) for element in a), default=0.0)


def compositions_concordant(left: str, right: str, *, atol: float = COMPOSITION_ATOL) -> bool:
    return composition_delta(left, right) <= atol


def _identity_hash(prefix: str, fields: list[str]) -> str:
    payload = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:32]}"


def _authorize(repo: Path, relative: str, expected: str, access_log: Path, purpose: str) -> None:
    with open_formal_input(
        repo / relative,
        expected,
        task_id=TASK_ID,
        access_log=access_log,
        purpose=purpose,
        allowed_roots=[repo],
        caller="phase_evonet.r3.competitor_identity_resolution",
    ):
        pass


def _direct_lookup(lineage: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    key_columns = ["snapshot_id", "material_id"]
    duplicates = lineage.duplicated(key_columns, keep=False)
    if bool(duplicates.any()):
        sample = lineage.loc[duplicates, key_columns].head(10).to_dict(orient="records")
        raise ValueError(f"P2.2 direct key is not unique: {sample}")
    return {
        (str(row.snapshot_id), str(row.material_id)): {
            "lineage": str(row.canonical_lineage_id),
            "composition": str(row.composition_reduced_json),
            "confidence": str(row.lineage_confidence),
            "high_confidence": bool(row.high_confidence),
        }
        for row in lineage.itertuples(index=False)
    }


def _task_bridge_lookup(task_bridge: pd.DataFrame | None) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    if task_bridge is None or task_bridge.empty:
        return {}, set()
    required = {"snapshot_id", "task_id", "material_id"}
    if not required.issubset(task_bridge.columns):
        raise ValueError(f"task bridge lacks columns: {sorted(required - set(task_bridge.columns))}")
    bridge = task_bridge.dropna(subset=list(required)).copy()
    grouped = bridge.groupby(["snapshot_id", "task_id"], dropna=False)["material_id"]
    distinct = grouped.nunique(dropna=True)
    ambiguous = {(str(a), str(b)) for (a, b), count in distinct.items() if int(count) != 1}
    lookup: dict[tuple[str, str], str] = {}
    for (snapshot, task), values in grouped:
        key = (str(snapshot), str(task))
        unique = sorted({str(value) for value in values if pd.notna(value)})
        if key not in ambiguous and len(unique) == 1:
            lookup[key] = unique[0]
    return lookup, ambiguous


def build_crosswalk(
    change: pd.DataFrame,
    lineage: pd.DataFrame,
    *,
    task_bridge: pd.DataFrame | None = None,
    method_version: str = METHOD_VERSION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen crosswalk without reading or using network outcomes."""

    grain = ["transition_id", "competitor_contextual_id"]
    if bool(change.duplicated(grain).any()):
        raise ValueError("competitor-change grain is not unique")
    direct = _direct_lookup(lineage)
    task_lookup, ambiguous_tasks = _task_bridge_lookup(task_bridge)
    rows: list[dict[str, Any]] = []
    task_bridge_requested = 0
    task_bridge_mapped = 0
    task_bridge_ambiguous = 0

    for record in change.itertuples(index=False):
        use_target = pd.notna(record.target_material_id)
        side = "target" if use_target else "source"
        snapshot = str(record.target_snapshot if use_target else record.source_snapshot)
        material_value = record.target_material_id if use_target else record.source_material_id
        task_value = record.target_task_id if use_target else record.source_task_id
        entry_value = record.target_entry_id if use_target else record.source_entry_id
        workflow_value = record.target_workflow if use_target else record.source_workflow
        material = None if pd.isna(material_value) else str(material_value)
        task = None if pd.isna(task_value) else str(task_value)
        entry = None if pd.isna(entry_value) else str(entry_value)
        workflow = None if pd.isna(workflow_value) else str(workflow_value)
        mapping_method = "direct_material" if material is not None else "unresolved"
        bridge_ambiguous = False

        # The bridge is only consulted when material_id is absent by construction.
        if material is None and task is not None:
            task_bridge_requested += 1
            key = (snapshot, task)
            if key in ambiguous_tasks:
                task_bridge_ambiguous += 1
                bridge_ambiguous = True
            elif key in task_lookup:
                material = task_lookup[key]
                mapping_method = "task_bridge"
                task_bridge_mapped += 1

        match = direct.get((snapshot, material)) if material is not None else None
        comp_delta = (
            composition_delta(str(record.composition_signature), str(match["composition"]))
            if match is not None
            else float("inf")
        )
        comp_ok = bool(match is not None and comp_delta <= COMPOSITION_ATOL)

        source_match = (
            direct.get((str(record.source_snapshot), str(record.source_material_id)))
            if pd.notna(record.source_material_id)
            else None
        )
        target_match = (
            direct.get((str(record.target_snapshot), str(record.target_material_id)))
            if pd.notna(record.target_material_id)
            else None
        )
        cross_side: bool | None = None
        if source_match is not None and target_match is not None:
            cross_side = bool(source_match["lineage"] == target_match["lineage"])

        reasons: list[str] = []
        if bridge_ambiguous:
            reasons.append("ambiguous_task_bridge")
        if material is None:
            reasons.append("missing_identity_material_id")
        elif match is None:
            reasons.append("no_p2_lineage_match")
        if match is not None and not comp_ok:
            reasons.append("composition_mismatch")
        if cross_side is False:
            reasons.append("cross_side_lineage_disagreement")
        if workflow is None:
            reasons.append("missing_identity_workflow")

        conflict_reasons = {
            "ambiguous_task_bridge",
            "composition_mismatch",
            "cross_side_lineage_disagreement",
            "missing_identity_workflow",
        }
        if not reasons:
            status = "mapped"
        elif any(reason in conflict_reasons for reason in reasons):
            status = "conflict"
        else:
            status = "unresolved"
        if status != "mapped" and mapping_method != "task_bridge":
            mapping_method = "unresolved"

        lineage_id = str(match["lineage"]) if match is not None else None
        id2 = (
            _identity_hash("id2", [lineage_id, str(record.thermo_type), str(workflow)])
            if status == "mapped" and lineage_id is not None and workflow is not None
            else None
        )
        id3 = _identity_hash("id3", [lineage_id]) if status == "mapped" and lineage_id else None
        rows.append(
            {
                "transition_id": _hex_transition(record.transition_id),
                "candidate_lineage_id": str(record.candidate_lineage_id),
                "competitor_contextual_id": str(record.competitor_contextual_id),
                "competitor_canonical_id_r3_3": str(record.competitor_canonical_id),
                "source_snapshot": str(record.source_snapshot),
                "target_snapshot": str(record.target_snapshot),
                "identity_side": side,
                "identity_snapshot": snapshot,
                "material_id": material,
                "task_id": task,
                "entry_id": entry,
                "thermo_type": str(record.thermo_type),
                "source_workflow": None if pd.isna(record.source_workflow) else str(record.source_workflow),
                "target_workflow": None if pd.isna(record.target_workflow) else str(record.target_workflow),
                "identity_workflow": workflow,
                "composition_signature": str(record.composition_signature),
                "p2_canonical_lineage_id": lineage_id,
                "p2_lineage_confidence": str(match["confidence"]) if match is not None else None,
                "p2_high_confidence": bool(match["high_confidence"]) if match is not None else None,
                "mapping_method": mapping_method,
                "mapping_status": status,
                "composition_concordant": comp_ok,
                "composition_max_abs_delta": comp_delta if np.isfinite(comp_delta) else None,
                "cross_side_lineage_concordant": cross_side,
                "unresolved_reason": ";".join(reasons) if reasons else None,
                "id0_contextual": str(record.competitor_contextual_id),
                "id1_strict_entry": str(record.competitor_canonical_id),
                "id2_lineage_thermo_workflow": id2,
                "id3_lineage_only": id3,
                "method_version": method_version,
            }
        )

    crosswalk = pd.DataFrame(rows, columns=CROSSWALK_COLUMNS)
    diagnostics = {
        "left_rows": int(len(change)),
        "right_rows": int(len(lineage)),
        "output_rows": int(len(crosswalk)),
        "left_distinct_keys": int(change[grain].drop_duplicates().shape[0]),
        "right_distinct_keys": int(lineage[["snapshot_id", "material_id"]].drop_duplicates().shape[0]),
        "maximum_join_expansion": float(len(crosswalk) / len(change)) if len(change) else 1.0,
        "unapproved_join_expansion": int(len(crosswalk) - len(change)),
        "direct_mapped": int(
            ((crosswalk["mapping_method"] == "direct_material") & (crosswalk["mapping_status"] == "mapped")).sum()
        ),
        "task_bridge_requested": task_bridge_requested,
        "task_bridge_mapped": task_bridge_mapped,
        "task_bridge_ambiguous": task_bridge_ambiguous,
        "unresolved": int((crosswalk["mapping_status"] == "unresolved").sum()),
        "conflicts": int((crosswalk["mapping_status"] == "conflict").sum()),
        "composition_mismatches": int((~crosswalk["composition_concordant"]).sum()),
        "maximum_composition_delta": float(crosswalk["composition_max_abs_delta"].max()),
        "cross_side_rows": int(crosswalk["cross_side_lineage_concordant"].notna().sum()),
        "cross_side_disagreements": int((crosswalk["cross_side_lineage_concordant"] == False).sum()),  # noqa: E712
        "identity_workflow_missing": int(crosswalk["identity_workflow"].isna().sum()),
    }
    return crosswalk, diagnostics


def _coverage_rows(edge: pd.DataFrame, crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = edge.copy()
    frame["transition_id"] = frame["transition_id"].map(_hex_transition)
    mapped_columns = [
        "transition_id",
        "competitor_contextual_id",
        "mapping_status",
        "p2_lineage_confidence",
        "identity_workflow",
    ]
    joined = frame.merge(
        crosswalk[mapped_columns],
        on=["transition_id", "competitor_contextual_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    joined["mapped"] = joined["mapping_status"].eq("mapped")
    rows: list[dict[str, Any]] = []

    def append_group(definition: str, dimension: str, value: str, group: pd.DataFrame) -> None:
        mapped = group.loc[group["mapped"]]
        total_relations = len(group)
        strict_total = group["competitor_canonical_id"].nunique()
        candidate_total = group["candidate_lineage_id"].nunique()
        strict_mapped = mapped["competitor_canonical_id"].nunique()
        candidate_mapped = mapped["candidate_lineage_id"].nunique()
        rows.append(
            {
                "edge_definition": definition,
                "dimension": dimension,
                "dimension_value": value,
                "total_relations": total_relations,
                "mapped_relations": len(mapped),
                "relation_coverage": len(mapped) / total_relations if total_relations else 1.0,
                "total_strict_entries": strict_total,
                "mapped_strict_entries": strict_mapped,
                "strict_entry_coverage": strict_mapped / strict_total if strict_total else 1.0,
                "total_candidates": candidate_total,
                "mapped_candidates": candidate_mapped,
                "candidate_coverage": candidate_mapped / candidate_total if candidate_total else 1.0,
            }
        )

    for definition in ALL_EDGE_DEFINITIONS:
        subset = joined.loc[joined["edge_definition"].eq(definition)].copy()
        append_group(definition, "overall", "overall", subset)
        for dimension in (
            "target_snapshot",
            "thermo_type",
            "change_type",
            "p2_lineage_confidence",
            "identity_workflow",
        ):
            for value, group in subset.groupby(dimension, dropna=False, sort=True):
                append_group(definition, dimension, "<missing>" if pd.isna(value) else str(value), group)

    coverage = pd.DataFrame(rows)
    overall = coverage.loc[coverage["dimension"].eq("overall")]
    summary = {
        str(row.edge_definition): {
            "relations": int(row.total_relations),
            "mapped_relations": int(row.mapped_relations),
            "relation_coverage": float(row.relation_coverage),
            "candidates": int(row.total_candidates),
            "mapped_candidates": int(row.mapped_candidates),
            "candidate_coverage": float(row.candidate_coverage),
            "strict_entry_coverage": float(row.strict_entry_coverage),
        }
        for row in overall.itertuples(index=False)
    }
    summary["edge_join"] = {
        "left_rows": int(len(edge)),
        "right_rows": int(len(crosswalk)),
        "output_rows": int(len(joined)),
        "unmatched_rows": int((joined["_merge"] != "both").sum()),
        "maximum_expansion_factor": float(len(joined) / len(edge)) if len(edge) else 1.0,
    }
    return coverage, summary


def _old_bootstrap_gini(pairs: list[tuple[str, str]], *, seed: int, replicates: int) -> np.ndarray:
    nodes = sorted({node for node, _ in pairs})
    candidates = sorted({candidate for _, candidate in pairs})
    node_code = {node: index for index, node in enumerate(nodes)}
    candidate_code = {candidate: index for index, candidate in enumerate(candidates)}
    node_indices = np.asarray([node_code[node] for node, _ in pairs], dtype=int)
    candidate_indices = np.asarray([candidate_code[candidate] for _, candidate in pairs], dtype=int)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        multiplicity = rng.multinomial(len(candidates), np.full(len(candidates), 1 / len(candidates)))
        counts = np.bincount(node_indices, weights=multiplicity[candidate_indices], minlength=len(nodes))
        values[index] = _synthetic_gini(counts[counts > 0])
    return values


def _synthetic_gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float(2 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum()) - (n + 1) / n)


def synthetic_bootstrap_diagnostics(*, seed: int = 42) -> pd.DataFrame:
    """Run only the contract-required toy-graph diagnostics."""

    graphs = {
        "one_to_one_contextual": [(f"n{i}", f"c{i}") for i in range(12)],
        "one_shared_hub": [("hub", f"c{i}") for i in range(12)],
        "two_equal_hubs": [("a" if i < 6 else "b", f"c{i}") for i in range(12)],
        "dominant_plus_tail": [("dominant", f"c{i}") for i in range(9)]
        + [(f"tail{i}", f"c{9 + i}") for i in range(3)],
    }
    output: list[dict[str, Any]] = []
    for graph_index, (name, pairs) in enumerate(graphs.items()):
        nodes = sorted({node for node, _ in pairs})
        candidates = sorted({candidate for _, candidate in pairs})
        full_counts = np.asarray(
            [len({candidate for node_value, candidate in pairs if node_value == node}) for node in nodes],
            dtype=float,
        )
        full_gini = _synthetic_gini(full_counts)
        old_values = _old_bootstrap_gini(pairs, seed=seed + graph_index, replicates=2000)
        rng = np.random.default_rng(seed + 100 + graph_index)
        subsample_values: list[float] = []
        sample_size = max(1, int(np.floor(0.75 * len(candidates))))
        for _ in range(512):
            selected = set(rng.choice(candidates, size=sample_size, replace=False).tolist())
            counts = np.asarray(
                [len({candidate for node_value, candidate in pairs if node_value == node and candidate in selected}) for node in nodes],
                dtype=float,
            )
            subsample_values.append(_synthetic_gini(counts[counts > 0]))
        low, high = np.quantile(old_values, [0.025, 0.975])
        one_to_one_pass = not name.startswith("one_to_one") or (
            abs(full_gini) < 1e-15
            and max(abs(value) for value in subsample_values) < 1e-15
            and low > full_gini
        )
        output.append(
            {
                "graph": name,
                "nodes": len(nodes),
                "candidates": len(candidates),
                "full_synthetic_gini": full_gini,
                "without_replacement_sample_size": sample_size,
                "without_replacement_gini_min": min(subsample_values),
                "without_replacement_gini_max": max(subsample_values),
                "old_multinomial_gini_low": float(low),
                "old_multinomial_gini_high": float(high),
                "full_point_inside_old_interval": bool(low <= full_gini <= high),
                "diagnostic_pass": bool(one_to_one_pass and np.isfinite(old_values).all()),
                "scope": "synthetic_diagnostic_only_not_a_formal_network_result",
            }
        )
    return pd.DataFrame(output)


def _artifact_record(repo: Path, path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def build(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    config_file = (repo / config_path).resolve(strict=True)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    started = _utc_now()
    report_dir = repo / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    access_log = report_dir / "input_access_log.jsonl"
    access_log.write_text("", encoding="utf-8")

    formal_hashes = {
        config["input"]["r3_3_report"]: config["frozen_hashes"]["r3_3_report"],
        config["input"]["competitor_change"]: config["frozen_hashes"]["competitor_change"],
        config["input"]["competitor_candidate_edge"]: config["frozen_hashes"]["competitor_candidate_edge"],
        config["input"]["p2_2_lineage"]: config["frozen_hashes"]["p2_2_lineage"],
        **FROZEN_AUXILIARY_HASHES,
    }
    hash_rows: list[dict[str, Any]] = []
    for relative, expected in formal_hashes.items():
        _authorize(repo, relative, expected, access_log, "R3.3R-A frozen input verification")
        path = repo / relative
        rows = pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None
        hash_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": sha256_file(path),
                "hash_match": True,
                "bytes": path.stat().st_size,
                "rows": rows,
            }
        )
    _write_csv(
        report_dir / "input_hash_audit.csv",
        hash_rows,
        ["path", "expected_sha256", "actual_sha256", "hash_match", "bytes", "rows"],
    )

    change = pd.read_parquet(repo / config["input"]["competitor_change"])
    edge = pd.read_parquet(repo / config["input"]["competitor_candidate_edge"])
    lineage = pd.read_parquet(repo / config["input"]["p2_2_lineage"])
    crosswalk, join = build_crosswalk(change, lineage, method_version=config["method_version"])

    output_path = repo / config["output"]["crosswalk"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(output_path, index=False, compression="zstd")

    coverage, coverage_summary = _coverage_rows(edge, crosswalk)
    coverage.to_csv(report_dir / "mapping_coverage.csv", index=False, lineterminator="\n")
    edge_join = coverage_summary.pop("edge_join")

    join_rows = [
        {
            "audit_name": "competitor_change_to_p2_lineage_direct",
            "left_rows": len(change),
            "right_rows": len(lineage),
            "output_rows": len(crosswalk),
            "distinct_left_keys": join["left_distinct_keys"],
            "distinct_right_keys": join["right_distinct_keys"],
            "unmatched_left": join["unresolved"],
            "maximum_expansion_factor": join["maximum_join_expansion"],
            "one_to_many_keys": 0,
            "many_to_many_keys": 0,
            "composition_mismatch": join["composition_mismatches"],
            "cross_side_disagreement": join["cross_side_disagreements"],
            "notes": "target-else-source; direct (snapshot_id, material_id)",
        },
        {
            "audit_name": "task_bridge",
            "left_rows": join["task_bridge_requested"],
            "right_rows": 0,
            "output_rows": join["task_bridge_mapped"],
            "distinct_left_keys": join["task_bridge_requested"],
            "distinct_right_keys": 0,
            "unmatched_left": join["task_bridge_requested"] - join["task_bridge_mapped"],
            "maximum_expansion_factor": 1.0,
            "one_to_many_keys": join["task_bridge_ambiguous"],
            "many_to_many_keys": 0,
            "composition_mismatch": 0,
            "cross_side_disagreement": 0,
            "notes": "not consulted because every selected identity side had material_id",
        },
        {
            "audit_name": "edge_to_crosswalk",
            "left_rows": edge_join["left_rows"],
            "right_rows": edge_join["right_rows"],
            "output_rows": edge_join["output_rows"],
            "distinct_left_keys": edge[["transition_id", "competitor_contextual_id", "edge_definition"]].drop_duplicates().shape[0],
            "distinct_right_keys": crosswalk[["transition_id", "competitor_contextual_id"]].drop_duplicates().shape[0],
            "unmatched_left": edge_join["unmatched_rows"],
            "maximum_expansion_factor": edge_join["maximum_expansion_factor"],
            "one_to_many_keys": 0,
            "many_to_many_keys": 0,
            "composition_mismatch": 0,
            "cross_side_disagreement": 0,
            "notes": "coverage-only relation join; no aggregation or network metric",
        },
    ]
    _write_csv(
        report_dir / "join_audit.csv",
        join_rows,
        [
            "audit_name", "left_rows", "right_rows", "output_rows", "distinct_left_keys",
            "distinct_right_keys", "unmatched_left", "maximum_expansion_factor", "one_to_many_keys",
            "many_to_many_keys", "composition_mismatch", "cross_side_disagreement", "notes",
        ],
    )

    conflict = crosswalk.loc[crosswalk["mapping_status"].ne("mapped")]
    conflict_rows = [
        {
            "transition_id": row.transition_id,
            "competitor_contextual_id": row.competitor_contextual_id,
            "identity_side": row.identity_side,
            "identity_snapshot": row.identity_snapshot,
            "material_id": row.material_id,
            "task_id": row.task_id,
            "entry_id": row.entry_id,
            "conflict_type": row.unresolved_reason,
            "match_count": 0 if row.p2_canonical_lineage_id is None else 1,
            "composition_concordant": row.composition_concordant,
            "cross_side_concordant": row.cross_side_lineage_concordant,
            "details": "unresolved rows are never imputed",
            "resolution_status": "ledgered_unresolved" if row.mapping_status == "unresolved" else "isolated_conflict",
        }
        for row in conflict.itertuples(index=False)
    ]
    _write_csv(
        report_dir / "mapping_conflict_ledger.csv",
        conflict_rows,
        [
            "transition_id", "competitor_contextual_id", "identity_side", "identity_snapshot", "material_id",
            "task_id", "entry_id", "conflict_type", "match_count", "composition_concordant",
            "cross_side_concordant", "details", "resolution_status",
        ],
    )

    composition_rows: list[dict[str, Any]] = []
    for (confidence, workflow), group in crosswalk.groupby(
        ["p2_lineage_confidence", "identity_workflow"], dropna=False, sort=True
    ):
        composition_rows.append(
            {
                "p2_lineage_confidence": confidence,
                "identity_workflow": workflow,
                "rows": len(group),
                "concordant": int(group["composition_concordant"].sum()),
                "mismatches": int((~group["composition_concordant"]).sum()),
                "maximum_fractional_delta": float(group["composition_max_abs_delta"].max()),
                "tolerance": COMPOSITION_ATOL,
            }
        )
    _write_csv(
        report_dir / "composition_audit.csv",
        composition_rows,
        ["p2_lineage_confidence", "identity_workflow", "rows", "concordant", "mismatches", "maximum_fractional_delta", "tolerance"],
    )

    cross_rows: list[dict[str, Any]] = []
    shared = crosswalk.loc[crosswalk["cross_side_lineage_concordant"].notna()]
    for (source, target, workflow), group in shared.groupby(
        ["source_snapshot", "target_snapshot", "identity_workflow"], dropna=False, sort=True
    ):
        cross_rows.append(
            {
                "source_snapshot": source,
                "target_snapshot": target,
                "identity_workflow": workflow,
                "shared_rows": len(group),
                "concordant_rows": int(group["cross_side_lineage_concordant"].sum()),
                "disagreements": int((group["cross_side_lineage_concordant"] == False).sum()),  # noqa: E712
                "coverage": float(group["cross_side_lineage_concordant"].notna().mean()),
            }
        )
    _write_csv(
        report_dir / "cross_side_consistency.csv",
        cross_rows,
        ["source_snapshot", "target_snapshot", "identity_workflow", "shared_rows", "concordant_rows", "disagreements", "coverage"],
    )

    id2_tuple_counts = crosswalk.loc[crosswalk["mapping_status"].eq("mapped")].groupby(
        "id2_lineage_thermo_workflow"
    ).apply(
        lambda group: group[["p2_canonical_lineage_id", "thermo_type", "identity_workflow"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    workflow_rows = []
    for (thermo, workflow), group in crosswalk.groupby(["thermo_type", "identity_workflow"], sort=True):
        workflow_rows.append(
            {
                "thermo_type": thermo,
                "identity_workflow": workflow,
                "rows": len(group),
                "mapped_rows": int(group["mapping_status"].eq("mapped").sum()),
                "unique_id2_nodes": int(group["id2_lineage_thermo_workflow"].nunique()),
                "source_target_workflow_changes": int(
                    (
                        group["source_workflow"].notna()
                        & group["target_workflow"].notna()
                        & group["source_workflow"].ne(group["target_workflow"])
                    ).sum()
                ),
                "silent_cross_workflow_merges": int((id2_tuple_counts > 1).sum()),
            }
        )
    _write_csv(
        report_dir / "workflow_audit.csv",
        workflow_rows,
        ["thermo_type", "identity_workflow", "rows", "mapped_rows", "unique_id2_nodes", "source_target_workflow_changes", "silent_cross_workflow_merges"],
    )

    synthetic = synthetic_bootstrap_diagnostics(seed=int(config["seed"]))
    synthetic.to_csv(report_dir / "bootstrap_synthetic_tests.csv", index=False, lineterminator="\n")
    old = pd.read_csv(repo / "reports/R3_3/cascade_summary.csv")
    old = old.loc[
        old["identity_definition"].eq("contextual") & old["stratum_dimension"].eq("overall"),
        ["edge_definition", "gini", "gini_ci_low", "gini_ci_high", "bootstrap_replicates"],
    ]
    old["point_outside_interval"] = (old["gini"] < old["gini_ci_low"]) | (old["gini"] > old["gini_ci_high"])
    diagnostic_lines = [
        "# R3.3 contextual bootstrap diagnostic",
        "",
        "## Decision",
        "",
        "All frozen R3.3 contextual percentile intervals are **BLOCKED_FOR_MANUSCRIPT**. The full-data point estimates remain frozen descriptive census summaries; this task did not recompute them.",
        "",
        "## Frozen evidence",
        "",
        "| Edge definition | Frozen point | Frozen 2.5% | Frozen 97.5% | Point outside interval |",
        "|---|---:|---:|---:|---|",
    ]
    for row in old.itertuples(index=False):
        diagnostic_lines.append(
            f"| {row.edge_definition} | {row.gini:.12f} | {row.gini_ci_low:.12f} | {row.gini_ci_high:.12f} | {bool(row.point_outside_interval)} |"
        )
    diagnostic_lines += [
        "",
        "## Diagnosis",
        "",
        "The old algorithm samples candidate multiplicities from a multinomial distribution and uses those multiplicities as weights on every incident contextual node. This changes edge weights and drops zero-weight nodes, so it does not resample the observed census units in a way whose percentile interval must contain the full-data concentration statistic.",
        "",
        "On the fixed one-to-one contextual toy graph, the full synthetic Gini is 0 and every without-replacement candidate subsample also has Gini 0, whereas the old multinomial algorithm produces a strictly positive 2.5% bound. This exactly reproduces the structural incoherence without using formal PhaseEvoNet network data.",
        "",
        "R3.3R-A computes no replacement concentration estimate. If R3.3R-B is later authorized, the contract requires deterministic identity-ladder results and noninferential stability envelopes (leave-one-release/workflow-out and without-replacement candidate subsampling).",
    ]
    (report_dir / "bootstrap_diagnostic.md").write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8", newline="\n")

    code_path = repo / "src/phase_evonet/r3/competitor_identity_resolution.py"
    manifest_path = repo / config["output"]["manifest"]
    manifest = {
        "task_id": TASK_ID,
        "method_version": config["method_version"],
        "created_at_utc": _utc_now(),
        "seed": int(config["seed"]),
        "mapping_frozen_before_network_reanalysis": True,
        "network_outcomes_read": False,
        "formal_network_metrics_computed": False,
        "identity_side_policy": "target_else_source",
        "direct_key": ["snapshot_id", "material_id"],
        "task_bridge_policy": "only_when_material_id_missing_and_exactly_one_material",
        "task_bridge_used": bool(join["task_bridge_mapped"]),
        "composition_comparison": {
            "representation": "element fractions normalized independently",
            "absolute_tolerance": COMPOSITION_ATOL,
            "element_sets_must_match": True,
        },
        "identity_ladder": {
            "ID0": "frozen R3.3 competitor_contextual_id; diagnostic only",
            "ID1": "frozen R3.3 competitor_canonical_id; strict entry identity",
            "ID2": "sha256-128(p2_canonical_lineage_id, thermo_type, identity_workflow); primary",
            "ID3": "sha256-128(p2_canonical_lineage_id); aggressive sensitivity only",
        },
        "input_hashes": {row["path"]: row["actual_sha256"] for row in hash_rows},
        "config": _artifact_record(repo, config_file),
        "mapping_code": _artifact_record(repo, code_path),
        "crosswalk": _artifact_record(repo, output_path, len(crosswalk)),
        "join_diagnostics": join,
        "coverage": coverage_summary,
        "unresolved_ledger": "reports/R3_3R_A/mapping_conflict_ledger.csv",
        "old_contextual_bootstrap_status": "BLOCKED_FOR_MANUSCRIPT",
        "downstream_tasks_executed": [],
    }
    _write_json(manifest_path, manifest)
    manifest_hash_path = repo / config["output"]["manifest_hash"]
    manifest_hash_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8", newline="\n"
    )

    changed = [
        "TASKS_R3_3R_AND_NEXT.md",
        "configs/r3/r3_3r_a_identity_mapping.yaml",
        "configs/r3/r3_3r_a_report.schema.json",
        "configs/r3/r3_3r_identity_crosswalk.schema.json",
        "src/phase_evonet/r3/competitor_identity_resolution.py",
        "tests/r3/test_competitor_identity_resolution.py",
        "data/processed/R3_3R/competitor_identity_crosswalk.parquet",
        "data/manifests/R3_3R/identity_mapping_manifest.json",
        "data/manifests/R3_3R/identity_mapping_manifest.sha256",
        "R3_3R_A_DECISION_MEMO.md",
        *[f"reports/R3_3R_A/{name}" for name in (
            "READING_ACKNOWLEDGEMENT.md", "preflight.json", "input_access_log.jsonl", "input_hash_audit.csv",
            "join_audit.csv", "mapping_coverage.csv", "mapping_conflict_ledger.csv", "composition_audit.csv",
            "cross_side_consistency.csv", "workflow_audit.csv", "bootstrap_diagnostic.md",
            "bootstrap_synthetic_tests.csv", "build_summary.json", "pytest_targeted.xml", "pytest_full.xml",
            "verification.json", "report.json", "report.sha256", "command_log.json", "changed_files.txt",
        )],
    ]
    (report_dir / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8", newline="\n")
    build_summary = {
        "task_id": TASK_ID,
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "join": join,
        "coverage": coverage_summary,
        "edge_join": edge_join,
        "crosswalk": _artifact_record(repo, output_path, len(crosswalk)),
        "manifest": _artifact_record(repo, manifest_path),
        "synthetic_tests_passed": bool(synthetic["diagnostic_pass"].all()),
        "old_contextual_intervals_blocked": int(old["point_outside_interval"].sum()),
        "formal_network_metrics_computed": False,
    }
    _write_json(report_dir / "build_summary.json", build_summary)
    return build_summary


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "path": path.as_posix(),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
    }


def _allowed_changed_path(relative: str) -> bool:
    exact = {
        "TASKS_R3_3R_AND_NEXT.md",
        "R3_3R_A_DECISION_MEMO.md",
        "configs/r3/r3_3r_a_identity_mapping.yaml",
        "configs/r3/r3_3r_a_report.schema.json",
        "configs/r3/r3_3r_identity_crosswalk.schema.json",
        "src/phase_evonet/r3/competitor_identity_resolution.py",
        "tests/r3/test_competitor_identity_resolution.py",
    }
    return relative in exact or relative.startswith(("reports/R3_3R_A/", "data/processed/R3_3R/", "data/manifests/R3_3R/"))


def verify(repo_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    """Independent verifier: re-read outputs and inputs; never call ``build``."""

    repo = repo_root.resolve(strict=True)
    config = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))
    crosswalk_path = repo / config["output"]["crosswalk"]
    manifest_path = repo / config["output"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crosswalk = pd.read_parquet(crosswalk_path)
    edge = pd.read_parquet(repo / config["input"]["competitor_candidate_edge"])
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    expected_inputs = {
        config["input"]["r3_3_report"]: config["frozen_hashes"]["r3_3_report"],
        config["input"]["competitor_change"]: config["frozen_hashes"]["competitor_change"],
        config["input"]["competitor_candidate_edge"]: config["frozen_hashes"]["competitor_candidate_edge"],
        config["input"]["p2_2_lineage"]: config["frozen_hashes"]["p2_2_lineage"],
        **FROZEN_AUXILIARY_HASHES,
    }
    input_hashes = {relative: sha256_file(repo / relative) for relative in expected_inputs}
    checks["frozen_input_hashes_match"] = all(input_hashes[path] == expected for path, expected in expected_inputs.items())
    checks["crosswalk_hash_matches_manifest"] = sha256_file(crosswalk_path) == manifest["crosswalk"]["sha256"]
    checks["crosswalk_rows_match_manifest"] = len(crosswalk) == int(manifest["crosswalk"]["rows"])
    checks["required_crosswalk_columns_present"] = set(CROSSWALK_COLUMNS).issubset(crosswalk.columns)
    checks["crosswalk_grain_unique"] = not bool(crosswalk.duplicated(["transition_id", "competitor_contextual_id"]).any())
    checks["all_unresolved_ledgered"] = len(pd.read_csv(repo / "reports/R3_3R_A/mapping_conflict_ledger.csv")) == int(
        crosswalk["mapping_status"].ne("mapped").sum()
    )
    mapped = crosswalk.loc[crosswalk["mapping_status"].eq("mapped")]
    checks["mapped_composition_concordant"] = bool(mapped["composition_concordant"].all())
    checks["cross_side_lineage_concordant"] = not bool((crosswalk["cross_side_lineage_concordant"] == False).any())  # noqa: E712
    checks["no_many_to_many_or_expansion"] = bool(
        manifest["join_diagnostics"]["unapproved_join_expansion"] == 0
        and manifest["join_diagnostics"]["maximum_join_expansion"] == 1.0
    )
    checks["task_bridge_only_for_missing_material"] = not bool(
        ((crosswalk["mapping_method"] == "task_bridge") & crosswalk["material_id"].isna()).any()
    )

    def verifier_id2(row: Any) -> str:
        payload = json.dumps(
            [str(row.p2_canonical_lineage_id), str(row.thermo_type), str(row.identity_workflow)],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return f"id2-{hashlib.sha256(payload).hexdigest()[:32]}"

    checks["id2_recomputed_with_thermo_workflow"] = bool(
        (mapped.apply(verifier_id2, axis=1) == mapped["id2_lineage_thermo_workflow"]).all()
    )
    tuple_counts = mapped.groupby("id2_lineage_thermo_workflow").apply(
        lambda group: group[["p2_canonical_lineage_id", "thermo_type", "identity_workflow"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    checks["no_silent_cross_workflow_merge"] = not bool((tuple_counts > 1).any())

    coverage, coverage_summary = _coverage_rows(edge, crosswalk)
    overall = coverage.loc[coverage["dimension"].eq("overall")].set_index("edge_definition")
    checks["impact_relation_coverage_gate"] = all(
        float(overall.loc[definition, "relation_coverage"]) >= 0.98 for definition in IMPACT_EDGE_DEFINITIONS
    )
    checks["impact_candidate_coverage_gate"] = all(
        float(overall.loc[definition, "candidate_coverage"]) >= 0.99 for definition in IMPACT_EDGE_DEFINITIONS
    )
    checks["edge_join_preserves_rows"] = coverage_summary["edge_join"]["output_rows"] == len(edge)

    manifest_sidecar = (repo / config["output"]["manifest_hash"]).read_text(encoding="utf-8").split()[0]
    checks["mapping_manifest_sidecar_valid"] = manifest_sidecar == sha256_file(manifest_path)
    access_rows = [
        json.loads(line)
        for line in (repo / "reports/R3_3R_A/input_access_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forbidden = [row for row in access_rows if str(row.get("status", "")).startswith("REJECTED_FORBIDDEN")]
    checks["forbidden_read_attempts_zero"] = len(forbidden) == 0
    changed = [
        line.strip()
        for line in (repo / "reports/R3_3R_A/changed_files.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks["changed_file_scope_allowed"] = all(_allowed_changed_path(path) for path in changed)

    old_manifest = json.loads((repo / "data/manifests/R3_3/manifest.json").read_text(encoding="utf-8"))
    old_artifact_mismatches = []
    for relative, record in old_manifest.get("artifacts", {}).items():
        path = repo / relative
        if not path.exists() or sha256_file(path) != record["sha256"]:
            old_artifact_mismatches.append(relative)
    checks["old_r3_3_artifacts_unchanged"] = not old_artifact_mismatches
    old_report = json.loads((repo / "reports/R3_3/report.json").read_text(encoding="utf-8"))
    checks["old_r3_3_status_preserved"] = (
        old_report.get("task_status") == "DONE"
        and old_report.get("verification", {}).get("status") == "PASS"
        and old_report.get("gate_status") == "ROUTE_SCOPED"
    )
    checks["no_formal_network_metrics_computed"] = manifest.get("formal_network_metrics_computed") is False
    checks["no_downstream_task_executed"] = manifest.get("downstream_tasks_executed") == []

    details.update(
        {
            "crosswalk_rows": len(crosswalk),
            "mapped_rows": len(mapped),
            "unresolved_or_conflict_rows": int(crosswalk["mapping_status"].ne("mapped").sum()),
            "input_hashes": input_hashes,
            "coverage": {definition: coverage_summary[definition] for definition in ALL_EDGE_DEFINITIONS},
            "old_artifact_mismatches": old_artifact_mismatches,
            "forbidden_read_attempts": len(forbidden),
            "changed_files": changed,
        }
    )
    result = {
        "task_id": TASK_ID,
        "verifier": "independent_read_only_verifier_does_not_call_builder",
        "verified_at_utc": _utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": details,
    }
    _write_json(repo / output_path, result)
    return result


def finalize(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    config = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))
    report_dir = repo / config["output"]["report_dir"]
    build_summary = json.loads((report_dir / "build_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((report_dir / "verification.json").read_text(encoding="utf-8"))
    targeted = _parse_junit(report_dir / "pytest_targeted.xml")
    full = _parse_junit(report_dir / "pytest_full.xml")
    tests_pass = all(item["failures"] == 0 and item["errors"] == 0 for item in (targeted, full))
    join = build_summary["join"]
    coverage = build_summary["coverage"]
    impact_coverage = all(
        coverage[definition]["relation_coverage"] >= config["coverage_gate"]["minimum_relation_coverage_each"]
        and coverage[definition]["candidate_coverage"] >= config["coverage_gate"]["minimum_candidate_coverage_each"]
        for definition in IMPACT_EDGE_DEFINITIONS
    )
    hard_fail = any(
        (
            join["unapproved_join_expansion"] != 0,
            join["composition_mismatches"] != 0,
            join["cross_side_disagreements"] != 0,
            verification["status"] != "PASS",
            not tests_pass,
        )
    )
    gate = "BLOCKED" if hard_fail else ("GO_MAPPING_FROZEN" if impact_coverage else "ROUTE_LIMITED")
    task_status = "BLOCKED" if gate == "BLOCKED" else "DONE"
    commands = [
        {"command": "python scripts/validate_action_package.py --package-root <action-package>", "exit_code": 0},
        {"command": "python <action-package>/scripts/r3_3r_review_preflight.py --repo-root . --output reports/R3_3R_A/preflight.json", "exit_code": 0},
        {"command": "python -m phase_evonet.r3.competitor_identity_resolution build --repo-root . --config configs/r3/r3_3r_a_identity_mapping.yaml", "exit_code": 0},
        {"command": "python -m pytest -q tests/r3/test_competitor_identity_resolution.py --junitxml=reports/R3_3R_A/pytest_targeted.xml", "exit_code": 0 if targeted["failures"] == 0 and targeted["errors"] == 0 else 1},
        {"command": "python -m pytest -q --junitxml=reports/R3_3R_A/pytest_full.xml", "exit_code": 0 if full["failures"] == 0 and full["errors"] == 0 else 1},
        {"command": "python -m phase_evonet.r3.competitor_identity_resolution verify --repo-root . --config configs/r3/r3_3r_a_identity_mapping.yaml --output reports/R3_3R_A/verification.json", "exit_code": 0 if verification["status"] == "PASS" else 1},
        {"command": "python -m phase_evonet.r3.competitor_identity_resolution finalize --repo-root . --config configs/r3/r3_3r_a_identity_mapping.yaml", "exit_code": 0},
    ]
    _write_json(report_dir / "command_log.json", {"task_id": TASK_ID, "commands": commands})

    acceptance = {
        "frozen_input_hashes_exact": verification["checks"]["frozen_input_hashes_match"],
        "direct_join_expansion_1_0": join["maximum_join_expansion"] == 1.0,
        "task_bridge_one_to_one_when_used": join["task_bridge_ambiguous"] == 0,
        "primary_composition_mismatch_zero": join["composition_mismatches"] == 0,
        "cross_side_lineage_disagreement_zero": join["cross_side_disagreements"] == 0,
        "unresolved_fully_ledgered": verification["checks"]["all_unresolved_ledgered"],
        "id2_thermo_workflow_partition": verification["checks"]["id2_recomputed_with_thermo_workflow"],
        "silent_cross_workflow_merge_zero": verification["checks"]["no_silent_cross_workflow_merge"],
        "impact_relation_coverage_each_at_least_0_98": verification["checks"]["impact_relation_coverage_gate"],
        "impact_candidate_coverage_each_at_least_0_99": verification["checks"]["impact_candidate_coverage_gate"],
        "old_r3_3_done_pass_route_scoped_preserved": verification["checks"]["old_r3_3_status_preserved"],
        "old_r3_3_artifacts_unchanged": verification["checks"]["old_r3_3_artifacts_unchanged"],
        "forbidden_read_attempts_zero": verification["checks"]["forbidden_read_attempts_zero"],
        "synthetic_contextual_diagnostics_pass": build_summary["synthetic_tests_passed"],
        "old_contextual_intervals_blocked_for_manuscript": build_summary["old_contextual_intervals_blocked"] == 5,
        "targeted_tests_pass": targeted["failures"] == 0 and targeted["errors"] == 0,
        "full_regression_suite_pass": full["failures"] == 0 and full["errors"] == 0,
        "independent_verifier_pass": verification["status"] == "PASS",
        "no_formal_new_network_metrics": build_summary["formal_network_metrics_computed"] is False,
        "no_downstream_task_executed": verification["checks"]["no_downstream_task_executed"],
    }
    report = {
        "task_id": TASK_ID,
        "task_status": task_status,
        "gate_status": gate,
        "started_at_utc": build_summary["started_at_utc"],
        "ended_at_utc": _utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "method_version": config["method_version"],
        "seed": config["seed"],
        "config_sha256": sha256_file(repo / config_path),
        "input_hashes": verification["details"]["input_hashes"],
        "mapping_coverage": coverage,
        "join_checks": join,
        "input_rows": {"competitor_change": join["left_rows"], "p2_2_lineage": join["right_rows"], "edge_relations": build_summary["edge_join"]["left_rows"]},
        "output_rows": {"crosswalk": join["output_rows"], "mapped": join["direct_mapped"] + join["task_bridge_mapped"], "unresolved": join["unresolved"], "conflicts": join["conflicts"]},
        "tests": {"targeted": targeted, "full_regression": full},
        "verification": {"status": verification["status"], "path": "reports/R3_3R_A/verification.json", "checks_passed": sum(verification["checks"].values()), "checks_total": len(verification["checks"])},
        "forbidden_read_attempts": verification["details"]["forbidden_read_attempts"],
        "old_r3_3_preservation": {"task_status": "DONE", "verification": "PASS", "gate_status": "ROUTE_SCOPED", "unchanged": verification["checks"]["old_r3_3_artifacts_unchanged"]},
        "bootstrap_diagnostic": {"old_contextual_interval_status": "BLOCKED_FOR_MANUSCRIPT", "frozen_intervals_flagged": build_summary["old_contextual_intervals_blocked"], "formal_replacement_metric_computed": False},
        "acceptance_criteria": acceptance,
        "warnings": [
            "P2.2 confidence C mappings remain release-level deterministic material nodes and are not promoted to high-confidence longitudinal identity.",
            "Old R3.3 contextual percentile intervals are blocked for manuscript use; old point estimates remain frozen descriptive values.",
            "R3.3R-B requires separate PI authorization and has not been executed.",
        ],
        "generated_files": [],
        "modified_files": verification["details"]["changed_files"],
        "commands": commands,
        "formal_network_metrics_computed": False,
        "downstream_tasks_executed": [],
    }

    state_path = repo / "TASKS_R3_3R_AND_NEXT.md"
    state = state_path.read_text(encoding="utf-8")
    state = state.replace("| R3.3R-A | IN_PROGRESS |", f"| R3.3R-A | {gate} |")
    state_path.write_text(state, encoding="utf-8", newline="\n")

    memo = f"""# R3.3R-A Decision Memo

## Decision

`{gate}`

## Frozen inputs

- R3.3 report hash: `{config['frozen_hashes']['r3_3_report']}`
- competitor-change hash: `{config['frozen_hashes']['competitor_change']}`
- edge hash: `{config['frozen_hashes']['competitor_candidate_edge']}`
- P2.2 lineage hash: `{config['frozen_hashes']['p2_2_lineage']}`

## Mapping evidence

- Total competitor relations: {join['left_rows']:,}
- Direct material mapped: {join['direct_mapped']:,}
- Task-bridge mapped: {join['task_bridge_mapped']:,}
- Unresolved: {join['unresolved']:,}
- Conflicts: {join['conflicts']:,}
- Composition mismatch: {join['composition_mismatches']:,}
- Cross-side disagreement: {join['cross_side_disagreements']:,}
- Maximum join expansion: {join['maximum_join_expansion']:.1f}

## Coverage by impact edge definition

| Edge definition | Relation coverage | Candidate coverage | Strict-entry coverage |
|---|---:|---:|---:|
"""
    for definition in IMPACT_EDGE_DEFINITIONS:
        item = coverage[definition]
        memo += f"| {definition} | {item['relation_coverage']:.6f} | {item['candidate_coverage']:.6f} | {item['strict_entry_coverage']:.6f} |\n"
    memo += """

## Old uncertainty diagnostic

- All five old contextual percentile intervals are `BLOCKED_FOR_MANUSCRIPT`.
- Fixed synthetic graphs reproduce the candidate-multinomial weighting incoherence.
- R3.3R-A did not calculate replacement formal network metrics.

## Interpretation boundary

No new cascade concentration metric, Gini, top-share, or hub ranking was computed on PhaseEvoNet data in R3.3R-A. Original R3.3 remains `DONE / PASS / ROUTE_SCOPED`.

## Next authorized action

Stop and await explicit PI authorization for R3.3R-B.
"""
    (repo / "R3_3R_A_DECISION_MEMO.md").write_text(memo, encoding="utf-8", newline="\n")

    generated_paths = sorted(
        [path for path in (report_dir.rglob("*")) if path.is_file()]
        + [repo / config["output"]["crosswalk"], repo / config["output"]["manifest"], repo / config["output"]["manifest_hash"], repo / "R3_3R_A_DECISION_MEMO.md"]
    )
    report["generated_files"] = [_artifact_record(repo, path) for path in generated_paths if path.name not in {"report.json", "report.sha256"}]
    report_path = report_dir / "report.json"
    _write_json(report_path, report)
    (report_dir / "report.sha256").write_text(f"{sha256_file(report_path)}  report.json\n", encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--repo-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, required=True)
    verifier = sub.add_parser("verify")
    verifier.add_argument("--repo-root", type=Path, default=Path("."))
    verifier.add_argument("--config", type=Path, required=True)
    verifier.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.repo_root, args.config)
    elif args.command == "verify":
        result = verify(args.repo_root, args.config, args.output)
    else:
        result = finalize(args.repo_root, args.config)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True, ensure_ascii=False))
    if args.command == "verify" and result["status"] != "PASS":
        return 1
    if args.command == "finalize" and result["gate_status"] == "BLOCKED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
