from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .identity_candidates import write_csv_atomic, write_json_atomic, write_jsonl_atomic
from .manifest import sha256_file


CONFIDENCE_RANK = {"A1": 0, "A2": 1, "B": 2, "C": 3}

IDENTITY_EDGE_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.binary(16), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("source_composition_key", pa.large_string(), nullable=False),
        pa.field("target_composition_key", pa.large_string(), nullable=False),
        pa.field("same_material_id", pa.bool_(), nullable=False),
        pa.field("shared_task_count", pa.int16(), nullable=False),
        pa.field("shared_external_id_count", pa.int16(), nullable=False),
        pa.field("exact_structure_fingerprint", pa.bool_(), nullable=False),
        pa.field("candidate_method_mask", pa.uint8(), nullable=False),
        pa.field("structure_match_evaluated", pa.bool_(), nullable=False),
        pa.field("primary_structure_match", pa.bool_()),
        pa.field("primary_rms_distance", pa.float64()),
        pa.field("primary_max_distance", pa.float64()),
        pa.field("confidence", pa.string(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("decision_reason", pa.string(), nullable=False),
        pa.field("source_accepted_degree", pa.int16(), nullable=False),
        pa.field("target_accepted_degree", pa.int16(), nullable=False),
        pa.field("relationship_event", pa.string(), nullable=False),
        pa.field("canonical_lineage_id", pa.string()),
        pa.field("source_object_sha256", pa.string(), nullable=False),
        pa.field("target_object_sha256", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("target_key", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("target_row_number", pa.int64(), nullable=False),
    ]
)

LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("material_id", pa.string(), nullable=False),
        pa.field("composition_reduced_json", pa.large_string(), nullable=False),
        pa.field("nsites", pa.int32()),
        pa.field("deprecated", pa.bool_()),
        pa.field("lineage_confidence", pa.string(), nullable=False),
        pa.field("high_confidence", pa.bool_(), nullable=False),
        pa.field("first_snapshot", pa.string(), nullable=False),
        pa.field("last_snapshot", pa.string(), nullable=False),
        pa.field("snapshot_count", pa.int16(), nullable=False),
        pa.field("member_count", pa.int32(), nullable=False),
        pa.field("has_merge_or_split", pa.bool_(), nullable=False),
        pa.field("source_object_sha256", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
    ]
)

LINEAGE_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("lineage_confidence", pa.string(), nullable=False),
        pa.field("high_confidence", pa.bool_(), nullable=False),
        pa.field("first_snapshot", pa.string(), nullable=False),
        pa.field("last_snapshot", pa.string(), nullable=False),
        pa.field("snapshot_count", pa.int16(), nullable=False),
        pa.field("member_count", pa.int32(), nullable=False),
        pa.field("has_merge_or_split", pa.bool_(), nullable=False),
        pa.field("representative_snapshot", pa.string(), nullable=False),
        pa.field("representative_material_id", pa.string(), nullable=False),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("snapshot_pair", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("material_id", pa.string(), nullable=False),
        pa.field("related_material_ids_json", pa.large_string(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("confidence", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def node_key(snapshot: str, material_id: str) -> str:
    return f"{snapshot}|{material_id}"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.size[left_root] < self.size[right_root] or (
            self.size[left_root] == self.size[right_root]
            and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return left_root


class AtomicParquetWriter:
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
        self.buffer: list[dict[str, Any]] = []
        self.rows = 0

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.row_group_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)


@dataclass(frozen=True, slots=True)
class MatchEvaluation:
    edge_id: bytes
    source_snapshot: str
    target_snapshot: str
    source_material_id: str
    target_material_id: str
    source_composition_key: str
    target_composition_key: str
    matched: bool
    rms: float | None
    maximum: float | None
    error: str | None = None

    @property
    def source_node(self) -> str:
        return node_key(self.source_snapshot, self.source_material_id)

    @property
    def target_node(self) -> str:
        return node_key(self.target_snapshot, self.target_material_id)


def matcher_from_config(config: dict[str, Any]) -> StructureMatcher:
    return StructureMatcher(
        ltol=float(config["ltol"]),
        stol=float(config["stol"]),
        angle_tol=float(config["angle_tol"]),
        primitive_cell=bool(config["primitive_cell"]),
        scale=bool(config["scale"]),
        attempt_supercell=bool(config["attempt_supercell"]),
        allow_subset=bool(config["allow_subset"]),
    )


def classify_nonaccepted(*, has_anchor: bool, evaluated_match: bool) -> tuple[str, str]:
    if evaluated_match:
        return "B", "structure_match_not_mutual_unique_or_a1_endpoint_conflict"
    if has_anchor:
        return "B", "provenance_anchor_without_accepted_structure_resolution"
    return "C", "same_composition_candidate_without_identity_evidence"


def deterministic_lineage_id(representative_node: str) -> str:
    digest = hashlib.blake2b(representative_node.encode(), digest_size=12).hexdigest()
    return f"lin-{digest}"


def event_id(event_type: str, snapshot_pair: str, material_id: str) -> str:
    body = f"{event_type}|{snapshot_pair}|{material_id}".encode()
    return "evt-" + hashlib.blake2b(body, digest_size=12).hexdigest()


def _candidate_columns() -> list[str]:
    return [field.name for field in pq.ParquetFile(
        "data/interim/P2_1/identity_candidate_edges.parquet"
    ).schema_arrow]


def _collect_a1_and_unresolved(
    candidate_path: Path,
) -> tuple[
    UnionFind,
    set[str],
    set[str],
    Counter[str],
    Counter[str],
    list[dict[str, Any]],
    int,
]:
    union_find = UnionFind()
    used_source: set[str] = set()
    used_target: set[str] = set()
    source_degree: Counter[str] = Counter()
    target_degree: Counter[str] = Counter()
    a1_count = 0
    parquet = pq.ParquetFile(candidate_path)
    columns = [
        "edge_id",
        "source_snapshot",
        "target_snapshot",
        "source_material_id",
        "target_material_id",
        "source_composition_key",
        "target_composition_key",
        "composition_block",
        "same_material_id",
        "shared_task_count",
        "shared_external_id_count",
        "exact_structure_fingerprint",
        "same_nsites",
        "same_spacegroup",
        "volume_per_site_relative_difference",
    ]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            if row["same_material_id"] and row["shared_task_count"] >= 1:
                source = node_key(row["source_snapshot"], row["source_material_id"])
                target = node_key(row["target_snapshot"], row["target_material_id"])
                union_find.union(source, target)
                used_source.add(source)
                used_target.add(target)
                source_degree[source] += 1
                target_degree[target] += 1
                a1_count += 1
    unresolved: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            if (
                node_key(row["source_snapshot"], row["source_material_id"])
                not in used_source
                and node_key(row["target_snapshot"], row["target_material_id"])
                not in used_target
                and bool(row["composition_block"])
            ):
                unresolved.append(row)
    return (
        union_find,
        used_source,
        used_target,
        source_degree,
        target_degree,
        unresolved,
        a1_count,
    )


def _load_structures(
    normalized_root: Path,
    required: dict[str, set[str]],
) -> dict[str, Structure]:
    structures: dict[str, Structure] = {}
    for snapshot, material_ids in required.items():
        path = normalized_root / f"snapshot={snapshot}" / "raw_material.parquet"
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=20_000, columns=["material_id", "structure_json"]
        ):
            for row in batch.to_pylist():
                material_id = str(row["material_id"])
                if material_id in material_ids:
                    structures[node_key(snapshot, material_id)] = Structure.from_dict(
                        json.loads(row["structure_json"])
                    )
    expected = sum(len(values) for values in required.values())
    if len(structures) != expected:
        raise RuntimeError(
            f"Loaded {len(structures)} required structures, expected {expected}"
        )
    return structures


def _evaluate_unresolved(
    rows: list[dict[str, Any]],
    normalized_root: Path,
    matcher_config: dict[str, Any],
) -> list[MatchEvaluation]:
    required: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        required[row["source_snapshot"]].add(str(row["source_material_id"]))
        required[row["target_snapshot"]].add(str(row["target_material_id"]))
    structures = _load_structures(normalized_root, required)
    matcher = matcher_from_config(matcher_config)
    evaluations: list[MatchEvaluation] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["source_snapshot"],
            item["source_material_id"],
            item["target_material_id"],
        ),
    ):
        source_node = node_key(row["source_snapshot"], row["source_material_id"])
        target_node = node_key(row["target_snapshot"], row["target_material_id"])
        matched = False
        rms: float | None = None
        maximum: float | None = None
        error: str | None = None
        try:
            source_structure = structures[source_node]
            target_structure = structures[target_node]
            matched = bool(matcher.fit(source_structure, target_structure))
            if matched:
                distances = matcher.get_rms_dist(source_structure, target_structure)
                if distances is not None:
                    rms, maximum = (float(distances[0]), float(distances[1]))
        except Exception as exc:  # recorded, never swallowed
            error = f"{type(exc).__name__}: {exc}"
        evaluations.append(
            MatchEvaluation(
                edge_id=bytes(row["edge_id"]),
                source_snapshot=str(row["source_snapshot"]),
                target_snapshot=str(row["target_snapshot"]),
                source_material_id=str(row["source_material_id"]),
                target_material_id=str(row["target_material_id"]),
                source_composition_key=str(row["source_composition_key"]),
                target_composition_key=str(row["target_composition_key"]),
                matched=matched,
                rms=rms,
                maximum=maximum,
                error=error,
            )
        )
    return evaluations


def select_mutual_unique_matches(
    evaluations: Iterable[MatchEvaluation], *, tie_tolerance: float = 1e-12
) -> tuple[set[bytes], list[dict[str, Any]]]:
    matched = [item for item in evaluations if item.matched and item.error is None]
    by_source: dict[str, list[MatchEvaluation]] = defaultdict(list)
    by_target: dict[str, list[MatchEvaluation]] = defaultdict(list)
    for item in matched:
        by_source[item.source_node].append(item)
        by_target[item.target_node].append(item)

    def unique_best(items: list[MatchEvaluation]) -> MatchEvaluation | None:
        ranked = sorted(
            items,
            key=lambda item: (
                math.inf if item.rms is None else item.rms,
                math.inf if item.maximum is None else item.maximum,
                item.target_material_id,
                item.source_material_id,
            ),
        )
        if len(ranked) > 1:
            first = math.inf if ranked[0].rms is None else ranked[0].rms
            second = math.inf if ranked[1].rms is None else ranked[1].rms
            if abs(first - second) <= tie_tolerance:
                return None
        return ranked[0]

    source_best = {key: unique_best(values) for key, values in by_source.items()}
    target_best = {key: unique_best(values) for key, values in by_target.items()}
    accepted: set[bytes] = set()
    for item in matched:
        if source_best.get(item.source_node) is item and target_best.get(
            item.target_node
        ) is item:
            accepted.add(item.edge_id)

    ambiguities: list[dict[str, Any]] = []
    for endpoint_type, grouped in (("source", by_source), ("target", by_target)):
        for endpoint, values in sorted(grouped.items()):
            if len(values) <= 1:
                continue
            ambiguities.append(
                {
                    "record_type": "structure_match_ambiguity",
                    "endpoint_type": endpoint_type,
                    "endpoint": endpoint,
                    "matched_edge_count": len(values),
                    "edge_ids": sorted(item.edge_id.hex() for item in values),
                    "resolution": "downgrade_non_mutual_or_tied_matches_to_B",
                }
            )
    return accepted, ambiguities


def _audit_a2(
    accepted: list[MatchEvaluation],
    normalized_root: Path,
    audit_config: dict[str, Any],
    sample_config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maximum = int(sample_config["maximum_sample_rows"])
    ranked = sorted(
        accepted,
        key=lambda item: hashlib.sha256(
            f"{seed}|{item.edge_id.hex()}".encode()
        ).hexdigest(),
    )
    sample = ranked if len(ranked) <= maximum else ranked[:maximum]
    required: dict[str, set[str]] = defaultdict(set)
    for item in sample:
        required[item.source_snapshot].add(item.source_material_id)
        required[item.target_snapshot].add(item.target_material_id)
    structures = _load_structures(normalized_root, required) if sample else {}
    matcher = matcher_from_config(audit_config)
    rows: list[dict[str, Any]] = []
    true_positive = 0
    for item in sample:
        source = structures[item.source_node]
        target = structures[item.target_node]
        audit_match = False
        audit_rms: float | None = None
        audit_maximum: float | None = None
        audit_error: str | None = None
        try:
            audit_match = bool(matcher.fit(source, target))
            if audit_match:
                distance = matcher.get_rms_dist(source, target)
                if distance is not None:
                    audit_rms, audit_maximum = float(distance[0]), float(distance[1])
        except Exception as exc:
            audit_error = f"{type(exc).__name__}: {exc}"
        reference_positive = (
            audit_match
            and audit_error is None
            and item.source_composition_key == item.target_composition_key
        )
        true_positive += int(reference_positive)
        rows.append(
            {
                "edge_id": item.edge_id.hex(),
                "source_snapshot": item.source_snapshot,
                "target_snapshot": item.target_snapshot,
                "source_material_id": item.source_material_id,
                "target_material_id": item.target_material_id,
                "same_composition": item.source_composition_key
                == item.target_composition_key,
                "primary_match": item.matched,
                "primary_rms": item.rms,
                "primary_maximum": item.maximum,
                "independent_strict_match": audit_match,
                "independent_rms": audit_rms,
                "independent_maximum": audit_maximum,
                "audit_error": audit_error or "",
                "reference_label": "TP" if reference_positive else "FP",
                "audit_basis": "independent stricter StructureMatcher plus composition concordance",
            }
        )
    precision = true_positive / len(rows) if rows else 0.0
    return rows, {
        "population_a2_edges": len(accepted),
        "sample_rows": len(rows),
        "true_positive": true_positive,
        "false_positive": len(rows) - true_positive,
        "observed_precision": precision,
        "minimum_precision": float(sample_config["minimum_a2_precision"]),
        "status": (
            "PASS"
            if rows and precision >= float(sample_config["minimum_a2_precision"])
            else "FAIL"
        ),
        "seed": seed,
        "sampling": "complete population" if len(accepted) <= maximum else "deterministic hash sample",
        "audit_independence": "A2 selection used the primary matcher; audit used separately configured stricter tolerances.",
    }


def _lineage_metadata(
    union_find: UnionFind,
    normalized_root: Path,
    snapshots: list[str],
    accepted_nodes: set[str],
    a2_nodes: set[str],
    event_nodes: set[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    material_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        path = normalized_root / f"snapshot={snapshot}" / "raw_material.parquet"
        parquet = pq.ParquetFile(path)
        columns = [
            "material_id",
            "composition_reduced_json",
            "nsites",
            "deprecated",
            "source_object_sha256",
            "source_key",
            "source_row_number",
        ]
        for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
            for row in batch.to_pylist():
                row["snapshot_id"] = snapshot
                key = node_key(snapshot, str(row["material_id"]))
                union_find.add(key)
                material_rows.append(row)

    grouped: dict[str, dict[str, Any]] = {}
    node_to_lineage: dict[str, str] = {}
    root_nodes: dict[str, list[str]] = defaultdict(list)
    for row in material_rows:
        key = node_key(row["snapshot_id"], str(row["material_id"]))
        root_nodes[union_find.find(key)].append(key)
    for root, nodes in root_nodes.items():
        representative = min(nodes)
        lineage_id = deterministic_lineage_id(representative)
        snapshots_present = sorted({item.split("|", 1)[0] for item in nodes})
        has_edge = any(item in accepted_nodes for item in nodes)
        confidence = "A2" if any(item in a2_nodes for item in nodes) else (
            "A1" if has_edge else "C"
        )
        metadata = {
            "canonical_lineage_id": lineage_id,
            "lineage_confidence": confidence,
            "high_confidence": has_edge and confidence in {"A1", "A2"},
            "first_snapshot": snapshots_present[0],
            "last_snapshot": snapshots_present[-1],
            "snapshot_count": len(snapshots_present),
            "member_count": len(nodes),
            "has_merge_or_split": any(item in event_nodes for item in nodes),
            "representative_snapshot": representative.split("|", 1)[0],
            "representative_material_id": representative.split("|", 1)[1],
        }
        grouped[root] = metadata
        for item in nodes:
            node_to_lineage[item] = lineage_id
    return grouped, node_to_lineage, material_rows


def _build_event_rows(
    accepted_pairs: list[tuple[str, str, str, str, str]],
    anchor_pairs: list[tuple[str, str, str, str]],
    source_degree: Counter[str],
    target_degree: Counter[str],
) -> list[dict[str, Any]]:
    by_source: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_snapshot, target_snapshot, source_id, target_id, _ in accepted_pairs:
        pair = f"{source_snapshot}_to_{target_snapshot}"
        by_source[(pair, source_id)].append(target_id)
        by_target[(pair, target_id)].append(source_id)
    events: list[dict[str, Any]] = []
    for (pair, material_id), related in sorted(by_source.items()):
        source_snapshot = pair.split("_to_", 1)[0]
        if source_degree[node_key(source_snapshot, material_id)] > 1:
            events.append(
                {
                    "event_id": event_id("split", pair, material_id),
                    "snapshot_pair": pair,
                    "event_type": "split",
                    "snapshot_id": source_snapshot,
                    "material_id": material_id,
                    "related_material_ids_json": json.dumps(sorted(related)),
                    "accepted": True,
                    "confidence": "A2",
                    "reason": "multiple accepted forward identity edges",
                }
            )
    for (pair, material_id), related in sorted(by_target.items()):
        target_snapshot = pair.split("_to_", 1)[1]
        if target_degree[node_key(target_snapshot, material_id)] > 1:
            events.append(
                {
                    "event_id": event_id("merge", pair, material_id),
                    "snapshot_pair": pair,
                    "event_type": "merge",
                    "snapshot_id": target_snapshot,
                    "material_id": material_id,
                    "related_material_ids_json": json.dumps(sorted(related)),
                    "accepted": True,
                    "confidence": "A2",
                    "reason": "multiple accepted incoming identity edges",
                }
            )
    anchor_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    anchor_by_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source_snapshot, target_snapshot, source_id, target_id in anchor_pairs:
        pair = f"{source_snapshot}_to_{target_snapshot}"
        anchor_by_source[(pair, source_id)].add(target_id)
        anchor_by_target[(pair, target_id)].add(source_id)
    for (pair, material_id), related in sorted(anchor_by_source.items()):
        source_snapshot = pair.split("_to_", 1)[0]
        if len(related) > 1 and source_degree[node_key(source_snapshot, material_id)] <= 1:
            events.append(
                {
                    "event_id": event_id("possible_split", pair, material_id),
                    "snapshot_pair": pair,
                    "event_type": "possible_split",
                    "snapshot_id": source_snapshot,
                    "material_id": material_id,
                    "related_material_ids_json": json.dumps(sorted(related)),
                    "accepted": False,
                    "confidence": "B",
                    "reason": "multiple strong-anchor candidates conflict with one-to-one high-confidence resolution",
                }
            )
    for (pair, material_id), related in sorted(anchor_by_target.items()):
        target_snapshot = pair.split("_to_", 1)[1]
        if len(related) > 1 and target_degree[node_key(target_snapshot, material_id)] <= 1:
            events.append(
                {
                    "event_id": event_id("possible_merge", pair, material_id),
                    "snapshot_pair": pair,
                    "event_type": "possible_merge",
                    "snapshot_id": target_snapshot,
                    "material_id": material_id,
                    "related_material_ids_json": json.dumps(sorted(related)),
                    "accepted": False,
                    "confidence": "B",
                    "reason": "multiple strong-anchor candidates conflict with one-to-one high-confidence resolution",
                }
            )
    return events


def _write_table(path: Path, schema: pa.Schema, rows: Iterable[dict[str, Any]], parquet_config: dict[str, Any]) -> int:
    writer = AtomicParquetWriter(path, schema, parquet_config)
    for row in rows:
        writer.add(row)
    writer.close()
    return writer.rows


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()


def _dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "confidence": "A1/A2/B/C edge confidence; only A1 and A2 edges are accepted.",
        "canonical_lineage_id": "Deterministic component ID derived from the earliest lexical snapshot/material member.",
        "primary_structure_match": "Pymatgen StructureMatcher result under preregistered primary tolerances.",
        "lineage_confidence": "A2 if a lineage contains an accepted A2 edge, otherwise A1 for longitudinal accepted components and C for singletons.",
        "high_confidence": "True only for a lineage containing at least one accepted A1/A2 cross-snapshot edge.",
        "has_merge_or_split": "True when an accepted node degree implies a recorded merge or split event.",
    }
    rows: list[dict[str, Any]] = []
    for table_name, schema in (
        ("identity_edge", IDENTITY_EDGE_SCHEMA),
        ("lineage", LINEAGE_SCHEMA),
        ("lineage_summary", LINEAGE_SUMMARY_SCHEMA),
        ("identity_events", EVENT_SCHEMA),
    ):
        for field in schema:
            rows.append(
                {
                    "table": table_name,
                    "field": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "description": descriptions.get(field.name, "Traceable P2.2 identity-resolution field."),
                }
            )
    return rows


def build_identity_lineages(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "P2.2":
        raise RuntimeError("P2.2 config must declare task_id: P2.2")
    started = utc_now()
    seed = int(config["seed"])
    snapshots = [str(item) for item in config["snapshots"]]
    input_config = config["input"]
    output_config = config["output"]
    candidate_path = Path(input_config["candidate_edges"])
    p2_1_report = json.loads(Path(input_config["p2_1_report"]).read_text(encoding="utf-8"))
    if not (
        p2_1_report.get("task_status") == "DONE"
        and p2_1_report.get("status") == "PASS"
        and p2_1_report.get("gate_status", {}).get("status") == "GO"
    ):
        raise RuntimeError("P2.1 DONE/PASS/GO is required before P2.2")
    expected_candidate_hash = p2_1_report["data_hashes"]["candidate_edge_parquet"]
    actual_candidate_hash = sha256_file(candidate_path)
    if actual_candidate_hash != expected_candidate_hash:
        raise RuntimeError("P2.1 candidate edge SHA256 does not match its passing report")

    (
        union_find,
        used_source,
        used_target,
        source_degree,
        target_degree,
        unresolved_rows,
        a1_count,
    ) = _collect_a1_and_unresolved(candidate_path)
    normalized_root = Path(input_config["normalized_root"])
    evaluations = _evaluate_unresolved(
        unresolved_rows, normalized_root, config["structure_matcher"]
    )
    a2_edge_ids, ambiguity_rows = select_mutual_unique_matches(evaluations)
    evaluation_by_id = {item.edge_id: item for item in evaluations}
    accepted_a2 = [evaluation_by_id[item] for item in sorted(a2_edge_ids)]

    accepted_pairs: list[tuple[str, str, str, str, str]] = []
    anchor_pairs: list[tuple[str, str, str, str]] = []
    accepted_nodes: set[str] = set(used_source) | set(used_target)
    a2_nodes: set[str] = set()
    for item in accepted_a2:
        union_find.union(item.source_node, item.target_node)
        source_degree[item.source_node] += 1
        target_degree[item.target_node] += 1
        accepted_nodes.update((item.source_node, item.target_node))
        a2_nodes.update((item.source_node, item.target_node))

    parquet = pq.ParquetFile(candidate_path)
    pass_columns = [
        "source_snapshot", "target_snapshot", "source_material_id", "target_material_id",
        "same_material_id", "shared_task_count", "shared_external_id_count",
        "exact_structure_fingerprint",
    ]
    for batch in parquet.iter_batches(batch_size=100_000, columns=pass_columns):
        for row in batch.to_pylist():
            if row["same_material_id"] and row["shared_task_count"] >= 1:
                accepted_pairs.append((
                    str(row["source_snapshot"]), str(row["target_snapshot"]),
                    str(row["source_material_id"]), str(row["target_material_id"]), "A1"
                ))
            if (
                row["same_material_id"]
                or row["shared_task_count"] > 0
                or row["shared_external_id_count"] > 0
                or row["exact_structure_fingerprint"]
            ):
                anchor_pairs.append((
                    str(row["source_snapshot"]), str(row["target_snapshot"]),
                    str(row["source_material_id"]), str(row["target_material_id"])
                ))
    accepted_pairs.extend(
        (item.source_snapshot, item.target_snapshot, item.source_material_id, item.target_material_id, "A2")
        for item in accepted_a2
    )
    event_rows = _build_event_rows(
        accepted_pairs, anchor_pairs, source_degree, target_degree
    )
    event_nodes = {
        node_key(row["snapshot_id"], row["material_id"])
        for row in event_rows
        if row["accepted"]
    }

    grouped, node_to_lineage, material_rows = _lineage_metadata(
        union_find, normalized_root, snapshots, accepted_nodes, a2_nodes, event_nodes
    )
    root_by_lineage = {metadata["canonical_lineage_id"]: root for root, metadata in grouped.items()}

    identity_path = Path(output_config["identity_edges"])
    edge_writer = AtomicParquetWriter(identity_path, IDENTITY_EDGE_SCHEMA, config["parquet"])
    edge_counts: Counter[str] = Counter()
    accepted_count = 0
    candidate_parquet = pq.ParquetFile(candidate_path)
    edge_columns = [
        "edge_id", "source_snapshot", "target_snapshot", "source_material_id", "target_material_id",
        "source_composition_key", "target_composition_key", "same_material_id", "shared_task_count",
        "shared_external_id_count", "exact_structure_fingerprint", "candidate_method_mask",
        "source_object_sha256", "target_object_sha256", "source_key", "target_key",
        "source_row_number", "target_row_number",
    ]
    for batch in candidate_parquet.iter_batches(batch_size=50_000, columns=edge_columns):
        for row in batch.to_pylist():
            edge = bytes(row["edge_id"])
            source = node_key(row["source_snapshot"], row["source_material_id"])
            target = node_key(row["target_snapshot"], row["target_material_id"])
            evaluation = evaluation_by_id.get(edge)
            is_a1 = bool(row["same_material_id"] and row["shared_task_count"] >= 1)
            is_a2 = edge in a2_edge_ids
            accepted = is_a1 or is_a2
            if is_a1:
                confidence, reason = "A1", "same_material_id_and_shared_task"
            elif is_a2:
                confidence, reason = "A2", "mutual_unique_periodic_structure_match_without_a1_conflict"
            else:
                has_anchor = bool(
                    row["shared_task_count"] > 0
                    or row["shared_external_id_count"] > 0
                    or row["exact_structure_fingerprint"]
                )
                confidence, reason = classify_nonaccepted(
                    has_anchor=has_anchor,
                    evaluated_match=bool(evaluation and evaluation.matched),
                )
            source_value = int(source_degree[source]) if accepted else 0
            target_value = int(target_degree[target]) if accepted else 0
            relationship = "none"
            if accepted and source_value > 1 and target_value > 1:
                relationship = "complex_merge_split"
            elif accepted and source_value > 1:
                relationship = "split"
            elif accepted and target_value > 1:
                relationship = "merge"
            lineage_id = node_to_lineage[source] if accepted else None
            if accepted and lineage_id != node_to_lineage[target]:
                raise RuntimeError("Accepted edge endpoints disagree on canonical lineage")
            edge_writer.add(
                {
                    **{key: row[key] for key in edge_columns},
                    "structure_match_evaluated": evaluation is not None,
                    "primary_structure_match": evaluation.matched if evaluation else None,
                    "primary_rms_distance": evaluation.rms if evaluation else None,
                    "primary_max_distance": evaluation.maximum if evaluation else None,
                    "confidence": confidence,
                    "accepted": accepted,
                    "decision_reason": reason,
                    "source_accepted_degree": source_value,
                    "target_accepted_degree": target_value,
                    "relationship_event": relationship,
                    "canonical_lineage_id": lineage_id,
                }
            )
            edge_counts[confidence] += 1
            accepted_count += int(accepted)
    edge_writer.close()

    lineage_path = Path(output_config["lineage_members"])
    lineage_writer = AtomicParquetWriter(lineage_path, LINEAGE_SCHEMA, config["parquet"])
    for row in material_rows:
        key = node_key(row["snapshot_id"], str(row["material_id"]))
        lineage_id = node_to_lineage[key]
        metadata = grouped[root_by_lineage[lineage_id]]
        lineage_writer.add(
            {
                "canonical_lineage_id": lineage_id,
                "snapshot_id": row["snapshot_id"],
                "material_id": str(row["material_id"]),
                "composition_reduced_json": str(row["composition_reduced_json"]),
                "nsites": row["nsites"],
                "deprecated": row["deprecated"],
                "lineage_confidence": metadata["lineage_confidence"],
                "high_confidence": metadata["high_confidence"],
                "first_snapshot": metadata["first_snapshot"],
                "last_snapshot": metadata["last_snapshot"],
                "snapshot_count": metadata["snapshot_count"],
                "member_count": metadata["member_count"],
                "has_merge_or_split": metadata["has_merge_or_split"],
                "source_object_sha256": str(row["source_object_sha256"]),
                "source_key": str(row["source_key"]),
                "source_row_number": int(row["source_row_number"]),
            }
        )
    lineage_writer.close()

    summary_path = Path(output_config["lineage_summary"])
    summary_rows = sorted(
        (metadata for metadata in grouped.values()),
        key=lambda item: item["canonical_lineage_id"],
    )
    summary_count = _write_table(
        summary_path, LINEAGE_SUMMARY_SCHEMA, summary_rows, config["parquet"]
    )
    event_path = Path(output_config["relationship_events"])
    event_count = _write_table(event_path, EVENT_SCHEMA, event_rows, config["parquet"])

    audit_rows, audit_summary = _audit_a2(
        accepted_a2,
        normalized_root,
        config["independent_audit_matcher"],
        config["audit"],
        seed,
    )
    audit_csv_path = Path(output_config["a2_audit_csv"])
    if audit_rows:
        write_csv_atomic(audit_csv_path, audit_rows)
    else:
        audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_csv_path.with_suffix(audit_csv_path.suffix + ".tmp")
        temporary.write_text("edge_id,reference_label\n", encoding="utf-8")
        os.replace(temporary, audit_csv_path)
    audit_json_path = Path(output_config["a2_audit_json"])
    audit_summary.update(
        {
            "task_id": "P2.2",
            "created_at_utc": utc_now(),
            "primary_matcher": config["structure_matcher"],
            "independent_audit_matcher": config["independent_audit_matcher"],
            "audit_csv": str(audit_csv_path),
            "audit_csv_sha256": sha256_file(audit_csv_path),
        }
    )
    write_json_atomic(audit_json_path, audit_summary)

    ambiguity_ledger = list(ambiguity_rows)
    for item in evaluations:
        if item.error:
            ambiguity_ledger.append(
                {
                    "record_type": "structure_match_error",
                    "edge_id": item.edge_id.hex(),
                    "source": item.source_node,
                    "target": item.target_node,
                    "error": item.error,
                    "resolution": "downgrade_to_B_or_C; do not silently drop",
                }
            )
    ambiguity_path = Path(output_config["ambiguity_ledger"])
    write_jsonl_atomic(ambiguity_path, ambiguity_ledger)
    dictionary_path = Path(output_config["data_dictionary"])
    write_csv_atomic(dictionary_path, _dictionary_rows())

    high_confidence_lineages = sum(
        bool(item["high_confidence"]) for item in summary_rows
    )
    confidence_counts = Counter(item["lineage_confidence"] for item in summary_rows)
    minimum_lineages = int(config["gate"]["minimum_high_confidence_lineages"])
    minimum_precision = float(config["gate"]["minimum_a2_precision"])
    gate_passed = (
        high_confidence_lineages >= minimum_lineages
        and audit_summary["sample_rows"] > 0
        and audit_summary["observed_precision"] >= minimum_precision
    )
    output_paths = [identity_path, lineage_path, summary_path, event_path]
    manifest = {
        "task_id": "P2.2",
        "status": "PASS" if gate_passed else "FAIL",
        "gate_status": "GO" if gate_passed else "NO-GO",
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "seed": seed,
        "network_access": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_candidate_sha256": actual_candidate_hash,
        "p2_1_report_sha256": sha256_file(input_config["p2_1_report"]),
        "candidate_rows": candidate_parquet.metadata.num_rows,
        "unresolved_candidates_evaluated": len(evaluations),
        "primary_structure_matches": sum(item.matched for item in evaluations),
        "edge_counts_by_confidence": dict(sorted(edge_counts.items())),
        "accepted_edges": accepted_count,
        "a1_edges": a1_count,
        "a2_edges": len(accepted_a2),
        "lineage_member_rows": lineage_writer.rows,
        "lineages": summary_count,
        "lineages_by_confidence": dict(sorted(confidence_counts.items())),
        "high_confidence_lineages": high_confidence_lineages,
        "minimum_high_confidence_lineages": minimum_lineages,
        "relationship_events": event_count,
        "ambiguity_records": len(ambiguity_ledger),
        "a2_precision_audit": audit_summary,
        "outputs": [
            {
                "path": str(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "sha256": sha256_file(path),
                "schema_sha256": _schema_hash(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
        "supporting_files": {
            str(audit_csv_path): sha256_file(audit_csv_path),
            str(audit_json_path): sha256_file(audit_json_path),
            str(ambiguity_path): sha256_file(ambiguity_path),
            str(dictionary_path): sha256_file(dictionary_path),
        },
        "gate": {
            "high_confidence_lineages": {
                "observed": high_confidence_lineages,
                "minimum": minimum_lineages,
                "passed": high_confidence_lineages >= minimum_lineages,
            },
            "a2_precision": {
                "observed": audit_summary["observed_precision"],
                "minimum": minimum_precision,
                "passed": audit_summary["sample_rows"] > 0
                and audit_summary["observed_precision"] >= minimum_precision,
            },
            "passed": gate_passed,
        },
    }
    manifest_path = Path(output_config["manifest"])
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_identity_lineages(config_path: str | Path) -> dict[str, Any]:
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
            failures.append(f"row_count:{path}")
        if sha256_file(path) != output["sha256"]:
            failures.append(f"sha256:{path}")
        if _schema_hash(path) != output["schema_sha256"]:
            failures.append(f"schema:{path}")
    for path_text, expected_hash in manifest["supporting_files"].items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"supporting_hash:{path}")
    edge_path = Path(config["output"]["identity_edges"])
    lineage_path = Path(config["output"]["lineage_members"])
    summary_path = Path(config["output"]["lineage_summary"])
    edge_table = pq.read_table(edge_path, columns=["edge_id", "accepted", "confidence", "canonical_lineage_id"])
    if pc.count_distinct(edge_table["edge_id"]).as_py() != edge_table.num_rows:
        failures.append("duplicate_edge_id")
    accepted = edge_table.filter(edge_table["accepted"])
    if accepted.num_rows != int(manifest["accepted_edges"]):
        failures.append("accepted_edge_count")
    if pc.any(pc.is_null(accepted["canonical_lineage_id"])).as_py():
        failures.append("accepted_edge_without_lineage")
    confidence_values = set(pc.unique(edge_table["confidence"]).to_pylist())
    if not confidence_values.issubset({"A1", "A2", "B", "C"}):
        failures.append("invalid_confidence")
    lineage_rows = pq.ParquetFile(lineage_path).metadata.num_rows
    expected_nodes = sum(
        pq.ParquetFile(
            Path(config["input"]["normalized_root"])
            / f"snapshot={snapshot}"
            / "raw_material.parquet"
        ).metadata.num_rows
        for snapshot in config["snapshots"]
    )
    if lineage_rows != expected_nodes:
        failures.append("lineage_node_coverage")
    summary = pq.read_table(summary_path, columns=["canonical_lineage_id", "high_confidence"])
    if pc.count_distinct(summary["canonical_lineage_id"]).as_py() != summary.num_rows:
        failures.append("duplicate_lineage_id")
    high_count = pc.sum(pc.cast(summary["high_confidence"], pa.int64())).as_py()
    if high_count != int(manifest["high_confidence_lineages"]):
        failures.append("high_confidence_count")
    if not manifest["gate"]["passed"]:
        failures.append("gate_not_passed")
    temporary_files = list(Path(config["output"]["root"]).rglob("*.tmp"))
    temporary_files += list(Path("reports/P2_2").rglob("*.tmp"))
    if temporary_files:
        failures.append("temporary_files")
    return {
        "task_id": "P2.2",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "identity_edges": edge_table.num_rows,
        "distinct_edge_ids": pc.count_distinct(edge_table["edge_id"]).as_py(),
        "accepted_edges": accepted.num_rows,
        "lineage_member_rows": lineage_rows,
        "lineages": summary.num_rows,
        "high_confidence_lineages": high_count,
        "a2_precision": manifest["a2_precision_audit"]["observed_precision"],
        "a2_audit_rows": manifest["a2_precision_audit"]["sample_rows"],
        "gate_status": manifest["gate_status"],
        "temporary_files": len(temporary_files),
        "network_access": False,
        "verified_at_utc": utc_now(),
    }
