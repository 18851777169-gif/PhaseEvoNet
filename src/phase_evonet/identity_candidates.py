from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from .manifest import sha256_file


ANCHOR_SAME_ID = 1
ANCHOR_SHARED_TASK = 2
ANCHOR_EXTERNAL_ID = 4
ANCHOR_EXACT_STRUCTURE = 8
ANCHOR_CATEGORIES = {
    "same_material_id": ANCHOR_SAME_ID,
    "shared_task_id": ANCHOR_SHARED_TASK,
    "shared_external_id": ANCHOR_EXTERNAL_ID,
    "exact_structure_fingerprint": ANCHOR_EXACT_STRUCTURE,
}


EDGE_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.binary(16), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("source_composition_key", pa.large_string(), nullable=False),
        pa.field("target_composition_key", pa.large_string(), nullable=False),
        pa.field("composition_block", pa.bool_(), nullable=False),
        pa.field("same_material_id", pa.bool_(), nullable=False),
        pa.field("shared_task_count", pa.int16(), nullable=False),
        pa.field("shared_external_id_count", pa.int16(), nullable=False),
        pa.field("exact_structure_fingerprint", pa.bool_(), nullable=False),
        pa.field("candidate_method_mask", pa.uint8(), nullable=False),
        pa.field("cross_composition_anchor", pa.bool_(), nullable=False),
        pa.field("source_nsites", pa.int32()),
        pa.field("target_nsites", pa.int32()),
        pa.field("same_nsites", pa.bool_(), nullable=False),
        pa.field("source_spacegroup", pa.int16()),
        pa.field("target_spacegroup", pa.int16()),
        pa.field("same_spacegroup", pa.bool_(), nullable=False),
        pa.field("source_volume_per_site", pa.float64()),
        pa.field("target_volume_per_site", pa.float64()),
        pa.field("volume_per_site_relative_difference", pa.float64()),
        pa.field("source_deprecated", pa.bool_()),
        pa.field("target_deprecated", pa.bool_()),
        pa.field("source_object_sha256", pa.string(), nullable=False),
        pa.field("target_object_sha256", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("target_key", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("target_row_number", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class MaterialDescriptor:
    snapshot: str
    material_id: str
    composition_key: str
    nsites: int | None
    spacegroup: int | None
    volume_per_site: float | None
    task_ids: frozenset[str]
    external_ids: frozenset[str]
    structure_fingerprint: str
    deprecated: bool | None
    source_object_sha256: str
    source_key: str
    source_row_number: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_float(value: Any, decimals: int) -> float:
    result = round(float(value), decimals)
    if abs(result) < 10 ** (-decimals):
        return 0.0
    return result


def normalized_structure_payload(
    structure: dict[str, Any], *, coordinate_decimals: int, lattice_decimals: int
) -> dict[str, Any]:
    lattice = structure.get("lattice") or {}
    matrix = [
        [round_float(value, lattice_decimals) for value in row]
        for row in (lattice.get("matrix") or [])
    ]
    sites = []
    for site in structure.get("sites") or []:
        species = []
        for item in site.get("species") or []:
            species.append(
                (
                    str(item.get("element")),
                    round_float(item.get("occu", 1.0), coordinate_decimals),
                )
            )
        species.sort()
        abc = []
        for value in site.get("abc") or []:
            wrapped = float(value) % 1.0
            rounded = round_float(wrapped, coordinate_decimals)
            if abs(rounded - 1.0) < 10 ** (-coordinate_decimals):
                rounded = 0.0
            abc.append(rounded)
        sites.append({"species": species, "abc": abc})
    sites.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {"lattice_matrix": matrix, "sites": sites}


def structure_fingerprint(
    structure_json: str,
    *,
    coordinate_decimals: int,
    lattice_decimals: int,
) -> str:
    structure = json.loads(structure_json)
    normalized = normalized_structure_payload(
        structure,
        coordinate_decimals=coordinate_decimals,
        lattice_decimals=lattice_decimals,
    )
    body = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def external_id_map(path: Path) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["material_id", "database_ids_json"]):
        for row in batch.to_pylist():
            if not row["material_id"] or not row["database_ids_json"]:
                continue
            payload = json.loads(row["database_ids_json"])
            if not isinstance(payload, dict):
                continue
            for database, values in payload.items():
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    if value is not None:
                        result[str(row["material_id"])].add(
                            f"{str(database).strip().lower()}:{str(value).strip().lower()}"
                        )
    return {key: frozenset(values) for key, values in result.items()}


def load_descriptors(
    root: Path,
    snapshot: str,
    *,
    coordinate_decimals: int,
    lattice_decimals: int,
) -> list[MaterialDescriptor]:
    snapshot_root = root / f"snapshot={snapshot}"
    external = external_id_map(snapshot_root / "raw_provenance.parquet")
    columns = [
        "material_id",
        "composition_reduced_json",
        "nsites",
        "symmetry_json",
        "volume",
        "task_ids_json",
        "structure_json",
        "deprecated",
        "source_object_sha256",
        "source_key",
        "source_row_number",
    ]
    descriptors: list[MaterialDescriptor] = []
    parquet = pq.ParquetFile(snapshot_root / "raw_material.parquet")
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        for row in batch.to_pylist():
            material_id = str(row["material_id"])
            symmetry = json.loads(row["symmetry_json"]) if row["symmetry_json"] else {}
            nsites = int(row["nsites"]) if row["nsites"] is not None else None
            volume_per_site = (
                float(row["volume"]) / nsites
                if row["volume"] is not None and nsites
                else None
            )
            task_ids = frozenset(
                str(item) for item in json.loads(row["task_ids_json"] or "[]")
            )
            descriptors.append(
                MaterialDescriptor(
                    snapshot=snapshot,
                    material_id=material_id,
                    composition_key=str(row["composition_reduced_json"]),
                    nsites=nsites,
                    spacegroup=(
                        int(symmetry["number"])
                        if symmetry.get("number") is not None
                        else None
                    ),
                    volume_per_site=volume_per_site,
                    task_ids=task_ids,
                    external_ids=external.get(material_id, frozenset()),
                    structure_fingerprint=structure_fingerprint(
                        row["structure_json"],
                        coordinate_decimals=coordinate_decimals,
                        lattice_decimals=lattice_decimals,
                    ),
                    deprecated=row["deprecated"],
                    source_object_sha256=str(row["source_object_sha256"]),
                    source_key=str(row["source_key"]),
                    source_row_number=int(row["source_row_number"]),
                )
            )
    descriptors.sort(key=lambda item: item.material_id)
    if len({item.material_id for item in descriptors}) != len(descriptors):
        raise RuntimeError(f"Duplicate material_id within snapshot {snapshot}")
    return descriptors


def add_anchor(
    anchors: dict[tuple[int, int], int], source_index: int, target_index: int, mask: int
) -> None:
    key = (source_index, target_index)
    anchors[key] = anchors.get(key, 0) | mask


def build_anchor_masks(
    source: list[MaterialDescriptor], target: list[MaterialDescriptor]
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    anchors: dict[tuple[int, int], int] = {}
    target_by_id = {item.material_id: index for index, item in enumerate(target)}
    target_by_task: dict[str, list[int]] = defaultdict(list)
    target_by_external: dict[str, list[int]] = defaultdict(list)
    target_by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(target):
        for task_id in item.task_ids:
            target_by_task[task_id].append(index)
        for external_id in item.external_ids:
            target_by_external[external_id].append(index)
        target_by_fingerprint[item.structure_fingerprint].append(index)
    for values in (target_by_task, target_by_external, target_by_fingerprint):
        for key in values:
            values[key].sort(key=lambda index: target[index].material_id)

    for source_index, item in enumerate(source):
        target_index = target_by_id.get(item.material_id)
        if target_index is not None:
            add_anchor(anchors, source_index, target_index, ANCHOR_SAME_ID)
        for task_id in sorted(item.task_ids):
            for target_index in target_by_task.get(task_id, []):
                add_anchor(anchors, source_index, target_index, ANCHOR_SHARED_TASK)
        for external_id in sorted(item.external_ids):
            for target_index in target_by_external.get(external_id, []):
                add_anchor(anchors, source_index, target_index, ANCHOR_EXTERNAL_ID)
        for target_index in target_by_fingerprint.get(item.structure_fingerprint, []):
            add_anchor(anchors, source_index, target_index, ANCHOR_EXACT_STRUCTURE)
    counts = {
        name: sum(bool(mask & flag) for mask in anchors.values())
        for name, flag in ANCHOR_CATEGORIES.items()
    }
    return anchors, counts


def edge_id(source: MaterialDescriptor, target: MaterialDescriptor) -> bytes:
    body = (
        f"{source.snapshot}|{source.material_id}|"
        f"{target.snapshot}|{target.material_id}"
    ).encode()
    return hashlib.blake2b(body, digest_size=16).digest()


def relative_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def candidate_row(
    source: MaterialDescriptor,
    target: MaterialDescriptor,
    *,
    composition_block: bool,
    anchor_mask: int,
) -> dict[str, Any]:
    shared_tasks = source.task_ids & target.task_ids
    shared_external = source.external_ids & target.external_ids
    exact_structure = source.structure_fingerprint == target.structure_fingerprint
    method_mask = anchor_mask | (16 if composition_block else 0)
    return {
        "edge_id": edge_id(source, target),
        "source_snapshot": source.snapshot,
        "target_snapshot": target.snapshot,
        "source_material_id": source.material_id,
        "target_material_id": target.material_id,
        "source_composition_key": source.composition_key,
        "target_composition_key": target.composition_key,
        "composition_block": composition_block,
        "same_material_id": source.material_id == target.material_id,
        "shared_task_count": len(shared_tasks),
        "shared_external_id_count": len(shared_external),
        "exact_structure_fingerprint": exact_structure,
        "candidate_method_mask": method_mask,
        "cross_composition_anchor": not composition_block,
        "source_nsites": source.nsites,
        "target_nsites": target.nsites,
        "same_nsites": source.nsites is not None and source.nsites == target.nsites,
        "source_spacegroup": source.spacegroup,
        "target_spacegroup": target.spacegroup,
        "same_spacegroup": (
            source.spacegroup is not None and source.spacegroup == target.spacegroup
        ),
        "source_volume_per_site": source.volume_per_site,
        "target_volume_per_site": target.volume_per_site,
        "volume_per_site_relative_difference": relative_difference(
            source.volume_per_site, target.volume_per_site
        ),
        "source_deprecated": source.deprecated,
        "target_deprecated": target.deprecated,
        "source_object_sha256": source.source_object_sha256,
        "target_object_sha256": target.source_object_sha256,
        "source_key": source.source_key,
        "target_key": target.source_key,
        "source_row_number": source.source_row_number,
        "target_row_number": target.source_row_number,
    }


class CandidateWriter:
    def __init__(self, path: Path, *, compression: str, row_group_size: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.row_group_size = row_group_size
        self.writer = pq.ParquetWriter(
            self.temporary,
            EDGE_SCHEMA,
            compression=compression,
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
        table = pa.Table.from_pylist(self.buffer, schema=EDGE_SCHEMA)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            # Reference-audit rows can legitimately have workflow-specific
            # diagnostic columns.  Preserve the first-seen column order while
            # taking the union across every row so no diagnostic is omitted
            # and csv.DictWriter does not fail on a later row.
            fieldnames: list[str] = []
            observed: set[str] = set()
            for row in rows:
                for field in row:
                    if field not in observed:
                        observed.add(field)
                        fieldnames.append(field)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def deterministic_anchor_sample(
    category: str,
    pairs: list[tuple[int, int]],
    source: list[MaterialDescriptor],
    target: list[MaterialDescriptor],
    recalled: set[tuple[int, int]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{category}|{source[pair[0]].material_id}|{target[pair[1]].material_id}".encode()
        ).hexdigest(),
    )[:limit]
    return [
        {
            "anchor_category": category,
            "source_snapshot": source_item.snapshot,
            "target_snapshot": target_item.snapshot,
            "source_material_id": source_item.material_id,
            "target_material_id": target_item.material_id,
            "same_composition": source_item.composition_key
            == target_item.composition_key,
            "candidate_present": pair in recalled,
        }
        for pair in ranked
        for source_item, target_item in [(source[pair[0]], target[pair[1]])]
    ]


def data_dictionary_rows() -> list[dict[str, Any]]:
    descriptions = {
        "edge_id": "Deterministic 128-bit BLAKE2b ID of the directed adjacent-snapshot pair.",
        "candidate_method_mask": "Bit mask: same ID=1, shared task=2, external ID=4, exact structure fingerprint=8, composition block=16.",
        "composition_block": "True when both records have identical canonical reduced composition.",
        "cross_composition_anchor": "True only for an explicit anchor retained despite composition disagreement.",
        "exact_structure_fingerprint": "Schema-tolerant rounded lattice/site fingerprint equality; not a P2.2 StructureMatcher decision.",
        "volume_per_site_relative_difference": "Absolute difference divided by the larger absolute volume per site.",
    }
    return [
        {
            "field": field.name,
            "arrow_type": str(field.type),
            "nullable": field.nullable,
            "description": descriptions.get(
                field.name,
                "Candidate-generation evidence or source-trace field; not a lineage confidence assignment.",
            ),
        }
        for field in EDGE_SCHEMA
    ]


def generate_identity_candidates(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("task_id") != "P2.1":
        raise ValueError("Candidate config task_id must be P2.1")
    snapshots = list(map(str, config["snapshots"]))
    if len(snapshots) < 2 or snapshots != sorted(snapshots):
        raise ValueError("Snapshots must contain at least two sorted frozen versions")
    input_manifest_path = Path(config["input_manifest"])
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("task_id") != "P1.2" or not input_manifest.get("gate_passed"):
        raise RuntimeError("P1.2 normalization manifest is not a passing prerequisite")
    started_at = utc_now()
    fingerprint_config = config["structure_fingerprint"]
    input_root = Path(config["input_root"])
    writer = CandidateWriter(
        Path(config["output_path"]),
        compression=str(config["compression"]),
        row_group_size=int(config["row_group_size"]),
    )
    audit_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    source = load_descriptors(
        input_root,
        snapshots[0],
        coordinate_decimals=int(fingerprint_config["coordinate_decimals"]),
        lattice_decimals=int(fingerprint_config["lattice_decimals"]),
    )
    total_recalled_anchors: dict[str, int] = defaultdict(int)
    total_expected_anchors: dict[str, int] = defaultdict(int)
    for pair_index, (source_snapshot, target_snapshot) in enumerate(
        zip(snapshots, snapshots[1:]), start=1
    ):
        target = load_descriptors(
            input_root,
            target_snapshot,
            coordinate_decimals=int(fingerprint_config["coordinate_decimals"]),
            lattice_decimals=int(fingerprint_config["lattice_decimals"]),
        )
        anchors, anchor_counts = build_anchor_masks(source, target)
        target_by_comp: dict[str, list[int]] = defaultdict(list)
        for target_index, item in enumerate(target):
            target_by_comp[item.composition_key].append(target_index)
        for key in target_by_comp:
            target_by_comp[key].sort(key=lambda index: target[index].material_id)

        recalled: set[tuple[int, int]] = set()
        source_with_candidate: set[int] = set()
        target_with_candidate: set[int] = set()
        pair_start_rows = writer.rows + len(writer.buffer)
        for source_index, source_item in enumerate(source):
            for target_index in target_by_comp.get(source_item.composition_key, []):
                pair = (source_index, target_index)
                writer.add(
                    candidate_row(
                        source_item,
                        target[target_index],
                        composition_block=True,
                        anchor_mask=anchors.get(pair, 0),
                    )
                )
                source_with_candidate.add(source_index)
                target_with_candidate.add(target_index)
                if pair in anchors:
                    recalled.add(pair)
        extra_anchors = sorted(
            (
                pair
                for pair in anchors
                if source[pair[0]].composition_key != target[pair[1]].composition_key
            ),
            key=lambda pair: (
                source[pair[0]].material_id,
                target[pair[1]].material_id,
            ),
        )
        for source_index, target_index in extra_anchors:
            pair = (source_index, target_index)
            writer.add(
                candidate_row(
                    source[source_index],
                    target[target_index],
                    composition_block=False,
                    anchor_mask=anchors[pair],
                )
            )
            source_with_candidate.add(source_index)
            target_with_candidate.add(target_index)
            recalled.add(pair)

        pair_rows = writer.rows + len(writer.buffer) - pair_start_rows
        category_results = {}
        for category, flag in ANCHOR_CATEGORIES.items():
            expected_pairs = [pair for pair, mask in anchors.items() if mask & flag]
            recalled_count = sum(pair in recalled for pair in expected_pairs)
            expected_count = len(expected_pairs)
            recall = recalled_count / expected_count if expected_count else 1.0
            category_results[category] = {
                "expected": expected_count,
                "recalled": recalled_count,
                "recall": recall,
            }
            total_expected_anchors[category] += expected_count
            total_recalled_anchors[category] += recalled_count
            sample_rows.extend(
                deterministic_anchor_sample(
                    category,
                    expected_pairs,
                    source,
                    target,
                    recalled,
                    int(config["audit_gate"]["sample_per_anchor_category"]),
                )
            )
        source_coverage = len(source_with_candidate) / len(source)
        target_coverage = len(target_with_candidate) / len(target)
        pair_summary = {
            "source_snapshot": source_snapshot,
            "target_snapshot": target_snapshot,
            "source_materials": len(source),
            "target_materials": len(target),
            "candidate_edges": pair_rows,
            "composition_block_edges": pair_rows - len(extra_anchors),
            "cross_composition_anchor_edges": len(extra_anchors),
            "source_materials_with_candidate": len(source_with_candidate),
            "source_coverage": source_coverage,
            "target_materials_with_candidate": len(target_with_candidate),
            "target_coverage": target_coverage,
            "anchor_recall": category_results,
        }
        pair_summaries.append(pair_summary)
        print(
            f"candidate_pair={pair_index}/{len(snapshots)-1} "
            f"{source_snapshot}->{target_snapshot} edges={pair_rows} "
            f"source_coverage={source_coverage:.6f}",
            flush=True,
        )
        source = target

    writer.close()
    audit_gate = config["audit_gate"]
    required_recall = float(audit_gate["required_anchor_recall"])
    min_source_coverage = float(audit_gate["minimum_source_material_coverage"])
    aggregate_anchor_recall = {}
    for category in ANCHOR_CATEGORIES:
        expected = total_expected_anchors[category]
        recalled_count = total_recalled_anchors[category]
        recall = recalled_count / expected if expected else 1.0
        aggregate_anchor_recall[category] = {
            "expected": expected,
            "recalled": recalled_count,
            "recall": recall,
            "status": "PASS" if recall >= required_recall else "FAIL",
        }
    gate_checks = {
        "all_anchor_categories_meet_recall": all(
            item["recall"] >= required_recall
            for item in aggregate_anchor_recall.values()
        ),
        "all_pairs_meet_source_coverage": all(
            item["source_coverage"] >= min_source_coverage for item in pair_summaries
        ),
        "candidate_edges_nonzero": writer.rows > 0,
        "adjacent_forward_snapshots_only": True,
        "unique_by_generation_contract": True,
    }
    gate_passed = all(gate_checks.values())
    output_path = Path(config["output_path"])
    parquet = pq.ParquetFile(output_path)
    if parquet.metadata.num_rows != writer.rows:
        raise RuntimeError("Candidate Parquet row count does not match writer total")
    if parquet.schema_arrow.remove_metadata() != EDGE_SCHEMA.remove_metadata():
        raise RuntimeError("Candidate Parquet schema does not match EDGE_SCHEMA")
    audit = {
        "task_id": "P2.1",
        "status": "PASS" if gate_passed else "FAIL",
        "required_anchor_recall": required_recall,
        "minimum_source_material_coverage": min_source_coverage,
        "anchor_recall": aggregate_anchor_recall,
        "pair_summaries": pair_summaries,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "audit_sample_rows": len(sample_rows),
        "interpretation": (
            "Anchor recall measures candidate-generation coverage only. It is not "
            "a precision estimate and does not assign lineage confidence."
        ),
    }
    report_dir = Path(config["report_dir"])
    write_json_atomic(report_dir / "candidate_recall_audit.json", audit)
    write_csv_atomic(report_dir / "recall_audit_sample.csv", sample_rows)
    dictionary = data_dictionary_rows()
    write_csv_atomic(report_dir / "data_dictionary.csv", dictionary)
    ledger_rows.extend(
        [
            {
                "record_type": "semantic_boundary",
                "reason": "candidate_edge_is_not_identity_decision",
                "action": "confidence assignment and lineage construction deferred to P2.2",
            },
            {
                "record_type": "semantic_boundary",
                "reason": "same_material_id_is_not_permanent_identity",
                "action": "retained only as a recall anchor, never as automatic lineage truth",
            },
            {
                "record_type": "semantic_boundary",
                "reason": "rounded_exact_structure_fingerprint_is_not_structurematcher_equivalence",
                "action": "retained as a high-specificity candidate anchor; full matching deferred to P2.2",
            },
            {
                "record_type": "candidate_policy",
                "reason": "all_same_reduced_composition_pairs_retained",
                "action": "no top-k pruning; precision work deferred to P2.2",
            },
        ]
    )
    manifest_dir = Path(config["manifest_dir"])
    write_jsonl_atomic(manifest_dir / "exclusion_ambiguity_ledger.jsonl", ledger_rows)
    manifest = {
        "task_id": "P2.1",
        "status": "PASS" if gate_passed else "FAIL",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "input_manifest": input_manifest_path.as_posix(),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "output_path": output_path.as_posix(),
        "output_rows": writer.rows,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "schema_sha256": hashlib.sha256(
            str(parquet.schema_arrow.remove_metadata()).encode()
        ).hexdigest(),
        "pair_summaries": pair_summaries,
        "audit_path": (report_dir / "candidate_recall_audit.json").as_posix(),
        "audit_sha256": sha256_file(report_dir / "candidate_recall_audit.json"),
        "audit_sample_path": (report_dir / "recall_audit_sample.csv").as_posix(),
        "audit_sample_sha256": sha256_file(report_dir / "recall_audit_sample.csv"),
        "data_dictionary_path": (report_dir / "data_dictionary.csv").as_posix(),
        "data_dictionary_sha256": sha256_file(report_dir / "data_dictionary.csv"),
        "ledger_path": (manifest_dir / "exclusion_ambiguity_ledger.jsonl").as_posix(),
        "ledger_sha256": sha256_file(manifest_dir / "exclusion_ambiguity_ledger.jsonl"),
        "gate_passed": gate_passed,
        "network_access": False,
        "seed": int(config["seed"]),
    }
    write_json_atomic(manifest_dir / "candidate_manifest.json", manifest)
    if not gate_passed:
        raise RuntimeError("P2.1 candidate recall audit gate failed")
    return manifest


def verify_identity_candidates(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest_dir"]) / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = Path(manifest["output_path"])
    parquet = pq.ParquetFile(output_path)
    failures = []
    if parquet.metadata.num_rows != manifest["output_rows"]:
        failures.append("row_count")
    if parquet.schema_arrow.remove_metadata() != EDGE_SCHEMA.remove_metadata():
        failures.append("schema")
    if sha256_file(output_path) != manifest["output_sha256"]:
        failures.append("sha256")
    verification_columns = parquet.read(
        columns=["edge_id", "source_snapshot", "target_snapshot"]
    )
    distinct_edge_ids = int(
        pc.count_distinct(verification_columns["edge_id"]).as_py()
    )
    if distinct_edge_ids != parquet.metadata.num_rows:
        failures.append("duplicate_edge_id")
    observed_pairs = set(
        zip(
            verification_columns["source_snapshot"].to_pylist(),
            verification_columns["target_snapshot"].to_pylist(),
        )
    )
    snapshots = list(map(str, config["snapshots"]))
    expected_pairs = set(zip(snapshots, snapshots[1:]))
    if observed_pairs != expected_pairs:
        failures.append("non_adjacent_or_reverse_snapshot_pair")
    audit = json.loads(Path(manifest["audit_path"]).read_text(encoding="utf-8"))
    if not audit.get("gate_passed"):
        failures.append("audit_gate")
    sample_path = Path(manifest["audit_sample_path"])
    with sample_path.open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    if not sample or any(row["candidate_present"].lower() != "true" for row in sample):
        failures.append("audit_sample")
    temporary_files = list(output_path.parent.rglob("*.tmp"))
    if temporary_files:
        failures.append("temporary_files")
    if failures:
        raise RuntimeError(f"P2.1 verification failed: {failures}")
    return {
        "status": "PASS",
        "verified_at_utc": utc_now(),
        "output_path": output_path.as_posix(),
        "output_rows": parquet.metadata.num_rows,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": manifest["output_sha256"],
        "distinct_edge_ids": distinct_edge_ids,
        "unique_edges": distinct_edge_ids == parquet.metadata.num_rows,
        "snapshot_pairs": [list(pair) for pair in sorted(observed_pairs)],
        "schema_sha256": manifest["schema_sha256"],
        "audit_gate": "PASS",
        "audit_sample_rows": len(sample),
        "temporary_files": 0,
        "network_access": False,
    }
