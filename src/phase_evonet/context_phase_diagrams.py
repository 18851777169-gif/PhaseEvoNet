from __future__ import annotations

import copy
import hashlib
import heapq
import itertools
import json
import math
import os
import warnings
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

from .identity_candidates import write_csv_atomic, write_json_atomic, write_jsonl_atomic
from .manifest import sha256_file
from .reported_transitions import DataFrameParquetWriter, utc_now
from .unified_phase_diagrams import DUPLICATE_SCHEMA, deduplicate_entries


ENTRY_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("phase_context_chemsys", pa.string(), nullable=False),
        pa.field("unified_entry_id", pa.binary(16), nullable=False),
        pa.field("entry_id", pa.string(), nullable=False),
        pa.field("is_target", pa.bool_(), nullable=False),
        pa.field("is_competitor", pa.bool_(), nullable=False),
        pa.field("source_workflow", pa.string(), nullable=False),
        pa.field("compatibility_mode", pa.string(), nullable=False),
        pa.field("task_id", pa.string()),
        pa.field("material_id", pa.string()),
        pa.field("thermo_id", pa.string()),
        pa.field("entry_label", pa.string()),
        pa.field("run_type", pa.string()),
        pa.field("composition_json", pa.large_string(), nullable=False),
        pa.field("reduced_formula", pa.string(), nullable=False),
        pa.field("chemsys", pa.string(), nullable=False),
        pa.field("nelements", pa.int16(), nullable=False),
        pa.field("num_atoms", pa.float64(), nullable=False),
        pa.field("uncorrected_energy", pa.float64(), nullable=False),
        pa.field("correction", pa.float64(), nullable=False),
        pa.field("corrected_energy", pa.float64(), nullable=False),
        pa.field("corrected_energy_per_atom", pa.float64(), nullable=False),
        pa.field("formation_energy_per_atom", pa.float64(), nullable=False),
        pa.field("energy_above_hull", pa.float64(), nullable=False),
        pa.field("is_stable", pa.bool_(), nullable=False),
        pa.field("decomposition_component_count", pa.int16(), nullable=False),
        pa.field("phase_diagram_status", pa.string(), nullable=False),
        pa.field("source_record_count", pa.int32(), nullable=False),
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
        pa.field("phase_context_chemsys", pa.string(), nullable=False),
        pa.field("unified_entry_id", pa.binary(16), nullable=False),
        pa.field("component_unified_entry_id", pa.binary(16), nullable=False),
        pa.field("component_entry_id", pa.string(), nullable=False),
        pa.field("component_formula", pa.string(), nullable=False),
        pa.field("amount", pa.float64(), nullable=False),
        pa.field("component_energy_per_atom", pa.float64(), nullable=False),
    ]
)

ENTRY_COLUMNS = [
    "snapshot_id", "material_id", "thermo_id", "thermo_type", "entry_label",
    "entry_id", "task_id", "energy", "correction", "composition_json",
    "energy_adjustments_json", "parameters_json", "run_type", "hubbards_json",
    "potcar_spec_json", "structure_json", "entry_data_json",
    "source_object_sha256", "source_key", "source_row_number",
]

RECONCILIATION_COLUMNS = [
    "snapshot_id", "source_workflow", "phase_context_chemsys", "thermo_id",
    "entry_id", "entry_label", "serialized_uncorrected_energy_eV",
    "serialized_corrected_energy_eV", "reported_uncorrected_energy_eV",
    "reported_corrected_energy_eV", "energy_delta_eV_per_atom",
    "correction_delta_eV_per_atom", "resolution", "source_object_sha256",
    "source_key", "source_row_number",
]

MIXING_ALIAS_COLUMNS = [
    "snapshot_id", "thermo_type", "phase_context_chemsys", "target_entry_id",
    "representative_entry_id", "target_source_workflow",
    "representative_source_workflow", "target_run_type",
    "representative_run_type", "target_energy_per_atom_eV",
    "representative_energy_per_atom_eV", "resolution",
    "source_object_sha256", "source_key", "source_row_number",
]

COMPATIBILITY_EXCLUSION_COLUMNS = [
    "snapshot_id", "thermo_type", "thermo_id", "selected_entry_id",
    "selected_entry_label", "phase_context_chemsys", "reason",
    "rule_version", "processed_target_present", "mixing_state_candidate_count",
    "structural_match_count", "candidate_representative_ids_json",
    "source_object_sha256", "source_key", "source_row_number",
]


def contextual_entry_id(
    snapshot: str, thermo_type: str, phase_context_chemsys: str, entry_id: str
) -> bytes:
    body = f"{snapshot}|{thermo_type}|{phase_context_chemsys}|{entry_id}".encode()
    return hashlib.blake2b(body, digest_size=16).digest()


def chemical_space(composition_json: str) -> tuple[str, ...]:
    return tuple(sorted(json.loads(composition_json)))


def chemical_space_text(space: Iterable[str]) -> str:
    return "-".join(sorted(space))


def subspaces(space: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for size in range(1, len(space) + 1):
        yield from itertools.combinations(space, size)


def _optional(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def _adjustments_json(entry: ComputedEntry) -> str:
    return json.dumps(
        [item.as_dict() for item in entry.energy_adjustments],
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_rows(root: Path, snapshot: str, thermo_type: str) -> pd.DataFrame:
    path = root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet"
    frame = pq.read_table(
        path, columns=ENTRY_COLUMNS, filters=[("thermo_type", "=", thermo_type)]
    ).to_pandas()
    frame["_context_only"] = False
    supplement = root / f"snapshot={snapshot}" / "raw_terminal_entry_supplement.parquet"
    if supplement.is_file():
        extra = pq.read_table(
            supplement,
            columns=ENTRY_COLUMNS,
            filters=[("thermo_type", "=", thermo_type)],
        ).to_pandas()
        if not extra.empty:
            extra["_context_only"] = True
            frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def _phase_source_hash(root: Path, snapshot: str) -> str:
    primary = root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet"
    supplement = root / f"snapshot={snapshot}" / "raw_terminal_entry_supplement.parquet"
    primary_hash = sha256_file(primary)
    if not supplement.is_file():
        return primary_hash
    body = f"raw_thermo_entry={primary_hash}\nterminal_supplement={sha256_file(supplement)}\n"
    return hashlib.sha256(body.encode()).hexdigest()


def _selected_rows(
    root: Path,
    snapshot: str,
    thermo_type: str,
    tolerance: float,
    *,
    reference_tolerance: float,
    reconcile_reported_energy: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    selected, duplicates, ambiguities = deduplicate_entries(
        _source_rows(root, snapshot, thermo_type), tolerance=tolerance
    )
    selected["_space"] = selected["composition_json"].map(chemical_space)
    reconciliations: list[dict[str, Any]] = []
    if reconcile_reported_energy:
        selected, reconciliation_errors, reconciliations = _reconcile_reported_energy(
            root=root,
            snapshot=snapshot,
            thermo_type=thermo_type,
            selected=selected,
            tolerance=reference_tolerance,
        )
        ambiguities.extend(reconciliation_errors)
    return selected, duplicates, ambiguities, reconciliations


def _row_index(frame: pd.DataFrame) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    result: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for space, group in frame.groupby("_space", sort=False):
        result[tuple(space)] = group.to_dict(orient="records")
    return result


def _context_rows(
    index: dict[tuple[str, ...], list[dict[str, Any]]], space: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in subspaces(space):
        rows.extend(index.get(subset, []))
    return rows


def _entry_row(
    *,
    source: dict[str, Any],
    entry: ComputedEntry,
    snapshot: str,
    thermo_type: str,
    context: str,
    source_workflow: str,
    compatibility_mode: str,
    phase_diagram: PhaseDiagram,
    hull: float,
    component_count: int,
    is_target: bool,
    is_competitor: bool,
) -> dict[str, Any]:
    composition = entry.composition
    correction = float(entry.energy - entry.uncorrected_energy)
    return {
        "snapshot_id": snapshot,
        "thermo_type": thermo_type,
        "phase_context_chemsys": context,
        "unified_entry_id": contextual_entry_id(snapshot, thermo_type, context, str(entry.entry_id)),
        "entry_id": str(entry.entry_id),
        "is_target": is_target,
        "is_competitor": is_competitor,
        "source_workflow": source_workflow,
        "compatibility_mode": compatibility_mode,
        "task_id": _optional(source.get("task_id")),
        "material_id": _optional(source.get("material_id")),
        "thermo_id": _optional(source.get("thermo_id")),
        "entry_label": _optional(source.get("entry_label")),
        "run_type": _optional(entry.parameters.get("run_type") if hasattr(entry, "parameters") else source.get("run_type")),
        "composition_json": str(source["composition_json"]),
        "reduced_formula": composition.reduced_formula,
        "chemsys": chemical_space_text(composition.get_el_amt_dict()),
        "nelements": len(composition.elements),
        "num_atoms": float(composition.num_atoms),
        "uncorrected_energy": float(entry.uncorrected_energy),
        "correction": correction,
        "corrected_energy": float(entry.energy),
        "corrected_energy_per_atom": float(entry.energy_per_atom),
        "formation_energy_per_atom": float(phase_diagram.get_form_energy_per_atom(entry)),
        "energy_above_hull": float(max(0.0, hull)),
        "is_stable": float(hull) <= PhaseDiagram.formation_energy_tol,
        "decomposition_component_count": int(component_count),
        "phase_diagram_status": "computed",
        "source_record_count": int(source.get("source_record_count", 1)),
        "energy_adjustments_json": _adjustments_json(entry),
        "parameters_json": _optional(source.get("parameters_json")),
        "hubbards_json": _optional(source.get("hubbards_json")),
        "potcar_spec_json": _optional(source.get("potcar_spec_json")),
        "entry_data_json": _optional(source.get("entry_data_json")),
        "source_object_sha256": str(source["source_object_sha256"]),
        "source_key": str(source["source_key"]),
        "source_row_number": int(source["source_row_number"]),
    }


def _add_target_and_decomposition(
    *,
    target_source: dict[str, Any],
    target_entry: ComputedEntry,
    entries_by_id: dict[str, ComputedEntry],
    sources_by_id: dict[str, dict[str, Any]],
    phase_diagram: PhaseDiagram,
    snapshot: str,
    thermo_type: str,
    context: str,
    source_workflow_by_id: dict[str, str],
    compatibility_mode: str,
    phase_rows: dict[bytes, dict[str, Any]],
    decomp_rows: list[dict[str, Any]],
    stable_tolerance: float,
) -> None:
    decomposition, hull = phase_diagram.get_decomp_and_e_above_hull(
        target_entry, check_stable=True, on_error="raise"
    )
    hull = float(hull)
    if hull < -stable_tolerance:
        raise RuntimeError(f"negative hull energy for {target_entry.entry_id}: {hull}")
    target_id = str(target_entry.entry_id)
    uid = contextual_entry_id(snapshot, thermo_type, context, target_id)
    components = sorted(decomposition.items(), key=lambda item: str(item[0].entry_id))
    phase_rows[uid] = _entry_row(
        source=target_source,
        entry=target_entry,
        snapshot=snapshot,
        thermo_type=thermo_type,
        context=context,
        source_workflow=source_workflow_by_id[target_id],
        compatibility_mode=compatibility_mode,
        phase_diagram=phase_diagram,
        hull=hull,
        component_count=len(components),
        is_target=True,
        is_competitor=uid in phase_rows and bool(phase_rows[uid].get("is_competitor")),
    )
    for component, amount in components:
        component_id = str(component.entry_id)
        component_uid = contextual_entry_id(snapshot, thermo_type, context, component_id)
        if component_uid not in phase_rows:
            component_source = sources_by_id[component_id]
            phase_rows[component_uid] = _entry_row(
                source=component_source,
                entry=entries_by_id[component_id],
                snapshot=snapshot,
                thermo_type=thermo_type,
                context=context,
                source_workflow=source_workflow_by_id[component_id],
                compatibility_mode=(
                    "official_material_terminal_supplement"
                    if bool(component_source.get("_context_only", False))
                    else compatibility_mode
                ),
                phase_diagram=phase_diagram,
                hull=0.0,
                component_count=0,
                is_target=False,
                is_competitor=True,
            )
        else:
            phase_rows[component_uid]["is_competitor"] = True
        decomp_rows.append(
            {
                "snapshot_id": snapshot,
                "thermo_type": thermo_type,
                "phase_context_chemsys": context,
                "unified_entry_id": uid,
                "component_unified_entry_id": component_uid,
                "component_entry_id": component_id,
                "component_formula": component.composition.reduced_formula,
                "amount": float(amount),
                "component_energy_per_atom": float(component.energy_per_atom),
            }
        )


def solve_homogeneous_contexts(
    selected: pd.DataFrame, *, stable_tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    snapshot = str(selected.iloc[0]["snapshot_id"])
    thermo_type = str(selected.iloc[0]["thermo_type"])
    index = _row_index(selected)
    errors: list[dict[str, Any]] = []
    warning_messages: set[str] = set()
    phase_rows: dict[bytes, dict[str, Any]] = {}
    decomp_rows: list[dict[str, Any]] = []
    for space in sorted(index, key=lambda item: (len(item), item)):
        context = chemical_space_text(space)
        source_rows = _context_rows(index, space)
        entries: list[ComputedEntry] = []
        sources_by_id: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            entry_id = str(row["entry_id"])
            entries.append(
                ComputedEntry(
                    Composition(json.loads(row["composition_json"])),
                    float(row["corrected_energy"]),
                    entry_id=entry_id,
                    parameters=json.loads(row["parameters_json"] or "{}"),
                )
            )
            sources_by_id[entry_id] = row
        entries_by_id = {str(entry.entry_id): entry for entry in entries}
        workflow_by_id = {entry_id: thermo_type for entry_id in entries_by_id}
        try:
            phase_diagram = PhaseDiagram(entries)
            targets = [
                row for row in index[space] if not bool(row.get("_context_only", False))
            ]
            for target in sorted(targets, key=lambda row: str(row["entry_id"])):
                _add_target_and_decomposition(
                    target_source=target,
                    target_entry=entries_by_id[str(target["entry_id"])],
                    entries_by_id=entries_by_id,
                    sources_by_id=sources_by_id,
                    phase_diagram=phase_diagram,
                    snapshot=snapshot,
                    thermo_type=thermo_type,
                    context=context,
                    source_workflow_by_id=workflow_by_id,
                    compatibility_mode="source_corrected_homogeneous",
                    phase_rows=phase_rows,
                    decomp_rows=decomp_rows,
                    stable_tolerance=stable_tolerance,
                )
        except Exception as exc:
            errors.append(
                {
                    "snapshot_id": snapshot,
                    "thermo_type": thermo_type,
                    "phase_context_chemsys": context,
                    "reason": "homogeneous_context_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return (
        pd.DataFrame(list(phase_rows.values()), columns=ENTRY_SCHEMA.names),
        pd.DataFrame(decomp_rows, columns=DECOMPOSITION_SCHEMA.names),
        errors,
        sorted(warning_messages),
    )


class _StructureEntryCache:
    def __init__(self, limit: int = 20000):
        self.limit = limit
        self.cache: OrderedDict[str, ComputedStructureEntry] = OrderedDict()
        self.warning_counts: dict[str, int] = {}

    def get(self, row: dict[str, Any]) -> ComputedStructureEntry:
        entry_id = str(row["entry_id"])
        if entry_id in self.cache:
            entry = self.cache.pop(entry_id)
            self.cache[entry_id] = entry
            return entry
        parameters = json.loads(row["parameters_json"] or "{}")
        run_type = str(parameters.get("run_type") or row.get("run_type") or "")
        if run_type.lower() == "r2scan":
            parameters["run_type"] = "R2SCAN"
        entry_data = json.loads(row["entry_data_json"] or "{}")
        oxidation_states = entry_data.get("oxidation_states")
        if isinstance(oxidation_states, list):
            if oxidation_states:
                raise ValueError(
                    "oxidation_states list must be empty before deterministic "
                    "mapping normalization"
                )
            entry_data["oxidation_states"] = {}
            warning = "empty oxidation_states list normalized to empty mapping"
            self.warning_counts[warning] = self.warning_counts.get(warning, 0) + 1
        elif oxidation_states is not None and not isinstance(oxidation_states, dict):
            raise TypeError(
                "oxidation_states must be a mapping, an empty list, or null; "
                f"received {type(oxidation_states).__name__}"
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = ComputedStructureEntry(
                Structure.from_dict(json.loads(row["structure_json"])),
                float(row["energy"]),
                parameters=parameters,
                data=entry_data,
                entry_id=entry_id,
            )
        for item in caught:
            text = str(item.message)
            self.warning_counts[text] = self.warning_counts.get(text, 0) + 1
        self.cache[entry_id] = entry
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return entry


def _thermo_selection(root: Path, snapshot: str, thermo_type: str) -> pd.DataFrame:
    path = root / f"snapshot={snapshot}" / "raw_thermo.parquet"
    frame = pq.read_table(
        path,
        columns=[
            "thermo_id", "thermo_type", "energy_type", "chemsys",
            "energy_per_atom", "uncorrected_energy_per_atom",
            "energy_above_hull", "formation_energy_per_atom",
            "source_object_sha256", "source_key", "source_row_number",
        ],
        filters=[("thermo_type", "=", thermo_type)],
    ).to_pandas()
    return frame.sort_values(
        ["thermo_id", "source_object_sha256", "source_key", "source_row_number"],
        kind="mergesort",
    ).drop_duplicates(["thermo_id", "thermo_type"], keep="first")


def _thermo_embedded_target_rows(
    root: Path, snapshot: str, thermo_type: str
) -> pd.DataFrame:
    """Read only columns needed to resolve a thermo document's target entry."""
    path = root / f"snapshot={snapshot}" / "raw_thermo_entry.parquet"
    return pq.read_table(
        path,
        columns=["thermo_id", "entry_label", "entry_id", "composition_json"],
        filters=[("thermo_type", "=", thermo_type)],
    ).to_pandas()


def _thermo_target_selection(
    root: Path, snapshot: str, thermo_type: str
) -> pd.DataFrame:
    """Map every frozen thermo source state to its explicitly selected entry.

    ``energy_type`` selects one embedded entry label for the thermo document.
    The contextual phase table is keyed by entry and chemical system rather
    than thermo provenance, so this mapping is the auditable bridge from each
    source state to its target and legitimately supports many source states
    referring to the same contextual entry.
    """
    source = _thermo_selection(root, snapshot, thermo_type).copy()
    embedded = _thermo_embedded_target_rows(root, snapshot, thermo_type).copy()
    embedded["_entry_context"] = embedded["composition_json"].map(
        lambda value: chemical_space_text(chemical_space(value))
    )
    embedded["_entry_label_key"] = embedded["entry_label"].astype(str).str.upper()
    source["_entry_label_key"] = source["energy_type"].astype(str).str.upper()
    candidates = source[
        ["thermo_id", "_entry_label_key", "chemsys"]
    ].merge(
        embedded,
        on=["thermo_id", "_entry_label_key"],
        how="left",
        validate="one_to_many",
        sort=False,
    )
    candidates = candidates[
        candidates["_entry_context"].eq(candidates["chemsys"])
    ].drop_duplicates(
        ["thermo_id", "entry_id", "entry_label", "_entry_context"],
        keep="first",
    )
    counts = candidates.groupby("thermo_id", sort=False)["entry_id"].nunique()
    invalid = source.loc[
        ~source["thermo_id"].map(counts).fillna(0).eq(1), "thermo_id"
    ].astype(str).tolist()
    if invalid:
        raise RuntimeError(
            "Frozen thermo target selection is not one-to-one for "
            f"{snapshot}/{thermo_type}: {invalid[:10]}"
        )
    selected = candidates.sort_values(
        ["thermo_id", "entry_id", "entry_label"], kind="mergesort"
    ).drop_duplicates("thermo_id", keep="first").rename(
        columns={
            "entry_id": "selected_entry_id",
            "entry_label": "selected_entry_label",
        }
    )
    result = source.drop(columns=["_entry_label_key"]).merge(
        selected[
            ["thermo_id", "selected_entry_id", "selected_entry_label"]
        ],
        on="thermo_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    return result


def _mixed_target_rows(
    mixed_selected: pd.DataFrame, target_selection: pd.DataFrame
) -> pd.DataFrame:
    """Return unique entries explicitly selected by at least one source state."""
    target_keys = target_selection[
        ["selected_entry_id", "selected_entry_label", "energy_type", "chemsys"]
    ].copy()
    target_keys["_entry_label_key"] = (
        target_keys["selected_entry_label"].astype(str).str.upper()
    )
    target_keys = target_keys.sort_values(
        ["selected_entry_id", "chemsys", "energy_type"], kind="mergesort"
    ).drop_duplicates(["selected_entry_id", "chemsys"], keep="first")
    chosen = mixed_selected.merge(
        target_keys,
        left_on="entry_id",
        right_on="selected_entry_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    chosen["_context"] = chosen["_space"].map(chemical_space_text)
    chosen = chosen[
        chosen["_context"].eq(chosen["chemsys"])
        & chosen["entry_label"].astype(str).str.upper().eq(
            chosen["_entry_label_key"]
        )
    ].copy()
    expected = set(
        zip(
            target_keys["selected_entry_id"].astype(str),
            target_keys["chemsys"].astype(str),
            strict=True,
        )
    )
    actual = set(
        zip(chosen["entry_id"].astype(str), chosen["_context"], strict=True)
    )
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Selected mixed targets absent after deduplication: {missing[:10]}")
    return chosen


def _reconcile_reported_energy(
    *,
    root: Path,
    snapshot: str,
    thermo_type: str,
    selected: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve frozen thermo top-level/embedded-entry energy inconsistencies.

    The top-level energy pair is authoritative only for the entry explicitly
    selected by ``energy_type`` in its native chemical system.  A correction is
    applied only when the reported corrected-minus-uncorrected energy agrees
    with the serialized entry correction within the locked reference tolerance.
    Otherwise the record is ambiguous and remains unmodified.
    """
    reported = _thermo_selection(root, snapshot, thermo_type)[
        [
            "thermo_id", "energy_type", "chemsys", "energy_per_atom",
            "uncorrected_energy_per_atom",
        ]
    ].rename(
        columns={
            "energy_type": "_reported_energy_type",
            "chemsys": "_reported_chemsys",
            "energy_per_atom": "_reported_corrected_energy_per_atom",
            "uncorrected_energy_per_atom": "_reported_uncorrected_energy_per_atom",
        }
    )
    frame = selected.merge(reported, on="thermo_id", how="left", validate="many_to_one", sort=False)
    atom_counts = frame["composition_json"].map(lambda value: float(sum(json.loads(value).values())))
    frame["_serialized_corrected_energy_per_atom"] = frame["corrected_energy"] / atom_counts
    eligible = (
        frame["entry_label"].eq(frame["_reported_energy_type"])
        & frame["_space"].map(chemical_space_text).eq(frame["_reported_chemsys"])
        & frame["_reported_corrected_energy_per_atom"].notna()
        & frame["_reported_uncorrected_energy_per_atom"].notna()
    )
    delta = (
        frame["_reported_corrected_energy_per_atom"]
        - frame["_serialized_corrected_energy_per_atom"]
    ).abs()
    correction_delta = (
        (
            frame["_reported_corrected_energy_per_atom"]
            - frame["_reported_uncorrected_energy_per_atom"]
        )
        - frame["correction"] / atom_counts
    ).abs()
    needs_resolution = eligible & delta.gt(tolerance)
    resolvable = needs_resolution & correction_delta.le(tolerance)
    unresolved = needs_resolution & ~correction_delta.le(tolerance)
    errors: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    for index in frame.index[unresolved]:
        row = frame.loc[index]
        errors.append(
            {
                "snapshot_id": snapshot,
                "thermo_type": thermo_type,
                "phase_context_chemsys": chemical_space_text(row["_space"]),
                "thermo_id": str(row["thermo_id"]),
                "entry_id": str(row["entry_id"]),
                "reason": "reported_entry_energy_inconsistent",
                "energy_delta_eV_per_atom": float(delta.loc[index]),
                "correction_delta_eV_per_atom": float(correction_delta.loc[index]),
            }
        )
    for index in frame.index[resolvable]:
        row = frame.loc[index]
        atoms = float(atom_counts.loc[index])
        serialized_uncorrected = float(row["energy"])
        serialized_corrected = float(row["corrected_energy"])
        reported_uncorrected = float(row["_reported_uncorrected_energy_per_atom"]) * atoms
        reported_corrected = float(row["_reported_corrected_energy_per_atom"]) * atoms
        frame.at[index, "energy"] = reported_uncorrected
        frame.at[index, "corrected_energy"] = reported_corrected
        reconciliations.append(
            {
                "snapshot_id": snapshot,
                "source_workflow": thermo_type,
                "phase_context_chemsys": chemical_space_text(row["_space"]),
                "thermo_id": str(row["thermo_id"]),
                "entry_id": str(row["entry_id"]),
                "entry_label": str(row["entry_label"]),
                "serialized_uncorrected_energy_eV": serialized_uncorrected,
                "serialized_corrected_energy_eV": serialized_corrected,
                "reported_uncorrected_energy_eV": reported_uncorrected,
                "reported_corrected_energy_eV": reported_corrected,
                "energy_delta_eV_per_atom": float(delta.loc[index]),
                "correction_delta_eV_per_atom": float(correction_delta.loc[index]),
                "resolution": "frozen_thermo_top_level_energy_pair",
                "source_object_sha256": str(row["source_object_sha256"]),
                "source_key": str(row["source_key"]),
                "source_row_number": int(row["source_row_number"]),
            }
        )
    internal = [column for column in frame.columns if column.startswith("_reported_")]
    internal.append("_serialized_corrected_energy_per_atom")
    frame = frame.drop(columns=internal)
    return frame, errors, reconciliations


def _read_part(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _has_exact_r2scan_target(rows: list[dict[str, Any]], context: str) -> bool:
    """Return whether this exact chemical system selects an R2SCAN target.

    A thermo document can contain both GGA and R2SCAN entries under one
    ``thermo_id``.  Entries reused as competitors can also retain the thermo
    document provenance of a larger chemical system.  Consequently neither a
    unique ``thermo_id`` assumption nor the mere presence of an R2SCAN entry is
    sufficient: label, selected energy type, and exact chemical system must all
    agree.
    """
    return any(
        str(row.get("entry_label", "")).upper() == "R2SCAN"
        and str(row.get("energy_type", "")).upper() == "R2SCAN"
        and str(row.get("chemsys", "")) == context
        for row in rows
    )


class _AuditedMixingScheme(MaterialsProjectDFTMixingScheme):
    """Make the frozen mixing implementation deterministic and auditable.

    The upstream implementation temporarily stores compatibility-processed
    entries in :class:`EntrySet`, whose backing ``set`` does not preserve input
    order.  Structure grouping uses the resulting order to break equal-energy
    ties, so separate Python processes can otherwise retain different entries.
    Sorting by the already-required unique entry ID restores a stable order
    without changing compatibility, matching, or energy rules.
    """

    mixing_state_data: pd.DataFrame | None = None

    def _filter_and_sort_entries(
        self, entries: list[ComputedStructureEntry], verbose: bool = False
    ) -> tuple[list[ComputedStructureEntry], list[ComputedStructureEntry]]:
        entries_type_1, entries_type_2 = super()._filter_and_sort_entries(
            entries, verbose=verbose
        )
        key = lambda entry: str(entry.entry_id)
        return sorted(entries_type_1, key=key), sorted(entries_type_2, key=key)

    def get_mixing_state_data(self, entries: list[ComputedStructureEntry]) -> pd.DataFrame:
        state = super().get_mixing_state_data(entries)
        self.mixing_state_data = state
        return state


def _resolve_mixing_duplicate_alias(
    target: ComputedStructureEntry,
    processed: list[ComputedStructureEntry],
    scheme: MaterialsProjectDFTMixingScheme,
    mixing_state_data: pd.DataFrame | None = None,
) -> tuple[ComputedStructureEntry, ComputedStructureEntry, str] | None:
    """Return the explicit or uniquely matched representative for a discarded target."""
    resolution, _ = _resolve_mixing_duplicate_alias_with_evidence(
        target, processed, scheme, mixing_state_data
    )
    return resolution


def _resolve_mixing_duplicate_alias_with_evidence(
    target: ComputedStructureEntry,
    processed: list[ComputedStructureEntry],
    scheme: MaterialsProjectDFTMixingScheme,
    mixing_state_data: pd.DataFrame | None = None,
) -> tuple[
    tuple[ComputedStructureEntry, ComputedStructureEntry, str] | None,
    dict[str, Any],
]:
    """Resolve a discarded target and retain label-blind eligibility evidence."""
    processed_by_id = {str(entry.entry_id): entry for entry in processed}
    mixing_candidate_ids: list[str] = []
    if mixing_state_data is not None:
        rows = mixing_state_data[
            mixing_state_data["entry_id_1"].eq(target.entry_id)
            | mixing_state_data["entry_id_2"].eq(target.entry_id)
        ]
        for _, row in rows.iterrows():
            mixing_candidate_ids.extend(
                str(value) for value in (row["entry_id_1"], row["entry_id_2"])
                if value is not None and not pd.isna(value) and str(value) != str(target.entry_id)
            )
        mixing_candidate_ids = sorted(set(mixing_candidate_ids))
        if len(rows) == 1:
            counterpart_ids = mixing_candidate_ids
            counterparts = [processed_by_id[value] for value in counterpart_ids if value in processed_by_id]
            if len(counterparts) == 1:
                representative = counterparts[0]
                alias = copy.deepcopy(representative)
                alias.entry_id = target.entry_id
                return (
                    alias,
                    representative,
                    "mixing_state_cross_workflow_counterpart",
                ), {
                    "processed_target_present": str(target.entry_id) in processed_by_id,
                    "mixing_state_candidate_count": len(mixing_candidate_ids),
                    "structural_match_count": 0,
                    "candidate_representative_ids": mixing_candidate_ids,
                }
    target_run_type = str(target.parameters.get("run_type", ""))
    target_family = (
        1 if target_run_type in scheme.valid_rtypes_1
        else 2 if target_run_type in scheme.valid_rtypes_2
        else 0
    )
    matches: list[ComputedStructureEntry] = []
    for candidate in processed:
        if candidate.composition != target.composition:
            continue
        try:
            structural_match = scheme.structure_matcher.fit(target.structure, candidate.structure)
        except Exception:
            structural_match = False
        if not structural_match:
            continue
        matches.append(candidate)
    same_family = [
        candidate for candidate in matches
        if (
            1 if str(candidate.parameters.get("run_type", "")) in scheme.valid_rtypes_1
            else 2 if str(candidate.parameters.get("run_type", "")) in scheme.valid_rtypes_2
            else 0
        ) == target_family
    ]
    candidates = same_family or matches
    candidate_ids = sorted(str(candidate.entry_id) for candidate in candidates)
    evidence = {
        "processed_target_present": str(target.entry_id) in processed_by_id,
        "mixing_state_candidate_count": len(mixing_candidate_ids),
        "structural_match_count": len(matches),
        "candidate_representative_ids": sorted(
            set(mixing_candidate_ids) | set(candidate_ids)
        ),
    }
    if len(candidates) != 1:
        return None, evidence
    representative = candidates[0]
    alias = copy.deepcopy(representative)
    alias.entry_id = target.entry_id
    return (
        alias,
        representative,
        "unique_processed_structure_representative",
    ), evidence


def solve_mixed_contexts(
    *,
    snapshot: str,
    mixed_selected: pd.DataFrame,
    gga_selected: pd.DataFrame,
    r2_selected: pd.DataFrame,
    gga_phase: pd.DataFrame,
    gga_decomp: pd.DataFrame,
    selection: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    mixed_type = "GGA_GGA+U_R2SCAN"
    stable_tolerance = float(config["energy_policy"]["stable_energy_tolerance_eV_per_atom"])
    chosen = _mixed_target_rows(mixed_selected, selection)
    gga_index = _row_index(gga_selected)
    r2_index = _row_index(r2_selected)
    phase_rows: dict[bytes, dict[str, Any]] = {}
    decomp_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    warning_counts: dict[str, int] = {}
    cache = _StructureEntryCache(int(config["execution"].get("structure_cache_entries", 20000)))

    gga_phase_by_uid = {bytes(row["unified_entry_id"]): row for row in gga_phase.to_dict(orient="records")}
    gga_decomp_groups = {
        bytes(uid): group.to_dict(orient="records")
        for uid, group in gga_decomp.groupby("unified_entry_id", sort=False)
    }
    for space, targets in chosen.groupby("_space", sort=False):
        space = tuple(space)
        context = chemical_space_text(space)
        target_rows = targets.to_dict(orient="records")
        has_r2_target = _has_exact_r2scan_target(target_rows, context)
        if not has_r2_target:
            for target in target_rows:
                source_uid = contextual_entry_id(snapshot, "GGA_GGA+U", context, str(target["entry_id"]))
                if source_uid not in gga_phase_by_uid:
                    errors.append({"snapshot_id": snapshot, "thermo_type": mixed_type, "phase_context_chemsys": context, "entry_id": str(target["entry_id"]), "reason": "missing_homogeneous_mirror"})
                    continue
                source_phase = copy.deepcopy(gga_phase_by_uid[source_uid])
                uid = contextual_entry_id(snapshot, mixed_type, context, str(target["entry_id"]))
                source_phase.update(
                    {
                        "thermo_type": mixed_type,
                        "unified_entry_id": uid,
                        "source_workflow": "GGA_GGA+U",
                        "compatibility_mode": "homogeneous_gga_mirror",
                        "material_id": _optional(target.get("material_id")),
                        "thermo_id": _optional(target.get("thermo_id")),
                        "entry_label": _optional(target.get("entry_label")),
                        "source_record_count": int(target.get("source_record_count", 1)),
                        "source_object_sha256": str(target["source_object_sha256"]),
                        "source_key": str(target["source_key"]),
                        "source_row_number": int(target["source_row_number"]),
                    }
                )
                phase_rows[uid] = source_phase
                for component in gga_decomp_groups.get(source_uid, []):
                    component_id = str(component["component_entry_id"])
                    source_component_uid = contextual_entry_id(snapshot, "GGA_GGA+U", context, component_id)
                    mixed_component_uid = contextual_entry_id(snapshot, mixed_type, context, component_id)
                    if mixed_component_uid not in phase_rows:
                        component_phase = copy.deepcopy(gga_phase_by_uid[source_component_uid])
                        component_phase.update(
                            {
                                "thermo_type": mixed_type,
                                "unified_entry_id": mixed_component_uid,
                                "is_target": False,
                                "is_competitor": True,
                                "source_workflow": "GGA_GGA+U",
                                "compatibility_mode": "homogeneous_gga_mirror",
                            }
                        )
                        phase_rows[mixed_component_uid] = component_phase
                    decomp_rows.append(
                        {
                            **component,
                            "thermo_type": mixed_type,
                            "unified_entry_id": uid,
                            "component_unified_entry_id": mixed_component_uid,
                        }
                    )
            continue

        base_rows = _context_rows(gga_index, space) + _context_rows(r2_index, space)
        context_source_by_id = {
            str(row["entry_id"]): row for row in base_rows
        }
        entries = [cache.get(row) for row in base_rows]
        input_entries_by_id = {str(entry.entry_id): entry for entry in entries}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scheme = _AuditedMixingScheme(
                    run_type_1=str(config["energy_policy"]["mixed_run_type_1"]),
                    run_type_2=str(config["energy_policy"]["mixed_run_type_2"]),
                    check_potcar=bool(config["energy_policy"]["check_potcar"]),
                )
                processed = scheme.process_entries(entries, clean=True, inplace=False)
            for item in caught:
                text = str(item.message)
                category = "oxidation_state_guess" if text.startswith("Failed to guess oxidation states") else text
                warning_counts[category] = warning_counts.get(category, 0) + 1
            entries_by_id = {str(entry.entry_id): entry for entry in processed}
            missing_sources = sorted(set(entries_by_id) - set(context_source_by_id))
            if missing_sources:
                raise RuntimeError(
                    "Compatibility processing returned entries without frozen "
                    f"context sources: {missing_sources[:10]}"
                )
            sources_by_id = {
                entry_id: context_source_by_id[entry_id]
                for entry_id in entries_by_id
            }
            workflow_by_id = {
                entry_id: str(sources_by_id[entry_id]["thermo_type"])
                for entry_id in entries_by_id
            }
            phase_diagram = PhaseDiagram(processed)
            for target in sorted(target_rows, key=lambda row: str(row["entry_id"])):
                entry_id = str(target["entry_id"])
                if entry_id not in entries_by_id:
                    target_input_entry = input_entries_by_id.get(entry_id)
                    if target_input_entry is None:
                        # Mixed thermo targets can use a workflow-specific ID
                        # (for example ``-GGA``) while the compatible frozen
                        # base entry uses ``-GGA+U``.  Resolve that identity
                        # difference structurally; never infer it by suffix.
                        target_input_entry = cache.get(target)
                    resolution, resolution_evidence = (
                        _resolve_mixing_duplicate_alias_with_evidence(
                        target_input_entry, processed, scheme,
                        scheme.mixing_state_data,
                        )
                    )
                    if resolution is None:
                        errors.append(
                            {
                                "snapshot_id": snapshot,
                                "thermo_type": mixed_type,
                                "phase_context_chemsys": context,
                                "entry_id": entry_id,
                                "reason": "unresolved_mixing_structure_duplicate",
                                **resolution_evidence,
                            }
                        )
                        continue
                    alias, representative, alias_resolution = resolution
                    representative_id = str(representative.entry_id)
                    entries_by_id[entry_id] = alias
                    sources_by_id[entry_id] = target
                    workflow_by_id[entry_id] = str(
                        context_source_by_id[representative_id]["thermo_type"]
                    )
                    aliases.append(
                        {
                            "snapshot_id": snapshot,
                            "thermo_type": mixed_type,
                            "phase_context_chemsys": context,
                            "target_entry_id": entry_id,
                            "representative_entry_id": representative_id,
                            "target_source_workflow": str(target["thermo_type"]),
                            "representative_source_workflow": str(context_source_by_id[representative_id]["thermo_type"]),
                            "target_run_type": str(target_input_entry.parameters.get("run_type", "")),
                            "representative_run_type": str(representative.parameters.get("run_type", "")),
                            "target_energy_per_atom_eV": float(target_input_entry.energy_per_atom),
                            "representative_energy_per_atom_eV": float(representative.energy_per_atom),
                            "resolution": alias_resolution,
                            "source_object_sha256": str(target["source_object_sha256"]),
                            "source_key": str(target["source_key"]),
                            "source_row_number": int(target["source_row_number"]),
                        }
                    )
                try:
                    _add_target_and_decomposition(
                        target_source=target,
                        target_entry=entries_by_id[entry_id],
                        entries_by_id=entries_by_id,
                        sources_by_id=sources_by_id,
                        phase_diagram=phase_diagram,
                        snapshot=snapshot,
                        thermo_type=mixed_type,
                        context=context,
                        source_workflow_by_id=workflow_by_id,
                        compatibility_mode=(
                            "regenerated_context_mixing_duplicate_alias"
                            if any(
                                row["target_entry_id"] == entry_id
                                and row["phase_context_chemsys"] == context
                                for row in aliases
                            )
                            else "regenerated_context_mixing"
                        ),
                        phase_rows=phase_rows,
                        decomp_rows=decomp_rows,
                        stable_tolerance=stable_tolerance,
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "snapshot_id": snapshot,
                            "thermo_type": mixed_type,
                            "phase_context_chemsys": context,
                            "entry_id": entry_id,
                            "reason": "mixed_target_failure",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        except Exception as exc:
            errors.append(
                {
                    "snapshot_id": snapshot,
                    "thermo_type": mixed_type,
                    "phase_context_chemsys": context,
                    "reason": "mixed_context_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for name, count in cache.warning_counts.items():
        warning_counts[name] = warning_counts.get(name, 0) + count
    warning_rows = [f"{name}: {count} occurrences" for name, count in sorted(warning_counts.items())]
    return (
        pd.DataFrame(list(phase_rows.values()), columns=ENTRY_SCHEMA.names),
        pd.DataFrame(decomp_rows, columns=DECOMPOSITION_SCHEMA.names),
        errors,
        warning_rows,
        aliases,
    )


def _write_table(path: Path, frame: pd.DataFrame, schema: pa.Schema, parquet: dict[str, Any]) -> None:
    writer = DataFrameParquetWriter(path, schema, parquet)
    writer.write(frame)
    writer.close()


def _write_reconciliation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_csv_atomic(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(columns=RECONCILIATION_COLUMNS).to_csv(
        temporary, index=False, lineterminator="\n"
    )
    os.replace(temporary, path)


def _write_mixing_alias_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_csv_atomic(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(columns=MIXING_ALIAS_COLUMNS).to_csv(
        temporary, index=False, lineterminator="\n"
    )
    os.replace(temporary, path)


def _write_compatibility_exclusion_csv(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=COMPATIBILITY_EXCLUSION_COLUMNS).to_csv(
        temporary, index=False, lineterminator="\n"
    )
    os.replace(temporary, path)


def _classify_compatibility_exclusions(
    *,
    root: Path,
    snapshot: str,
    thermo_type: str,
    ambiguities: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate approved source-state eligibility exclusions from failures.

    Classification is deliberately limited to two predeclared, label-blind
    failure modes.  Runtime, schema, context, and reconstruction errors remain
    blocking ambiguities.
    """
    if thermo_type != "GGA_GGA+U_R2SCAN":
        return ambiguities, []
    reason_map = {
        "unresolved_mixing_structure_duplicate": (
            "no_unique_processed_representative"
        ),
        "missing_homogeneous_mirror": "homogeneous_source_unavailable",
    }
    allowed = set(config["reference_test"]["allowed_exclusion_reasons"])
    rule_version = str(config["reference_test"]["eligibility_rule_version"])
    selection = _thermo_target_selection(root, snapshot, thermo_type)
    remaining: list[dict[str, Any]] = []
    exclusions_by_state: dict[tuple[str, str, str], dict[str, Any]] = {}
    for error in ambiguities:
        approved_reason = reason_map.get(str(error.get("reason")))
        if approved_reason not in allowed:
            remaining.append(error)
            continue
        entry_id = str(error.get("entry_id", ""))
        context = str(error.get("phase_context_chemsys", ""))
        matched = selection[
            selection["selected_entry_id"].astype(str).eq(entry_id)
            & selection["chemsys"].astype(str).eq(context)
        ]
        if matched.empty:
            remaining.append(
                {
                    **error,
                    "classification_error": (
                        "approved eligibility reason did not map to a frozen "
                        "selected source state"
                    ),
                }
            )
            continue
        candidate_ids = sorted(
            {str(value) for value in error.get("candidate_representative_ids", [])}
        )
        for source in matched.to_dict(orient="records"):
            row = {
                "snapshot_id": snapshot,
                "thermo_type": thermo_type,
                "thermo_id": str(source["thermo_id"]),
                "selected_entry_id": str(source["selected_entry_id"]),
                "selected_entry_label": str(source["selected_entry_label"]),
                "phase_context_chemsys": context,
                "reason": approved_reason,
                "rule_version": rule_version,
                "processed_target_present": bool(
                    error.get("processed_target_present", False)
                ),
                "mixing_state_candidate_count": int(
                    error.get("mixing_state_candidate_count", 0)
                ),
                "structural_match_count": int(
                    error.get("structural_match_count", 0)
                ),
                "candidate_representative_ids_json": json.dumps(
                    candidate_ids, separators=(",", ":")
                ),
                "source_object_sha256": str(source["source_object_sha256"]),
                "source_key": str(source["source_key"]),
                "source_row_number": int(source["source_row_number"]),
            }
            key = (snapshot, thermo_type, row["thermo_id"])
            prior = exclusions_by_state.get(key)
            if prior is not None and prior != row:
                remaining.append(
                    {
                        **error,
                        "classification_error": (
                            "conflicting eligibility evidence for one source state"
                        ),
                    }
                )
                continue
            exclusions_by_state[key] = row
    exclusions = [
        exclusions_by_state[key] for key in sorted(exclusions_by_state)
    ]
    return remaining, exclusions


def _part_paths(parts: Path, snapshot: str, thermo_type: str) -> dict[str, Path]:
    token = hashlib.sha256(f"context|{snapshot}|{thermo_type}".encode()).hexdigest()[:16]
    return {
        "phase": parts / f"{token}.phase.parquet",
        "decomposition": parts / f"{token}.decomposition.parquet",
        "duplicates": parts / f"{token}.duplicates.parquet",
        "ambiguities": parts / f"{token}.ambiguities.jsonl",
        "stats": parts / f"{token}.stats.json",
    }


def _stored_path(value: str | Path) -> Path:
    """Interpret persisted relative artifact paths on Windows or POSIX."""
    return Path(str(value).replace("\\", "/"))


def _homogeneous_mirror_reference(
    *, snapshot: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Load the frozen homogeneous rows that mixed mirror targets must copy."""
    paths = _part_paths(
        Path(config["output"]["root"]) / "_context_parts",
        snapshot,
        "GGA_GGA+U",
    )
    return pq.read_table(
        paths["phase"],
        columns=[
            "unified_entry_id",
            "entry_id",
            "phase_context_chemsys",
            "corrected_energy_per_atom",
            "energy_above_hull",
            "formation_energy_per_atom",
        ],
    ).to_pandas()


def _audit_homogeneous_mirrors(
    *,
    snapshot: str,
    targets: pd.DataFrame,
    config: dict[str, Any],
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    """Compare every mixed mirror with its exact frozen homogeneous artifact."""
    mirrors = targets[
        targets["compatibility_mode"].eq("homogeneous_gga_mirror")
    ].copy()
    if mirrors.empty:
        return 1.0, {
            "mirror_reference_mode": "exact_homogeneous_context_artifact",
            "mirror_states": 0,
            "mirror_matched_states": 0,
            "max_mirror_corrected_energy_absolute_error_eV_per_atom": 0.0,
            "max_mirror_hull_absolute_error_eV_per_atom": 0.0,
            "max_mirror_formation_absolute_error_eV_per_atom": 0.0,
        }
    mirrors["homogeneous_unified_entry_id"] = mirrors.apply(
        lambda row: contextual_entry_id(
            snapshot,
            "GGA_GGA+U",
            str(row["phase_context_chemsys"]),
            str(row["entry_id"]),
        ),
        axis=1,
    )
    reference = _homogeneous_mirror_reference(
        snapshot=snapshot, config=config
    ).rename(
        columns={
            "unified_entry_id": "homogeneous_unified_entry_id",
            "corrected_energy_per_atom": "homogeneous_corrected_energy_per_atom",
            "energy_above_hull": "homogeneous_energy_above_hull",
            "formation_energy_per_atom": "homogeneous_formation_energy_per_atom",
        }
    )
    reference = reference[
        [
            "homogeneous_unified_entry_id",
            "homogeneous_corrected_energy_per_atom",
            "homogeneous_energy_above_hull",
            "homogeneous_formation_energy_per_atom",
        ]
    ]
    mirror_audit = mirrors.merge(
        reference,
        on="homogeneous_unified_entry_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    mirror_audit["mirror_corrected_energy_absolute_error"] = (
        mirror_audit["corrected_energy_per_atom"]
        - mirror_audit["homogeneous_corrected_energy_per_atom"]
    ).abs()
    mirror_audit["mirror_hull_absolute_error"] = (
        mirror_audit["rebuilt_energy_above_hull"]
        - mirror_audit["homogeneous_energy_above_hull"]
    ).abs()
    mirror_audit["mirror_formation_absolute_error"] = (
        mirror_audit["rebuilt_formation_energy_per_atom"]
        - mirror_audit["homogeneous_formation_energy_per_atom"]
    ).abs()
    mirror_audit["mirror_matched"] = (
        mirror_audit["homogeneous_corrected_energy_per_atom"].notna()
        & mirror_audit["mirror_corrected_energy_absolute_error"].le(tolerance)
        & mirror_audit["mirror_hull_absolute_error"].le(tolerance)
        & mirror_audit["mirror_formation_absolute_error"].le(tolerance)
    )
    return float(mirror_audit["mirror_matched"].mean()), {
        "mirror_reference_mode": "exact_homogeneous_context_artifact",
        "mirror_states": len(mirror_audit),
        "mirror_matched_states": int(mirror_audit["mirror_matched"].sum()),
        "max_mirror_corrected_energy_absolute_error_eV_per_atom": float(
            mirror_audit["mirror_corrected_energy_absolute_error"].max()
        ),
        "max_mirror_hull_absolute_error_eV_per_atom": float(
            mirror_audit["mirror_hull_absolute_error"].max()
        ),
        "max_mirror_formation_absolute_error_eV_per_atom": float(
            mirror_audit["mirror_formation_absolute_error"].max()
        ),
    }


def _reference_audit(
    *, snapshot: str, thermo_type: str, root: Path, phase: pd.DataFrame,
    decomposition: pd.DataFrame, config: dict[str, Any],
    compatibility_exclusions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    compatibility_exclusions = compatibility_exclusions or []
    if thermo_type != "GGA_GGA+U_R2SCAN" and compatibility_exclusions:
        raise RuntimeError("Compatibility exclusions are only valid for mixed workflow")
    source = _thermo_target_selection(root, snapshot, thermo_type)
    targets = phase[phase["is_target"]][
        [
            "unified_entry_id", "entry_label", "entry_id",
            "phase_context_chemsys", "compatibility_mode",
            "corrected_energy_per_atom", "energy_above_hull",
            "formation_energy_per_atom",
        ]
    ].rename(
        columns={
            "energy_above_hull": "rebuilt_energy_above_hull",
            "formation_energy_per_atom": "rebuilt_formation_energy_per_atom",
        }
    )
    source["_entry_label_key"] = source["selected_entry_label"].astype(str).str.upper()
    targets["_entry_label_key"] = targets["entry_label"].astype(str).str.upper()
    audit = source.merge(
        targets,
        left_on=["selected_entry_id", "_entry_label_key", "chemsys"],
        right_on=["entry_id", "_entry_label_key", "phase_context_chemsys"],
        how="left",
        validate="many_to_one",
    )
    audit["hull_absolute_error"] = (audit["rebuilt_energy_above_hull"] - audit["energy_above_hull"]).abs()
    audit["formation_absolute_error"] = (audit["rebuilt_formation_energy_per_atom"] - audit["formation_energy_per_atom"]).abs()
    tolerance = float(config["reference_test"]["absolute_tolerance_eV_per_atom"])
    audit["matched"] = audit["hull_absolute_error"].le(tolerance) & audit["formation_absolute_error"].le(tolerance)
    limit = int(config["reference_test"]["sample_per_snapshot_thermo_type"])
    if thermo_type == "GGA_GGA+U_R2SCAN":
        exclusions = pd.DataFrame(compatibility_exclusions)
        if exclusions.empty:
            exclusions = pd.DataFrame(columns=["thermo_id", "reason", "rule_version"])
        elif exclusions["thermo_id"].astype(str).duplicated().any():
            raise RuntimeError(
                f"Duplicate compatibility exclusions for {snapshot}/{thermo_type}"
            )
        audit = audit.merge(
            exclusions[["thermo_id", "reason", "rule_version"]].rename(
                columns={
                    "reason": "compatibility_exclusion_reason",
                    "rule_version": "compatibility_eligibility_rule_version",
                }
            ),
            on="thermo_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        audit["compatibility_excluded"] = audit[
            "compatibility_exclusion_reason"
        ].notna()
        audit["target_available"] = audit["rebuilt_energy_above_hull"].notna()
        eligible_mask = ~audit["compatibility_excluded"]
        eligible_audit = audit[eligible_mask]
        raw_coverage_rate = (
            float(audit["target_available"].mean()) if len(audit) else 0.0
        )
        eligible_coverage_rate = (
            float(eligible_audit["target_available"].mean())
            if len(eligible_audit)
            else 0.0
        )
        unclassified_missing = int(
            ((~audit["target_available"]) & eligible_mask).sum()
        )
        excluded_with_target = int(
            (audit["target_available"] & audit["compatibility_excluded"]).sum()
        )
        exclusion_fraction = (
            float(audit["compatibility_excluded"].mean()) if len(audit) else 0.0
        )
        exclusion_limit = float(
            config["reference_test"]["maximum_exclusion_fraction"]
        )
        exclusion_cap_passed = exclusion_fraction <= exclusion_limit
        weighted = decomposition.assign(
            _weighted_component_energy=(
                decomposition["amount"] * decomposition["component_energy_per_atom"]
            )
        ).groupby("unified_entry_id", sort=False, as_index=False).agg(
            decomposition_amount_sum=("amount", "sum"),
            decomposition_energy_per_atom=("_weighted_component_energy", "sum"),
        )
        internal = targets.merge(weighted, on="unified_entry_id", how="left", validate="one_to_one")
        internal["reconstructed_energy_above_hull"] = (
            internal["corrected_energy_per_atom"] - internal["decomposition_energy_per_atom"]
        ).clip(lower=0.0)
        internal["decomposition_hull_absolute_error"] = (
            internal["reconstructed_energy_above_hull"] - internal["rebuilt_energy_above_hull"]
        ).abs()
        internal["decomposition_reconstruction_matched"] = (
            internal["decomposition_hull_absolute_error"].le(tolerance)
            & (internal["decomposition_amount_sum"] - 1.0).abs().le(tolerance)
        )
        internal_rate = (
            float(internal["decomposition_reconstruction_matched"].mean())
            if len(internal)
            else 0.0
        )
        mirror_rate, mirror_summary = _audit_homogeneous_mirrors(
            snapshot=snapshot,
            targets=targets,
            config=config,
            tolerance=tolerance,
        )
        eligibility_integrity = (
            1.0
            if exclusion_cap_passed
            and unclassified_missing == 0
            and excluded_with_target == 0
            else 0.0
        )
        gate_rate = min(
            eligible_coverage_rate,
            internal_rate,
            mirror_rate,
            eligibility_integrity,
        )
        # Unmatched source states intentionally retain a null contextual ID in
        # ``audit`` so they reduce the locked target-coverage rate.  They have
        # no target row to annotate in the internal reconstruction sample and
        # must not participate in the one-to-one diagnostic join.
        diagnostic = eligible_audit.loc[
            eligible_audit["unified_entry_id"].notna(),
            [
                "unified_entry_id",
                "hull_absolute_error",
                "formation_absolute_error",
                "matched",
            ],
        ].groupby("unified_entry_id", sort=False, as_index=False).agg(
            serialized_source_hull_absolute_error=(
                "hull_absolute_error", "max"
            ),
            serialized_source_formation_absolute_error=(
                "formation_absolute_error", "max"
            ),
            serialized_source_matched=("matched", "all"),
            serialized_source_state_count=("matched", "size"),
        )
        # A mixed thermo document can legitimately retain both its GGA(+U)
        # and R2SCAN entries under one thermo_id.  Only the entry selected by
        # the document's energy_type has a serialized top-level diagnostic, so
        # thermo_id is not a valid join key here.  The contextual entry ID is
        # the locked P3.2/P3.3 foreign key and distinguishes those entries
        # without dropping either one.
        sample_frame = internal.merge(
            diagnostic,
            on="unified_entry_id",
            how="left",
            validate="one_to_one",
        )
        hashes = sample_frame.apply(
            lambda row: hashlib.sha256(
                f"{config['seed']}|{bytes(row['unified_entry_id']).hex()}".encode()
            ).hexdigest(),
            axis=1,
        )
        sample = sample_frame.loc[hashes.sort_values(kind="mergesort").index[:limit]].copy()
        summary = {
            "snapshot_id": snapshot,
            "thermo_type": thermo_type,
            "reference_mode": (
                "compatibility_eligible_context_coverage_decomposition_and_mirror"
            ),
            "source_states": len(audit),
            "eligible_source_states": int(eligible_mask.sum()),
            "excluded_source_states": int(audit["compatibility_excluded"].sum()),
            "comparable_states": int(eligible_audit["target_available"].sum()),
            "matched_states": int(round(gate_rate * int(eligible_mask.sum()))),
            "match_rate": gate_rate,
            "raw_target_coverage_rate": raw_coverage_rate,
            "eligible_target_coverage_rate": eligible_coverage_rate,
            "target_coverage_rate": eligible_coverage_rate,
            "compatibility_exclusion_fraction": exclusion_fraction,
            "maximum_compatibility_exclusion_fraction": exclusion_limit,
            "compatibility_exclusion_cap_passed": exclusion_cap_passed,
            "unclassified_missing_source_states": unclassified_missing,
            "excluded_states_with_target": excluded_with_target,
            "eligibility_rule_version": str(
                config["reference_test"]["eligibility_rule_version"]
            ),
            "decomposition_reconstruction_match_rate": internal_rate,
            "decomposition_reconstruction_states": len(internal),
            "decomposition_reconstruction_matched_states": int(
                internal["decomposition_reconstruction_matched"].sum()
            ),
            "mirror_source_match_rate": mirror_rate,
            **mirror_summary,
            "serialized_source_match_rate_diagnostic": float(
                eligible_audit["matched"].mean()
            ),
            "serialized_source_matched_states_diagnostic": int(
                eligible_audit["matched"].sum()
            ),
            "max_decomposition_hull_absolute_error_eV_per_atom": float(
                internal["decomposition_hull_absolute_error"].max()
            ),
            "max_serialized_source_hull_absolute_error_eV_per_atom_diagnostic": float(
                eligible_audit["hull_absolute_error"].max()
            ),
            "max_serialized_source_formation_absolute_error_eV_per_atom_diagnostic": float(
                eligible_audit["formation_absolute_error"].max()
            ),
            "tolerance_eV_per_atom": tolerance,
        }
    else:
        hashes = audit.apply(
            lambda row: hashlib.sha256(f"{config['seed']}|{row['thermo_id']}".encode()).hexdigest(),
            axis=1,
        )
        sample = audit.loc[hashes.sort_values(kind="mergesort").index[:limit]].copy()
        summary = {
            "snapshot_id": snapshot,
            "thermo_type": thermo_type,
            "reference_mode": "serialized_source_exact",
            "source_states": len(audit),
            "comparable_states": int(audit["rebuilt_energy_above_hull"].notna().sum()),
            "matched_states": int(audit["matched"].sum()),
            "match_rate": float(audit["matched"].mean()),
            "max_hull_absolute_error_eV_per_atom": float(audit["hull_absolute_error"].max()),
            "max_formation_absolute_error_eV_per_atom": float(audit["formation_absolute_error"].max()),
            "tolerance_eV_per_atom": tolerance,
        }
    if "unified_entry_id" in sample.columns:
        sample["unified_entry_id"] = sample["unified_entry_id"].map(
            lambda value: bytes(value).hex()
            if isinstance(value, (bytes, bytearray, memoryview))
            else value
        )
    sample.insert(0, "snapshot_id", snapshot)
    return summary, sample.to_dict(orient="records")


def _finish_part(
    *, paths: dict[str, Path], phase: pd.DataFrame, decomposition: pd.DataFrame,
    duplicates: pd.DataFrame, ambiguities: list[dict[str, Any]], warnings_: list[str],
    reconciliations: list[dict[str, Any]],
    mixing_aliases: list[dict[str, Any]],
    snapshot: str, thermo_type: str, raw_rows: int, config: dict[str, Any],
    source_hash: str, config_hash: str, root: Path,
) -> dict[str, Any]:
    ambiguities, compatibility_exclusions = _classify_compatibility_exclusions(
        root=root,
        snapshot=snapshot,
        thermo_type=thermo_type,
        ambiguities=ambiguities,
        config=config,
    )
    _write_table(paths["phase"], phase, ENTRY_SCHEMA, config["parquet"])
    _write_table(paths["decomposition"], decomposition, DECOMPOSITION_SCHEMA, config["parquet"])
    _write_table(paths["duplicates"], duplicates, DUPLICATE_SCHEMA, config["parquet"])
    write_jsonl_atomic(paths["ambiguities"], ambiguities)
    reference, reference_sample = _reference_audit(
        snapshot=snapshot, thermo_type=thermo_type, root=root, phase=phase,
        decomposition=decomposition, config=config,
        compatibility_exclusions=compatibility_exclusions,
    )
    outputs = [
        {"path": str(path), "rows": pq.ParquetFile(path).metadata.num_rows, "sha256": sha256_file(path)}
        for path in (paths["phase"], paths["decomposition"], paths["duplicates"])
    ]
    outputs.append({"path": str(paths["ambiguities"]), "rows": len(ambiguities), "sha256": sha256_file(paths["ambiguities"])})
    stats = {
        "complete": True, "snapshot_id": snapshot, "thermo_type": thermo_type,
        "source_sha256": source_hash, "config_sha256": config_hash,
        "raw_rows": raw_rows, "target_entries": int(phase["is_target"].sum()),
        "context_entry_rows": len(phase), "competitor_rows": int(phase["is_competitor"].sum()),
        "decomposition_rows": len(decomposition), "duplicate_rows": len(duplicates),
        "ambiguity_rows": len(ambiguities),
        "compatibility_exclusion_rows": len(compatibility_exclusions),
        "compatibility_exclusions": compatibility_exclusions,
        "energy_reconciliation_rows": len(reconciliations),
        "energy_reconciliations": reconciliations,
        "mixing_alias_rows": len(mixing_aliases),
        "mixing_aliases": mixing_aliases,
        "warnings": warnings_ + (
            [f"frozen thermo top-level energy reconciled: {len(reconciliations)} entries"]
            if reconciliations else []
        ),
        "reference_test": reference, "reference_sample": reference_sample, "outputs": outputs,
    }
    write_json_atomic(paths["stats"], stats)
    return stats


def _valid_prior(
    paths: dict[str, Path], source_hash: str, config_hash: str, required_match_rate: float
) -> dict[str, Any] | None:
    if not paths["stats"].exists():
        return None
    prior = json.loads(paths["stats"].read_text(encoding="utf-8"))
    if prior.get("source_sha256") != source_hash or prior.get("config_sha256") != config_hash:
        return None
    if float(prior.get("reference_test", {}).get("match_rate", 0.0)) < required_match_rate:
        return None
    if not all(
        _stored_path(item["path"]).exists()
        and sha256_file(_stored_path(item["path"])) == item["sha256"]
        for item in prior["outputs"]
    ):
        return None
    return prior


def _mixed_context_assignment(
    *,
    mixed_selected: pd.DataFrame,
    gga_selected: pd.DataFrame,
    r2_selected: pd.DataFrame,
    selection: pd.DataFrame,
    shard_count: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Assign exact chemical contexts to deterministic, load-balanced shards.

    Expensive contexts are those that invoke the structure-aware mixing scheme.
    Their cost is approximated by the square of the number of homogeneous base
    entries in the context; homogeneous mirrors use their target-row count.
    Longest-processing-time assignment with stable context/shard tie breaking
    gives every process the same partition without a mutable scheduling file.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    chosen = _mixed_target_rows(mixed_selected, selection)
    gga_counts = {
        tuple(space): int(len(group))
        for space, group in gga_selected.groupby("_space", sort=False)
    }
    r2_counts = {
        tuple(space): int(len(group))
        for space, group in r2_selected.groupby("_space", sort=False)
    }
    contexts: list[dict[str, Any]] = []
    for space, targets in chosen.groupby("_space", sort=False):
        space = tuple(space)
        context = chemical_space_text(space)
        target_rows = targets.to_dict(orient="records")
        mixed = _has_exact_r2scan_target(target_rows, context)
        base_entries = sum(
            gga_counts.get(subset, 0) + r2_counts.get(subset, 0)
            for subset in subspaces(space)
        )
        weight = max(1, base_entries * base_entries if mixed else len(targets))
        contexts.append(
            {
                "context": context,
                "target_rows": int(len(targets)),
                "base_entries": int(base_entries),
                "uses_mixing": bool(mixed),
                "weight": int(weight),
            }
        )
    loads = [0] * shard_count
    available = [(0, index) for index in range(shard_count)]
    heapq.heapify(available)
    assignments: dict[str, int] = {}
    for item in sorted(contexts, key=lambda row: (-row["weight"], row["context"])):
        load, shard = heapq.heappop(available)
        assignments[str(item["context"])] = shard
        loads[shard] = load + int(item["weight"])
        heapq.heappush(available, (loads[shard], shard))
        item["shard"] = shard
    assignment_body = json.dumps(
        [
            [item["context"], int(item["shard"]), int(item["weight"])]
            for item in sorted(contexts, key=lambda row: row["context"])
        ],
        separators=(",", ":"),
    )
    metadata = {
        "shard_count": shard_count,
        "context_count": len(contexts),
        "target_rows": int(len(chosen)),
        "mixing_contexts": sum(bool(item["uses_mixing"]) for item in contexts),
        "estimated_loads": loads,
        "assignment_sha256": hashlib.sha256(assignment_body.encode()).hexdigest(),
    }
    return assignments, metadata


def _mixed_shard_paths(
    root: Path, snapshot: str, shard_index: int, shard_count: int
) -> dict[str, Path]:
    token = hashlib.sha256(
        f"mixed-context-shard|{snapshot}|{shard_index}|{shard_count}".encode()
    ).hexdigest()[:20]
    directory = root / f"snapshot={snapshot}"
    return {
        "phase": directory / f"{token}.phase.parquet",
        "decomposition": directory / f"{token}.decomposition.parquet",
        "ambiguities": directory / f"{token}.ambiguities.jsonl",
        "stats": directory / f"{token}.stats.json",
    }


def _valid_mixed_shard(
    paths: dict[str, Path],
    *,
    source_hash: str,
    config_hash: str,
    assignment_hash: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any] | None:
    if not paths["stats"].exists():
        return None
    stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
    expected = {
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "assignment_sha256": assignment_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    if not stats.get("complete") or any(stats.get(key) != value for key, value in expected.items()):
        return None
    if not all(
        Path(item["path"]).exists()
        and sha256_file(Path(item["path"])) == item["sha256"]
        for item in stats.get("outputs", [])
    ):
        return None
    return stats


def _build_mixed_shard_job(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    snapshot = str(job["snapshot"])
    shard_index = int(job["shard_index"])
    shard_count = int(job["shard_count"])
    root = Path(config["input"]["normalized_root"])
    tolerance = float(config["energy_policy"]["duplicate_float_absolute_tolerance"])
    reference_tolerance = float(config["reference_test"]["absolute_tolerance_eV_per_atom"])
    mixed, _, _, _ = _selected_rows(
        root,
        snapshot,
        "GGA_GGA+U_R2SCAN",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=False,
    )
    gga, _, _, _ = _selected_rows(
        root,
        snapshot,
        "GGA_GGA+U",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=True,
    )
    r2, _, _, _ = _selected_rows(
        root,
        snapshot,
        "R2SCAN",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=True,
    )
    selection = _thermo_target_selection(root, snapshot, "GGA_GGA+U_R2SCAN")
    assignment, assignment_meta = _mixed_context_assignment(
        mixed_selected=mixed,
        gga_selected=gga,
        r2_selected=r2,
        selection=selection,
        shard_count=shard_count,
    )
    paths = _mixed_shard_paths(
        Path(job["shards_root"]), snapshot, shard_index, shard_count
    )
    if config["execution"].get("resume_group_parts"):
        prior = _valid_mixed_shard(
            paths,
            source_hash=job["source_hash"],
            config_hash=job["config_hash"],
            assignment_hash=assignment_meta["assignment_sha256"],
            shard_index=shard_index,
            shard_count=shard_count,
        )
        if prior:
            return prior
    selected_contexts = {
        context for context, assigned in assignment.items() if assigned == shard_index
    }
    shard_mixed = mixed[
        mixed["_space"].map(chemical_space_text).isin(selected_contexts)
    ].copy()
    shard_selection = selection[
        selection["chemsys"].astype(str).isin(selected_contexts)
    ].copy()
    phase, decomposition, errors, warning_rows, aliases = solve_mixed_contexts(
        snapshot=snapshot,
        mixed_selected=shard_mixed,
        gga_selected=gga,
        r2_selected=r2,
        gga_phase=_read_part(job["gga_phase"]),
        gga_decomp=_read_part(job["gga_decomp"]),
        selection=shard_selection,
        config=config,
    )
    _write_table(paths["phase"], phase, ENTRY_SCHEMA, config["parquet"])
    _write_table(
        paths["decomposition"], decomposition, DECOMPOSITION_SCHEMA, config["parquet"]
    )
    write_jsonl_atomic(paths["ambiguities"], errors)
    outputs = [
        {
            "path": str(path),
            "rows": (
                pq.ParquetFile(path).metadata.num_rows
                if path.suffix == ".parquet"
                else len(errors)
            ),
            "sha256": sha256_file(path),
        }
        for path in (paths["phase"], paths["decomposition"], paths["ambiguities"])
    ]
    stats = {
        "complete": True,
        "snapshot_id": snapshot,
        "thermo_type": "GGA_GGA+U_R2SCAN",
        "source_sha256": job["source_hash"],
        "config_sha256": job["config_hash"],
        "assignment_sha256": assignment_meta["assignment_sha256"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_contexts": len(selected_contexts),
        "assigned_target_rows": int(len(shard_mixed)),
        "estimated_load": int(assignment_meta["estimated_loads"][shard_index]),
        "context_entry_rows": int(len(phase)),
        "decomposition_rows": int(len(decomposition)),
        "ambiguity_rows": len(errors),
        "warnings": warning_rows,
        "mixing_aliases": aliases,
        "outputs": outputs,
    }
    write_json_atomic(paths["stats"], stats)
    return stats


def _finalize_mixed_shards_job(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    snapshot = str(job["snapshot"])
    shard_count = int(job["shard_count"])
    root = Path(config["input"]["normalized_root"])
    tolerance = float(config["energy_policy"]["duplicate_float_absolute_tolerance"])
    reference_tolerance = float(config["reference_test"]["absolute_tolerance_eV_per_atom"])
    mixed, duplicates, ambiguities, _ = _selected_rows(
        root,
        snapshot,
        "GGA_GGA+U_R2SCAN",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=False,
    )
    gga, _, base_errors_1, reconciliations_1 = _selected_rows(
        root,
        snapshot,
        "GGA_GGA+U",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=True,
    )
    r2, _, base_errors_2, reconciliations_2 = _selected_rows(
        root,
        snapshot,
        "R2SCAN",
        tolerance,
        reference_tolerance=reference_tolerance,
        reconcile_reported_energy=True,
    )
    selection = _thermo_target_selection(root, snapshot, "GGA_GGA+U_R2SCAN")
    _, assignment_meta = _mixed_context_assignment(
        mixed_selected=mixed,
        gga_selected=gga,
        r2_selected=r2,
        selection=selection,
        shard_count=shard_count,
    )
    shard_stats: list[dict[str, Any]] = []
    paths_by_shard: list[dict[str, Path]] = []
    for shard_index in range(shard_count):
        paths = _mixed_shard_paths(
            Path(job["shards_root"]), snapshot, shard_index, shard_count
        )
        stats = _valid_mixed_shard(
            paths,
            source_hash=job["source_hash"],
            config_hash=job["config_hash"],
            assignment_hash=assignment_meta["assignment_sha256"],
            shard_index=shard_index,
            shard_count=shard_count,
        )
        if stats is None:
            raise RuntimeError(f"Missing or invalid mixed shard {snapshot}/{shard_index}")
        shard_stats.append(stats)
        paths_by_shard.append(paths)
    phase = pd.concat(
        [_read_part(str(paths["phase"])) for paths in paths_by_shard],
        ignore_index=True,
    )
    decomposition = pd.concat(
        [_read_part(str(paths["decomposition"])) for paths in paths_by_shard],
        ignore_index=True,
    )
    if phase["unified_entry_id"].duplicated().any():
        raise RuntimeError(f"Duplicate contextual entry IDs across shards for {snapshot}")
    ambiguities.extend(base_errors_1)
    ambiguities.extend(base_errors_2)
    for paths in paths_by_shard:
        ambiguities.extend(
            json.loads(line)
            for line in paths["ambiguities"].read_text(encoding="utf-8").splitlines()
            if line
        )
    aliases_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for stats in shard_stats:
        for alias in stats.get("mixing_aliases", []):
            aliases_by_key[(str(alias["phase_context_chemsys"]), str(alias["target_entry_id"]))] = alias
    reconciliations = reconciliations_1 + reconciliations_2
    warning_rows = sorted(
        {warning for stats in shard_stats for warning in stats.get("warnings", [])}
    )
    warning_rows.append(
        f"deterministic mixed context sharding: {shard_count} shards; "
        f"assignment_sha256={assignment_meta['assignment_sha256']}"
    )
    return _finish_part(
        paths=_part_paths(Path(job["parts_root"]), snapshot, "GGA_GGA+U_R2SCAN"),
        phase=phase,
        decomposition=decomposition,
        duplicates=duplicates,
        ambiguities=ambiguities,
        warnings_=warning_rows,
        reconciliations=reconciliations,
        mixing_aliases=[aliases_by_key[key] for key in sorted(aliases_by_key)],
        snapshot=snapshot,
        thermo_type="GGA_GGA+U_R2SCAN",
        raw_rows=len(_source_rows(root, snapshot, "GGA_GGA+U_R2SCAN")),
        config=config,
        source_hash=job["source_hash"],
        config_hash=job["config_hash"],
        root=root,
    )


def _build_homogeneous_job(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    snapshot, thermo_type = job["snapshot"], job["thermo_type"]
    paths = _part_paths(Path(job["parts_root"]), snapshot, thermo_type)
    if config["execution"].get("resume_group_parts"):
        prior = _valid_prior(
            paths, job["source_hash"], job["config_hash"],
            float(config["reference_test"]["required_match_rate"]),
        )
        if prior:
            return prior
    root = Path(config["input"]["normalized_root"])
    selected, duplicates, ambiguities, reconciliations = _selected_rows(
        root,
        snapshot,
        thermo_type,
        float(config["energy_policy"]["duplicate_float_absolute_tolerance"]),
        reference_tolerance=float(config["reference_test"]["absolute_tolerance_eV_per_atom"]),
        reconcile_reported_energy=True,
    )
    raw_rows = len(selected) + len(duplicates)
    phase, decomposition, errors, warning_rows = solve_homogeneous_contexts(
        selected, stable_tolerance=float(config["energy_policy"]["stable_energy_tolerance_eV_per_atom"])
    )
    ambiguities.extend(errors)
    return _finish_part(
        paths=paths, phase=phase, decomposition=decomposition, duplicates=duplicates,
        ambiguities=ambiguities, warnings_=warning_rows, reconciliations=reconciliations,
        mixing_aliases=[],
        snapshot=snapshot,
        thermo_type=thermo_type, raw_rows=raw_rows, config=config,
        source_hash=job["source_hash"], config_hash=job["config_hash"], root=root,
    )


def _build_mixed_job(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    snapshot = job["snapshot"]
    thermo_type = "GGA_GGA+U_R2SCAN"
    paths = _part_paths(Path(job["parts_root"]), snapshot, thermo_type)
    if config["execution"].get("resume_group_parts"):
        prior = _valid_prior(
            paths, job["source_hash"], job["config_hash"],
            float(config["reference_test"]["required_match_rate"]),
        )
        if prior:
            return prior
    root = Path(config["input"]["normalized_root"])
    tolerance = float(config["energy_policy"]["duplicate_float_absolute_tolerance"])
    reference_tolerance = float(config["reference_test"]["absolute_tolerance_eV_per_atom"])
    mixed, duplicates, ambiguities, _ = _selected_rows(
        root, snapshot, thermo_type, tolerance,
        reference_tolerance=reference_tolerance, reconcile_reported_energy=False,
    )
    gga, _, base_ambiguities_1, reconciliations_1 = _selected_rows(
        root, snapshot, "GGA_GGA+U", tolerance,
        reference_tolerance=reference_tolerance, reconcile_reported_energy=True,
    )
    r2, _, base_ambiguities_2, reconciliations_2 = _selected_rows(
        root, snapshot, "R2SCAN", tolerance,
        reference_tolerance=reference_tolerance, reconcile_reported_energy=True,
    )
    reconciliations = reconciliations_1 + reconciliations_2
    ambiguities.extend(base_ambiguities_1)
    ambiguities.extend(base_ambiguities_2)
    selection = _thermo_target_selection(root, snapshot, thermo_type)
    phase, decomposition, errors, warning_rows, mixing_aliases = solve_mixed_contexts(
        snapshot=snapshot, mixed_selected=mixed, gga_selected=gga, r2_selected=r2,
        gga_phase=_read_part(job["gga_phase"]), gga_decomp=_read_part(job["gga_decomp"]),
        selection=selection, config=config,
    )
    ambiguities.extend(errors)
    return _finish_part(
        paths=paths, phase=phase, decomposition=decomposition, duplicates=duplicates,
        ambiguities=ambiguities, warnings_=warning_rows, reconciliations=reconciliations,
        mixing_aliases=mixing_aliases,
        snapshot=snapshot,
        thermo_type=thermo_type, raw_rows=len(_source_rows(root, snapshot, thermo_type)),
        config=config, source_hash=job["source_hash"], config_hash=job["config_hash"], root=root,
    )


def _combine(paths: list[Path], target: Path, schema: pa.Schema, parquet: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression=str(parquet["compression"]), use_dictionary=True, write_statistics=True)
    try:
        for path in paths:
            source = pq.ParquetFile(path)
            if source.schema_arrow != schema:
                raise RuntimeError(f"schema mismatch: {path}")
            for batch in source.iter_batches(batch_size=int(parquet["row_group_size"])):
                writer.write_batch(batch, row_group_size=int(parquet["row_group_size"]))
    finally:
        writer.close()
    os.replace(temporary, target)


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()


def _dictionary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    descriptions = {
        "phase_context_chemsys": "Exact sorted element set in which compatibility and hull evaluation were performed.",
        "unified_entry_id": "BLAKE2b-128 of snapshot, workflow, phase context, and source entry ID.",
        "is_target": "True for a source thermo target evaluated in its exact chemical system.",
        "is_competitor": "True when the contextual entry appears in at least one stored decomposition.",
        "compatibility_mode": "Homogeneous source correction, homogeneous mirror, or regenerated local mixed compatibility.",
    }
    for table, schema in (("phase_entry_unified", ENTRY_SCHEMA), ("phase_decomposition", DECOMPOSITION_SCHEMA), ("entry_duplicate_ledger", DUPLICATE_SCHEMA)):
        for field in schema:
            rows.append({
                "table": table, "field": field.name, "arrow_type": str(field.type),
                "nullable": field.nullable,
                "unit": "eV/atom" if "per_atom" in field.name or "above_hull" in field.name else ("eV" if "energy" in field.name or field.name == "correction" else ""),
                "description": descriptions.get(field.name, "Traceable P3.2 contextual phase entry, decomposition, or ledger field."),
            })
    return rows


def _validated_phase_task_id(config: dict[str, Any]) -> str:
    task_id = str(config.get("task_id", ""))
    amended = (
        config.get("energy_policy", {}).get("context_contract_status")
        == "AMENDED_2026-08-24"
    )
    if task_id == "P3.2" and amended:
        return task_id
    if (
        task_id == "R4.0"
        and amended
        and config.get("authorization_status") == "PI_AMENDMENT_2026-08-30"
    ):
        return task_id
    raise RuntimeError(
        "Phase reconstruction requires either the frozen P3.2 contract or "
        "the approved R4.0 current-release amendment"
    )


def build_unified_phase_diagrams(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase_task_id = _validated_phase_task_id(config)
    p3_report = json.loads(Path(config["input"]["p3_1_report"]).read_text(encoding="utf-8"))
    if not (p3_report.get("task_status") == "DONE" and p3_report.get("status") == "PASS" and p3_report.get("gate_decision") == "GO"):
        raise RuntimeError("P3.1 DONE/PASS/GO is required")
    started = utc_now()
    root = Path(config["input"]["normalized_root"])
    snapshots = [str(item) for item in config["snapshots"]]
    config_hash = sha256_file(config_path)
    source_hashes = {snapshot: _phase_source_hash(root, snapshot) for snapshot in snapshots}
    parts = Path(config["output"]["root"]) / "_context_parts"
    parts.mkdir(parents=True, exist_ok=True)
    homogeneous_jobs = [
        {"snapshot": snapshot, "thermo_type": workflow, "config": config, "parts_root": str(parts), "source_hash": source_hashes[snapshot], "config_hash": config_hash}
        for snapshot in snapshots for workflow in config["energy_policy"]["mixed_base_workflows"]
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(config["execution"]["workers"])) as executor:
        futures = {executor.submit(_build_homogeneous_job, job): (job["snapshot"], job["thermo_type"]) for job in homogeneous_jobs}
        for future in as_completed(futures):
            results.append(future.result())
    by_group = {(item["snapshot_id"], item["thermo_type"]): item for item in results}
    mixed_jobs = []
    for snapshot in snapshots:
        gga = by_group[(snapshot, "GGA_GGA+U")]
        mixed_jobs.append(
            {
                "snapshot": snapshot, "config": config, "parts_root": str(parts),
                "source_hash": source_hashes[snapshot], "config_hash": config_hash,
            "gga_phase": str(_stored_path(next(item["path"] for item in gga["outputs"] if item["path"].endswith(".phase.parquet")))),
            "gga_decomp": str(_stored_path(next(item["path"] for item in gga["outputs"] if item["path"].endswith(".decomposition.parquet")))),
            }
        )
    # Structure-bearing mixed entries are memory intensive (about 5 GB per
    # worker on the largest frozen snapshot).  Cap this stage at two workers;
    # homogeneous phase diagrams can still use the configured wider pool.
    with ProcessPoolExecutor(max_workers=min(2, int(config["execution"]["workers"]))) as executor:
        futures = {executor.submit(_build_mixed_job, job): job["snapshot"] for job in mixed_jobs}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["snapshot_id"], item["thermo_type"]))
    def artifact(result: dict[str, Any], suffix: str) -> Path:
        return _stored_path(
            next(item["path"] for item in result["outputs"] if item["path"].endswith(suffix))
        )
    phase_path = Path(config["output"]["phase_entries"])
    decomp_path = Path(config["output"]["decompositions"])
    duplicate_path = Path(config["output"]["duplicate_ledger"])
    _combine([artifact(item, ".phase.parquet") for item in results], phase_path, ENTRY_SCHEMA, config["parquet"])
    _combine([artifact(item, ".decomposition.parquet") for item in results], decomp_path, DECOMPOSITION_SCHEMA, config["parquet"])
    _combine([artifact(item, ".duplicates.parquet") for item in results], duplicate_path, DUPLICATE_SCHEMA, config["parquet"])
    ambiguity_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    reconciliation_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    alias_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    exclusion_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        ambiguity_path = artifact(result, ".ambiguities.jsonl")
        ambiguity_rows.extend(json.loads(line) for line in ambiguity_path.read_text(encoding="utf-8").splitlines() if line)
        reference_rows.extend(result["reference_sample"])
        for row in result.get("energy_reconciliations", []):
            key = (
                str(row["snapshot_id"]), str(row["source_workflow"]),
                str(row["thermo_id"]), str(row["entry_id"]),
            )
            reconciliation_by_key[key] = row
        for row in result.get("mixing_aliases", []):
            key = (
                str(row["snapshot_id"]), str(row["phase_context_chemsys"]),
                str(row["target_entry_id"]),
            )
            alias_by_key[key] = row
        for row in result.get("compatibility_exclusions", []):
            key = (
                str(row["snapshot_id"]), str(row["thermo_type"]),
                str(row["thermo_id"]),
            )
            prior = exclusion_by_key.get(key)
            if prior is not None and prior != row:
                raise RuntimeError(
                    f"Conflicting compatibility exclusion rows for {key}"
                )
            exclusion_by_key[key] = row
    ambiguity_path = Path(config["output"]["ambiguity_ledger"])
    write_jsonl_atomic(ambiguity_path, ambiguity_rows)
    audit_csv = Path(config["output"]["reference_audit_csv"])
    write_csv_atomic(audit_csv, reference_rows)
    reconciliation_rows = [reconciliation_by_key[key] for key in sorted(reconciliation_by_key)]
    reconciliation_path = audit_csv.with_name("energy_reconciliation_ledger.csv")
    _write_reconciliation_csv(reconciliation_path, reconciliation_rows)
    alias_rows = [alias_by_key[key] for key in sorted(alias_by_key)]
    alias_path = audit_csv.with_name("mixing_duplicate_alias_ledger.csv")
    _write_mixing_alias_csv(alias_path, alias_rows)
    exclusion_rows = [exclusion_by_key[key] for key in sorted(exclusion_by_key)]
    exclusion_path = Path(config["output"]["compatibility_exclusion_ledger"])
    _write_compatibility_exclusion_csv(exclusion_path, exclusion_rows)
    source_states = sum(item["reference_test"]["source_states"] for item in results)
    eligible_source_states = sum(
        item["reference_test"].get(
            "eligible_source_states", item["reference_test"]["source_states"]
        )
        for item in results
    )
    matched_states = sum(item["reference_test"]["matched_states"] for item in results)
    group_match_rate = min(item["reference_test"]["match_rate"] for item in results)
    reference = {
        "task_id": phase_task_id, "created_at_utc": utc_now(), "source_states": source_states,
        "eligible_source_states": eligible_source_states,
        "excluded_source_states": len(exclusion_rows),
        "matched_states": matched_states, "match_rate": group_match_rate,
        "aggregation": "minimum_group_gate_rate",
        "required_match_rate": float(config["reference_test"]["required_match_rate"]),
        "absolute_tolerance_eV_per_atom": float(config["reference_test"]["absolute_tolerance_eV_per_atom"]),
        "groups": [item["reference_test"] for item in results],
    }
    reference["status"] = "PASS" if reference["match_rate"] >= reference["required_match_rate"] else "FAIL"
    audit_json = Path(config["output"]["reference_audit_json"])
    write_json_atomic(audit_json, reference)
    dictionary = Path(config["output"]["data_dictionary"])
    write_csv_atomic(dictionary, _dictionary_rows())
    gate_passed = not ambiguity_rows and reference["status"] == "PASS"
    outputs = [phase_path, decomp_path, duplicate_path]
    mixed_groups = [
        item["reference_test"] for item in results
        if item["thermo_type"] == "GGA_GGA+U_R2SCAN"
    ]
    mixed_source_states = sum(item["source_states"] for item in mixed_groups)
    exclusion_summary = {
        "rule_version": str(config["reference_test"]["eligibility_rule_version"]),
        "source_states": mixed_source_states,
        "eligible_source_states": sum(
            item["eligible_source_states"] for item in mixed_groups
        ),
        "excluded_source_states": len(exclusion_rows),
        "overall_exclusion_fraction": (
            len(exclusion_rows) / mixed_source_states if mixed_source_states else 0.0
        ),
        "maximum_exclusion_fraction": float(
            config["reference_test"]["maximum_exclusion_fraction"]
        ),
        "reason_counts": dict(sorted(Counter(
            row["reason"] for row in exclusion_rows
        ).items())),
        "chemistry_counts": dict(sorted(Counter(
            row["phase_context_chemsys"] for row in exclusion_rows
        ).items())),
        "by_snapshot": [
            {
                "snapshot_id": item["snapshot_id"],
                "source_states": item["source_states"],
                "eligible_source_states": item["eligible_source_states"],
                "excluded_source_states": item["excluded_source_states"],
                "exclusion_fraction": item["compatibility_exclusion_fraction"],
                "cap_passed": item["compatibility_exclusion_cap_passed"],
            }
            for item in mixed_groups
        ],
    }
    manifest = {
        "task_id": phase_task_id, "status": "PASS" if gate_passed else "FAIL", "gate_status": "GO" if gate_passed else "NO-GO",
        "started_at_utc": started, "ended_at_utc": utc_now(), "seed": int(config["seed"]), "network_access": False,
        "phase_context_contract": config["energy_policy"]["context_contract_status"],
        "phase_context_key": config["energy_policy"]["phase_context_key"],
        "config_path": str(config_path), "config_sha256": config_hash, "source_hashes": source_hashes,
        "groups": [{key: value for key, value in item.items() if key not in {"reference_sample", "outputs"}} for item in results],
        "target_entries": sum(item["target_entries"] for item in results),
        "context_entry_rows": sum(item["context_entry_rows"] for item in results),
        "competitor_rows": sum(item["competitor_rows"] for item in results),
        "decomposition_rows": sum(item["decomposition_rows"] for item in results),
        "duplicate_rows": sum(item["duplicate_rows"] for item in results),
        "energy_reconciliation_rows": len(reconciliation_rows),
        "mixing_alias_rows": len(alias_rows),
        "compatibility_exclusion_rows": len(exclusion_rows),
        "compatibility_exclusion_summary": exclusion_summary,
        "ambiguity_rows": len(ambiguity_rows), "warnings": sorted({warning for item in results for warning in item["warnings"]}),
        "reference_test": reference,
        "outputs": [{"path": str(path), "rows": pq.ParquetFile(path).metadata.num_rows, "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema_sha256": _schema_hash(path)} for path in outputs],
        "supporting_files": {
            str(path): sha256_file(path)
            for path in (
                ambiguity_path, audit_csv, audit_json, reconciliation_path,
                alias_path, exclusion_path, dictionary,
            )
        },
        "gate": {"unit_tests_required": True, "reference_tests_passed": reference["status"] == "PASS", "no_unledgered_errors": not ambiguity_rows, "passed": gate_passed},
    }
    manifest_path = Path(config["output"]["manifest"])
    write_json_atomic(manifest_path, manifest)
    # Retain deterministic group parts as an ignored, resumable audit cache.
    # They are not formal P3.2 deliverables and are excluded from the
    # manifest, but keeping them avoids destructive cleanup and makes a
    # failed verification reproducible without recomputing every context.
    return manifest


def build_unified_phase_diagrams_sharded(
    config_path: str | Path,
    *,
    shards_per_snapshot: int,
    workers: int,
) -> dict[str, Any]:
    """Build mixed P3.2 contexts in resumable deterministic process shards.

    Scientific configuration and canonical output contracts remain owned by
    ``config_path``.  Shard and worker counts affect scheduling only, so they
    are intentionally command-line execution parameters rather than research
    configuration fields.
    """
    if shards_per_snapshot < 1 or workers < 1:
        raise ValueError("shards_per_snapshot and workers must be positive")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validated_phase_task_id(config)
    p3_report = json.loads(
        Path(config["input"]["p3_1_report"]).read_text(encoding="utf-8")
    )
    if not (
        p3_report.get("task_status") == "DONE"
        and p3_report.get("status") == "PASS"
        and p3_report.get("gate_decision") == "GO"
    ):
        raise RuntimeError("P3.1 DONE/PASS/GO is required")
    root = Path(config["input"]["normalized_root"])
    snapshots = [str(item) for item in config["snapshots"]]
    config_hash = sha256_file(config_path)
    source_hashes = {snapshot: _phase_source_hash(root, snapshot) for snapshot in snapshots}
    parts = Path(config["output"]["root"]) / "_context_parts"
    shards_root = Path(config["output"]["root"]) / "_mixed_context_shards"
    parts.mkdir(parents=True, exist_ok=True)
    shards_root.mkdir(parents=True, exist_ok=True)
    homogeneous_jobs = [
        {
            "snapshot": snapshot,
            "thermo_type": workflow,
            "config": config,
            "parts_root": str(parts),
            "source_hash": source_hashes[snapshot],
            "config_hash": config_hash,
        }
        for snapshot in snapshots
        for workflow in config["energy_policy"]["mixed_base_workflows"]
    ]
    homogeneous: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(homogeneous_jobs))) as executor:
        futures = [executor.submit(_build_homogeneous_job, job) for job in homogeneous_jobs]
        for future in as_completed(futures):
            homogeneous.append(future.result())
    homogeneous_by_group = {
        (item["snapshot_id"], item["thermo_type"]): item for item in homogeneous
    }
    shard_jobs: list[dict[str, Any]] = []
    for snapshot in snapshots:
        gga = homogeneous_by_group[(snapshot, "GGA_GGA+U")]
        gga_phase = str(_stored_path(next(
            item["path"] for item in gga["outputs"] if item["path"].endswith(".phase.parquet")
        )))
        gga_decomp = str(_stored_path(next(
            item["path"]
            for item in gga["outputs"]
            if item["path"].endswith(".decomposition.parquet")
        )))
        for shard_index in range(shards_per_snapshot):
            shard_jobs.append(
                {
                    "snapshot": snapshot,
                    "shard_index": shard_index,
                    "shard_count": shards_per_snapshot,
                    "config": config,
                    "parts_root": str(parts),
                    "shards_root": str(shards_root),
                    "source_hash": source_hashes[snapshot],
                    "config_hash": config_hash,
                    "gga_phase": gga_phase,
                    "gga_decomp": gga_decomp,
                }
            )
    with ProcessPoolExecutor(max_workers=min(workers, len(shard_jobs))) as executor:
        futures = [executor.submit(_build_mixed_shard_job, job) for job in shard_jobs]
        for future in as_completed(futures):
            future.result()
    finalize_jobs = [
        {
            "snapshot": snapshot,
            "shard_count": shards_per_snapshot,
            "config": config,
            "parts_root": str(parts),
            "shards_root": str(shards_root),
            "source_hash": source_hashes[snapshot],
            "config_hash": config_hash,
        }
        for snapshot in snapshots
    ]
    with ProcessPoolExecutor(max_workers=min(workers, len(finalize_jobs))) as executor:
        futures = [executor.submit(_finalize_mixed_shards_job, job) for job in finalize_jobs]
        for future in as_completed(futures):
            future.result()
    return build_unified_phase_diagrams(config_path)


def verify_unified_phase_diagrams(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["output"]["manifest"]).read_text(encoding="utf-8"))
    failures: list[str] = []
    for artifact in manifest["outputs"]:
        path = Path(artifact["path"])
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"hash_or_missing:{path}")
        elif pq.ParquetFile(path).metadata.num_rows != artifact["rows"]:
            failures.append(f"rows:{path}")
        elif _schema_hash(path) != artifact["schema_sha256"]:
            failures.append(f"schema:{path}")
    for path_text, expected in manifest["supporting_files"].items():
        if not Path(path_text).exists() or sha256_file(path_text) != expected:
            failures.append(f"supporting:{path_text}")
    phase = pq.read_table(config["output"]["phase_entries"], columns=["unified_entry_id", "snapshot_id", "thermo_type", "phase_context_chemsys", "entry_id", "is_target", "energy_above_hull"])
    decomposition = pq.read_table(config["output"]["decompositions"], columns=["unified_entry_id", "component_unified_entry_id", "amount"])
    if pc.count_distinct(phase["unified_entry_id"]).as_py() != phase.num_rows:
        failures.append("duplicate_contextual_entry_id")
    expected_ids = [contextual_entry_id(*row) for row in zip(phase["snapshot_id"].to_pylist(), phase["thermo_type"].to_pylist(), phase["phase_context_chemsys"].to_pylist(), phase["entry_id"].to_pylist(), strict=True)]
    if expected_ids != phase["unified_entry_id"].to_pylist():
        failures.append("contextual_entry_id_mismatch")
    phase_ids = set(phase["unified_entry_id"].to_pylist())
    if any(value not in phase_ids for value in pc.unique(decomposition["component_unified_entry_id"]).to_pylist()):
        failures.append("missing_context_component_entry")
    sums = decomposition.group_by("unified_entry_id").aggregate([("amount", "sum")])
    max_sum_error = max((abs(float(value) - 1.0) for value in sums["amount_sum"].to_pylist()), default=0.0)
    if max_sum_error > 1e-8:
        failures.append("decomposition_sum")
    if pc.any(pc.less(phase["energy_above_hull"], -float(config["energy_policy"]["stable_energy_tolerance_eV_per_atom"]))).as_py():
        failures.append("negative_hull")
    target_rows = pc.sum(pc.cast(phase["is_target"], pa.int64())).as_py()
    if target_rows != manifest["target_entries"]:
        failures.append("target_row_count")
    if not manifest["gate"]["passed"]:
        failures.append("gate_not_passed")
    temporary = list(Path(config["output"]["root"]).rglob("*.tmp"))
    if temporary:
        failures.append("temporary_files")
    return {
        "task_id": str(config["task_id"]), "status": "PASS" if not failures else "FAIL", "failures": failures,
        "context_entry_rows": phase.num_rows, "target_entries": target_rows,
        "decomposition_rows": decomposition.num_rows, "max_decomposition_sum_error": max_sum_error,
        "reference_match_rate": manifest["reference_test"]["match_rate"],
        "gate_status": manifest["gate_status"], "network_access": False,
        "temporary_files": len(temporary), "verified_at_utc": utc_now(),
    }
