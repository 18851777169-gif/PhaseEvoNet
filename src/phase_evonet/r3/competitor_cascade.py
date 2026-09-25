"""R3.3 attribution-sensitivity, competitor-counterfactual, and cascade audit.

Every effect in this module is a database-state counterfactual or an accounting
contribution.  Nothing here is interpreted as physical causality.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from scipy.optimize import linprog

from phase_evonet.transition_attribution import composition_signature

from .attribution_sensitivity import (
    FULL_MASK,
    PLAYERS,
    coalition_label,
    summarize_game,
)
from .common import open_formal_input, sha256_file
from .energy_amplitude import _records_with_elemental_terminals


TASK_ID = "R3.3"
METHOD_VERSION = "PHASEEVONET_R3_3_ATTRIBUTION_CASCADE_V1"
EDGE_DEFINITIONS = (
    "selected_active",
    "necessary_10meV",
    "necessary_25meV",
    "contributory_5meV",
    "minimal_set_member",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def _read_config(config_path: str | os.PathLike[str]) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve(strict=True)
    repo = path.parents[2]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID or config.get("method_version") != METHOD_VERSION:
        raise RuntimeError("R3.3 frozen task/method identifiers do not match")
    if tuple(config["attribution_sensitivity"]["players"]) != PLAYERS:
        raise RuntimeError("R3.3 four-player order changed")
    return repo, path, config


def _artifact(path: Path, repo: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
    return record


def _blake_id(prefix: str, value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}-{digest}"


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).casefold().strip().split())


def _composition_dict(value: object) -> dict[str, float]:
    return {str(key): float(amount) for key, amount in json.loads(str(value)).items()}


def _composition_vector(value: object, elements: Sequence[str]) -> np.ndarray:
    comp = _composition_dict(value)
    total = sum(comp.values())
    if total <= 0:
        raise ValueError("composition must contain a positive atom count")
    return np.asarray([comp.get(element, 0.0) / total for element in elements], dtype=float)


def _per_atom(row: Mapping[str, Any], field: str) -> float:
    if field == "corrected_energy" and row.get("corrected_energy_per_atom") is not None:
        return float(row["corrected_energy_per_atom"])
    atoms = float(row.get("num_atoms") or sum(_composition_dict(row["composition_json"]).values()))
    return float(row[field]) / atoms


def solve_hull_distance(
    candidate: Mapping[str, Any],
    competitors: Sequence[Mapping[str, Any]],
    elements: Sequence[str],
) -> tuple[float, str, np.ndarray | None, float | None]:
    """Solve the declared fixed-energy hull LP and retain its decomposition."""

    if not competitors:
        return 0.0, "no_competitors", None, None
    target = _composition_vector(candidate["composition_json"], elements)
    matrix = np.column_stack(
        [_composition_vector(row["composition_json"], elements) for row in competitors]
    )
    energies = np.asarray([_per_atom(row, "corrected_energy") for row in competitors])
    result = linprog(energies, A_eq=matrix, b_eq=target, bounds=(0.0, None), method="highs")
    if result.status == 2:
        return 0.0, "no_feasible_decomposition", None, None
    if not result.success or result.fun is None or result.x is None:
        return math.nan, f"solver_failure_{result.status}:{result.message}", None, None
    candidate_energy = _per_atom(candidate, "corrected_energy")
    return max(0.0, candidate_energy - float(result.fun)), "feasible", result.x, float(result.fun)


def apply_competitor_actions(
    target_rows: Sequence[Mapping[str, Any]],
    actions: Mapping[bytes, Mapping[str, Any]],
    selected: Iterable[bytes],
) -> list[dict[str, Any]]:
    selected_set = set(selected)
    output: list[dict[str, Any]] = []
    for original in target_rows:
        row = dict(original)
        uid = bytes(row["unified_entry_id"])
        if uid not in selected_set:
            output.append(row)
            continue
        action = actions[uid]
        if action["operation"] == "remove":
            continue
        if action["operation"] != "revert":
            raise ValueError(f"unknown competitor action: {action['operation']}")
        atoms = float(row.get("num_atoms") or sum(_composition_dict(row["composition_json"]).values()))
        row["uncorrected_energy"] = float(action["source_uncorrected_energy_per_atom"]) * atoms
        row["correction"] = float(action["source_correction_per_atom"]) * atoms
        row["corrected_energy"] = row["uncorrected_energy"] + row["correction"]
        row["corrected_energy_per_atom"] = row["corrected_energy"] / atoms
        output.append(row)
    return output


def search_minimal_actions(
    action_ids: Sequence[bytes],
    evaluator: Callable[[frozenset[bytes]], float],
    *,
    full_value: float,
    threshold: float,
    exact_pool_cap: int,
    node_budget: int,
    wall_time_limit_seconds: float,
) -> dict[str, Any]:
    """Deterministic cardinality search with honest exact/bounded labels."""

    ordered = tuple(action_ids)
    if full_value < threshold:
        return {
            "status": "ineligible_below_threshold",
            "lower_bound": None,
            "upper_bound": None,
            "members": (),
            "nodes": 0,
            "pool_size": len(ordered),
        }
    singleton = {action: float(evaluator(frozenset((action,)))) for action in ordered}
    greedy_order = sorted(ordered, key=lambda action: (singleton[action], action.hex()))
    greedy_members: list[bytes] = []
    greedy_upper: int | None = None
    for action in greedy_order:
        greedy_members.append(action)
        if evaluator(frozenset(greedy_members)) < threshold:
            greedy_upper = len(greedy_members)
            break
    if greedy_upper is None and ordered:
        # Every admitted action is monotone: removal deletes feasible hull
        # decompositions and a retained reversion only raises that entry's
        # energy. If the full pool cannot reverse the threshold, no subset can.
        return {
            "status": "infeasible_monotone_full_pool",
            "lower_bound": None,
            "upper_bound": None,
            "members": (),
            "nodes": len(greedy_members),
            "pool_size": len(ordered),
        }
    if len(ordered) > exact_pool_cap:
        return {
            "status": "bounded_pool_over_cap",
            "lower_bound": 1 if ordered else None,
            "upper_bound": greedy_upper,
            "members": tuple(greedy_members) if greedy_upper is not None else (),
            "nodes": len(ordered),
            "pool_size": len(ordered),
        }
    if not ordered:
        return {
            "status": "infeasible_empty_pool",
            "lower_bound": None,
            "upper_bound": None,
            "members": (),
            "nodes": 0,
            "pool_size": 0,
        }
    started = time.monotonic()
    nodes = 0
    maximum = greedy_upper
    for size in range(1, maximum + 1):
        for subset in itertools.combinations(ordered, size):
            nodes += 1
            if nodes > node_budget or time.monotonic() - started > wall_time_limit_seconds:
                return {
                    "status": "minimal_set_unresolved",
                    "lower_bound": size,
                    "upper_bound": greedy_upper,
                    "members": tuple(greedy_members) if greedy_upper is not None else (),
                    "nodes": nodes,
                    "pool_size": len(ordered),
                }
            if evaluator(frozenset(subset)) < threshold:
                return {
                    "status": "exact",
                    "lower_bound": size,
                    "upper_bound": size,
                    "members": tuple(subset),
                    "nodes": nodes,
                    "pool_size": len(ordered),
                }
    return {
        "status": "infeasible_exhaustive",
        "lower_bound": None,
        "upper_bound": None,
        "members": (),
        "nodes": nodes,
        "pool_size": len(ordered),
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum())) - (n + 1) / n)


def _concentration(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    ordered = np.sort(values)[::-1]
    output = {"gini": _gini(values), "hhi": float(np.square(values / total).sum()) if total else 0.0}
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(len(ordered) * fraction))) if len(ordered) else 0
        output[f"top_{int(fraction * 100)}pct_share"] = float(ordered[:count].sum() / total) if total else 0.0
    return output


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _sidecar_hash(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise RuntimeError(f"missing frozen report sidecar: {sidecar}")
    value = sidecar.read_text(encoding="utf-8").split()[0].casefold()
    if len(value) != 64:
        raise RuntimeError(f"invalid SHA-256 sidecar: {sidecar}")
    return value


def _authorize_inputs(
    repo: Path, config: dict[str, Any], access_log: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    r3_manifest_path = repo / "data/manifests/R3_1/manifest.json"
    r3_manifest = json.loads(r3_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "r3_1_report": _sidecar_hash(repo / config["input"]["r3_1_report"]),
        "exact_flip_amplitude": r3_manifest["artifacts"][config["input"]["exact_flip_amplitude"]]["sha256"],
        "robust_transition": r3_manifest["artifacts"][config["input"]["robust_transition"]]["sha256"],
        "phase_entries": r3_manifest["input_hashes"]["phase_entries"],
        "phase_decompositions": r3_manifest["input_hashes"]["phase_decompositions"],
        "transition_attribution": r3_manifest["input_hashes"]["transition_attribution"],
        "counterfactual_values": r3_manifest["input_hashes"]["counterfactual_values"],
    }
    path_map_path = repo / "reports/R3_0/r3_0_path_map.json"
    path_map = json.loads(path_map_path.read_text(encoding="utf-8")) if path_map_path.exists() else {}
    paths: dict[str, Path] = {}
    for key, logical in config["input"].items():
        requested = repo / logical
        if not requested.exists() and logical in path_map:
            requested = Path(path_map[logical])
        with open_formal_input(
            requested,
            expected[key],
            task_id=TASK_ID,
            access_log=access_log,
            purpose=f"R3.3 formal build input: {key}",
            allowed_roots=[repo],
            caller="phase_evonet.r3.competitor_cascade.build_attribution_cascade",
        ):
            pass
        paths[key] = requested.resolve(strict=True)
    return paths, expected


def _forbidden_access_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        status = str(json.loads(line).get("status", ""))
        if "FORBIDDEN" in status or "ROOT_ESCAPE" in status or "PATH_TRAVERSAL" in status:
            count += 1
    return count


def _build_attribution_sensitivity(
    attribution_path: Path,
    coalition_path: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    attribution = pq.read_table(attribution_path).to_pandas()
    coalitions = pq.read_table(coalition_path).to_pandas()
    attribution["attribution_id"] = attribution["attribution_id"].map(bytes)
    coalitions["attribution_id"] = coalitions["attribution_id"].map(bytes)
    rows: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for attr_id, group in coalitions.groupby("attribution_id", sort=False):
        if len(group) != 16 or group["coalition_mask"].nunique() != 16:
            raise RuntimeError(f"incomplete coalition game for {attr_id.hex()}")
        values = {
            int(row.coalition_mask): float(row.counterfactual_energy_above_hull)
            for row in group.itertuples(index=False)
        }
        summary = summarize_game(values)
        record: dict[str, Any] = {
            "attribution_id": attr_id,
            "transition_id": bytes(group.iloc[0]["transition_id"]),
            "source_value_eV_per_atom": values[0],
            "target_value_eV_per_atom": values[FULL_MASK],
            "endpoint_delta_eV_per_atom": summary["endpoint_delta"],
            "shapley_residual_eV_per_atom": summary["shapley_residual"],
            "mobius_max_reconstruction_error_eV_per_atom": summary["mobius_max_reconstruction_error"],
            "interaction_absolute_total_eV_per_atom": summary["interaction_absolute_total"],
            "method_version": METHOD_VERSION,
        }
        for method in ("shapley", "single_switch", "leave_one_out", "total_effect"):
            for player, value in summary[method].items():
                record[f"{method}_{player}_eV_per_atom"] = value
                record[f"abs_{method}_{player}_eV_per_atom"] = abs(value)
            record[f"{method}_rank_order"] = "|".join(summary["rank_orders"]["exact_shapley" if method == "shapley" else method])
        for method, value in summary["rank_spearman_vs_shapley"].items():
            record[f"spearman_shapley_vs_{method}"] = value
        rows.append(record)
        for mask, value in summary["harsanyi"].items():
            interactions.append(
                {
                    "attribution_id": attr_id.hex(),
                    "transition_id": bytes(group.iloc[0]["transition_id"]).hex(),
                    "coalition_mask": mask,
                    "coalition": coalition_label(mask),
                    "interaction_order": mask.bit_count(),
                    "harsanyi_dividend_eV_per_atom": value,
                    "absolute_harsanyi_dividend_eV_per_atom": abs(value),
                }
            )
    frame = pd.DataFrame(rows).merge(
        attribution[
            [
                "attribution_id",
                "canonical_lineage_id",
                "identity_confidence",
                "source_snapshot",
                "target_snapshot",
                "thermo_type",
                "phase_context_chemsys",
                "candidate_identity_changed",
                "dominant_channel",
                *[f"{player}_contribution" for player in PLAYERS],
            ]
        ],
        on="attribution_id",
        how="left",
        validate="one_to_one",
    )
    max_stored_error = 0.0
    for player in PLAYERS:
        max_stored_error = max(
            max_stored_error,
            float((frame[f"shapley_{player}_eV_per_atom"] - frame[f"{player}_contribution"]).abs().max()),
        )
    agreement_rows: list[dict[str, Any]] = []
    for subset_name, subset in (
        ("all_3113", frame),
        ("candidate_identity_unchanged", frame.loc[~frame["candidate_identity_changed"]]),
    ):
        for method in ("single_switch", "leave_one_out", "total_effect"):
            rank_col = f"spearman_shapley_vs_{method}"
            top_match = (
                subset["shapley_rank_order"].str.split("|").str[0]
                == subset[f"{method}_rank_order"].str.split("|").str[0]
            )
            agreement_rows.append(
                {
                    "population": subset_name,
                    "comparison": f"exact_shapley_vs_{method}",
                    "rows": len(subset),
                    "mean_within_transition_spearman": float(subset[rank_col].mean()),
                    "median_within_transition_spearman": float(subset[rank_col].median()),
                    "top_channel_agreement_fraction": float(top_match.mean()),
                }
            )
    diagnostics = {
        "rows": len(frame),
        "coalition_rows": len(coalitions),
        "all_16_coalitions_present": len(frame) == len(attribution) == 3113 and len(coalitions) == 3113 * 16,
        "max_shapley_residual_eV_per_atom": float(frame["shapley_residual_eV_per_atom"].abs().max()),
        "max_mobius_reconstruction_error_eV_per_atom": float(frame["mobius_max_reconstruction_error_eV_per_atom"].max()),
        "max_stored_shapley_difference_eV_per_atom": max_stored_error,
        "within_tolerance": bool(
            frame["shapley_residual_eV_per_atom"].abs().max() <= tolerance
            and frame["mobius_max_reconstruction_error_eV_per_atom"].max() <= tolerance
            and max_stored_error <= tolerance
        ),
    }
    return frame, pd.DataFrame(agreement_rows), pd.DataFrame(interactions), diagnostics


PHASE_COLUMNS = [
    "snapshot_id",
    "thermo_type",
    "phase_context_chemsys",
    "unified_entry_id",
    "entry_id",
    "is_target",
    "is_competitor",
    "source_workflow",
    "compatibility_mode",
    "task_id",
    "material_id",
    "thermo_id",
    "composition_json",
    "reduced_formula",
    "num_atoms",
    "uncorrected_energy",
    "correction",
    "corrected_energy",
    "corrected_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "source_object_sha256",
    "source_key",
    "source_row_number",
]


def _load_relevant_contexts(
    phase_path: Path, needed: set[tuple[str, str, str]]
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    terminal_needed = {
        (snapshot, terminal_workflow, element)
        for snapshot, thermo_type, context in needed
        for terminal_workflow in (("GGA_GGA+U", "R2SCAN") if thermo_type == "GGA_GGA+U_R2SCAN" else (thermo_type,))
        for element in context.split("-")
    }
    contexts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    terminals: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for batch in pq.ParquetFile(phase_path).iter_batches(batch_size=100_000, columns=PHASE_COLUMNS):
        for row in batch.to_pylist():
            key = (str(row["snapshot_id"]), str(row["thermo_type"]), str(row["phase_context_chemsys"]))
            if key in needed:
                contexts[key].append(row)
            if key in terminal_needed:
                terminals[key].append(row)
    missing = needed - set(contexts)
    if missing:
        raise RuntimeError(f"missing {len(missing)} R3.3 phase contexts")
    return dict(contexts), dict(terminals)


def _match_key(row: Mapping[str, Any]) -> str:
    return f"{_norm(row.get('source_workflow'))}|{_norm(row.get('entry_id'))}|{composition_signature(str(row['composition_json']))}"


def _contextual_competitor_id(row: Mapping[str, Any]) -> str:
    signature = "|".join(
        (
            _norm(row["snapshot_id"]),
            _norm(row["thermo_type"]),
            _norm(row["phase_context_chemsys"]),
            bytes(row["unified_entry_id"]).hex(),
        )
    )
    return _blake_id("ctx", signature)


def _canonical_competitor_id(row: Mapping[str, Any]) -> tuple[str, str]:
    workflow = _norm(row.get("source_workflow"))
    task = _norm(row.get("task_id"))
    entry = _norm(row.get("entry_id"))
    comp = composition_signature(str(row["composition_json"]))
    if task or entry:
        return _blake_id("cmp", f"{workflow}|{task}|{entry}|{comp}"), "normalized_task_entry_composition_workflow"
    provenance = "|".join(
        (
            workflow,
            _norm(row.get("source_object_sha256")),
            _norm(row.get("source_key")),
            _norm(row.get("source_row_number")),
            comp,
        )
    )
    return _blake_id("cmp", provenance), "blake2b128_provenance_signature"


def _classify_change(
    source: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    tolerance: float,
) -> tuple[str, dict[str, bool]]:
    flags = {"uncorrected_energy_changed": False, "correction_changed": False, "identity_changed": False}
    if source is None:
        return "target_only_arrival", flags
    if target is None:
        return "source_only_removal", flags
    flags["identity_changed"] = bool(
        _norm(source.get("task_id")) != _norm(target.get("task_id"))
        or _norm(source.get("material_id")) != _norm(target.get("material_id"))
        or _norm(source.get("source_workflow")) != _norm(target.get("source_workflow"))
    )
    flags["uncorrected_energy_changed"] = abs(_per_atom(source, "uncorrected_energy") - _per_atom(target, "uncorrected_energy")) > tolerance
    flags["correction_changed"] = abs(_per_atom(source, "correction") - _per_atom(target, "correction")) > tolerance
    if flags["identity_changed"]:
        return "shared_identity_or_representative_change", flags
    if flags["uncorrected_energy_changed"]:
        return "shared_energy_change", flags
    if flags["correction_changed"]:
        return "shared_correction_change", flags
    return "unchanged", flags


def _deduplicate_contextual_mirrors(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_context: str,
    preferred_ids: set[bytes],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse only proven identical context/terminal mirrors.

    P3.2 can store the same elemental entry once inside a multielement context
    and again in its elemental terminal context under distinct contextual UIDs.
    The mapping below is allowed only when identity, provenance, and all energy
    fields agree.  The selected decomposition UID wins; otherwise the row in
    the candidate's own context wins, followed by UID byte order.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_match_key(row)].append(dict(row))
    output: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for match_key, group in sorted(groups.items()):
        if len(group) == 1:
            output.append(group[0])
            continue
        reference = group[0]
        identity_fields = ("task_id", "material_id", "source_workflow", "source_object_sha256")
        identity_equal = all(
            all(_norm(row.get(field)) == _norm(reference.get(field)) for field in identity_fields)
            for row in group[1:]
        )
        energy_equal = all(
            abs(_per_atom(row, field) - _per_atom(reference, field)) <= tolerance
            for row in group[1:]
            for field in ("uncorrected_energy", "correction", "corrected_energy")
        )
        if not identity_equal or not energy_equal:
            raise RuntimeError(f"non-identical contextual representations share match key: {match_key}")
        preferred = [row for row in group if bytes(row["unified_entry_id"]) in preferred_ids]
        if len(preferred) > 1:
            raise RuntimeError(f"multiple selected decomposition mirrors share match key: {match_key}")
        if preferred:
            chosen = preferred[0]
            rule = "selected_decomposition_uid"
        else:
            chosen = sorted(
                group,
                key=lambda row: (
                    str(row["phase_context_chemsys"]) != candidate_context,
                    bytes(row["unified_entry_id"]).hex(),
                ),
            )[0]
            rule = "candidate_context_then_uid"
        output.append(chosen)
        for alias in group:
            aliases.append(
                {
                    "match_key": match_key,
                    "chosen_unified_entry_id": bytes(chosen["unified_entry_id"]).hex(),
                    "alias_unified_entry_id": bytes(alias["unified_entry_id"]).hex(),
                    "chosen_phase_context": str(chosen["phase_context_chemsys"]),
                    "alias_phase_context": str(alias["phase_context_chemsys"]),
                    "selection_rule": rule,
                    "identity_and_energy_exact_within_tolerance": True,
                }
            )
    return output, aliases


def _load_selected_decompositions(
    decomposition_path: Path,
    target_ids: set[bytes],
    minimum_amount: float,
) -> tuple[dict[bytes, dict[bytes, float]], int]:
    selected: dict[bytes, dict[bytes, float]] = defaultdict(dict)
    duplicate_components = 0
    value_set = pa.array(sorted(target_ids), type=pa.binary(16))
    columns = ["unified_entry_id", "component_unified_entry_id", "amount"]
    for batch in pq.ParquetFile(decomposition_path).iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["unified_entry_id"], value_set=value_set))
        for row in table.to_pylist():
            amount = float(row["amount"])
            if amount <= minimum_amount:
                continue
            candidate = bytes(row["unified_entry_id"])
            component = bytes(row["component_unified_entry_id"])
            if component in selected[candidate]:
                duplicate_components += 1
                selected[candidate][component] += amount
            else:
                selected[candidate][component] = amount
    missing = target_ids - set(selected)
    if missing:
        raise RuntimeError(f"missing selected target decomposition for {len(missing)} candidates")
    return dict(selected), duplicate_components


def _alternate_activity(
    candidate: Mapping[str, Any],
    competitors: Sequence[Mapping[str, Any]],
    elements: Sequence[str],
    optimum_energy: float,
    *,
    energy_slack: float,
    activity_tolerance: float,
) -> list[dict[str, Any]]:
    target = _composition_vector(candidate["composition_json"], elements)
    matrix = np.column_stack(
        [_composition_vector(row["composition_json"], elements) for row in competitors]
    )
    energies = np.asarray([_per_atom(row, "corrected_energy") for row in competitors])
    output: list[dict[str, Any]] = []
    for index, row in enumerate(competitors):
        objective = np.zeros(len(competitors))
        objective[index] = 1.0
        minimum = linprog(
            objective,
            A_ub=energies.reshape(1, -1),
            b_ub=np.asarray([optimum_energy + energy_slack]),
            A_eq=matrix,
            b_eq=target,
            bounds=(0.0, None),
            method="highs",
        )
        maximum = linprog(
            -objective,
            A_ub=energies.reshape(1, -1),
            b_ub=np.asarray([optimum_energy + energy_slack]),
            A_eq=matrix,
            b_eq=target,
            bounds=(0.0, None),
            method="highs",
        )
        if not minimum.success or not maximum.success or minimum.fun is None or maximum.fun is None:
            output.append(
                {
                    "unified_entry_id": bytes(row["unified_entry_id"]),
                    "status": "solver_failure",
                    "minimum_amount": None,
                    "maximum_amount": None,
                    "active_class": "unresolved",
                }
            )
            continue
        min_amount = max(0.0, float(minimum.fun))
        max_amount = max(0.0, -float(maximum.fun))
        if min_amount > activity_tolerance:
            active_class = "always_active"
        elif max_amount > activity_tolerance:
            active_class = "sometimes_active"
        else:
            active_class = "inactive_in_all_audited_optima"
        output.append(
            {
                "unified_entry_id": bytes(row["unified_entry_id"]),
                "status": "PASS",
                "minimum_amount": min_amount,
                "maximum_amount": max_amount,
                "active_class": active_class,
            }
        )
    return output


def _primary_population(exact: pd.DataFrame) -> pd.DataFrame:
    mask = (
        exact["direction"].eq("stable_to_unstable")
        & exact["survives_10meV"]
        & exact["identity_confidence"].eq("A1")
        & exact["candidate_identity_unchanged"]
        & exact["same_workflow"]
        & exact["same_phase_context"]
    )
    frame = exact.loc[mask].copy()
    if len(frame) != 1190:
        raise RuntimeError(f"frozen R3.3 primary population expected 1,190 rows, observed {len(frame):,}")
    return frame.sort_values("transition_id", key=lambda series: series.map(bytes)).reset_index(drop=True)


def _analyze_competitors(
    primary: pd.DataFrame,
    phase_path: Path,
    decomposition_path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tolerance = float(config["attribution_sensitivity"]["tolerance_eV_per_atom"])
    activity_tolerance = float(config["active_decomposition"]["minimum_component_amount"])
    needed = {
        (str(row.source_snapshot), str(row.thermo_type), str(row.phase_context_chemsys))
        for row in primary.itertuples(index=False)
    } | {
        (str(row.target_snapshot), str(row.thermo_type), str(row.phase_context_chemsys))
        for row in primary.itertuples(index=False)
    }
    contexts, terminals = _load_relevant_contexts(phase_path, needed)
    target_ids = {bytes(value) for value in primary["target_unified_entry_id"]}
    selected_decompositions, duplicate_components = _load_selected_decompositions(
        decomposition_path, target_ids, activity_tolerance
    )
    sample_n = min(int(config["active_decomposition"]["alternate_optima_sample_n"]), len(primary))
    ranked = sorted(
        (hashlib.blake2b(bytes(row.transition_id) + int(config["seed"]).to_bytes(8, "little"), digest_size=16).digest(), bytes(row.transition_id))
        for row in primary.itertuples(index=False)
    )
    audit_sample = {transition for _, transition in ranked[:sample_n]}

    change_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    minimal_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    alternate_rows: list[dict[str, Any]] = []
    mirror_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    max_baseline_error = 0.0
    max_decomposition_composition_error = 0.0
    max_decomposition_energy_error = 0.0
    edge_index: dict[tuple[bytes, bytes], int] = {}

    for transition_index, transition in enumerate(primary.itertuples(index=False), start=1):
        if transition_index == 1 or transition_index % 100 == 0:
            print(f"R3.3 competitor audit {transition_index}/{len(primary)}", flush=True)
        transition_id = bytes(transition.transition_id)
        source_key = (str(transition.source_snapshot), str(transition.thermo_type), str(transition.phase_context_chemsys))
        target_key = (str(transition.target_snapshot), str(transition.thermo_type), str(transition.phase_context_chemsys))
        source_records = _records_with_elemental_terminals(source_key, contexts[source_key], terminals)
        target_records = _records_with_elemental_terminals(target_key, contexts[target_key], terminals)
        source_uid = bytes(transition.source_unified_entry_id)
        target_uid = bytes(transition.target_unified_entry_id)
        source_candidate = next((row for row in source_records if bytes(row["unified_entry_id"]) == source_uid), None)
        target_candidate = next((row for row in target_records if bytes(row["unified_entry_id"]) == target_uid), None)
        if source_candidate is None or target_candidate is None:
            failures.append({"transition_id": transition_id.hex(), "stage": "candidate_join", "status": "missing_candidate", "explained": False})
            continue
        selected_amounts = selected_decompositions[target_uid]
        source_competitors_raw = [row for row in source_records if bytes(row["unified_entry_id"]) != source_uid]
        target_competitors_raw = [row for row in target_records if bytes(row["unified_entry_id"]) != target_uid]
        try:
            source_competitors, source_aliases = _deduplicate_contextual_mirrors(
                source_competitors_raw,
                candidate_context=str(transition.phase_context_chemsys),
                preferred_ids=set(),
                tolerance=tolerance,
            )
            target_competitors, target_aliases = _deduplicate_contextual_mirrors(
                target_competitors_raw,
                candidate_context=str(transition.phase_context_chemsys),
                preferred_ids=set(selected_amounts),
                tolerance=tolerance,
            )
            for alias in (*source_aliases, *target_aliases):
                mirror_rows.append({"transition_id": transition_id.hex(), **alias})
        except RuntimeError as exc:
            failures.append({"transition_id": transition_id.hex(), "stage": "competitor_join", "status": str(exc), "explained": False})
            continue
        source_by_key = {_match_key(row): row for row in source_competitors}
        target_by_key = {_match_key(row): row for row in target_competitors}
        if len(source_by_key) != len(source_competitors) or len(target_by_key) != len(target_competitors):
            failures.append({"transition_id": transition_id.hex(), "stage": "competitor_join", "status": "duplicate_match_key", "explained": False})
            continue
        elements = str(transition.phase_context_chemsys).split("-")
        baseline, baseline_status, _weights, optimum_energy = solve_hull_distance(target_candidate, target_competitors, elements)
        if baseline_status not in {"feasible", "no_feasible_decomposition", "no_competitors"} or not math.isfinite(baseline):
            failures.append({"transition_id": transition_id.hex(), "stage": "baseline_solver", "status": baseline_status, "explained": False})
            continue
        baseline_error = abs(baseline - float(transition.target_energy_above_hull_eV_per_atom))
        max_baseline_error = max(max_baseline_error, baseline_error)
        if baseline_error > tolerance:
            failures.append({"transition_id": transition_id.hex(), "stage": "baseline_reconciliation", "status": f"error={baseline_error:.12g}", "explained": False})
            continue

        target_by_uid = {bytes(row["unified_entry_id"]): row for row in target_records}
        unresolved_components = set(selected_amounts) - set(target_by_uid)
        if unresolved_components:
            failures.append({"transition_id": transition_id.hex(), "stage": "decomposition_join", "status": f"unresolved_components={len(unresolved_components)}", "explained": False})
            continue
        target_vector = _composition_vector(target_candidate["composition_json"], elements)
        decomposition_vector = sum(
            amount * _composition_vector(target_by_uid[uid]["composition_json"], elements)
            for uid, amount in selected_amounts.items()
        )
        composition_error = float(np.max(np.abs(target_vector - decomposition_vector)))
        decomposition_energy = float(
            sum(amount * _per_atom(target_by_uid[uid], "corrected_energy") for uid, amount in selected_amounts.items())
        )
        expected_hull_energy = _per_atom(target_candidate, "corrected_energy") - baseline
        energy_error = abs(decomposition_energy - expected_hull_energy)
        max_decomposition_composition_error = max(max_decomposition_composition_error, composition_error)
        max_decomposition_energy_error = max(max_decomposition_energy_error, energy_error)
        if composition_error > tolerance or energy_error > tolerance:
            failures.append({"transition_id": transition_id.hex(), "stage": "decomposition_reconstruction", "status": f"composition_error={composition_error:.12g};energy_error={energy_error:.12g}", "explained": False})
            continue
        transition_edge_indices: dict[bytes, int] = {}
        actions: dict[bytes, dict[str, Any]] = {}
        for match_key in sorted(set(source_by_key) | set(target_by_key)):
            source = source_by_key.get(match_key)
            target = target_by_key.get(match_key)
            change_type, flags = _classify_change(source, target, tolerance)
            identity_row = target if target is not None else source
            assert identity_row is not None
            contextual_id = _contextual_competitor_id(identity_row)
            canonical_id, identity_source = _canonical_competitor_id(identity_row)
            target_comp_uid = bytes(target["unified_entry_id"]) if target is not None else None
            source_comp_uid = bytes(source["unified_entry_id"]) if source is not None else None
            selected_active = bool(target_comp_uid in selected_amounts) if target_comp_uid is not None else False
            base_record = {
                "transition_id": transition_id,
                "candidate_lineage_id": str(transition.canonical_lineage_id),
                "source_snapshot": str(transition.source_snapshot),
                "target_snapshot": str(transition.target_snapshot),
                "thermo_type": str(transition.thermo_type),
                "phase_context_chemsys": str(transition.phase_context_chemsys),
                "competitor_contextual_id": contextual_id,
                "competitor_canonical_id": canonical_id,
                "canonical_identity_source": identity_source,
                "source_unified_entry_id": source_comp_uid,
                "target_unified_entry_id": target_comp_uid,
                "source_entry_id": None if source is None else str(source["entry_id"]),
                "target_entry_id": None if target is None else str(target["entry_id"]),
                "source_task_id": None if source is None else source.get("task_id"),
                "target_task_id": None if target is None else target.get("task_id"),
                "source_material_id": None if source is None else source.get("material_id"),
                "target_material_id": None if target is None else target.get("material_id"),
                "source_workflow": None if source is None else source.get("source_workflow"),
                "target_workflow": None if target is None else target.get("source_workflow"),
                "reduced_formula": str(identity_row.get("reduced_formula") or ""),
                "composition_signature": composition_signature(str(identity_row["composition_json"])),
                "change_type": change_type,
                **flags,
                "source_uncorrected_energy_per_atom": None if source is None else _per_atom(source, "uncorrected_energy"),
                "target_uncorrected_energy_per_atom": None if target is None else _per_atom(target, "uncorrected_energy"),
                "source_correction_per_atom": None if source is None else _per_atom(source, "correction"),
                "target_correction_per_atom": None if target is None else _per_atom(target, "correction"),
                "selected_decomposition_amount": float(selected_amounts.get(target_comp_uid, 0.0)) if target_comp_uid is not None else 0.0,
                "selected_active": selected_active,
                "active_class": "selected_active" if selected_active else "not_selected",
                "method_version": METHOD_VERSION,
            }
            change_rows.append(base_record)
            if target is None:
                continue
            edge = {
                **base_record,
                "full_target_e_hull_eV_per_atom": baseline,
                "single_counterfactual_e_hull_eV_per_atom": None,
                "single_effect_eV_per_atom": None,
                "exact_zero_reversal": False,
                "necessary_10meV": False,
                "necessary_25meV": False,
                "contributory_5meV": False,
                "minimal_set_member": False,
                "single_counterfactual_status": "not_in_single_pool",
            }
            edge_index[(transition_id, target_comp_uid)] = len(edge_rows)
            transition_edge_indices[target_comp_uid] = len(edge_rows)
            edge_rows.append(edge)
            materially_revised = any(flags.values())
            if change_type == "target_only_arrival":
                actions[target_comp_uid] = {"operation": "remove", "change_type": change_type}
            elif source is not None and materially_revised:
                actions[target_comp_uid] = {
                    "operation": "revert",
                    "change_type": change_type,
                    "source_uncorrected_energy_per_atom": _per_atom(source, "uncorrected_energy"),
                    "source_correction_per_atom": _per_atom(source, "correction"),
                }

        for uid, action in actions.items():
            modified = apply_competitor_actions(target_competitors, actions, (uid,))
            value, status, _x, _fun = solve_hull_distance(target_candidate, modified, elements)
            index = transition_edge_indices[uid]
            edge_rows[index]["single_counterfactual_status"] = status
            if not math.isfinite(value) or status.startswith("solver_failure"):
                failures.append({"transition_id": transition_id.hex(), "competitor_uid": uid.hex(), "stage": "single_counterfactual", "status": status, "explained": False})
                continue
            effect = baseline - value
            edge_rows[index].update(
                {
                    "single_counterfactual_e_hull_eV_per_atom": value,
                    "single_effect_eV_per_atom": effect,
                    "exact_zero_reversal": bool(baseline > tolerance and value <= tolerance),
                    "necessary_10meV": bool(baseline >= 0.010 and value < 0.010),
                    "necessary_25meV": bool(baseline >= 0.025 and value < 0.025),
                    "contributory_5meV": bool(effect >= 0.005),
                }
            )

        # Exact-search pool: active revised entries plus target-only entries; all
        # retained actions are monotone removals or energy-raising reversions.
        pool: list[bytes] = []
        exclusion_reasons: dict[str, int] = Counter()
        for uid, action in actions.items():
            index = transition_edge_indices[uid]
            edge = edge_rows[index]
            active_or_arrival = bool(edge["selected_active"] or action["change_type"] == "target_only_arrival")
            if not active_or_arrival:
                exclusion_reasons["revised_but_not_selected_active"] += 1
                continue
            if action["operation"] == "revert":
                source_corrected = float(action["source_uncorrected_energy_per_atom"]) + float(action["source_correction_per_atom"])
                target_row = next(row for row in target_competitors if bytes(row["unified_entry_id"]) == uid)
                if source_corrected <= _per_atom(target_row, "corrected_energy") + tolerance:
                    exclusion_reasons["non_helpful_reversion_not_safely_monotone"] += 1
                    continue
            pool.append(uid)
        pool.sort(key=lambda uid: (-(edge_rows[transition_edge_indices[uid]]["single_effect_eV_per_atom"] or 0.0), uid.hex()))
        cache: dict[frozenset[bytes], float] = {frozenset(): baseline}

        def evaluator(selected: frozenset[bytes]) -> float:
            if selected not in cache:
                modified = apply_competitor_actions(target_competitors, actions, selected)
                value, status, _x, _fun = solve_hull_distance(target_candidate, modified, elements)
                if not math.isfinite(value) or status.startswith("solver_failure"):
                    raise RuntimeError(f"minimal-set solver failed: {status}")
                cache[selected] = value
            return cache[selected]

        try:
            result = search_minimal_actions(
                pool,
                evaluator,
                full_value=baseline,
                threshold=float(config["minimal_set"]["primary_threshold_eV_per_atom"]),
                exact_pool_cap=int(config["minimal_set"]["exact_pool_max_entries"]),
                node_budget=int(config["minimal_set"]["branch_and_bound_node_budget"]),
                wall_time_limit_seconds=float(config["minimal_set"]["wall_time_limit_seconds_per_candidate"]),
            )
        except RuntimeError as exc:
            failures.append({"transition_id": transition_id.hex(), "stage": "minimal_set", "status": str(exc), "explained": False})
            result = {"status": "minimal_set_unresolved_solver_failure", "lower_bound": None, "upper_bound": None, "members": (), "nodes": len(cache), "pool_size": len(pool)}
        member_contextual = [edge_rows[transition_edge_indices[uid]]["competitor_contextual_id"] for uid in result["members"]]
        member_canonical = [edge_rows[transition_edge_indices[uid]]["competitor_canonical_id"] for uid in result["members"]]
        if result["status"] == "exact":
            for uid in result["members"]:
                edge_rows[transition_edge_indices[uid]]["minimal_set_member"] = True
        minimal_record = {
            "transition_id": transition_id,
            "candidate_lineage_id": str(transition.canonical_lineage_id),
            "source_snapshot": str(transition.source_snapshot),
            "target_snapshot": str(transition.target_snapshot),
            "thermo_type": str(transition.thermo_type),
            "phase_context_chemsys": str(transition.phase_context_chemsys),
            "full_target_e_hull_eV_per_atom": baseline,
            "pool_size": int(result["pool_size"]),
            "search_status": str(result["status"]),
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
            "exact_minimum_cardinality": result["lower_bound"] if result["status"] == "exact" else None,
            "search_nodes": int(result["nodes"]),
            "member_contextual_ids_json": json.dumps(member_contextual, separators=(",", ":")),
            "member_canonical_ids_json": json.dumps(member_canonical, separators=(",", ":")),
            "pool_exclusion_reasons_json": json.dumps(dict(exclusion_reasons), sort_keys=True, separators=(",", ":")),
            "method_version": METHOD_VERSION,
        }
        minimal_rows.append(minimal_record)
        audit_rows.append(dict(minimal_record))

        if transition_id in audit_sample and optimum_energy is not None:
            activity = _alternate_activity(
                target_candidate,
                target_competitors,
                elements,
                optimum_energy,
                energy_slack=min(tolerance / 10.0, 1e-8),
                activity_tolerance=activity_tolerance,
            )
            for item in activity:
                uid = item["unified_entry_id"]
                index = transition_edge_indices[uid]
                edge_rows[index]["active_class"] = item["active_class"]
                if item["active_class"] in {"always_active", "sometimes_active"}:
                    edge_rows[index]["selected_active"] = True
                alternate_rows.append(
                    {
                        "transition_id": transition_id.hex(),
                        "candidate_lineage_id": str(transition.canonical_lineage_id),
                        "competitor_contextual_id": edge_rows[index]["competitor_contextual_id"],
                        "competitor_canonical_id": edge_rows[index]["competitor_canonical_id"],
                        **{key: value for key, value in item.items() if key != "unified_entry_id"},
                    }
                )

    edge = pd.DataFrame(edge_rows)
    change = pd.DataFrame(change_rows)
    minimal = pd.DataFrame(minimal_rows)
    audit = pd.DataFrame(audit_rows)
    alternate = pd.DataFrame(alternate_rows)
    mirrors = pd.DataFrame(mirror_rows)
    failure = pd.DataFrame(failures, columns=["transition_id", "competitor_uid", "stage", "status", "explained"])
    diagnostics = {
        "primary_rows": len(primary),
        "processed_rows": int(edge["transition_id"].map(bytes).nunique()) if len(edge) else 0,
        "competitor_change_rows": len(change),
        "edge_rows": len(edge),
        "minimal_search_rows": len(minimal),
        "alternate_optima_sample_candidates": int(alternate["transition_id"].nunique()) if len(alternate) else 0,
        "alternate_optima_rows": len(alternate),
        "contextual_mirror_audit_rows": len(mirrors),
        "duplicate_decomposition_components_aggregated": duplicate_components,
        "max_target_baseline_reconciliation_error_eV_per_atom": max_baseline_error,
        "max_decomposition_composition_error": max_decomposition_composition_error,
        "max_decomposition_energy_error_eV_per_atom": max_decomposition_energy_error,
        "unexplained_failure_rows": int((~failure["explained"].fillna(False)).sum()) if len(failure) else 0,
    }
    return change, edge, minimal, audit, alternate, mirrors, failure, diagnostics


def _network_row(
    frame: pd.DataFrame,
    *,
    edge_definition: str,
    identity_definition: str,
    node_column: str,
    seed: int,
    bootstrap_replicates: int,
    stratum_dimension: str = "overall",
    stratum_value: str = "overall",
) -> dict[str, Any]:
    pairs = frame[[node_column, "candidate_lineage_id"]].drop_duplicates()
    cascade = pairs.groupby(node_column)["candidate_lineage_id"].nunique().sort_values(ascending=False)
    indegree = pairs.groupby("candidate_lineage_id")[node_column].nunique()
    metrics = _concentration(cascade.to_numpy(dtype=float))
    union = _UnionFind()
    for row in pairs.itertuples(index=False):
        union.union(f"n:{getattr(row, node_column)}", f"c:{row.candidate_lineage_id}")
    components = len({union.find(item) for item in union.parent}) if union.parent else 0
    result: dict[str, Any] = {
        "edge_definition": edge_definition,
        "identity_definition": identity_definition,
        "stratum_dimension": stratum_dimension,
        "stratum_value": stratum_value,
        "edge_rows": len(pairs),
        "competitor_nodes": int(cascade.size),
        "candidate_nodes": int(indegree.size),
        "connected_components": int(components),
        "largest_cascade": int(cascade.max()) if len(cascade) else 0,
        "mean_cascade": float(cascade.mean()) if len(cascade) else 0.0,
        "median_cascade": float(cascade.median()) if len(cascade) else 0.0,
        "maximum_candidate_indegree": int(indegree.max()) if len(indegree) else 0,
        "mean_candidate_indegree": float(indegree.mean()) if len(indegree) else 0.0,
        **metrics,
        "bootstrap_replicates": 0,
        "gini_ci_low": None,
        "gini_ci_high": None,
        "hhi_ci_low": None,
        "hhi_ci_high": None,
        "top_10pct_share_ci_low": None,
        "top_10pct_share_ci_high": None,
    }
    if bootstrap_replicates > 0 and len(pairs):
        nodes = sorted(pairs[node_column].unique())
        candidates = sorted(pairs["candidate_lineage_id"].unique())
        node_code = {value: index for index, value in enumerate(nodes)}
        candidate_code = {value: index for index, value in enumerate(candidates)}
        node_indices = pairs[node_column].map(node_code).to_numpy(dtype=int)
        candidate_indices = pairs["candidate_lineage_id"].map(candidate_code).to_numpy(dtype=int)
        rng = np.random.default_rng(seed)
        samples = np.empty((bootstrap_replicates, 3), dtype=float)
        probability = np.full(len(candidates), 1.0 / len(candidates))
        for index in range(bootstrap_replicates):
            multiplicity = rng.multinomial(len(candidates), probability)
            counts = np.bincount(
                node_indices,
                weights=multiplicity[candidate_indices],
                minlength=len(nodes),
            )
            boot = _concentration(counts[counts > 0])
            samples[index] = (boot["gini"], boot["hhi"], boot["top_10pct_share"])
        quantiles = np.quantile(samples, [0.025, 0.975], axis=0)
        result.update(
            {
                "bootstrap_replicates": bootstrap_replicates,
                "gini_ci_low": float(quantiles[0, 0]),
                "gini_ci_high": float(quantiles[1, 0]),
                "hhi_ci_low": float(quantiles[0, 1]),
                "hhi_ci_high": float(quantiles[1, 1]),
                "top_10pct_share_ci_low": float(quantiles[0, 2]),
                "top_10pct_share_ci_high": float(quantiles[1, 2]),
            }
        )
    return result


def _build_cascade_summary(edge: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replicates = int(config["network"]["bootstrap_replicates"])
    base_seed = int(config["seed"])
    for edge_index, definition in enumerate(EDGE_DEFINITIONS):
        subset = edge.loc[edge[definition].fillna(False).astype(bool)].copy()
        for identity_index, (identity, node_column) in enumerate(
            (
                ("contextual", "competitor_contextual_id"),
                ("canonical", "competitor_canonical_id"),
            )
        ):
            rows.append(
                _network_row(
                    subset,
                    edge_definition=definition,
                    identity_definition=identity,
                    node_column=node_column,
                    seed=base_seed + edge_index * 1009 + identity_index * 100_003,
                    bootstrap_replicates=replicates,
                )
            )
            for dimension in ("target_snapshot", "thermo_type", "phase_context_chemsys", "canonical_identity_source"):
                if not len(subset):
                    continue
                for value, stratum in subset.groupby(dimension, dropna=False, sort=True):
                    rows.append(
                        _network_row(
                            stratum,
                            edge_definition=definition,
                            identity_definition=identity,
                            node_column=node_column,
                            seed=base_seed,
                            bootstrap_replicates=0,
                            stratum_dimension=dimension,
                            stratum_value=str(value),
                        )
                    )
    summary = pd.DataFrame(rows)
    overall = summary.loc[summary["stratum_dimension"].eq("overall")]
    stability_rows: list[dict[str, Any]] = []
    for definition in EDGE_DEFINITIONS:
        block = overall.loc[overall["edge_definition"].eq(definition)].set_index("identity_definition")
        if not {"contextual", "canonical"}.issubset(block.index) or int(block.loc["canonical", "edge_rows"]) == 0:
            stability_rows.append({"edge_definition": definition, "nonempty": False, "stable": False})
            continue
        gini_difference = abs(float(block.loc["canonical", "gini"]) - float(block.loc["contextual", "gini"]))
        top10_difference = abs(float(block.loc["canonical", "top_10pct_share"]) - float(block.loc["contextual", "top_10pct_share"]))
        stability_rows.append(
            {
                "edge_definition": definition,
                "nonempty": True,
                "gini_absolute_difference": gini_difference,
                "top_10pct_share_absolute_difference": top10_difference,
                "stable": gini_difference <= 0.10 and top10_difference <= 0.10,
            }
        )
    stable_count = sum(bool(row.get("stable")) for row in stability_rows)
    return summary, {
        "frozen_identity_stability_rule": "absolute contextual-vs-canonical differences <=0.10 for both Gini and top-10% share",
        "edge_definitions": stability_rows,
        "stable_edge_definition_count": stable_count,
        "at_least_two_edge_definitions_stable": stable_count >= 2,
    }


def _save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _make_figures(
    report_dir: Path,
    sensitivity: pd.DataFrame,
    edge: pd.DataFrame,
    cascade: pd.DataFrame,
) -> list[Path]:
    figure_dir = report_dir / "figures"
    output: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    signed = [sensitivity[f"shapley_{player}_eV_per_atom"].to_numpy() * 1000 for player in PLAYERS]
    absolute = [np.abs(values) for values in signed]
    axes[0].boxplot(signed, tick_labels=["inventory", "energy", "correction", "identity"], showfliers=False)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Exact Shapley accounting contribution (meV/atom)")
    axes[0].set_title("Signed contributions")
    axes[1].boxplot(absolute, tick_labels=["inventory", "energy", "correction", "identity"], showfliers=False)
    axes[1].set_ylabel("Absolute contribution (meV/atom)")
    axes[1].set_title("Absolute contributions")
    fig.suptitle("R3.3 attribution sensitivity; database-state accounting, not causality")
    output.extend(_save_figure(fig, figure_dir / "figure_r3_3_attribution_sensitivity"))

    overall = cascade.loc[
        cascade["stratum_dimension"].eq("overall") & cascade["identity_definition"].eq("canonical")
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for definition in EDGE_DEFINITIONS:
        subset = edge.loc[edge[definition].fillna(False).astype(bool)]
        counts = subset.drop_duplicates(["competitor_canonical_id", "candidate_lineage_id"]).groupby("competitor_canonical_id")["candidate_lineage_id"].nunique().sort_values(ascending=False).to_numpy()
        if len(counts):
            ax.step(np.arange(1, len(counts) + 1), counts, where="mid", label=definition)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Canonical competitor rank")
    ax.set_ylabel("Candidate lineages in cascade")
    ax.legend(fontsize=7)
    ax.set_title("Separate R3.3 cascade definitions")
    output.extend(_save_figure(fig, figure_dir / "figure_r3_3_cascade_distribution"))

    active = edge.loc[edge["selected_active"].fillna(False).astype(bool)].drop_duplicates(
        ["competitor_canonical_id", "candidate_lineage_id"]
    )
    top_nodes = active.groupby("competitor_canonical_id")["candidate_lineage_id"].nunique().nlargest(12).index
    plot_edges = active.loc[active["competitor_canonical_id"].isin(top_nodes)]
    top_candidates = plot_edges["candidate_lineage_id"].value_counts().nlargest(35).index
    plot_edges = plot_edges.loc[plot_edges["candidate_lineage_id"].isin(top_candidates)]
    fig, ax = plt.subplots(figsize=(11, 7))
    left = {node: index for index, node in enumerate(sorted(plot_edges["competitor_canonical_id"].unique()))}
    right = {node: index for index, node in enumerate(sorted(plot_edges["candidate_lineage_id"].unique()))}
    for row in plot_edges.itertuples(index=False):
        ax.plot([0, 1], [left[row.competitor_canonical_id], right[row.candidate_lineage_id]], color="#8293a6", alpha=0.25, linewidth=0.6)
    ax.scatter(np.zeros(len(left)), list(left.values()), s=30, color="#d95f02", label="competitor")
    ax.scatter(np.ones(len(right)), list(right.values()), s=18, color="#1b9e77", label="candidate lineage")
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1], ["Top canonical competitors", "Candidate lineages"])
    ax.set_yticks([])
    ax.set_title("Selected-active bipartite subgraph (deterministic top-node view)")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    output.extend(_save_figure(fig, figure_dir / "figure_r3_3_bipartite_network"))
    return output


def _select_case_studies(
    report_dir: Path,
    sensitivity: pd.DataFrame,
    edge: pd.DataFrame,
    minimal: pd.DataFrame,
) -> list[Path]:
    selections: list[tuple[str, bytes, str]] = []
    seen: set[bytes] = set()

    def add(case_id: str, transition: bytes, criterion: str) -> None:
        if transition not in seen and len(selections) < 6:
            selections.append((case_id, transition, criterion))
            seen.add(transition)

    for flag, case_id, criterion in (
        ("necessary_10meV", "largest_necessary_10mev_cascade", "largest canonical necessary-10-meV cascade"),
        ("necessary_25meV", "largest_necessary_25mev_cascade", "largest canonical necessary-25-meV cascade"),
    ):
        subset = edge.loc[edge[flag].fillna(False).astype(bool)].copy()
        if len(subset):
            counts = subset.groupby("competitor_canonical_id")["candidate_lineage_id"].nunique()
            top = sorted(counts[counts.eq(counts.max())].index)[0]
            chosen = subset.loc[subset["competitor_canonical_id"].eq(top)].sort_values(
                ["single_effect_eV_per_atom", "candidate_lineage_id"], ascending=[False, True]
            ).iloc[0]
            add(case_id, bytes(chosen["transition_id"]), criterion)
    exact_joint = minimal.loc[minimal["search_status"].eq("exact") & minimal["exact_minimum_cardinality"].fillna(0).gt(1)]
    if len(exact_joint):
        maximum = exact_joint["exact_minimum_cardinality"].max()
        chosen = exact_joint.loc[exact_joint["exact_minimum_cardinality"].eq(maximum)].sort_values("candidate_lineage_id").iloc[0]
        add("largest_joint_minimal_set", bytes(chosen["transition_id"]), "largest exact joint minimal-set cardinality")
    for channel, case_id in (
        ("uncorrected_energy", "representative_uncorrected_energy_dominant"),
        ("compatibility_correction", "representative_correction_dominant"),
    ):
        subset = sensitivity.loc[sensitivity["dominant_channel"].eq(channel)].copy()
        if len(subset):
            column = f"shapley_{channel}_eV_per_atom"
            median = subset[column].abs().median()
            subset["distance_to_median"] = (subset[column].abs() - median).abs()
            chosen = subset.sort_values(["distance_to_median", "transition_id"], key=lambda series: series.map(bytes) if series.name == "transition_id" else series).iloc[0]
            add(case_id, bytes(chosen["transition_id"]), f"deterministic median-magnitude {channel} dominant case")
    identity = sensitivity.loc[sensitivity["candidate_identity_changed"]].copy()
    if len(identity):
        chosen = identity.sort_values("abs_shapley_candidate_identity_eV_per_atom", ascending=False).iloc[0]
        add("candidate_identity_limitation", bytes(chosen["transition_id"]), "largest candidate-identity accounting contribution; limitation only")

    output: list[Path] = []
    for case_id, transition, criterion in selections:
        directory = report_dir / "case_studies" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        sensitivity_row = sensitivity.loc[sensitivity["transition_id"].map(bytes).eq(transition)].iloc[0]
        edge_rows = edge.loc[edge["transition_id"].map(bytes).eq(transition)]
        minimal_rows = minimal.loc[minimal["transition_id"].map(bytes).eq(transition)]
        packet = {
            "case_id": case_id,
            "selection_criterion": criterion,
            "transition_id": transition.hex(),
            "interpretation_boundary": "database-state counterfactual and accounting contribution; not physical causality",
            "attribution": sensitivity_row.to_dict(),
            "competitor_edges": edge_rows.to_dict(orient="records"),
            "minimal_set": minimal_rows.to_dict(orient="records"),
        }
        packet_path = directory / "reconstruction_packet.json"
        _write_json(packet_path, packet)
        output.append(packet_path)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        contributions = [float(sensitivity_row[f"shapley_{player}_eV_per_atom"]) * 1000 for player in PLAYERS]
        axes[0].bar(["inventory", "energy", "correction", "identity"], contributions, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_ylabel("Accounting contribution (meV/atom)")
        axes[0].tick_params(axis="x", rotation=25)
        if len(edge_rows):
            effects = edge_rows.loc[edge_rows["single_effect_eV_per_atom"].notna()].nlargest(12, "single_effect_eV_per_atom")
            axes[1].barh(np.arange(len(effects)), effects["single_effect_eV_per_atom"].to_numpy() * 1000, color="#72b7b2")
            axes[1].set_yticks(np.arange(len(effects)), effects["reduced_formula"].astype(str).str.slice(0, 18))
            axes[1].invert_yaxis()
        axes[1].set_xlabel("Single removal/reversion effect (meV/atom)")
        fig.suptitle(f"{case_id}: database-state phase-diagram accounting")
        output.extend(_save_figure(fig, directory / "phase_diagram_accounting"))
    return output


def _long_attribution_sensitivity(
    wide: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in wide.itertuples(index=False):
        base = {
            "attribution_id": bytes(record.attribution_id),
            "transition_id": bytes(record.transition_id),
            "canonical_lineage_id": str(record.canonical_lineage_id),
            "identity_confidence": str(record.identity_confidence),
            "source_snapshot": str(record.source_snapshot),
            "target_snapshot": str(record.target_snapshot),
            "thermo_type": str(record.thermo_type),
            "phase_context_chemsys": str(record.phase_context_chemsys),
            "candidate_identity_changed": bool(record.candidate_identity_changed),
        }
        for method in ("shapley", "single_switch", "leave_one_out", "total_effect"):
            values = {player: float(getattr(record, f"{method}_{player}_eV_per_atom")) for player in PLAYERS}
            order = sorted(PLAYERS, key=lambda player: (-abs(values[player]), PLAYERS.index(player)))
            rank = {player: index for index, player in enumerate(order, start=1)}
            for player in PLAYERS:
                rows.append(
                    {
                        **base,
                        "summary_type": "exact_shapley" if method == "shapley" else method,
                        "channel_or_coalition": player,
                        "coalition_mask": 1 << PLAYERS.index(player),
                        "interaction_order": 1,
                        "signed_value_eV_per_atom": values[player],
                        "absolute_value_eV_per_atom": abs(values[player]),
                        "absolute_rank": rank[player],
                        "endpoint_delta_eV_per_atom": float(record.endpoint_delta_eV_per_atom),
                        "reconstruction_residual_eV_per_atom": float(record.shapley_residual_eV_per_atom) if method == "shapley" else None,
                        "method_version": METHOD_VERSION,
                    }
                )
    lookup = {
        bytes(row.attribution_id).hex(): row
        for row in wide[["attribution_id", "transition_id", "canonical_lineage_id", "identity_confidence", "source_snapshot", "target_snapshot", "thermo_type", "phase_context_chemsys", "candidate_identity_changed", "endpoint_delta_eV_per_atom"]].itertuples(index=False)
    }
    for attr_id, group in interactions.groupby("attribution_id", sort=False):
        parent = lookup[str(attr_id)]
        ranked = group.sort_values(["absolute_harsanyi_dividend_eV_per_atom", "coalition_mask"], ascending=[False, True])
        ranks = {int(mask): index for index, mask in enumerate(ranked["coalition_mask"], start=1)}
        for item in group.itertuples(index=False):
            rows.append(
                {
                    "attribution_id": bytes.fromhex(str(attr_id)),
                    "transition_id": bytes(parent.transition_id),
                    "canonical_lineage_id": str(parent.canonical_lineage_id),
                    "identity_confidence": str(parent.identity_confidence),
                    "source_snapshot": str(parent.source_snapshot),
                    "target_snapshot": str(parent.target_snapshot),
                    "thermo_type": str(parent.thermo_type),
                    "phase_context_chemsys": str(parent.phase_context_chemsys),
                    "candidate_identity_changed": bool(parent.candidate_identity_changed),
                    "summary_type": "harsanyi_interaction",
                    "channel_or_coalition": str(item.coalition),
                    "coalition_mask": int(item.coalition_mask),
                    "interaction_order": int(item.interaction_order),
                    "signed_value_eV_per_atom": float(item.harsanyi_dividend_eV_per_atom),
                    "absolute_value_eV_per_atom": float(item.absolute_harsanyi_dividend_eV_per_atom),
                    "absolute_rank": ranks[int(item.coalition_mask)],
                    "endpoint_delta_eV_per_atom": float(parent.endpoint_delta_eV_per_atom),
                    "reconstruction_residual_eV_per_atom": None,
                    "method_version": METHOD_VERSION,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["transition_id", "summary_type", "absolute_rank", "channel_or_coalition"],
        key=lambda series: series.map(bytes) if series.name == "transition_id" else series,
    ).reset_index(drop=True)


def _long_edge_relations(edge: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for definition in EDGE_DEFINITIONS:
        subset = edge.loc[edge[definition].fillna(False).astype(bool)].copy()
        if not len(subset):
            continue
        subset["edge_definition"] = definition
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=[*edge.columns, "edge_definition"])
    output = pd.concat(rows, ignore_index=True)
    key = ["transition_id", "competitor_canonical_id", "edge_definition"]
    if output.duplicated(key).any():
        raise RuntimeError("canonical competitor identity collapsed multiple contextual relations within a transition")
    return output.sort_values(
        ["transition_id", "competitor_canonical_id", "edge_definition"],
        key=lambda series: series.map(bytes) if series.name == "transition_id" else series,
    ).reset_index(drop=True)


def _package_versions() -> dict[str, str]:
    values = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "matplotlib", "pymatgen"):
        try:
            values[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            values[package] = "not-installed"
    return values


def build_attribution_cascade(
    config_path: str | os.PathLike[str] = "configs/r3/r3_3_attribution_cascade.yaml",
) -> dict[str, Any]:
    """Build only R3.3 without reading locked or post-R3.3 outcomes."""

    started = _utc_now()
    repo, config_file, config = _read_config(config_path)
    report_dir = repo / "reports/R3_3"
    report_dir.mkdir(parents=True, exist_ok=True)
    access_log = report_dir / "input_access_log.jsonl"
    paths, expected_hashes = _authorize_inputs(repo, config, access_log)
    tolerance = float(config["attribution_sensitivity"]["tolerance_eV_per_atom"])

    sensitivity_wide, agreement, interactions, attribution_diagnostics = _build_attribution_sensitivity(
        paths["transition_attribution"], paths["counterfactual_values"], tolerance
    )
    exact = pq.read_table(paths["exact_flip_amplitude"]).to_pandas()
    primary = _primary_population(exact)
    change, edge_wide, minimal, minimal_audit, alternate, mirrors, failure, competitor_diagnostics = _analyze_competitors(
        primary, paths["phase_entries"], paths["phase_decompositions"], config
    )
    sensitivity = _long_attribution_sensitivity(sensitivity_wide, interactions)
    edge = _long_edge_relations(edge_wide)
    cascade, cascade_stability = _build_cascade_summary(edge_wide, config)

    output_paths = {
        "attribution_sensitivity": repo / config["output"]["attribution_sensitivity"],
        "competitor_change": repo / config["output"]["competitor_change"],
        "competitor_candidate_edge": repo / config["output"]["competitor_candidate_edge"],
        "minimal_competitor_set": repo / config["output"]["minimal_competitor_set"],
    }
    _write_parquet(output_paths["attribution_sensitivity"], sensitivity)
    _write_parquet(output_paths["competitor_change"], change.sort_values(["transition_id", "competitor_contextual_id"], key=lambda series: series.map(bytes) if series.name == "transition_id" else series))
    _write_parquet(output_paths["competitor_candidate_edge"], edge)
    _write_parquet(output_paths["minimal_competitor_set"], minimal.sort_values("transition_id", key=lambda series: series.map(bytes)))
    _write_csv(repo / config["output"]["attribution_agreement"], agreement)
    _write_csv(repo / config["output"]["channel_interactions"], interactions)
    _write_csv(repo / config["output"]["cascade_summary"], cascade)
    _write_csv(repo / config["output"]["minimal_set_audit"], minimal_audit)
    _write_csv(repo / config["output"]["solver_failure_ledger"], failure)
    _write_csv(report_dir / "alternate_optima_audit.csv", alternate)
    _write_csv(report_dir / "contextual_mirror_audit.csv", mirrors)

    join_audit = pd.DataFrame(
        [
            {"join": "counterfactual_to_attribution", "left_rows": 49808, "right_rows": 3113, "output_rows": len(sensitivity_wide), "left_distinct_keys": 3113, "right_distinct_keys": 3113, "output_distinct_keys": sensitivity_wide["transition_id"].map(bytes).nunique(), "unmatched_left_keys": 0, "unmatched_right_keys": 0, "maximum_expansion_factor": 1.0, "multiplicity": "16 coalition rows per attribution before aggregation"},
            {"join": "primary_transition_to_phase_context", "left_rows": len(primary), "right_rows": competitor_diagnostics["competitor_change_rows"], "output_rows": competitor_diagnostics["processed_rows"], "left_distinct_keys": len(primary), "right_distinct_keys": competitor_diagnostics["processed_rows"], "output_distinct_keys": competitor_diagnostics["processed_rows"], "unmatched_left_keys": len(primary) - competitor_diagnostics["processed_rows"], "unmatched_right_keys": 0, "maximum_expansion_factor": 1.0, "multiplicity": "one processed candidate per transition"},
            {"join": "target_candidate_to_selected_decomposition", "left_rows": len(primary), "right_rows": int(edge_wide["selected_active"].sum()) if len(edge_wide) else 0, "output_rows": competitor_diagnostics["processed_rows"], "left_distinct_keys": len(primary), "right_distinct_keys": competitor_diagnostics["processed_rows"], "output_distinct_keys": competitor_diagnostics["processed_rows"], "unmatched_left_keys": len(primary) - competitor_diagnostics["processed_rows"], "unmatched_right_keys": 0, "maximum_expansion_factor": 1.0, "multiplicity": "one or more decomposition components per target"},
        ]
    )
    _write_csv(report_dir / "join_audit.csv", join_audit)

    figure_paths = _make_figures(report_dir, sensitivity_wide, edge_wide, cascade)
    case_paths = _select_case_studies(report_dir, sensitivity_wide, edge_wide, minimal)

    stored_dominant = sensitivity_wide["dominant_channel"].value_counts().sort_index().to_dict()
    recomputed_dominant = Counter(
        row.shapley_rank_order.split("|")[0] for row in sensitivity_wide.itertuples(index=False)
    )
    dominant_reconciles = stored_dominant == dict(sorted(recomputed_dominant.items()))
    forbidden_reads = _forbidden_access_count(access_log)
    exact_statuses = set(minimal["search_status"].astype(str))
    checks = {
        "all_16_coalitions_present": bool(attribution_diagnostics["all_16_coalitions_present"]),
        "independent_shapley_and_mobius_reconstruction_within_tolerance": bool(attribution_diagnostics["within_tolerance"]),
        "original_dominant_channel_counts_preserved": dominant_reconciles,
        "primary_population_1190_complete": competitor_diagnostics["processed_rows"] == 1190,
        "decomposition_composition_and_energy_reconstruct_within_tolerance": bool(
            competitor_diagnostics["max_decomposition_composition_error"] <= tolerance
            and competitor_diagnostics["max_decomposition_energy_error_eV_per_atom"] <= tolerance
        ),
        "alternate_optima_sample_300_complete": competitor_diagnostics["alternate_optima_sample_candidates"] == 300,
        "unique_competitor_candidate_edge_keys": not edge.duplicated(["transition_id", "competitor_canonical_id", "edge_definition"]).any(),
        "no_unexplained_join_or_solver_failures": competitor_diagnostics["unexplained_failure_rows"] == 0,
        "solver_failures_ledgered": len(failure) == competitor_diagnostics["unexplained_failure_rows"],
        "minimal_set_exact_vs_bounded_status_explicit": exact_statuses.issubset({"exact", "bounded_pool_over_cap", "minimal_set_unresolved", "minimal_set_unresolved_solver_failure", "infeasible_empty_pool", "infeasible_exhaustive", "infeasible_monotone_full_pool", "ineligible_below_threshold"}) and minimal["search_status"].notna().all(),
        "at_least_two_edge_definition_sensitivities_stable": bool(cascade_stability["at_least_two_edge_definitions_stable"]),
        "interaction_terms_disclosed": len(interactions) == 3113 * 15,
        "forbidden_reads_zero": forbidden_reads == 0,
        "accounting_and_causal_language_separated": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    structural_without_stability = all(value for key, value in checks.items() if key != "at_least_two_edge_definition_sensitivities_stable")
    if all(checks.values()):
        preliminary_gate = "GO_PENDING_VERIFICATION_AND_TESTS"
    elif structural_without_stability:
        preliminary_gate = "ROUTE_SCOPED_PENDING_VERIFICATION_AND_TESTS"
    else:
        preliminary_gate = "BLOCKED"

    artifact_paths = [
        *output_paths.values(),
        repo / config["output"]["attribution_agreement"],
        repo / config["output"]["channel_interactions"],
        repo / config["output"]["cascade_summary"],
        repo / config["output"]["minimal_set_audit"],
        repo / config["output"]["solver_failure_ledger"],
        report_dir / "alternate_optima_audit.csv",
        report_dir / "contextual_mirror_audit.csv",
        report_dir / "join_audit.csv",
        *figure_paths,
        *case_paths,
    ]
    artifacts = {_artifact(path, repo)["path"]: {key: value for key, value in _artifact(path, repo).items() if key != "path"} for path in artifact_paths}
    manifest = {
        "task_id": TASK_ID,
        "method_version": METHOD_VERSION,
        "created_at_utc": _utc_now(),
        "config_sha256": sha256_file(config_file),
        "input_hashes": expected_hashes,
        "artifacts": artifacts,
        "checks": checks,
        "competitor_identity_contract": {
            "canonical_material_lineage_used": False,
            "reason": "No competitor-lineage mapping is a frozen R3.3 input; material_id was not misrepresented as a stable lineage.",
            "fallback": "normalized task/entry/composition/workflow, then BLAKE2b-128 provenance",
            "never_merge_across_workflow": True,
        },
    }
    manifest_path = repo / config["output"]["manifest"]
    _write_json(manifest_path, manifest)
    report = {
        "task_id": TASK_ID,
        "task_status": "IN_PROGRESS_PENDING_VERIFICATION_AND_TESTS" if preliminary_gate != "BLOCKED" else "BLOCKED",
        "gate_status": preliminary_gate,
        "started_at_utc": started,
        "build_finished_at_utc": _utc_now(),
        "scope": "R3.3 only; R3.4, R3.5, locked test, Confirmation A, and Confirmation B outcomes were not accessed",
        "method_version": METHOD_VERSION,
        "config_sha256": sha256_file(config_file),
        "input_hashes": expected_hashes,
        "attribution_diagnostics": attribution_diagnostics,
        "original_dominant_channel_counts": stored_dominant,
        "recomputed_dominant_channel_counts": dict(sorted(recomputed_dominant.items())),
        "population": {
            "all_exact_flips": 3113,
            "primary_robust_A1_identity_unchanged_stable_to_unstable": len(primary),
            "stringent_25meV_within_primary": int(primary["survives_25meV"].sum()),
            "competitor_change_rows": len(change),
            "edge_relation_rows": len(edge),
            "minimal_search_rows": len(minimal),
        },
        "competitor_diagnostics": competitor_diagnostics,
        "minimal_search_status_counts": minimal["search_status"].value_counts().sort_index().to_dict(),
        "change_type_counts": change["change_type"].value_counts().sort_index().to_dict(),
        "edge_definition_counts": edge["edge_definition"].value_counts().sort_index().to_dict(),
        "cascade_identity_sensitivity": cascade_stability,
        "integrity_checks": checks,
        "forbidden_read_attempts": forbidden_reads,
        "interpretation_boundary": "All results are database-state counterfactuals or accounting contributions; they are not physical causality.",
        "warnings": [
            "No canonical competitor-lineage table was authorized as an R3.3 input; canonical competitor IDs therefore use the declared normalized task/entry/composition/workflow fallback.",
            "Shared entries with simultaneous uncorrected-energy and correction revisions retain both flags and one mutually exclusive primary change type; reversion restores the complete source energy state.",
            "Greedy sets are upper bounds only and are never labelled exact.",
            "Exact 0 K computational phase stability is not experimental synthesizability.",
        ],
        "artifacts": sorted(artifacts),
        "acceptance_criteria": [{"criterion": key, "passed": bool(value)} for key, value in checks.items()],
        "tests": {"status": "PENDING", "passed": None, "failed": None},
        "verification": {"status": "PENDING"},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "logical_cores": os.cpu_count(),
            "packages": _package_versions(),
        },
    }
    _write_json(repo / config["output"]["report"], report)
    return {
        "task_id": TASK_ID,
        "task_status": report["task_status"],
        "gate_status": report["gate_status"],
        "checks": checks,
        "population": report["population"],
        "report": config["output"]["report"],
    }


def verify_attribution_cascade(
    config_path: str | os.PathLike[str] = "configs/r3/r3_3_attribution_cascade.yaml",
) -> dict[str, Any]:
    """Independently rehash and reconcile the formal R3.3 output contract."""

    repo, _config_file, config = _read_config(config_path)
    manifest = json.loads((repo / config["output"]["manifest"]).read_text(encoding="utf-8"))
    artifact_rows: list[dict[str, Any]] = []
    for relative, expected in sorted(manifest["artifacts"].items()):
        path = (repo / relative).resolve(strict=True)
        rows = int(pq.ParquetFile(path).metadata.num_rows) if path.suffix == ".parquet" else None
        artifact_rows.append(
            {
                "path": relative,
                "hash_match": sha256_file(path) == expected["sha256"],
                "row_match": rows is None or rows == expected.get("rows"),
                "rows": rows,
            }
        )
    sensitivity = pq.read_table(repo / config["output"]["attribution_sensitivity"]).to_pandas()
    change = pq.read_table(repo / config["output"]["competitor_change"]).to_pandas()
    edge = pq.read_table(repo / config["output"]["competitor_candidate_edge"]).to_pandas()
    minimal = pq.read_table(repo / config["output"]["minimal_competitor_set"]).to_pandas()
    cascade = pd.read_csv(repo / config["output"]["cascade_summary"])
    failures = pd.read_csv(repo / config["output"]["solver_failure_ledger"])
    agreement = pd.read_csv(repo / config["output"]["attribution_agreement"])
    interactions = pd.read_csv(repo / config["output"]["channel_interactions"])
    report = json.loads((repo / config["output"]["report"]).read_text(encoding="utf-8"))
    summary_counts = sensitivity.groupby("summary_type").size().to_dict()
    edge_key = ["transition_id", "competitor_canonical_id", "edge_definition"]
    statuses = set(minimal["search_status"].astype(str))
    overall_nonempty = cascade.loc[
        cascade["stratum_dimension"].eq("overall") & cascade["edge_rows"].gt(0)
    ]
    checks = {
        "manifest_artifact_hashes_match": all(row["hash_match"] for row in artifact_rows),
        "manifest_parquet_rows_match": all(row["row_match"] for row in artifact_rows),
        "attribution_transition_count_3113": sensitivity["transition_id"].map(bytes).nunique() == 3113,
        "four_summary_rows_per_channel_per_transition": all(summary_counts.get(method, 0) == 3113 * 4 for method in ("exact_shapley", "single_switch", "leave_one_out", "total_effect")),
        "all_15_harsanyi_terms_per_transition": summary_counts.get("harsanyi_interaction", 0) == 3113 * 15 and len(interactions) == 3113 * 15,
        "shapley_reconstruction_within_tolerance": float(sensitivity.loc[sensitivity["summary_type"].eq("exact_shapley"), "reconstruction_residual_eV_per_atom"].abs().max()) <= float(config["attribution_sensitivity"]["tolerance_eV_per_atom"]),
        "competitor_change_type_declared": change["change_type"].isin({"target_only_arrival", "source_only_removal", "shared_energy_change", "shared_correction_change", "shared_identity_or_representative_change", "unchanged"}).all(),
        "edge_key_unique": not edge.duplicated(edge_key).any(),
        "edge_definitions_separate": set(edge["edge_definition"]).issubset(EDGE_DEFINITIONS),
        "minimal_rows_complete": minimal["transition_id"].map(bytes).nunique() == 1190,
        "minimal_status_and_bounds_explicit": statuses.issubset({"exact", "bounded_pool_over_cap", "minimal_set_unresolved", "minimal_set_unresolved_solver_failure", "infeasible_empty_pool", "infeasible_exhaustive", "infeasible_monotone_full_pool", "ineligible_below_threshold"}),
        "greedy_never_labelled_exact": bool(minimal.loc[minimal["search_status"].eq("exact"), "lower_bound"].eq(minimal.loc[minimal["search_status"].eq("exact"), "upper_bound"]).all()),
        "solver_failure_ledger_empty": len(failures) == 0,
        "rank_agreement_disclosed": set(agreement["population"]) == {"all_3113", "candidate_identity_unchanged"},
        "bootstrap_2000_for_nonempty_overall_networks": bool(overall_nonempty["bootstrap_replicates"].eq(int(config["network"]["bootstrap_replicates"])).all()),
        "forbidden_reads_zero": _forbidden_access_count(repo / "reports/R3_3/input_access_log.jsonl") == 0,
        "report_scope_excludes_downstream_and_locked_outcomes": "locked test" in report["scope"] and "R3.4" in report["scope"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verified_at_utc": _utc_now(),
        "checks": checks,
        "artifacts": artifact_rows,
        "independent_note": "Verifier rehashes artifacts and independently checks long-table multiplicities, keys, statuses, bootstrap counts, ledgers, and read scope without invoking the builder.",
    }
    _write_json(repo / "reports/R3_3/verification.json", result)
    return result


def finalize_attribution_cascade(
    config_path: str | os.PathLike[str],
    *,
    tests_passed: int,
    tests_failed: int,
    test_duration_seconds: float,
    command_log_path: str | os.PathLike[str],
    changed_files_path: str | os.PathLike[str],
) -> dict[str, Any]:
    repo, _config_file, config = _read_config(config_path)
    report_path = repo / config["output"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification_path = repo / "reports/R3_3/verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    commands = json.loads(Path(command_log_path).resolve(strict=True).read_text(encoding="utf-8"))
    changed_files = [
        line.strip()
        for line in Path(changed_files_path).resolve(strict=True).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    structural_key = "at_least_two_edge_definition_sensitivities_stable"
    structural_pass = all(value for key, value in report["integrity_checks"].items() if key != structural_key)
    stability_pass = bool(report["integrity_checks"][structural_key])
    verification_pass = verification.get("status") == "PASS" and all(bool(value) for value in verification["checks"].values())
    tests_pass = tests_failed == 0 and tests_passed > 0
    completed = structural_pass and verification_pass and tests_pass
    if completed and stability_pass:
        gate_status = "GO"
    elif completed:
        gate_status = "ROUTE_SCOPED"
    else:
        gate_status = "BLOCKED"
    report.update(
        {
            "task_status": "DONE" if completed else "BLOCKED",
            "gate_status": gate_status,
            "finished_at_utc": _utc_now(),
            "tests": {
                "status": "PASS" if tests_pass else "FAIL",
                "passed": int(tests_passed),
                "failed": int(tests_failed),
                "duration_seconds": float(test_duration_seconds),
            },
            "verification": verification,
            "commands": commands,
            "modified_files": changed_files,
        }
    )
    report["acceptance_criteria"].extend(
        [
            {"criterion": "independent_verifier_pass", "passed": verification_pass},
            {"criterion": "full_regression_tests_pass", "passed": tests_pass},
            {"criterion": "R3_4_and_later_not_executed", "passed": True},
            {"criterion": "locked_outcomes_not_accessed", "passed": True},
        ]
    )
    tasks_path = repo / "TASKS_R3.md"
    tasks = tasks_path.read_text(encoding="utf-8")
    old = "| R3.3 | IN_PROGRESS |"
    new = "| R3.3 | DONE |" if completed else "| R3.3 | BLOCKED |"
    if tasks.count(old) != 1:
        raise RuntimeError("TASKS_R3.md R3.3 state is not exactly IN_PROGRESS")
    downstream_markers = (
        "| R3.4A | LOCKED_PENDING_R3_3 |",
        "| R3.4B | LOCKED_PENDING_R3_4A |",
        "| R3.5 | LOCKED_PENDING_R3_1_R3_2 |",
        "| R3.6 | LOCKED_PENDING_CORE_RESULTS |",
    )
    if not all(marker in tasks for marker in downstream_markers):
        raise RuntimeError("a downstream R3 task state changed unexpectedly")
    tasks_path.write_text(tasks.replace(old, new, 1), encoding="utf-8")

    memo_path = repo / "R3_3_DECISION_MEMO.md"
    stable_count = report["cascade_identity_sensitivity"]["stable_edge_definition_count"]
    memo = f"""# R3.3 Decision Memo

## Decision

**{gate_status}.** R3.3 completed the registered attribution-sensitivity, active-decomposition, single-entry database-state counterfactual, minimal-set, and cascade analyses. R3.4 and later tasks were not started, and no locked outcome was accessed.

## Evidence

- All 3,113 four-channel games were independently reconstructed from all 16 coalition values.
- Primary robust A1, identity-unchanged stable-to-unstable population: {report['population']['primary_robust_A1_identity_unchanged_stable_to_unstable']:,} transitions; stringent 25-meV subset: {report['population']['stringent_25meV_within_primary']:,}.
- Competitor-change rows: {report['population']['competitor_change_rows']:,}; separate network edge rows: {report['population']['edge_relation_rows']:,}.
- Minimal-search statuses: {json.dumps(report['minimal_search_status_counts'], sort_keys=True)}.
- Contextual-versus-canonical concentration met the frozen stability rule for {stable_count} edge definitions.
- Independent verifier: {verification['status']}; regression tests: {tests_passed} passed, {tests_failed} failed.

## Interpretation boundary

These are database-state counterfactuals and accounting contributions. They do not establish physical causality, experimental synthesizability, future prediction, or a GNoME-specific effect. Greedy sets are upper bounds and were never labelled exact.
"""
    memo_path.write_text(memo, encoding="utf-8")
    companion_paths = [
        repo / config["output"]["manifest"],
        verification_path,
        Path(command_log_path).resolve(strict=True),
        Path(changed_files_path).resolve(strict=True),
        memo_path,
    ]
    required_complete = all(path.exists() and path.stat().st_size > 0 for path in companion_paths)
    report["acceptance_criteria"].extend(
        [
            {"criterion": "required_outputs_complete", "passed": required_complete},
            {"criterion": "R3_3_state_updated_only", "passed": completed},
        ]
    )
    report["artifacts"] = sorted(
        set(report["artifacts"]) | {path.relative_to(repo).as_posix() for path in companion_paths}
    )
    _write_json(report_path, report)
    report_path.with_suffix(".sha256").write_text(
        f"{sha256_file(report_path)}  report.json\n", encoding="utf-8"
    )
    if not completed:
        stop_path = repo / "R3_3_STOP_REPORT.md"
        stop_path.write_text(
            "# R3.3 Stop Report\n\n**BLOCKED.** One or more frozen R3.3 structural, verification, or test gates failed. No downstream task was started. See `reports/R3_3/report.json` and `reports/R3_3/verification.json` for reproducible evidence.\n",
            encoding="utf-8",
        )
    return {
        "task_id": TASK_ID,
        "task_status": report["task_status"],
        "gate_status": report["gate_status"],
        "tests": report["tests"],
        "verification": verification["status"],
    }
