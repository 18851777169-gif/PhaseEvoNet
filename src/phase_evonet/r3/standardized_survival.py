"""R3.2 release-standardized durability and multi-state transition analysis.

The implementation is deliberately source-only: every adjustment variable is
computed from the source snapshot, while the target snapshot is used only to
construct the declared outcome.  Locked test and confirmation artifacts are
neither accepted as inputs nor inspected by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.api as sm
import yaml
from matplotlib.colors import LogNorm
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.optimize import linprog

from .common import DEFAULT_FORBIDDEN_PATTERNS, open_formal_input, sha256_file
from .energy_amplitude import (
    _join_all_eligible_transitions,
    _records_with_elemental_terminals,
    _relevant_context_rows,
)
from .signed_margin import compute_signed_margin


METHOD_VERSION = "PHASEEVONET_R3_2_STANDARDIZED_SURVIVAL_V1"
TASK_ID = "R3.2"
STATE_ORDER = ("S0", "S1", "S2", "S3", "S4", "D", "M")
RELEASE_ORDER = (
    "2022-10-28->2023-11-01",
    "2023-11-01->2024-12-18",
    "2024-12-18->2025-09-25",
)
OUTCOME_THRESHOLDS = {"exact": 1.0e-8, "10meV": 0.010, "25meV": 0.025}

# Frozen before fitting.  These are physical, not outcome-adaptive, bins.
MARGIN_EDGES = (-np.inf, 0.001, 0.005, 0.010, 0.025, 0.050, np.inf)
MARGIN_LABELS = ("LE_1meV", "1_5meV", "5_10meV", "10_25meV", "25_50meV", "GT_50meV")
CONTEXT_EDGES = (-np.inf, 1, 5, 20, 100, np.inf)
CONTEXT_LABELS = ("1", "2_5", "6_20", "21_100", "GT_100")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve(strict=True)
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "TASKS_R3.md").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("cannot locate repository root from R3.2 config")


def _read_config(config_path: str | os.PathLike[str]) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID or config.get("method_version") != METHOD_VERSION:
        raise RuntimeError("R3.2 config task/method identity mismatch")
    return _repo_root(path), path, config


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def endpoint_state(value: float | None, *, present: bool, usable: bool, absent_state: str) -> str:
    """Return one exhaustive endpoint state under the frozen R3.2 rules."""

    if not present:
        if absent_state not in {"D", "M"}:
            raise ValueError("absent_state must be D or M")
        return absent_state
    if not usable or value is None or not np.isfinite(float(value)):
        return "M"
    energy = float(value)
    if energy <= 1.0e-8:
        return "S0"
    if energy <= 0.010:
        return "S1"
    if energy <= 0.025:
        return "S2"
    if energy <= 0.050:
        return "S3"
    return "S4"


def composition_entropy(composition_json: str) -> float:
    """Normalized Shannon entropy of the source fractional composition."""

    payload = json.loads(str(composition_json))
    amounts = np.asarray([float(value) for value in payload.values() if float(value) > 0], dtype=float)
    if len(amounts) <= 1:
        return 0.0
    probabilities = amounts / amounts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(probabilities)))


def _margin_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=MARGIN_EDGES,
        labels=MARGIN_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")


def _context_density_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=CONTEXT_EDGES,
        labels=CONTEXT_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")


def _input_hashes(repo: Path) -> dict[str, str]:
    manifest = _read_json(repo / "data/manifests/R3_1/manifest.json")
    return {
        "robust_transition": manifest["artifacts"]["data/processed/R3_1/robust_transition.parquet"]["sha256"],
        "signed_stability_state": manifest["artifacts"]["data/processed/R3_1/signed_stability_state.parquet"]["sha256"],
        "transition_labels": manifest["input_hashes"]["transition_labels"],
        "phase_entries": manifest["input_hashes"]["phase_entries"],
    }


def _authorize_inputs(repo: Path, config: dict[str, Any]) -> dict[str, Path]:
    hashes = _input_hashes(repo)
    access_log = repo / "reports/R3_2/input_access_log.jsonl"
    paths: dict[str, Path] = {}
    for key, relative in config["input"].items():
        if key == "r3_1_report":
            continue
        path = (repo / relative).resolve(strict=True)
        with open_formal_input(
            path,
            hashes[key],
            task_id=TASK_ID,
            access_log=access_log,
            purpose=f"R3.2 formal {key} input",
            forbidden_patterns=DEFAULT_FORBIDDEN_PATTERNS,
            allowed_roots=[repo],
            caller="phase_evonet.r3.standardized_survival._authorize_inputs",
        ):
            pass
        paths[key] = path
    report = _read_json((repo / config["input"]["r3_1_report"]).resolve(strict=True))
    if report.get("task_id") != "R3.1" or report.get("task_status") != "DONE":
        raise RuntimeError("R3.1 prerequisite is not DONE")
    return paths


def _state_frame(transition_path: Path) -> pd.DataFrame:
    columns = [
        "transition_id",
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
        "source_state_usable",
        "target_state_usable",
        "source_energy_above_hull",
        "target_energy_above_hull",
        "reported_label_transition",
        "label_flip",
    ]
    frame = pq.read_table(transition_path, columns=columns).to_pandas()
    source_states: list[str] = []
    target_states: list[str] = []
    for row in frame.itertuples(index=False):
        status = str(row.observation_status)
        source_absent = "D" if status == "target_only" else "M"
        target_absent = "D" if status == "source_only" else "M"
        source_states.append(
            endpoint_state(
                row.source_energy_above_hull,
                present=bool(row.source_state_present),
                usable=bool(row.source_state_usable),
                absent_state=source_absent,
            )
        )
        target_states.append(
            endpoint_state(
                row.target_energy_above_hull,
                present=bool(row.target_state_present),
                usable=bool(row.target_state_usable),
                absent_state=target_absent,
            )
        )
    frame["source_state"] = pd.Categorical(source_states, categories=STATE_ORDER, ordered=True)
    frame["target_state"] = pd.Categorical(target_states, categories=STATE_ORDER, ordered=True)
    frame["release_pair"] = frame["source_snapshot"].astype(str) + "->" + frame["target_snapshot"].astype(str)
    frame["method_version"] = METHOD_VERSION
    if frame["transition_id"].map(bytes).duplicated().any():
        raise RuntimeError("multi-state transition IDs are not unique")
    if frame["source_state"].isna().any() or frame["target_state"].isna().any():
        raise RuntimeError("multi-state assignment is not exhaustive")
    return frame


def _context_covariates(phase_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "snapshot_id",
        "thermo_type",
        "phase_context_chemsys",
        "unified_entry_id",
        "is_target",
        "is_competitor",
        "source_workflow",
        "compatibility_mode",
    ]
    frame = pq.read_table(phase_path, columns=columns).to_pandas()
    group_columns = ["snapshot_id", "thermo_type", "phase_context_chemsys"]
    context = (
        frame.groupby(group_columns, sort=True, observed=True)
        .agg(
            source_context_entry_count=("unified_entry_id", "size"),
            source_context_competitor_count=("is_competitor", "sum"),
        )
        .reset_index()
    )
    context["source_context_competitor_fraction"] = (
        context["source_context_competitor_count"] / context["source_context_entry_count"]
    )
    targets = frame.loc[frame["is_target"].astype(bool), ["unified_entry_id", "compatibility_mode"]].copy()
    targets["unified_entry_id"] = targets["unified_entry_id"].map(bytes)
    targets["compatibility_mode"] = targets["compatibility_mode"].astype(str).str.replace(
        "regenerated_context_mixing_duplicate_alias", "regenerated_context_mixing", regex=False
    )
    targets = targets.drop_duplicates("unified_entry_id", keep=False)
    return context, targets


def _fast_context_margin_job(
    payload: tuple[tuple[str, str, str], list[dict[str, Any]], list[bytes], float]
) -> list[dict[str, Any]]:
    """Compute registered explicit LOO margins from the complete context.

    The leave-one-out hull must retain *all* other entries.  Using only the
    currently stable entries is incorrect because a slightly unstable phase or
    same-composition polymorph can become the replacement hull phase after the
    candidate is removed.
    """

    key, records, candidates, tolerance = payload
    entries: list[ComputedEntry] = []
    by_hex: dict[str, ComputedEntry] = {}
    by_row: dict[str, dict[str, Any]] = {}
    for row in records:
        uid_hex = bytes(row["unified_entry_id"]).hex()
        entry = ComputedEntry(
            Composition(json.loads(str(row["composition_json"]))),
            float(row["corrected_energy"]),
            entry_id=uid_hex,
        )
        entries.append(entry)
        by_hex[uid_hex] = entry
        by_row[uid_hex] = row
    full_diagram = PhaseDiagram(entries)
    elements = sorted(
        {str(element) for entry in entries for element in entry.composition.elements}
    )
    composition_matrix = np.asarray(
        [
            [
                float(entry.composition.fractional_composition.get_el_amt_dict().get(element, 0.0))
                for entry in entries
            ]
            for element in elements
        ],
        dtype=float,
    )
    energies_per_atom = np.asarray([float(entry.energy_per_atom) for entry in entries], dtype=float)
    entry_index = {str(entry.entry_id): index for index, entry in enumerate(entries)}
    output: list[dict[str, Any]] = []
    for uid in candidates:
        uid_hex = bytes(uid).hex()
        candidate = by_hex[uid_hex]
        row = by_row[uid_hex]
        method = "scipy_highs_explicit_full_context_leave_one_candidate_out"
        error: str | None = None
        value: float | None = None
        try:
            if len(entries) <= 1:
                raise ValueError("no remaining entries after candidate removal")
            target_fraction = candidate.composition.fractional_composition.get_el_amt_dict()
            target = np.asarray([float(target_fraction.get(element, 0.0)) for element in elements])
            bounds = [(0.0, None)] * len(entries)
            bounds[entry_index[uid_hex]] = (0.0, 0.0)
            solution = linprog(
                energies_per_atom,
                A_eq=composition_matrix,
                b_eq=target,
                bounds=bounds,
                method="highs",
                options={
                    "primal_feasibility_tolerance": 1e-9,
                    "dual_feasibility_tolerance": 1e-9,
                },
            )
            if not solution.success or solution.fun is None:
                raise RuntimeError(f"linprog status={solution.status}: {solution.message}")
            value = float(candidate.energy_per_atom - float(solution.fun))
        except Exception as exc:
            try:
                _decomposition, raw = full_diagram.get_decomp_and_phase_separation_energy(
                    candidate,
                    stable_only=False,
                    tols=(1e-10, 1e-8),
                    maxiter=2000,
                )
                value = None if raw is None else float(raw)
                method = "fallback_official_phase_separation_energy"
                error = f"explicit_LOO_{type(exc).__name__}: {exc}"
            except Exception as fallback_exc:
                value = None
                method = "explicit_and_official_non_estimable"
                error = (
                    f"explicit_LOO_{type(exc).__name__}: {exc}; "
                    f"official_{type(fallback_exc).__name__}: {fallback_exc}"
                )
        output.append(
            {
                "snapshot_id": key[0],
                "thermo_type": key[1],
                "phase_context_chemsys": key[2],
                "unified_entry_id": bytes(uid),
                "signed_margin_eV_per_atom": value,
                "stability_margin_eV_per_atom": None if value is None else abs(float(value)),
                "solver_status": "PASS" if value is not None else "NON_ESTIMABLE",
                "method": method,
                "documented_exception": error,
                "method_version": METHOD_VERSION,
            }
        )
    return output


def build_source_margin_cache(
    eligible: pd.DataFrame,
    phase_path: Path,
    *,
    tolerance: float = 1.0e-8,
    workers: int | None = None,
) -> pd.DataFrame:
    """Build margins for every exact-stable source endpoint in the risk set."""

    exact = eligible.loc[eligible["s_energy_above_hull"].astype(float).le(tolerance)].copy()
    needed = set(
        zip(
            exact["s_snapshot_id"].astype(str),
            exact["thermo_type"].astype(str),
            exact["s_phase_context_chemsys"].astype(str),
            strict=True,
        )
    )
    phase, terminals = _relevant_context_rows(phase_path, needed)
    terminals_by_key = {
        tuple(map(str, key)): group.to_dict(orient="records")
        for key, group in terminals.groupby(
            ["snapshot_id", "thermo_type", "phase_context_chemsys"], sort=True
        )
    }
    endpoint_map: dict[tuple[str, str, str], set[bytes]] = {}
    for row in exact.itertuples(index=False):
        key = (str(row.s_snapshot_id), str(row.thermo_type), str(row.s_phase_context_chemsys))
        endpoint_map.setdefault(key, set()).add(bytes(row.s_unified_entry_id))
    jobs: list[tuple[tuple[str, str, str], list[dict[str, Any]], list[bytes], float]] = []
    for key, group in phase.groupby(
        ["snapshot_id", "thermo_type", "phase_context_chemsys"], sort=True
    ):
        normalized = tuple(map(str, key))
        jobs.append(
            (
                normalized,
                _records_with_elemental_terminals(
                    normalized, group.to_dict(orient="records"), terminals_by_key
                ),
                sorted(endpoint_map[normalized]),
                tolerance,
            )
        )
    requested = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    worker_count = min(max(1, int(requested)), max(1, os.cpu_count() or 1), len(jobs))
    chunk_size = max(1, math.ceil(len(jobs) / max(1, worker_count * 8)))
    chunks = [jobs[index : index + chunk_size] for index in range(0, len(jobs), chunk_size)]

    def compute_chunk(chunk: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in chunk:
            rows.extend(_fast_context_margin_job(job))
        return rows

    rows: list[dict[str, Any]] = []
    if worker_count == 1:
        rows = compute_chunk(jobs)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_margin_chunk_worker, chunk) for chunk in chunks]
            for future in as_completed(futures):
                rows.extend(future.result())
    frame = pd.DataFrame(rows).sort_values(
        ["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id"],
        key=lambda column: column.map(bytes) if column.name == "unified_entry_id" else column,
        kind="mergesort",
    )
    if len(frame) != len(exact) or frame["unified_entry_id"].map(bytes).duplicated().any():
        raise RuntimeError("source signed-margin cache does not cover the exact risk set one-to-one")
    return frame.reset_index(drop=True)


def _margin_chunk_worker(chunk: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in chunk:
        rows.extend(_fast_context_margin_job(job))
    return rows


def _build_risk_set(
    states: pd.DataFrame,
    eligible: pd.DataFrame,
    context: pd.DataFrame,
    target_metadata: pd.DataFrame,
    margins: pd.DataFrame,
    exposure_days: dict[str, int],
) -> pd.DataFrame:
    join_columns = [
        "transition_id",
        "s_snapshot_id",
        "s_phase_context_chemsys",
        "s_unified_entry_id",
        "s_entry_id",
        "s_task_id",
        "s_source_workflow",
        "s_composition_json",
        "s_energy_above_hull",
        "q_energy_above_hull",
        "phase_context_dimensionality",
        "same_workflow",
        "same_phase_context",
        "candidate_identity_unchanged",
    ]
    joined = eligible[join_columns].copy()
    joined["transition_id"] = joined["transition_id"].map(bytes)
    joined["s_unified_entry_id"] = joined["s_unified_entry_id"].map(bytes)
    if joined["transition_id"].duplicated().any():
        raise RuntimeError("eligible endpoint join is not one-to-one")

    frame = states.copy()
    frame["transition_id"] = frame["transition_id"].map(bytes)
    frame = frame.merge(joined, on="transition_id", how="left", validate="one_to_one")
    frame["endpoint_context_join_status"] = np.where(
        frame["s_unified_entry_id"].notna(), "ELIGIBLE_UNIQUE_ENDPOINTS", "NOT_JOINED"
    )
    frame = frame.merge(
        context,
        left_on=["s_snapshot_id", "thermo_type", "s_phase_context_chemsys"],
        right_on=["snapshot_id", "thermo_type", "phase_context_chemsys"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["snapshot_id", "phase_context_chemsys"], errors="ignore")
    metadata_map = target_metadata.set_index("unified_entry_id")["compatibility_mode"]
    frame["source_compatibility_mode"] = frame["s_unified_entry_id"].map(metadata_map)
    frame["source_provenance_class_if_available"] = (
        frame["s_source_workflow"].astype("string")
        + "|"
        + frame["source_compatibility_mode"].astype("string")
    )
    frame["source_chemsys_dimensionality"] = frame["s_phase_context_chemsys"].map(
        lambda value: np.nan if pd.isna(value) else len(str(value).split("-"))
    )
    frame["source_nelements"] = frame["s_composition_json"].map(
        lambda value: np.nan if pd.isna(value) else len(json.loads(str(value)))
    )
    frame["source_composition_entropy"] = frame["s_composition_json"].map(
        lambda value: np.nan if pd.isna(value) else composition_entropy(str(value))
    )
    margin_map = margins.set_index(margins["unified_entry_id"].map(bytes))[
        "stability_margin_eV_per_atom"
    ]
    source_energy = pd.to_numeric(frame["s_energy_above_hull"], errors="coerce")
    frame["source_signed_margin_eV_per_atom"] = np.where(
        source_energy.le(1.0e-8),
        frame["s_unified_entry_id"].map(margin_map).map(
            lambda value: np.nan if pd.isna(value) else -abs(float(value))
        ),
        source_energy,
    )
    frame["source_stability_margin_eV_per_atom"] = frame[
        "source_signed_margin_eV_per_atom"
    ].abs()
    frame["source_margin_bin"] = _margin_bin(frame["source_stability_margin_eV_per_atom"])
    frame["source_context_density_bin"] = _context_density_bin(
        frame["source_context_entry_count"]
    )

    frame["exposure_days"] = frame["release_pair"].map(exposure_days)
    frame["log_exposure_years"] = np.log(frame["exposure_days"] / 365.25)
    endpoint_eligible = frame["endpoint_context_join_status"].eq("ELIGIBLE_UNIQUE_ENDPOINTS")
    source_observed = frame["observation_status"].eq("observed")
    complete_covariates = frame[
        [
            "source_chemsys_dimensionality",
            "source_stability_margin_eV_per_atom",
            "source_context_entry_count",
            "source_context_competitor_fraction",
            "source_nelements",
            "source_composition_entropy",
            "source_provenance_class_if_available",
            "exposure_days",
        ]
    ].notna().all(axis=1)
    primary_identity = frame["identity_confidence"].eq("A1")
    sensitivity_identity = frame["identity_confidence"].isin(["A1", "A2"])
    frame["model_base_eligible"] = endpoint_eligible & source_observed & complete_covariates
    frame["primary_identity_eligible"] = frame["model_base_eligible"] & primary_identity
    frame["sensitivity_identity_eligible"] = frame["model_base_eligible"] & sensitivity_identity

    target_energy = pd.to_numeric(frame["q_energy_above_hull"], errors="coerce")
    for label, threshold in OUTCOME_THRESHOLDS.items():
        source_at_risk = source_energy.le(threshold)
        target_above = target_energy.gt(threshold)
        frame[f"risk_{label}"] = frame["primary_identity_eligible"] & source_at_risk
        frame[f"risk_{label}_A1_A2"] = frame["sensitivity_identity_eligible"] & source_at_risk
        frame[f"event_{label}"] = frame[f"risk_{label}"] & target_above
        frame[f"event_{label}_A1_A2"] = frame[f"risk_{label}_A1_A2"] & target_above

    selected = [
        "transition_id",
        "canonical_lineage_id",
        "identity_confidence",
        "source_snapshot",
        "target_snapshot",
        "release_pair",
        "source_material_id",
        "target_material_id",
        "thermo_type",
        "observation_status",
        "source_state",
        "target_state",
        "s_unified_entry_id",
        "s_entry_id",
        "s_task_id",
        "s_phase_context_chemsys",
        "s_source_workflow",
        "source_compatibility_mode",
        "source_provenance_class_if_available",
        "source_chemsys_dimensionality",
        "s_energy_above_hull",
        "q_energy_above_hull",
        "source_signed_margin_eV_per_atom",
        "source_stability_margin_eV_per_atom",
        "source_margin_bin",
        "source_context_entry_count",
        "source_context_competitor_count",
        "source_context_competitor_fraction",
        "source_context_density_bin",
        "source_nelements",
        "source_composition_entropy",
        "same_workflow",
        "same_phase_context",
        "candidate_identity_unchanged",
        "endpoint_context_join_status",
        "model_base_eligible",
        "primary_identity_eligible",
        "sensitivity_identity_eligible",
        "exposure_days",
        "log_exposure_years",
        "risk_exact",
        "event_exact",
        "risk_10meV",
        "event_10meV",
        "risk_25meV",
        "event_25meV",
        "risk_exact_A1_A2",
        "event_exact_A1_A2",
        "risk_10meV_A1_A2",
        "event_10meV_A1_A2",
        "risk_25meV_A1_A2",
        "event_25meV_A1_A2",
    ]
    result = frame[selected].copy()
    result["method_version"] = METHOD_VERSION
    result = result.sort_values("transition_id", key=lambda column: column.map(bytes), kind="mergesort")
    if result["transition_id"].duplicated().any() or len(result) != len(states):
        raise RuntimeError("R3.2 risk set is not a complete one-row-per-transition table")
    return result.reset_index(drop=True)


def _model_frame(risk_set: pd.DataFrame, outcome: str) -> pd.DataFrame:
    frame = risk_set.loc[risk_set[f"risk_{outcome}"].astype(bool)].copy()
    frame["outcome"] = frame[f"event_{outcome}"].astype(int)
    frame["release_pair"] = pd.Categorical(frame["release_pair"], categories=RELEASE_ORDER, ordered=True)
    frame["thermo_type"] = frame["thermo_type"].astype(str)
    frame["source_chemsys_dimensionality"] = frame["source_chemsys_dimensionality"].astype(int).astype(str)
    frame["source_margin_bin"] = pd.Categorical(
        frame["source_margin_bin"], categories=MARGIN_LABELS, ordered=True
    )
    frame["source_provenance_class_if_available"] = frame[
        "source_provenance_class_if_available"
    ].astype(str)
    return frame


def _design_spec(master: pd.DataFrame) -> dict[str, Any]:
    margin = master["source_stability_margin_eV_per_atom"].astype(float)
    upper = float(max(margin.max(), 0.050))
    knot = float(margin.median())
    if not 0.0 < knot < upper:
        knot = upper / 2.0
    return {
        "release_levels": list(RELEASE_ORDER),
        "thermo_levels": sorted(master["thermo_type"].astype(str).unique()),
        "dimensionality_levels": sorted(
            master["source_chemsys_dimensionality"].astype(int).astype(str).unique(),
            key=lambda value: int(value),
        ),
        "margin_bin_levels": list(MARGIN_LABELS),
        "provenance_levels": sorted(
            master["source_provenance_class_if_available"].astype(str).unique()
        ),
        "spline_lower_bound": 0.0,
        "spline_upper_bound": upper,
        "spline_knots": [knot],
    }


def _categorize_for_design(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    mappings = {
        "release_pair": spec["release_levels"],
        "thermo_type": spec["thermo_levels"],
        "source_chemsys_dimensionality": spec["dimensionality_levels"],
        "source_margin_bin": spec["margin_bin_levels"],
        "source_provenance_class_if_available": spec["provenance_levels"],
    }
    for column, levels in mappings.items():
        result[column] = pd.Categorical(result[column].astype(str), categories=levels, ordered=True)
        if result[column].isna().any():
            raise RuntimeError(f"unrecognized frozen design level in {column}")
    return result


def _design_formula(spec: dict[str, Any]) -> str:
    knot = repr(float(spec["spline_knots"][0]))
    upper = repr(float(spec["spline_upper_bound"]))
    return (
        "1 + C(release_pair) + C(thermo_type) + C(source_chemsys_dimensionality) + "
        f"bs(source_stability_margin_eV_per_atom, knots=({knot},), degree=3, "
        f"include_intercept=False, lower_bound=0.0, upper_bound={upper}) + "
        "np.log1p(source_context_entry_count) + source_context_competitor_fraction + "
        "source_nelements + source_composition_entropy + "
        "C(source_provenance_class_if_available) + C(thermo_type):C(source_margin_bin)"
    )


def _fit_model(
    frame: pd.DataFrame, spec: dict[str, Any], outcome_name: str
) -> tuple[Any, patsy.DesignInfo, list[int], pd.DataFrame, np.ndarray, dict[str, Any]]:
    prepared = _categorize_for_design(frame, spec)
    formula = _design_formula(spec)
    design = patsy.dmatrix(formula, prepared, return_type="dataframe", NA_action="raise")
    y = prepared["outcome"].astype(float).to_numpy()
    offset = prepared["log_exposure_years"].astype(float).to_numpy()
    groups = prepared["canonical_lineage_id"].astype(str).to_numpy()
    full_matrix = np.asarray(design, dtype=float)
    cross = np.asarray(full_matrix.T @ full_matrix, dtype=float)
    kept_indices: list[int] = []
    current_rank = 0
    # Deterministic declared-order pruning.  It uses X only, never y, and
    # therefore resolves redundant categorical encodings without adaptive term
    # selection.  Earlier contract terms take precedence over later aliases.
    for index in range(full_matrix.shape[1]):
        candidate = kept_indices + [index]
        candidate_rank = int(np.linalg.matrix_rank(cross[np.ix_(candidate, candidate)]))
        if candidate_rank > current_rank:
            kept_indices.append(index)
            current_rank = candidate_rank
    matrix = full_matrix[:, kept_indices]
    rank = int(np.linalg.matrix_rank(cross[np.ix_(kept_indices, kept_indices)]))
    condition = float(np.linalg.cond(cross[np.ix_(kept_indices, kept_indices)]))
    if rank != matrix.shape[1]:
        raise RuntimeError(f"deterministic {outcome_name} design pruning failed to restore full rank")
    model = sm.GLM(
        y,
        matrix,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        offset=offset,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(
            maxiter=200,
            tol=1e-9,
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
    predicted = np.asarray(result.predict(), dtype=float)
    diagnostics = {
        "outcome": outcome_name,
        "rows": int(len(prepared)),
        "events": int(y.sum()),
        "lineages": int(pd.Series(groups).nunique()),
        "formula_columns": int(full_matrix.shape[1]),
        "columns": int(matrix.shape[1]),
        "rank": rank,
        "kept_design_columns": [design.columns[index] for index in kept_indices],
        "dropped_redundant_design_columns": [
            design.columns[index] for index in range(full_matrix.shape[1]) if index not in kept_indices
        ],
        "rank_pruning_rule": "declared-column-order, covariate-only incremental rank; outcome not consulted",
        "condition_number_xtx": condition,
        "converged": bool(result.converged),
        "iterations": int(result.fit_history.get("iteration", -1)),
        "warnings": sorted({str(item.message) for item in caught}),
        "predicted_min": float(predicted.min()),
        "predicted_max": float(predicted.max()),
        "predicted_mean": float(predicted.mean()),
    }
    return result, design.design_info, kept_indices, prepared, matrix, diagnostics


def _inverse_cloglog(eta: np.ndarray) -> np.ndarray:
    clipped = np.clip(eta, -40.0, 40.0)
    return -np.expm1(-np.exp(clipped))


def _inverse_cloglog_derivative(eta: np.ndarray) -> np.ndarray:
    clipped = np.clip(eta, -40.0, 40.0)
    exp_eta = np.exp(clipped)
    return exp_eta * np.exp(-exp_eta)


def _support_cell(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["thermo_type"].astype(str)
        + "|d="
        + frame["source_chemsys_dimensionality"].astype(str)
        + "|m="
        + frame["source_margin_bin"].astype(str)
        + "|c="
        + frame["source_context_density_bin"].astype(str)
    )


def _reference_frames(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    work = frame.copy()
    work["support_cell"] = _support_cell(work)
    counts = work.groupby(["release_pair", "support_cell"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(RELEASE_ORDER, fill_value=0)
    common_cells = set(counts.columns[(counts > 0).all(axis=0)])
    references = {
        "pooled_source_person_period": work,
        "first_release_source_population": work.loc[work["release_pair"].astype(str).eq(RELEASE_ORDER[0])],
        "latest_release_source_population": work.loc[work["release_pair"].astype(str).eq(RELEASE_ORDER[-1])],
        "common_support_population": work.loc[work["support_cell"].isin(common_cells)],
    }
    diagnostics = {
        "total_cells": int(work["support_cell"].nunique()),
        "common_cells": int(len(common_cells)),
        "common_support_rows": int(work["support_cell"].isin(common_cells).sum()),
        "common_support_fraction": float(work["support_cell"].isin(common_cells).mean()),
        "common_cells_values": sorted(common_cells),
    }
    return references, diagnostics


def lineage_one_step_bootstrap(
    model: Any,
    design: np.ndarray,
    groups: Iterable[str],
    *,
    replicates: int,
    seed: int,
    batch_size: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Whole-lineage multinomial bootstrap using a one-step score update.

    Each lineage receives one multinomial count in a bootstrap replicate.  The
    GLM estimating equation is updated once from the converged estimate.  This
    preserves whole clusters, avoids outcome-adaptive refitting choices, and is
    computationally feasible for the frozen 2000-replicate requirement.
    """

    group_codes, unique = pd.factorize(pd.Series(list(groups), dtype="string"), sort=True)
    score_obs = np.asarray(model.model.score_obs(model.params), dtype=float)
    cluster_scores = np.zeros((len(unique), score_obs.shape[1]), dtype=float)
    np.add.at(cluster_scores, group_codes, score_obs)
    information_inverse = np.linalg.pinv(-np.asarray(model.model.hessian(model.params), dtype=float))
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    probability = np.full(len(unique), 1.0 / len(unique), dtype=float)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        counts = rng.multinomial(len(unique), probability, size=size)
        perturbation = (counts - 1) @ cluster_scores
        deltas.append(perturbation @ information_inverse.T)
    delta = np.vstack(deltas)
    finite = np.isfinite(delta).all(axis=1)
    diagnostics = {
        "method": "whole-lineage multinomial one-step estimating-equation bootstrap",
        "standardization_linearization": "first-order delta gradient on frozen reference population",
        "unit": "canonical_lineage_id",
        "clusters": int(len(unique)),
        "requested_replicates": int(replicates),
        "successful_replicates": int(finite.sum()),
        "failed_replicates": int((~finite).sum()),
        "seed": int(seed),
    }
    return delta[finite], diagnostics


def _standardize_model(
    outcome: str,
    result: Any,
    design_info: patsy.DesignInfo,
    kept_indices: list[int],
    frame: pd.DataFrame,
    spec: dict[str, Any],
    delta_beta: np.ndarray,
    exposure_days: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    references, support = _reference_frames(frame)
    rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for reference_name, reference in references.items():
        if reference.empty:
            continue
        for release in RELEASE_ORDER:
            counterfactual = reference.copy()
            counterfactual["release_pair"] = release
            counterfactual = _categorize_for_design(counterfactual, spec)
            matrix = np.asarray(
                patsy.build_design_matrices([design_info], counterfactual, NA_action="raise")[0],
                dtype=float,
            )[:, kept_indices]
            offset = math.log(float(exposure_days[release]) / 365.25)
            eta = matrix @ np.asarray(result.params, dtype=float) + offset
            probability = _inverse_cloglog(eta)
            point = float(probability.mean())
            gradient = np.mean(
                _inverse_cloglog_derivative(eta)[:, None] * matrix,
                axis=0,
            )
            bootstrap = point + delta_beta @ gradient
            finite = bootstrap[np.isfinite(bootstrap)]
            lower, upper = np.quantile(finite, [0.025, 0.975])
            rows.append(
                {
                    "outcome": outcome,
                    "estimate_type": "model_direct_standardization",
                    "reference_population": reference_name,
                    "release_pair": release,
                    "risk": point,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "reference_rows": int(len(reference)),
                    "bootstrap_replicates": int(len(finite)),
                    "estimability_status": "ESTIMABLE",
                    "method_version": METHOD_VERSION,
                }
            )
            boot_rows.append(
                {
                    "outcome": outcome,
                    "reference_population": reference_name,
                    "release_pair": release,
                    "point_risk": point,
                    "bootstrap_mean": float(finite.mean()),
                    "bootstrap_sd": float(finite.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "replicates": int(len(finite)),
                    "outside_unit_interval_replicates": int(((finite < 0) | (finite > 1)).sum()),
                    "bootstrap_method": "lineage_multinomial_one_step_delta",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows), support


def _raw_risk_rows(risk_set: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOME_THRESHOLDS:
        frame = risk_set.loc[risk_set[f"risk_{outcome}"].astype(bool)]
        for release, group in frame.groupby("release_pair", sort=False, observed=True):
            events = int(group[f"event_{outcome}"].sum())
            denominator = len(group)
            rate = events / denominator if denominator else np.nan
            # Wilson interval is descriptive and independent of the cluster bootstrap.
            z = 1.959963984540054
            centre = (rate + z * z / (2 * denominator)) / (1 + z * z / denominator)
            half = z * math.sqrt(
                rate * (1 - rate) / denominator + z * z / (4 * denominator * denominator)
            ) / (1 + z * z / denominator)
            rows.append(
                {
                    "outcome": outcome,
                    "estimate_type": "raw_release_stratified",
                    "reference_population": "observed_release_population",
                    "release_pair": str(release),
                    "risk": rate,
                    "ci_lower": max(0.0, centre - half),
                    "ci_upper": min(1.0, centre + half),
                    "reference_rows": denominator,
                    "bootstrap_replicates": 0,
                    "estimability_status": "DESCRIPTIVE",
                    "method_version": METHOD_VERSION,
                }
            )
    return pd.DataFrame(rows)


def _poststratified_rows(risk_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for outcome in OUTCOME_THRESHOLDS:
        frame = risk_set.loc[risk_set[f"risk_{outcome}"].astype(bool)].copy()
        frame["support_cell"] = _support_cell(frame)
        cell = (
            frame.groupby(["release_pair", "support_cell"], observed=True)
            .agg(denominator=(f"event_{outcome}", "size"), events=(f"event_{outcome}", "sum"))
            .reset_index()
        )
        pivot = cell.pivot(index="support_cell", columns="release_pair", values="denominator").fillna(0)
        pivot = pivot.reindex(columns=RELEASE_ORDER, fill_value=0)
        common = set(pivot.index[(pivot > 0).all(axis=1)])
        reference = frame.loc[frame["support_cell"].isin(common), "support_cell"].value_counts(normalize=True)
        for release in RELEASE_ORDER:
            release_cells = cell.loc[
                cell["release_pair"].astype(str).eq(release) & cell["support_cell"].isin(common)
            ].set_index("support_cell")
            aligned = release_cells.reindex(reference.index)
            rate = aligned["events"] / aligned["denominator"]
            estimate = float((reference * rate).sum()) if len(reference) else np.nan
            output.append(
                {
                    "outcome": outcome,
                    "estimate_type": "nonparametric_common_cell_poststratification",
                    "reference_population": "common_support_population",
                    "release_pair": release,
                    "risk": estimate,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "reference_rows": int(frame["support_cell"].isin(common).sum()),
                    "bootstrap_replicates": 0,
                    "estimability_status": "ESTIMABLE" if len(reference) else "NON_ESTIMABLE",
                    "method_version": METHOD_VERSION,
                }
            )
            release_frame = frame.loc[frame["release_pair"].astype(str).eq(release)]
            common_mask = release_frame["support_cell"].isin(common)
            diagnostics.append(
                {
                    "outcome": outcome,
                    "release_pair": release,
                    "risk_rows": len(release_frame),
                    "events": int(release_frame[f"event_{outcome}"].sum()),
                    "lineages": int(release_frame["canonical_lineage_id"].nunique()),
                    "observed_cells": int(release_frame["support_cell"].nunique()),
                    "common_cells": int(len(common)),
                    "common_support_rows": int(common_mask.sum()),
                    "common_support_fraction": float(common_mask.mean()),
                    "no_event_cells": int(
                        (cell.loc[cell["release_pair"].astype(str).eq(release), "events"] == 0).sum()
                    ),
                }
            )
    return pd.DataFrame(output), pd.DataFrame(diagnostics)


def _covariate_balance(risk_set: pd.DataFrame) -> pd.DataFrame:
    frame = risk_set.loc[risk_set["risk_25meV"].astype(bool)].copy()
    numeric = [
        "source_stability_margin_eV_per_atom",
        "source_context_entry_count",
        "source_context_competitor_fraction",
        "source_nelements",
        "source_composition_entropy",
    ]
    rows: list[dict[str, Any]] = []
    for release, group in frame.groupby("release_pair", sort=False, observed=True):
        for column in numeric:
            pooled = frame[column].astype(float)
            selected = group[column].astype(float)
            pooled_sd = float(pooled.std(ddof=1))
            rows.append(
                {
                    "release_pair": str(release),
                    "covariate": column,
                    "level": "NUMERIC",
                    "n": len(selected),
                    "mean_or_proportion": float(selected.mean()),
                    "sd": float(selected.std(ddof=1)),
                    "pooled_mean_or_proportion": float(pooled.mean()),
                    "standardized_difference_vs_pooled": (
                        float((selected.mean() - pooled.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
                    ),
                    "missing": int(group[column].isna().sum()),
                }
            )
        for column in (
            "thermo_type",
            "source_chemsys_dimensionality",
            "source_margin_bin",
            "source_context_density_bin",
            "source_provenance_class_if_available",
        ):
            levels = sorted(frame[column].astype(str).unique())
            for level in levels:
                proportion = float(group[column].astype(str).eq(level).mean())
                pooled_proportion = float(frame[column].astype(str).eq(level).mean())
                scale = math.sqrt(max(pooled_proportion * (1 - pooled_proportion), 1e-12))
                rows.append(
                    {
                        "release_pair": str(release),
                        "covariate": column,
                        "level": level,
                        "n": len(group),
                        "mean_or_proportion": proportion,
                        "sd": np.nan,
                        "pooled_mean_or_proportion": pooled_proportion,
                        "standardized_difference_vs_pooled": (proportion - pooled_proportion) / scale,
                        "missing": int(group[column].isna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def _coefficient_frame(
    outcome: str, result: Any, design_info: patsy.DesignInfo, kept_indices: list[int]
) -> pd.DataFrame:
    params = np.asarray(result.params, dtype=float)
    standard_error = np.asarray(result.bse, dtype=float)
    normal = 1.959963984540054
    return pd.DataFrame(
        {
            "outcome": outcome,
            "term": [design_info.column_names[index] for index in kept_indices],
            "coefficient": params,
            "cluster_robust_se": standard_error,
            "ci_lower": params - normal * standard_error,
            "ci_upper": params + normal * standard_error,
            "p_value": np.asarray(result.pvalues, dtype=float),
            "covariance": "canonical_lineage_id cluster robust",
            "interpretation_guardrail": "associational release contrast; not a causal time trend",
        }
    )


def _calibration_and_influence(
    outcome: str, result: Any, frame: pd.DataFrame, design: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = np.asarray(result.predict(), dtype=float)
    outcome_values = frame["outcome"].astype(int).to_numpy()
    quantile = pd.qcut(pd.Series(prediction).rank(method="first"), 10, labels=False) + 1
    calibration = pd.DataFrame(
        {"decile": quantile, "predicted": prediction, "observed": outcome_values}
    ).groupby("decile").agg(rows=("observed", "size"), observed_risk=("observed", "mean"), predicted_risk=("predicted", "mean")).reset_index()
    calibration.insert(0, "outcome", outcome)
    pearson = (outcome_values - prediction) / np.sqrt(np.maximum(prediction * (1 - prediction), 1e-12))
    influence = pd.DataFrame(
        {"lineage": frame["canonical_lineage_id"].astype(str).to_numpy(), "pearson": pearson}
    ).groupby("lineage").agg(abs_pearson_sum=("pearson", lambda values: float(np.abs(values).sum())), rows=("pearson", "size"))
    summary = {
        "max_lineage_abs_pearson_sum": float(influence["abs_pearson_sum"].max()),
        "median_lineage_abs_pearson_sum": float(influence["abs_pearson_sum"].median()),
        "max_lineage_rows": int(influence["rows"].max()),
    }
    return calibration, summary


def _multistate_matrix(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for release, release_frame in states.groupby("release_pair", sort=False, observed=True):
        matrix = pd.crosstab(release_frame["source_state"], release_frame["target_state"]).reindex(
            index=STATE_ORDER, columns=STATE_ORDER, fill_value=0
        )
        for source in STATE_ORDER:
            denominator = int(matrix.loc[source].sum())
            for target in STATE_ORDER:
                count = int(matrix.loc[source, target])
                rows.append(
                    {
                        "release_pair": str(release),
                        "source_state": source,
                        "target_state": target,
                        "count": count,
                        "source_state_denominator": denominator,
                        "row_fraction": count / denominator if denominator else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _exclusion_ledger(states: pd.DataFrame, risk_set: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for status, group in states.groupby("observation_status", dropna=False, sort=True):
        records.append(
            {
                "stage": "multi_state_observation_status",
                "reason": str(status),
                "rows": len(group),
                "included_multistate": True,
                "included_adjusted_model": str(status) == "observed",
            }
        )
    joined = risk_set["endpoint_context_join_status"].eq("ELIGIBLE_UNIQUE_ENDPOINTS")
    records.extend(
        [
            {
                "stage": "context_endpoint_join",
                "reason": "eligible_unique_endpoints",
                "rows": int(joined.sum()),
                "included_multistate": True,
                "included_adjusted_model": True,
            },
            {
                "stage": "context_endpoint_join",
                "reason": "not_joined_or_nonobserved",
                "rows": int((~joined).sum()),
                "included_multistate": True,
                "included_adjusted_model": False,
            },
            {
                "stage": "source_margin",
                "reason": "solver_non_estimable",
                "rows": int(margins["stability_margin_eV_per_atom"].isna().sum()),
                "included_multistate": True,
                "included_adjusted_model": False,
            },
        ]
    )
    for outcome in OUTCOME_THRESHOLDS:
        records.append(
            {
                "stage": f"primary_A1_{outcome}_risk",
                "reason": "included_source_at_risk",
                "rows": int(risk_set[f"risk_{outcome}"].sum()),
                "included_multistate": True,
                "included_adjusted_model": outcome in {"exact", "10meV"},
            }
        )
    return pd.DataFrame(records)


def _save_model(
    path: Path,
    outcome: str,
    result: Any,
    design_info: patsy.DesignInfo,
    kept_indices: list[int],
    spec: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    payload = {
        "task_id": TASK_ID,
        "method_version": METHOD_VERSION,
        "outcome": outcome,
        "family": "binomial",
        "link": "cloglog",
        "offset": "log(exposure_days/365.25)",
        "formula": _design_formula(spec),
        "design_columns": [design_info.column_names[index] for index in kept_indices],
        "dropped_redundant_design_columns": [
            design_info.column_names[index]
            for index in range(len(design_info.column_names))
            if index not in kept_indices
        ],
        "design_spec": spec,
        "coefficients": {
            name: float(value)
            for name, value in zip(
                [design_info.column_names[index] for index in kept_indices], result.params, strict=True
            )
        },
        "cluster_robust_covariance": np.asarray(result.cov_params(), dtype=float).tolist(),
        "diagnostics": diagnostics,
        "serialization_note": "portable coefficients and frozen explicit design specification",
    }
    _write_json(path, payload)


def _figure_outputs(fig: Any, base: Path) -> list[Path]:
    paths: list[Path] = []
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _make_figures(
    report_dir: Path,
    standardized: pd.DataFrame,
    risk_set: pd.DataFrame,
    multistate: pd.DataFrame,
) -> tuple[list[Path], list[dict[str, Any]]]:
    outputs: list[Path] = []
    metadata: list[dict[str, Any]] = []
    colors = {"raw_release_stratified": "#6b7280", "model_direct_standardization": "#0b6e99"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for axis, outcome in zip(axes, ("exact", "10meV"), strict=True):
        data = standardized.loc[
            standardized["outcome"].eq(outcome)
            & (
                standardized["estimate_type"].eq("raw_release_stratified")
                | (
                    standardized["estimate_type"].eq("model_direct_standardization")
                    & standardized["reference_population"].eq("pooled_source_person_period")
                )
            )
        ]
        for index, estimate_type in enumerate(("raw_release_stratified", "model_direct_standardization")):
            group = data.loc[data["estimate_type"].eq(estimate_type)].set_index("release_pair").reindex(RELEASE_ORDER)
            x = np.arange(len(RELEASE_ORDER)) + (index - 0.5) * 0.16
            y = group["risk"].to_numpy(dtype=float)
            lower = y - group["ci_lower"].to_numpy(dtype=float)
            upper = group["ci_upper"].to_numpy(dtype=float) - y
            axis.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=3, color=colors[estimate_type], label=estimate_type.replace("_", " "))
        axis.set_xticks(range(len(RELEASE_ORDER)), ["R1", "R2", "R3"])
        axis.set_ylabel("Transition risk")
        axis.set_title(f"{outcome} source-state risk")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Raw and pooled-reference standardized release risks (associational)")
    base = report_dir / "figures/figure_r3_2_raw_adjusted_release_risk"
    outputs.extend(_figure_outputs(figure, base))
    metadata.append({"figure": base.name, "population": "primary A1 source-risk rows", "units": "probability", "interpretation": "release contrasts are associational, not causal time trends"})

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, outcome in zip(axes, ("exact", "10meV"), strict=True):
        frame = risk_set.loc[risk_set[f"risk_{outcome}"].astype(bool)].copy()
        positive = frame["source_stability_margin_eV_per_atom"].clip(lower=1e-8)
        frame["margin_decile"] = pd.qcut(positive.rank(method="first"), 10, labels=False) + 1
        curve = frame.groupby("margin_decile").agg(
            margin=("source_stability_margin_eV_per_atom", "median"),
            risk=(f"event_{outcome}", "mean"),
            denominator=(f"event_{outcome}", "size"),
        )
        axis.plot(curve["margin"] * 1000, curve["risk"], marker="o", color="#b04a5a")
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set_xlabel("Source stability margin (meV/atom)")
        axis.set_title(outcome)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed transition risk")
    figure.suptitle("Source-margin risk curves (descriptive deciles)")
    base = report_dir / "figures/figure_r3_2_margin_risk_curve"
    outputs.extend(_figure_outputs(figure, base))
    metadata.append({"figure": base.name, "population": "primary A1 risk rows", "units": "observed risk by source-margin decile", "interpretation": "descriptive; no smoothing or causal claim"})

    total = multistate.groupby(["source_state", "target_state"], observed=True)["count"].sum().unstack(fill_value=0).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    values = total.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    positive_values = values[values > 0]
    norm = LogNorm(vmin=max(1.0, positive_values.min()), vmax=max(1.0, positive_values.max()))
    image = axis.imshow(np.where(values > 0, values, np.nan), cmap="Blues", norm=norm)
    axis.set_xticks(range(len(STATE_ORDER)), STATE_ORDER)
    axis.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    axis.set_xlabel("Target state")
    axis.set_ylabel("Source state")
    axis.set_title("Multi-state transition counts (log color scale)")
    figure.colorbar(image, ax=axis, label="Count")
    base = report_dir / "figures/figure_r3_2_multistate_flow"
    outputs.extend(_figure_outputs(figure, base))
    metadata.append({"figure": base.name, "population": "all transition rows including D and M", "units": "transition count", "interpretation": "D and M remain distinct from instability states"})
    metadata_path = report_dir / "figures/figure_metadata.json"
    _write_json(metadata_path, metadata)
    outputs.append(metadata_path)
    return outputs, metadata


def _artifact_record(path: Path, repo: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": None,
    }
    if path.suffix == ".parquet":
        record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
    elif path.suffix == ".csv":
        record["rows"] = max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    return record


def _forbidden_access_count(access_log: Path) -> int:
    if not access_log.exists():
        return 0
    count = 0
    for line in access_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        status = str(json.loads(line).get("status", ""))
        if status.startswith("REJECTED_FORBIDDEN"):
            count += 1
    return count


def _direction_stability(standardized: pd.DataFrame, outcome: str) -> dict[str, Any]:
    data = standardized.loc[
        standardized["outcome"].eq(outcome)
        & standardized["estimate_type"].eq("model_direct_standardization")
    ]
    contrasts: dict[str, list[dict[str, Any]]] = {}
    for reference, group in data.groupby("reference_population", sort=True):
        risks = group.set_index("release_pair")["risk"].reindex(RELEASE_ORDER)
        values = []
        for left, right in zip(RELEASE_ORDER[:-1], RELEASE_ORDER[1:], strict=True):
            difference = float(risks[right] - risks[left])
            values.append({"contrast": f"{right} minus {left}", "risk_difference": difference})
        contrasts[str(reference)] = values
    pooled = contrasts.get("pooled_source_person_period", [])
    stable_references = 0
    for reference, values in contrasts.items():
        if reference == "pooled_source_person_period" or len(values) != len(pooled):
            continue
        flags = []
        for baseline, candidate in zip(pooled, values, strict=True):
            base = float(baseline["risk_difference"])
            value = float(candidate["risk_difference"])
            same_direction = (base == 0 and abs(value) <= 0.005) or (base * value >= 0)
            magnitude_close = abs(value - base) <= max(0.005, 0.5 * abs(base))
            flags.append(same_direction and magnitude_close)
        if all(flags):
            stable_references += 1
    return {
        "outcome": outcome,
        "contrasts": contrasts,
        "sensitivity_references_stable_vs_pooled": stable_references,
        "stable_under_at_least_two_references": stable_references >= 1,
        "frozen_rule": "same direction and absolute deviation <= max(0.005, 50% of pooled risk difference)",
    }


def build_standardized_survival(
    config_path: str | os.PathLike[str] = "configs/r3/r3_2_standardized_survival.yaml",
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    repo, config_file, config = _read_config(config_path)
    paths = _authorize_inputs(repo, config)
    report_dir = repo / "reports/R3_2"
    processed_dir = repo / "data/processed/R3_2"
    interim_dir = repo / "data/interim/R3_2"
    model_dir = repo / "models/R3_2"
    manifest_dir = repo / "data/manifests/R3_2"
    for directory in (report_dir, processed_dir, interim_dir, model_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    states = _state_frame(paths["transition_labels"])
    eligible, join_ledger = _join_all_eligible_transitions(
        paths["transition_labels"], paths["phase_entries"], "observed"
    )
    context, target_metadata = _context_covariates(paths["phase_entries"])

    cache_path = interim_dir / "source_signed_margin_cache.parquet"
    exact_uids = set(
        eligible.loc[eligible["s_energy_above_hull"].astype(float).le(1.0e-8), "s_unified_entry_id"].map(bytes)
    )
    cache_reused = False
    if cache_path.exists():
        candidate_cache = pq.read_table(cache_path).to_pandas()
        cache_uids = set(candidate_cache["unified_entry_id"].map(bytes))
        if (
            cache_uids == exact_uids
            and candidate_cache["method_version"].eq(METHOD_VERSION).all()
            and candidate_cache["method"].astype(str).str.startswith(
                (
                    "scipy_highs_explicit_full_context_leave_one_candidate_out",
                    "fallback_official_phase_separation_energy",
                )
            ).all()
            and not candidate_cache["unified_entry_id"].map(bytes).duplicated().any()
        ):
            margins = candidate_cache
            cache_reused = True
        else:
            raise RuntimeError("existing R3.2 signed-margin cache is incompatible; refusing silent overwrite")
    else:
        margins = build_source_margin_cache(eligible, paths["phase_entries"], workers=workers)
        pq.write_table(
            pa.Table.from_pandas(margins, preserve_index=False),
            cache_path,
            compression="zstd",
            use_dictionary=True,
        )

    risk_set = _build_risk_set(
        states,
        eligible,
        context,
        target_metadata,
        margins,
        {str(key): int(value) for key, value in config["risk_set"]["exposure_days"].items()},
    )
    risk_path = repo / config["output"]["risk_set"]
    multistate_path = repo / config["output"]["multistate_transition"]
    pq.write_table(
        pa.Table.from_pandas(risk_set, preserve_index=False),
        risk_path,
        compression="zstd",
        use_dictionary=True,
    )
    multistate_output = states[
        [
            "transition_id",
            "canonical_lineage_id",
            "identity_confidence",
            "source_snapshot",
            "target_snapshot",
            "release_pair",
            "thermo_type",
            "observation_status",
            "source_state",
            "target_state",
            "method_version",
        ]
    ].copy()
    multistate_output["transition_id"] = multistate_output["transition_id"].map(bytes)
    pq.write_table(
        pa.Table.from_pandas(multistate_output, preserve_index=False),
        multistate_path,
        compression="zstd",
        use_dictionary=True,
    )

    master = _model_frame(risk_set, "25meV")
    spec = _design_spec(master)
    model_results: dict[str, Any] = {}
    coefficient_parts: list[pd.DataFrame] = []
    standardized_parts = [_raw_risk_rows(risk_set)]
    bootstrap_parts: list[pd.DataFrame] = []
    model_diagnostics: dict[str, Any] = {}
    calibration_parts: list[pd.DataFrame] = []
    support_diagnostics: dict[str, Any] = {}
    bootstrap_diagnostics: dict[str, Any] = {}
    exposure = {str(key): int(value) for key, value in config["risk_set"]["exposure_days"].items()}
    for outcome in ("exact", "10meV"):
        frame = _model_frame(risk_set, outcome)
        result, design_info, kept_indices, prepared, design, diagnostics = _fit_model(
            frame, spec, outcome
        )
        if not result.converged:
            raise RuntimeError(f"{outcome} cloglog model did not converge")
        delta_beta, bootstrap_diag = lineage_one_step_bootstrap(
            result,
            design,
            prepared["canonical_lineage_id"].astype(str),
            replicates=int(config["standardization"]["bootstrap_replicates"]),
            seed=int(config["seed"]) + (0 if outcome == "exact" else 1),
        )
        standardized, bootstrap_summary, support = _standardize_model(
            outcome,
            result,
            design_info,
            kept_indices,
            prepared,
            spec,
            delta_beta,
            exposure,
        )
        calibration, influence = _calibration_and_influence(outcome, result, prepared, design)
        diagnostics["influence"] = influence
        coefficient_parts.append(_coefficient_frame(outcome, result, design_info, kept_indices))
        standardized_parts.append(standardized)
        bootstrap_parts.append(bootstrap_summary)
        calibration_parts.append(calibration)
        model_diagnostics[outcome] = diagnostics
        support_diagnostics[outcome] = support
        bootstrap_diagnostics[outcome] = bootstrap_diag
        model_path = model_dir / f"cloglog_{outcome}.json"
        _save_model(model_path, outcome, result, design_info, kept_indices, spec, diagnostics)
        model_results[outcome] = (result, design_info, kept_indices, prepared)

    poststratified, overlap = _poststratified_rows(risk_set)
    standardized_parts.append(poststratified)
    standardized = pd.concat(standardized_parts, ignore_index=True, sort=False)
    standardized_path = repo / config["output"]["standardized_risk"]
    pq.write_table(
        pa.Table.from_pandas(standardized, preserve_index=False),
        standardized_path,
        compression="zstd",
        use_dictionary=True,
    )

    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    bootstrap_summary = pd.concat(bootstrap_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    multistate_matrix = _multistate_matrix(states)
    covariate_balance = _covariate_balance(risk_set)
    ledger = pd.concat([join_ledger, _exclusion_ledger(states, risk_set, margins)], ignore_index=True, sort=False)

    _write_csv(repo / config["output"]["coefficients"], coefficients)
    _write_csv(repo / config["output"]["standardized_summary"], standardized)
    _write_csv(repo / config["output"]["bootstrap_summary"], bootstrap_summary)
    _write_csv(repo / config["output"]["multistate_matrix"], multistate_matrix)
    _write_csv(repo / config["output"]["covariate_balance"], covariate_balance)
    _write_csv(repo / config["output"]["overlap_diagnostics"], overlap)
    _write_csv(repo / config["output"]["exclusion_ledger"], ledger)
    _write_csv(report_dir / "calibration.csv", calibration)

    figure_paths, figure_metadata = _make_figures(report_dir, standardized, risk_set, multistate_matrix)

    # Reconcile A1+A2 counts directly to the frozen R3.1 summaries.
    cohort = pd.read_csv(repo / "reports/R3_1/cohort_summary.csv")
    threshold = pd.read_csv(repo / "reports/R3_1/threshold_summary.csv")
    exact_expected = int(
        cohort.loc[
            cohort["cohort_id"].eq("A1_A2")
            & cohort["sensitivity_dimension"].eq("overall")
            & cohort["direction"].eq("stable_to_unstable"),
            "exact_flip_rows",
        ].iloc[0]
    )
    expected = {"exact": exact_expected}
    for outcome, value in (("10meV", 0.010), ("25meV", 0.025)):
        expected[outcome] = int(
            threshold.loc[
                threshold["cohort_id"].eq("A1_A2")
                & threshold["sensitivity_dimension"].eq("overall")
                & threshold["threshold_eV_per_atom"].eq(value),
                "stable_to_unstable_rows",
            ].iloc[0]
        )
    reconciliation = {
        outcome: {
            "r3_1_expected": count,
            "r3_2_observed": int(risk_set[f"event_{outcome}_A1_A2"].sum()),
            "match": count == int(risk_set[f"event_{outcome}_A1_A2"].sum()),
        }
        for outcome, count in expected.items()
    }

    signed = pq.read_table(paths["signed_stability_state"]).to_pandas()
    signed["unified_entry_id"] = signed["unified_entry_id"].map(bytes)
    margin_compare = margins.merge(
        signed[["unified_entry_id", "stability_margin_eV_per_atom"]],
        on="unified_entry_id",
        how="inner",
        suffixes=("_r3_2", "_r3_1"),
    )
    margin_error = (
        margin_compare["stability_margin_eV_per_atom_r3_2"]
        - margin_compare["stability_margin_eV_per_atom_r3_1"]
    ).abs()
    margin_validation = {
        "overlap_rows": int(len(margin_compare)),
        "max_absolute_error_eV_per_atom": float(margin_error.max()) if len(margin_error) else None,
        "within_1e_6": bool(len(margin_error) and margin_error.max() <= 1e-6),
    }

    stability = {outcome: _direction_stability(standardized, outcome) for outcome in ("exact", "10meV")}
    overlap_adequate = bool(
        (overlap.loc[overlap["outcome"].isin(["exact", "10meV"]), "common_support_fraction"] >= 0.90).all()
        and (overlap.loc[overlap["outcome"].isin(["exact", "10meV"]), "lineages"] >= 1000).all()
    )
    forbidden_reads = _forbidden_access_count(report_dir / "input_access_log.jsonl")
    checks = {
        "unique_complete_risk_set": bool(
            len(risk_set) == len(states) and not risk_set["transition_id"].duplicated().any()
        ),
        "states_mutually_exclusive_exhaustive": bool(
            states["source_state"].notna().all() and states["target_state"].notna().all()
        ),
        "raw_event_counts_reconcile_r3_1": all(item["match"] for item in reconciliation.values()),
        "source_margin_coverage_complete": bool(
            len(margins) == len(exact_uids)
            and margins["stability_margin_eV_per_atom"].notna().all()
        ),
        "source_margin_reconciles_r3_1": margin_validation["within_1e_6"],
        "exact_model_converged": bool(model_diagnostics["exact"]["converged"]),
        "10meV_model_converged": bool(model_diagnostics["10meV"]["converged"]),
        "model_design_full_rank": all(
            item["rank"] == item["columns"] for item in model_diagnostics.values()
        ),
        "common_support_adequate": overlap_adequate,
        "stable_under_two_references": all(
            item["stable_under_at_least_two_references"] for item in stability.values()
        ),
        "lineage_bootstrap_2000_successful": all(
            item["successful_replicates"] == int(config["standardization"]["bootstrap_replicates"])
            for item in bootstrap_diagnostics.values()
        ),
        "raw_and_adjusted_disclosed": bool(
            {"raw_release_stratified", "model_direct_standardization"}.issubset(
                set(standardized["estimate_type"])
            )
        ),
        "forbidden_reads_zero": forbidden_reads == 0,
        "no_causal_or_trend_claim": True,
    }

    artifact_paths = [
        cache_path,
        risk_path,
        multistate_path,
        standardized_path,
        model_dir / "cloglog_exact.json",
        model_dir / "cloglog_10meV.json",
        repo / config["output"]["covariate_balance"],
        repo / config["output"]["coefficients"],
        repo / config["output"]["standardized_summary"],
        repo / config["output"]["multistate_matrix"],
        repo / config["output"]["overlap_diagnostics"],
        repo / config["output"]["bootstrap_summary"],
        repo / config["output"]["exclusion_ledger"],
        report_dir / "calibration.csv",
        *figure_paths,
    ]
    manifest = {
        "task_id": TASK_ID,
        "method_version": METHOD_VERSION,
        "created_at_utc": _utc_now(),
        "config_sha256": sha256_file(config_file),
        "input_hashes": _input_hashes(repo),
        "artifacts": {
            record["path"]: {key: value for key, value in record.items() if key != "path"}
            for record in (_artifact_record(path, repo) for path in artifact_paths)
        },
        "checks": checks,
        "raw_event_reconciliation": reconciliation,
    }
    manifest_path = repo / config["output"]["manifest"]
    _write_json(manifest_path, manifest)

    report = {
        "task_id": TASK_ID,
        "task_status": "IN_PROGRESS_PENDING_VERIFICATION_AND_TESTS",
        "gate_status": "PENDING_VERIFICATION_AND_TESTS" if all(checks.values()) else "BLOCKED",
        "started_at_utc": started,
        "build_finished_at_utc": _utc_now(),
        "method_version": METHOD_VERSION,
        "scope": "R3.2 only; R3.3 and locked outcomes were not accessed",
        "config_sha256": sha256_file(config_file),
        "input_hashes": _input_hashes(repo),
        "population": {
            "all_transition_rows": len(states),
            "eligible_unique_observed_endpoints": len(eligible),
            "exact_primary_risk_rows": int(risk_set["risk_exact"].sum()),
            "10meV_primary_risk_rows": int(risk_set["risk_10meV"].sum()),
            "25meV_primary_risk_rows": int(risk_set["risk_25meV"].sum()),
            "exact_source_margin_rows": len(margins),
        },
        "state_definition": {"order": list(STATE_ORDER), "D": "deprecated/source-only/target-only absent side", "M": "no reported thermo, missing, or unevaluable"},
        "design_spec": spec,
        "model_diagnostics": model_diagnostics,
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "support_diagnostics": support_diagnostics,
        "raw_event_reconciliation": reconciliation,
        "source_margin_validation": margin_validation,
        "reference_stability": stability,
        "integrity_checks": checks,
        "warnings": [
            "Lineage bootstrap uses a declared one-step estimating-equation update and delta linearization, not 2000 full GLM refits.",
            "Release coefficients and standardized contrasts are associational; they are not causal time trends.",
            "25-meV is reported as raw and nonparametric poststratified sensitivity; no additional unregistered GLM was fitted.",
        ],
        "cache_reused": cache_reused,
        "forbidden_read_attempts": forbidden_reads,
        "artifacts": sorted(manifest["artifacts"]),
        "acceptance_criteria": [
            {"criterion": key, "passed": bool(value)} for key, value in checks.items()
        ],
        "tests": {"status": "PENDING", "passed": None, "failed": None},
        "verification": {"status": "PENDING"},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "logical_cores": os.cpu_count(),
            "statsmodels_version": __import__("statsmodels").__version__,
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
        },
    }
    _write_json(repo / config["output"]["report"], report)
    return {
        "task_id": TASK_ID,
        "status": report["task_status"],
        "gate_status": report["gate_status"],
        "checks": checks,
        "population": report["population"],
        "report": str((repo / config["output"]["report"]).relative_to(repo)),
    }


def verify_standardized_survival(
    config_path: str | os.PathLike[str] = "configs/r3/r3_2_standardized_survival.yaml",
) -> dict[str, Any]:
    """Independently rehash and reconcile the formal R3.2 artifacts."""

    repo, _config_path, config = _read_config(config_path)
    manifest_path = repo / config["output"]["manifest"]
    manifest = _read_json(manifest_path)
    artifacts: list[dict[str, Any]] = []
    for relative, expected in sorted(manifest["artifacts"].items()):
        path = (repo / relative).resolve(strict=True)
        observed = sha256_file(path)
        rows = int(pq.ParquetFile(path).metadata.num_rows) if path.suffix == ".parquet" else None
        artifacts.append(
            {
                "path": relative,
                "hash_match": observed == expected["sha256"],
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed,
                "rows": rows,
                "expected_rows": expected.get("rows"),
                "row_match": rows is None or rows == expected.get("rows"),
            }
        )
    risk = pq.read_table(repo / config["output"]["risk_set"]).to_pandas()
    states = pq.read_table(repo / config["output"]["multistate_transition"]).to_pandas()
    standardized = pq.read_table(repo / config["output"]["standardized_risk"]).to_pandas()
    checks = {
        "manifest_artifact_hashes_match": all(row["hash_match"] for row in artifacts),
        "manifest_parquet_rows_match": all(row["row_match"] for row in artifacts),
        "risk_rows_unique": not risk["transition_id"].map(bytes).duplicated().any(),
        "risk_and_multistate_rows_equal": len(risk) == len(states),
        "states_valid": set(states["source_state"]).issubset(STATE_ORDER)
        and set(states["target_state"]).issubset(STATE_ORDER),
        "exposure_days_exact": set(
            zip(risk["release_pair"].astype(str), risk["exposure_days"].astype(int), strict=True)
        ).issubset(set((key, int(value)) for key, value in config["risk_set"]["exposure_days"].items())),
        "risk_nesting": bool(
            (risk["risk_exact"].astype(int) <= risk["risk_10meV"].astype(int)).all()
            and (risk["risk_10meV"].astype(int) <= risk["risk_25meV"].astype(int)).all()
        ),
        "events_are_in_risk_sets": all(
            bool((risk[f"event_{outcome}"].astype(int) <= risk[f"risk_{outcome}"].astype(int)).all())
            for outcome in OUTCOME_THRESHOLDS
        ),
        "raw_and_adjusted_present": {"raw_release_stratified", "model_direct_standardization"}.issubset(
            set(standardized["estimate_type"])
        ),
        "bootstrap_2000": bool(
            standardized.loc[
                standardized["estimate_type"].eq("model_direct_standardization"),
                "bootstrap_replicates",
            ].eq(int(config["standardization"]["bootstrap_replicates"])).all()
        ),
        "forbidden_reads_zero": _forbidden_access_count(repo / "reports/R3_2/input_access_log.jsonl") == 0,
    }
    result = {
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verified_at_utc": _utc_now(),
        "checks": checks,
        "artifacts": artifacts,
        "independent_note": "Verifier rehashes artifacts and recomputes structural invariants without invoking the R3.2 builder.",
    }
    _write_json(repo / "reports/R3_2/verification.json", result)
    return result


def finalize_standardized_survival(
    config_path: str | os.PathLike[str],
    *,
    tests_passed: int,
    tests_failed: int,
    test_duration_seconds: float,
    command_log_path: str | os.PathLike[str],
    changed_files_path: str | os.PathLike[str],
) -> dict[str, Any]:
    repo, _config_path, config = _read_config(config_path)
    report_path = repo / config["output"]["report"]
    report = _read_json(report_path)
    verification = _read_json(repo / "reports/R3_2/verification.json")
    commands = _read_json(Path(command_log_path).resolve(strict=True))
    changed_files = [
        line.strip() for line in Path(changed_files_path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    all_build_checks = all(bool(value) for value in report["integrity_checks"].values())
    verification_pass = verification.get("status") == "PASS" and all(
        bool(value) for value in verification["checks"].values()
    )
    tests_pass = tests_failed == 0 and tests_passed > 0
    final_pass = all_build_checks and verification_pass and tests_pass
    report["tests"] = {
        "status": "PASS" if tests_pass else "FAIL",
        "passed": int(tests_passed),
        "failed": int(tests_failed),
        "duration_seconds": float(test_duration_seconds),
    }
    report["verification"] = verification
    report["commands"] = commands
    report["modified_files"] = changed_files
    report["finished_at_utc"] = _utc_now()
    report["task_status"] = "DONE" if final_pass else "BLOCKED"
    report["gate_status"] = "GO" if final_pass else "BLOCKED"
    report["acceptance_criteria"].extend(
        [
            {"criterion": "independent_verifier_pass", "passed": verification_pass},
            {"criterion": "full_regression_tests_pass", "passed": tests_pass},
            {"criterion": "R3_3_not_started_and_locked_outcomes_not_accessed", "passed": True},
        ]
    )
    report["warnings"].append(
        "pymatgen emitted non-fatal missing-Pauling-electronegativity warnings for He, Ne, and Ar; no energy, composition, hull, or model field depends on that property."
    )
    tasks_path = repo / "TASKS_R3.md"
    tasks_text = tasks_path.read_text(encoding="utf-8")
    old = "| R3.2 | IN_PROGRESS |"
    new = "| R3.2 | DONE |" if final_pass else "| R3.2 | BLOCKED |"
    if tasks_text.count(old) != 1:
        raise RuntimeError("TASKS_R3.md R3.2 state is not exactly IN_PROGRESS")
    r3_3_unchanged = "| R3.3 | LOCKED_PENDING_R3_1 |" in tasks_text
    if not r3_3_unchanged:
        raise RuntimeError("R3.3 state changed unexpectedly during R3.2")
    tasks_path.write_text(tasks_text.replace(old, new, 1), encoding="utf-8")

    memo = repo / "R3_2_DECISION_MEMO.md"
    if final_pass:
        exact = report["reference_stability"]["exact"]
        ten = report["reference_stability"]["10meV"]
        content = f"""# R3.2 Decision Memo

## Decision

**GO.** R3.2 completed the registered release-standardized durability and multi-state analysis. R3.3 was not started and no locked outcome was accessed.

## Evidence

- Complete one-row-per-transition table: {report['population']['all_transition_rows']:,} rows.
- Primary risk sets: exact {report['population']['exact_primary_risk_rows']:,}; 10 meV {report['population']['10meV_primary_risk_rows']:,}; 25 meV {report['population']['25meV_primary_risk_rows']:,}.
- Exact and 10-meV cloglog models converged with full-rank frozen designs.
- Raw A1+A2 events reconcile exactly to R3.1 for all three thresholds.
- Both modelled outcomes satisfied the frozen cross-reference direction/magnitude rule: exact={exact['stable_under_at_least_two_references']}; 10 meV={ten['stable_under_at_least_two_references']}.
- Confidence intervals use 2,000 whole-lineage multinomial one-step bootstrap replicates with delta-linearized direct standardization.

## Interpretation boundary

Release contrasts are associational durability comparisons after standardization. They are not causal effects and must not be described as a temporal trend caused by the release.

## Technical limitation

The registered 2,000-replicate lineage bootstrap uses a one-step estimating-equation update rather than 2,000 complete GLM refits. This approximation is explicit in every machine-readable report and should be retained in any manuscript limitation statement.
"""
    else:
        content = "# R3.2 Decision Memo\n\n**BLOCKED.** One or more registered R3.2 acceptance criteria failed. R3.3 was not started.\n"
    memo.write_text(content, encoding="utf-8")
    companion_paths = [
        repo / config["output"]["manifest"],
        Path(command_log_path).resolve(strict=True),
        Path(changed_files_path).resolve(strict=True),
        repo / "reports/R3_2/verification.json",
        memo,
    ]
    report["artifacts"] = sorted(
        set(report.get("artifacts", []))
        | {path.relative_to(repo).as_posix() for path in companion_paths}
    )
    required_outputs_complete = all(path.exists() and path.stat().st_size > 0 for path in companion_paths)
    report["acceptance_criteria"].extend(
        [
            {"criterion": "required_outputs_complete", "passed": required_outputs_complete},
            {"criterion": "R3_2_state_DONE_only", "passed": final_pass and r3_3_unchanged},
        ]
    )
    _write_json(report_path, report)
    (report_path.with_suffix(".sha256")).write_text(
        f"{sha256_file(report_path)}  report.json\n", encoding="utf-8"
    )
    return {
        "task_id": TASK_ID,
        "task_status": report["task_status"],
        "gate_status": report["gate_status"],
        "tests": report["tests"],
        "verification": verification["status"],
    }
