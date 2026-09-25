"""Scoped R3.4B-S GNoME database-state counterfactual.

The computation removes frozen provenance-classified database entries while
holding every retained energy, correction, workflow, phase context, and
candidate representative fixed.  It is not a physical intervention or a
causal estimate of a discovery program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .common import open_formal_input, sha256_file
from .competitor_cascade import (
    _deduplicate_contextual_mirrors,
    _load_relevant_contexts,
    solve_hull_distance,
)
from .energy_amplitude import _records_with_elemental_terminals


TASK_ID = "R3.4B-S"
METHOD_VERSION = "PHASEEVONET_R3_4B_S_GNOME_COUNTERFACTUAL_V1"
SCOPES = {
    "A_ONLY": frozenset({"A"}),
    "A_PLUS_B": frozenset({"A", "B"}),
}


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
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def _hex(value: object) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)


def _bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return bytes.fromhex(str(value))


def _read_config(config_path: str | os.PathLike[str]) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve(strict=True)
    repo = path.parents[2]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID or config.get("method_version") != METHOD_VERSION:
        raise RuntimeError("R3.4B-S task or method identifier changed")
    if config["route"]["upstream_r3_3r_b"] != "ROUTE_BOUNDED":
        raise RuntimeError("R3.4B-S must preserve the frozen ROUTE_BOUNDED route")
    if any(str(item).casefold() in {"gini", "hhi", "top_share", "hub_ranking"}
           for item in config["route"]["allowed_outputs"]):
        raise RuntimeError("network concentration output was improperly authorized")
    return repo, path, config


def _authorize_inputs(
    repo: Path, config: dict[str, Any], access_log: Path
) -> tuple[dict[str, Path], pd.DataFrame]:
    paths: dict[str, Path] = {}
    audit: list[dict[str, Any]] = []
    if access_log.exists():
        access_log.unlink()
    for name, relative in config["input"].items():
        path = (repo / relative).resolve(strict=True)
        expected = str(config["expected_sha256"][name])
        with open_formal_input(
            path,
            expected,
            task_id=TASK_ID,
            access_log=access_log,
            purpose=f"R3.4B-S formal input: {name}",
            allowed_roots=[repo],
            caller="phase_evonet.r3.gnome_counterfactual.build",
        ):
            pass
        observed = sha256_file(path)
        audit.append(
            {
                "input_name": name,
                "path": path.relative_to(repo).as_posix(),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "bytes": path.stat().st_size,
                "status": "MATCH",
            }
        )
        paths[name] = path
    return paths, pd.DataFrame(audit)


def _validate_prerequisites(paths: Mapping[str, Path]) -> dict[str, Any]:
    r3_4a = json.loads(paths["r3_4a_report"].read_text(encoding="utf-8"))
    verify_a = json.loads(paths["r3_4a_verification"].read_text(encoding="utf-8"))
    attestation = json.loads(paths["r3_4a_attestation"].read_text(encoding="utf-8"))
    manifest_a = json.loads(paths["r3_4a_manifest"].read_text(encoding="utf-8"))
    route = json.loads(paths["r3_3r_b_report"].read_text(encoding="utf-8"))
    api_probe = json.loads(paths["api_probe"].read_text(encoding="utf-8"))
    checks = {
        "r3_4a_go": r3_4a.get("gate_status") == "GO" and r3_4a.get("task_status") in {"DONE", "GO"},
        "r3_4a_verifier_pass": verify_a.get("status") == "PASS",
        "human_review_complete": bool(attestation.get("reviewed_all_rows")) and int(attestation.get("review_rows", 0)) >= 400,
        "human_review_zero_disagreements": int(attestation.get("disagreements", -1)) == 0,
        "license_attested": bool(attestation.get("license_acceptance_attested")),
        "r3_4b_s_eligible": bool(manifest_a.get("r3_4b_s_eligible")),
        "route_bounded": route.get("gate_status") == "ROUTE_BOUNDED",
        "api_access": bool(api_probe.get("summary_probe", {}).get("access_granted")),
        "api_key_not_logged": api_probe.get("api_key_logged") is False,
        "api_batch_nonempty": int(api_probe.get("summary_probe", {}).get("total_doc", 0)) > 0,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise RuntimeError(f"R3.4B-S prerequisite failed: {failed}")
    return {"checks": checks, "api_total_doc": int(api_probe["summary_probe"]["total_doc"])}


def select_primary_population(exact: pd.DataFrame, expected_rows: int = 1190) -> pd.DataFrame:
    mask = (
        exact["direction"].eq("stable_to_unstable")
        & exact["survives_10meV"]
        & exact["identity_confidence"].eq("A1")
        & exact["candidate_identity_unchanged"]
        & exact["same_workflow"]
        & exact["same_phase_context"]
    )
    frame = exact.loc[mask].copy()
    if len(frame) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} frozen primary rows, observed {len(frame)}")
    frame["transition_id"] = frame["transition_id"].map(_hex)
    frame["target_unified_entry_id"] = frame["target_unified_entry_id"].map(_hex)
    return frame.sort_values("transition_id").reset_index(drop=True)


def _stability_class(value: object, exact_tolerance: float, primary_threshold: float) -> str:
    if value is None or pd.isna(value):
        return "missing"
    number = float(value)
    if number <= exact_tolerance:
        return "stable"
    if number <= primary_threshold:
        return "near_hull"
    return "metastable"


def _retained_value_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    values = [
        (
            _hex(row["unified_entry_id"]),
            float(row["uncorrected_energy"]),
            float(row["correction"]),
            float(row["corrected_energy"]),
            str(row["source_workflow"]),
        )
        for row in rows
    ]
    payload = json.dumps(sorted(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _relation_inputs(
    primary: pd.DataFrame,
    classification: pd.DataFrame,
    changes: pd.DataFrame,
    identity_edges: pd.DataFrame,
) -> pd.DataFrame:
    for frame in (classification, changes, identity_edges):
        frame["transition_id"] = frame["transition_id"].map(_hex)
    for frame in (changes, identity_edges):
        for column in ("source_unified_entry_id", "target_unified_entry_id"):
            if column in frame:
                frame[column + "_hex"] = frame[column].map(
                    lambda value: None if value is None or pd.isna(value) else _hex(value)
                )
    tids = set(primary["transition_id"])
    changes = changes.loc[changes["transition_id"].isin(tids)].copy()
    identity_columns = [
        "transition_id", "competitor_contextual_id", "id0_contextual",
        "id1_strict_entry", "id2_lineage_thermo_workflow", "id3_lineage_only",
    ]
    identity_map = identity_edges[identity_columns].drop_duplicates()
    identity_multiplicity = identity_map.groupby(
        ["transition_id", "competitor_contextual_id"], sort=False
    ).size()
    if int(identity_multiplicity.max()) != 1:
        raise RuntimeError("one competitor relation maps to conflicting R3.3R identity definitions")
    relation = classification.merge(
        changes,
        on=["transition_id", "competitor_contextual_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_provenance", "_change"),
    )
    relation = relation.merge(
        identity_map,
        on=["transition_id", "competitor_contextual_id"],
        how="left",
        validate="one_to_one",
    )
    if len(relation) != len(changes):
        raise RuntimeError("provenance-to-change relation join did not preserve rows")
    return relation.sort_values(["transition_id", "competitor_contextual_id"]).reset_index(drop=True)


def _records_by_context(
    paths: Mapping[str, Path], primary: pd.DataFrame
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[tuple[str, str, str], list[dict[str, Any]]],
]:
    needed = {
        (str(row.source_snapshot), str(row.thermo_type), str(row.phase_context_chemsys))
        for row in primary.itertuples(index=False)
    } | {
        (str(row.target_snapshot), str(row.thermo_type), str(row.phase_context_chemsys))
        for row in primary.itertuples(index=False)
    }
    return _load_relevant_contexts(paths["phase_entries"], needed)


def _context_rows(
    key: tuple[str, str, str],
    contexts: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    terminals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _records_with_elemental_terminals(key, contexts[key], dict(terminals))
    deduplicated, aliases = _deduplicate_contextual_mirrors(
        records,
        candidate_context=key[2],
        preferred_ids=set(),
        tolerance=tolerance,
    )
    return deduplicated, aliases


def freeze_matched_design(
    relation: pd.DataFrame,
    primary: pd.DataFrame,
    context_cache: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    contexts: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze deterministic nearest controls using pre-outcome covariates only."""

    prohibited = set(config["matching"]["prohibited_variables"])
    used = set(config["matching"]["exact_strata"]) | set(config["matching"]["numeric_covariates"])
    if prohibited & used:
        raise RuntimeError("a prohibited post-outcome matching variable was requested")
    primary_meta = primary.set_index("transition_id")
    rows: list[dict[str, Any]] = []
    for unit_id, group in relation.groupby("competitor_contextual_id", sort=True):
        first = group.iloc[0]
        if str(first["change_type"]) != "target_only_arrival":
            continue
        target_uid = str(first["target_unified_entry_id_hex"])
        target_key = (
            str(first["target_snapshot"]),
            str(first["thermo_type_change"]),
            str(first["phase_context_chemsys_change"]),
        )
        candidates = [row for row in context_cache[target_key] if _hex(row["unified_entry_id"]) == target_uid]
        if len(candidates) != 1:
            raise RuntimeError(f"arrival unit {unit_id} did not resolve to one contextual entry")
        entry = candidates[0]
        transition = primary_meta.loc[str(first["transition_id"])]
        source_key = (
            str(transition.source_snapshot),
            str(transition.thermo_type),
            str(transition.phase_context_chemsys),
        )
        source_records = contexts[source_key]
        provenance_completeness = sum(
            bool(first.get(name, False))
            for name in ("raw_record_found", "raw_provenance_present", "source_manifest_match")
        )
        rows.append(
            {
                "unit_id": str(unit_id),
                "provenance_class": str(first["provenance_class"]),
                "target_snapshot": target_key[0],
                "thermo_type": target_key[1],
                "phase_context_chemsys": target_key[2],
                "target_unified_entry_id": target_uid,
                "entry_stability_class_at_arrival": _stability_class(
                    entry.get("energy_above_hull"),
                    float(config["counterfactual"]["exact_zero_tolerance_eV_per_atom"]),
                    float(config["population"]["primary_threshold_eV_per_atom"]),
                ),
                "chemical_dimensionality": len(target_key[2].split("-")),
                "composition_complexity": len(json.loads(str(entry["composition_json"]))),
                "source_context_entry_count": len(source_records),
                "source_context_competitor_count": sum(bool(row["is_competitor"]) for row in source_records),
                "provenance_completeness": provenance_completeness,
                "relation_count": int(group["transition_id"].nunique()),
                "role": "TREATMENT" if str(first["provenance_class"]) == "A" else "CONTROL_POOL",
            }
        )
    units = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    treatment = units.loc[units["role"].eq("TREATMENT")].copy()
    controls = units.loc[units["role"].eq("CONTROL_POOL")].copy()
    numeric = list(config["matching"]["numeric_covariates"])
    exact = list(config["matching"]["exact_strata"])
    means = units[numeric].astype(float).mean()
    scales = units[numeric].astype(float).std(ddof=0).replace(0.0, 1.0)
    z = (units[numeric].astype(float) - means) / scales
    z.index = units["unit_id"]
    control_groups = {key: frame for key, frame in controls.groupby(exact, dropna=False, sort=True)}
    design_rows: list[dict[str, Any]] = []
    caliper = float(config["matching"]["distance_caliper"])
    for row in treatment.itertuples(index=False):
        key_values = tuple(getattr(row, name) for name in exact)
        key: object = key_values[0] if len(key_values) == 1 else key_values
        pool = control_groups.get(key)
        if pool is None or pool.empty:
            design_rows.append({
                "pair_id": f"pair-{row.unit_id}", "treatment_unit_id": row.unit_id,
                "control_unit_id": None, "distance": None, "matched": False,
                "unmatched_reason": "no_exact_stratum_control",
            })
            continue
        left = z.loc[row.unit_id].to_numpy(dtype=float)
        distances = []
        for control_id in pool["unit_id"]:
            distance = float(np.linalg.norm(left - z.loc[control_id].to_numpy(dtype=float)))
            distances.append((distance, str(control_id)))
        distance, control_id = min(distances, key=lambda item: (item[0], item[1]))
        matched = distance <= caliper
        design_rows.append({
            "pair_id": f"pair-{row.unit_id}", "treatment_unit_id": row.unit_id,
            "control_unit_id": control_id if matched else None,
            "distance": distance, "matched": matched,
            "unmatched_reason": None if matched else "distance_caliper",
        })
    design = pd.DataFrame(design_rows).sort_values("pair_id").reset_index(drop=True)
    design = design.merge(
        treatment.add_prefix("treatment_"),
        left_on="treatment_unit_id", right_on="treatment_unit_id", how="left", validate="one_to_one",
    )
    matched_controls = units.add_prefix("control_")
    design = design.merge(
        matched_controls,
        left_on="control_unit_id", right_on="control_unit_id", how="left", validate="many_to_one",
    )
    diagnostics: list[dict[str, Any]] = [
        {"diagnostic": "treatment_units", "value": len(treatment)},
        {"diagnostic": "control_pool_units", "value": len(controls)},
        {"diagnostic": "matched_units", "value": int(design["matched"].sum())},
        {"diagnostic": "matched_fraction", "value": float(design["matched"].mean())},
        {"diagnostic": "with_replacement", "value": bool(config["matching"]["with_replacement"])},
    ]
    matched = design.loc[design["matched"]].copy()
    for name in numeric:
        left = matched[f"treatment_{name}"].astype(float)
        right = matched[f"control_{name}"].astype(float)
        pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0) if len(matched) > 1 else 0.0
        smd = 0.0 if pooled == 0 else float((left.mean() - right.mean()) / pooled)
        diagnostics.append({"diagnostic": f"absolute_smd:{name}", "value": abs(smd)})
    return design, pd.DataFrame(diagnostics)


def _threshold_reversal(full: float, cf: float, threshold: float, exact_tolerance: float) -> bool:
    if threshold == 0.0:
        return full > exact_tolerance and cf <= exact_tolerance
    return full >= threshold and cf < threshold


def _deterministic_validation_sample(states: pd.DataFrame, sample_n: int, seed: int) -> pd.DataFrame:
    base = states.loc[states["scope"].eq("A_ONLY")].copy()
    base["stratum"] = base["thermo_type"].astype(str) + "|" + base["target_snapshot"].astype(str)
    ranked: list[pd.DataFrame] = []
    for _, group in base.groupby("stratum", sort=True):
        take = max(1, int(round(sample_n * len(group) / len(base))))
        group = group.assign(
            _rank=group["transition_id"].map(
                lambda value: hashlib.blake2b(
                    f"{seed}|{value}".encode("utf-8"), digest_size=16
                ).hexdigest()
            )
        ).sort_values(["_rank", "transition_id"])
        ranked.append(group.head(take))
    sample = pd.concat(ranked, ignore_index=True).sort_values(["_rank", "transition_id"])
    if len(sample) < sample_n:
        remaining = base.loc[~base["transition_id"].isin(sample["transition_id"])].assign(
            _rank=lambda frame: frame["transition_id"].map(
                lambda value: hashlib.blake2b(
                    f"{seed}|{value}".encode("utf-8"), digest_size=16
                ).hexdigest()
            )
        ).sort_values(["_rank", "transition_id"])
        sample = pd.concat([sample, remaining.head(sample_n - len(sample))], ignore_index=True)
    return sample.head(sample_n)[
        ["transition_id", "target_snapshot", "thermo_type", "phase_context_chemsys",
         "target_unified_entry_id", "full_e_hull_eV_per_atom", "counterfactual_e_hull_eV_per_atom",
         "removed_entry_ids_json"]
    ].sort_values("transition_id").reset_index(drop=True)


def _bootstrap_matched(
    design: pd.DataFrame,
    relation_outcomes: pd.DataFrame,
    thresholds: Sequence[float],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    matched = design.loc[design["matched"], ["pair_id", "treatment_unit_id", "control_unit_id"]]
    lineages = sorted(relation_outcomes["candidate_lineage_id"].unique())
    lineage_index = {value: index for index, value in enumerate(lineages)}
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros((replicates, len(lineages)), dtype=np.int16)
    for index in range(replicates):
        draw = rng.integers(0, len(lineages), size=len(lineages))
        multiplicities[index] = np.bincount(draw, minlength=len(lineages))
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        column = f"single_reversal_{int(round(threshold * 1000))}meV" if threshold else "single_reversal_exact"
        unit_lineage = relation_outcomes.pivot_table(
            index="unit_id", columns="candidate_lineage_id", values=column,
            aggfunc="sum", fill_value=0.0,
        ).reindex(columns=lineages, fill_value=0.0)
        coefficients = np.zeros(len(lineages), dtype=float)
        observed_differences: list[float] = []
        for row in matched.itertuples(index=False):
            treated = unit_lineage.loc[row.treatment_unit_id].to_numpy(dtype=float) if row.treatment_unit_id in unit_lineage.index else np.zeros(len(lineages))
            control = unit_lineage.loc[row.control_unit_id].to_numpy(dtype=float) if row.control_unit_id in unit_lineage.index else np.zeros(len(lineages))
            difference = treated - control
            coefficients += difference
            observed_differences.append(float(difference.sum()))
        coefficients /= max(1, len(matched))
        values = multiplicities @ coefficients
        output.append({
            "threshold_eV_per_atom": threshold,
            "matched_pairs": len(matched),
            "treated_mean_reversals_per_entry": float(np.mean([
                unit_lineage.loc[row.treatment_unit_id].to_numpy(dtype=float).sum()
                if row.treatment_unit_id in unit_lineage.index else 0.0
                for row in matched.itertuples(index=False)
            ])),
            "control_mean_reversals_per_entry": float(np.mean([
                unit_lineage.loc[row.control_unit_id].to_numpy(dtype=float).sum()
                if row.control_unit_id in unit_lineage.index else 0.0
                for row in matched.itertuples(index=False)
            ])),
            "mean_difference": float(np.mean(observed_differences)),
            "lineage_stability_interval_lower": float(np.quantile(values, 0.025)),
            "lineage_stability_interval_upper": float(np.quantile(values, 0.975)),
            "interval_label": "paired fixed-design candidate-lineage stability interval; not a causal population CI",
        })
    return pd.DataFrame(output)


def _make_figures(report_dir: Path, summary: pd.DataFrame, edges: pd.DataFrame) -> list[Path]:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = summary.loc[summary["scope"].eq("A_ONLY")].sort_values("threshold_eV_per_atom")
    labels = ["exact" if value == 0 else f"{int(value * 1000)} meV" for value in primary["threshold_eV_per_atom"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(labels, primary["reversed_transitions"], color="#3478a9")
    ax.set_ylabel("candidate transitions reversed")
    ax.set_xlabel("counterfactual threshold")
    ax.set_title("A-only GNoME database-state counterfactual")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    waterfall = figure_dir / "figure_r3_4_counterfactual_waterfall.png"
    fig.savefig(waterfall, dpi=200)
    plt.close(fig)

    counts = edges.loc[edges["provenance_class"].eq("A")].groupby("id1_strict_entry")["single_reversal_10meV"].sum()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    maximum = max(1, int(counts.max()) if len(counts) else 1)
    ax.hist(counts.to_numpy(dtype=float), bins=np.arange(-0.5, maximum + 1.5, 1.0), color="#cc6b49", edgecolor="white")
    ax.set_xlabel("10-meV candidate reversals per strict entry")
    ax.set_ylabel("entry count")
    ax.set_title("Relation-count distribution (no hub ranking)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    cascades = figure_dir / "figure_r3_4_gnome_cascades.png"
    fig.savefig(cascades, dpi=200)
    plt.close(fig)
    return [waterfall, cascades]


def _public_release_inventory(repo: Path, report_dir: Path, outputs: Sequence[Path]) -> pd.DataFrame:
    public_names = {
        "counterfactual_summary.csv", "matched_comparison.csv", "overlap_diagnostics.csv",
        "license_safe_release_inventory.csv", "public_release_summary.json",
        "figure_r3_4_counterfactual_waterfall.png", "figure_r3_4_gnome_cascades.png",
    }
    rows = []
    for path in outputs:
        classification = "PUBLIC_AGGREGATE" if path.name in public_names else "RESTRICTED_LOCAL_EVIDENCE"
        rows.append({
            "path": path.relative_to(repo).as_posix(),
            "content_class": classification,
            "public_release_allowed": classification == "PUBLIC_AGGREGATE",
            "contains_raw_gnome_identifier": False if classification == "PUBLIC_AGGREGATE" else "not_scanned_for_release",
            "contains_raw_structure_or_energy": False if classification == "PUBLIC_AGGREGATE" else "restricted_local_possible",
            "required_action": "release aggregate only" if classification == "PUBLIC_AGGREGATE" else "exclude from public bundle",
        })
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scipy", "matplotlib", "pymatgen")
    values = {}
    for name in names:
        try:
            values[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            values[name] = "not-installed"
    return values


def build(
    config_path: str | os.PathLike[str] = "configs/r3/r3_4b_s_gnome_counterfactual.yaml",
) -> dict[str, Any]:
    repo, config_file, config = _read_config(config_path)
    report_dir = repo / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    access_log = report_dir / "input_access_log.jsonl"
    paths, hash_audit = _authorize_inputs(repo, config, access_log)
    _write_csv(report_dir / "input_hash_audit.csv", hash_audit)
    prerequisite = _validate_prerequisites(paths)

    exact = pd.read_parquet(paths["exact_flip_amplitude"])
    primary = select_primary_population(exact, int(config["population"]["expected_rows"]))
    classification = pd.read_parquet(paths["r3_4a_classification"])
    changes = pd.read_parquet(paths["competitor_change"])
    identity_edges = pd.read_parquet(paths["identity_resolved_edge"])
    relation = _relation_inputs(primary, classification, changes, identity_edges)
    a_b = relation.loc[relation["provenance_class"].isin(["A", "B"])]
    if not a_b["change_type"].eq("target_only_arrival").all():
        raise RuntimeError("A/B provenance set contains a non-arrival action")
    if not a_b["eligible_after_human_review"].all():
        raise RuntimeError("A/B provenance set contains a row not cleared by human review")
    if relation.loc[relation["provenance_class"].isin(["C", "U"]), "official_batch_match"].any():
        raise RuntimeError("C/U record unexpectedly matches the frozen official batch")

    print("R3.4B-S loading frozen phase contexts", flush=True)
    contexts, terminals = _records_by_context(paths, primary)
    tolerance = float(config["counterfactual"]["tolerance_eV_per_atom"])
    context_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    alias_rows: list[dict[str, Any]] = []
    target_keys = {
        (str(row.target_snapshot), str(row.thermo_type), str(row.phase_context_chemsys))
        for row in primary.itertuples(index=False)
    }
    for key in sorted(target_keys):
        context_cache[key], aliases = _context_rows(key, contexts, terminals, tolerance)
        alias_rows.extend({"snapshot_id": key[0], "thermo_type": key[1], "phase_context_chemsys": key[2], **row} for row in aliases)

    design_path = repo / config["output"]["matched_design"]
    design_manifest_path = repo / config["output"]["matched_design_manifest"]
    overlap_path = report_dir / "overlap_diagnostics.csv"
    if design_path.exists() and design_manifest_path.exists() and overlap_path.exists():
        design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
        frozen_design_hash = str(design_manifest["design_sha256"])
        if sha256_file(design_path) != frozen_design_hash:
            raise RuntimeError("pre-outcome matched design hash changed")
        design = pd.read_parquet(design_path)
        overlap = pd.read_csv(overlap_path)
    else:
        design, overlap = freeze_matched_design(relation, primary, context_cache, contexts, config)
        _write_parquet(design_path, design)
        design_manifest = {
            "task_id": TASK_ID,
            "method_version": METHOD_VERSION,
            "created_at_utc": _utc_now(),
            "design_path": design_path.relative_to(repo).as_posix(),
            "design_sha256": sha256_file(design_path),
            "rows": len(design),
            "matched_rows": int(design["matched"].sum()),
            "matching_variables": list(config["matching"]["exact_strata"]) + list(config["matching"]["numeric_covariates"]),
            "prohibited_variables": list(config["matching"]["prohibited_variables"]),
            "outcome_fields_read_before_freeze": [],
        }
        _write_json(design_manifest_path, design_manifest)
        (design_manifest_path.parent / "matched_design_manifest.sha256").write_text(
            f"{sha256_file(design_manifest_path)}  {design_manifest_path.name}\n", encoding="ascii"
        )
        frozen_design_hash = sha256_file(design_path)
        _write_csv(overlap_path, overlap)

    primary_index = primary.set_index("transition_id")
    removal_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    removal_classes: dict[tuple[str, str, str], str] = {}
    for row in a_b.itertuples(index=False):
        uid = str(row.target_unified_entry_id_hex)
        removal_map[(str(row.transition_id), str(row.provenance_class))].add(uid)
        removal_classes[(str(row.transition_id), str(row.competitor_contextual_id), uid)] = str(row.provenance_class)

    full_cache: dict[str, dict[str, Any]] = {}
    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    max_baseline_error = 0.0
    max_retained_change = 0
    exact_tolerance = float(config["counterfactual"]["exact_zero_tolerance_eV_per_atom"])
    thresholds = tuple(float(value) for value in config["counterfactual"]["thresholds_eV_per_atom"])

    for index, transition in enumerate(primary.itertuples(index=False), start=1):
        if index == 1 or index % 100 == 0:
            print(f"R3.4B-S group counterfactual {index}/{len(primary)}", flush=True)
        transition_id = str(transition.transition_id)
        key = (str(transition.target_snapshot), str(transition.thermo_type), str(transition.phase_context_chemsys))
        records = context_cache[key]
        candidate_uid = str(transition.target_unified_entry_id)
        candidates = [row for row in records if _hex(row["unified_entry_id"]) == candidate_uid]
        if len(candidates) != 1:
            raise RuntimeError(f"transition {transition_id} candidate did not resolve uniquely")
        candidate = candidates[0]
        competitors = [row for row in records if _hex(row["unified_entry_id"]) != candidate_uid]
        elements = key[2].split("-")
        full, full_status, weights, _ = solve_hull_distance(candidate, competitors, elements)
        baseline_error = abs(full - float(transition.target_energy_above_hull_eV_per_atom))
        max_baseline_error = max(max_baseline_error, baseline_error)
        if baseline_error > tolerance or full_status != "feasible":
            raise RuntimeError(
                f"full-state reconstruction failed for {transition_id}: {full_status}, error={baseline_error}"
            )
        weight_by_uid = {
            _hex(row["unified_entry_id"]): float(weight)
            for row, weight in zip(competitors, weights, strict=True)
        }
        full_cache[transition_id] = {
            "full": full, "status": full_status, "key": key, "candidate_uid": candidate_uid,
            "candidate": candidate, "competitors": competitors, "weight_by_uid": weight_by_uid,
        }
        for scope, included in SCOPES.items():
            requested = set().union(*(removal_map.get((transition_id, label), set()) for label in included))
            candidate_excluded = candidate_uid in requested
            if candidate_excluded:
                requested.remove(candidate_uid)
                exclusion_rows.append({
                    "transition_id": transition_id, "scope": scope, "stage": "group_removal",
                    "reason": "evaluated_candidate_protected", "unified_entry_id": candidate_uid,
                    "resolved": True,
                })
            present = {str(uid) for uid in requested if any(_hex(row["unified_entry_id"]) == uid for row in competitors)}
            missing = requested - present
            for uid in sorted(missing):
                exclusion_rows.append({
                    "transition_id": transition_id, "scope": scope, "stage": "group_removal",
                    "reason": "classified_entry_missing_from_rebuilt_context", "unified_entry_id": uid,
                    "resolved": False,
                })
            retained = [row for row in competitors if _hex(row["unified_entry_id"]) not in present]
            before_hash = _retained_value_hash(retained)
            cf, cf_status, _, _ = solve_hull_distance(candidate, retained, elements)
            after_hash = _retained_value_hash(retained)
            values_unchanged = before_hash == after_hash
            max_retained_change = max(max_retained_change, int(not values_unchanged))
            estimable = cf_status == "feasible" and not missing
            if cf_status != "feasible":
                exclusion_rows.append({
                    "transition_id": transition_id, "scope": scope, "stage": "group_removal",
                    "reason": f"counterfactual_{cf_status}", "unified_entry_id": None,
                    "resolved": False,
                })
            state_rows.append({
                "transition_id": transition_id,
                "candidate_lineage_id": str(transition.canonical_lineage_id),
                "identity_confidence": str(transition.identity_confidence),
                "target_snapshot": key[0], "thermo_type": key[1], "phase_context_chemsys": key[2],
                "target_unified_entry_id": candidate_uid,
                "scope": scope,
                "requested_removal_count": len(requested),
                "removed_entry_count": len(present),
                "removed_entry_ids_json": json.dumps(sorted(present), separators=(",", ":")),
                "candidate_protected": candidate_excluded,
                "active_removed_entry_count": sum(weight_by_uid.get(uid, 0.0) > float(config["counterfactual"]["active_amount_tolerance"]) for uid in present),
                "full_e_hull_eV_per_atom": full,
                "stored_full_e_hull_eV_per_atom": float(transition.target_energy_above_hull_eV_per_atom),
                "baseline_error_eV_per_atom": baseline_error,
                "counterfactual_e_hull_eV_per_atom": cf if estimable else np.nan,
                "full_minus_counterfactual_eV_per_atom": full - cf if estimable else np.nan,
                "full_solver_status": full_status,
                "counterfactual_solver_status": cf_status,
                "estimable": estimable,
                "retained_value_hash_before": before_hash,
                "retained_value_hash_after": after_hash,
                "retained_values_unchanged": values_unchanged,
                "method_version": METHOD_VERSION,
            })
            for threshold in thresholds:
                threshold_eligible = (
                    full > exact_tolerance if threshold == 0.0 else full >= threshold
                )
                transition_rows.append({
                    "transition_id": transition_id,
                    "candidate_lineage_id": str(transition.canonical_lineage_id),
                    "target_snapshot": key[0], "thermo_type": key[1], "phase_context_chemsys": key[2],
                    "scope": scope, "threshold_eV_per_atom": threshold,
                    "threshold_eligible": threshold_eligible,
                    "full_e_hull_eV_per_atom": full,
                    "counterfactual_e_hull_eV_per_atom": cf if estimable else np.nan,
                    "reversed_below_threshold": _threshold_reversal(full, cf, threshold, exact_tolerance) if estimable else False,
                    "estimable": estimable, "removed_entry_count": len(present),
                    "method_version": METHOD_VERSION,
                })

    states = pd.DataFrame(state_rows).sort_values(["transition_id", "scope"]).reset_index(drop=True)
    transitions = pd.DataFrame(transition_rows).sort_values(["transition_id", "scope", "threshold_eV_per_atom"]).reset_index(drop=True)
    if states.duplicated(["transition_id", "scope"]).any():
        raise RuntimeError("duplicate counterfactual state key")
    if transitions.duplicated(["transition_id", "scope", "threshold_eV_per_atom"]).any():
        raise RuntimeError("duplicate counterfactual transition key")

    all_treatment_units = set(
        relation.loc[
            relation["provenance_class"].eq("A")
            & relation["change_type"].eq("target_only_arrival"),
            "competitor_contextual_id",
        ]
    )
    matched_units = all_treatment_units | set(design.loc[design["matched"], "control_unit_id"])
    evaluation_relation = relation.loc[
        relation["competitor_contextual_id"].isin(matched_units)
        & relation["change_type"].eq("target_only_arrival")
    ].copy()
    relation_outcomes: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_relation.itertuples(index=False), start=1):
        if index == 1 or index % 500 == 0:
            print(f"R3.4B-S single-entry comparison {index}/{len(evaluation_relation)}", flush=True)
        transition_id = str(row.transition_id)
        cached = full_cache[transition_id]
        uid = str(row.target_unified_entry_id_hex)
        if uid == cached["candidate_uid"]:
            exclusion_rows.append({
                "transition_id": transition_id, "scope": "SINGLE_ENTRY", "stage": "single_removal",
                "reason": "evaluated_candidate_protected", "unified_entry_id": uid, "resolved": True,
            })
            continue
        retained = [entry for entry in cached["competitors"] if _hex(entry["unified_entry_id"]) != uid]
        if len(retained) == len(cached["competitors"]):
            exclusion_rows.append({
                "transition_id": transition_id, "scope": "SINGLE_ENTRY", "stage": "single_removal",
                "reason": "arrival_entry_missing_from_rebuilt_context", "unified_entry_id": uid, "resolved": False,
            })
            continue
        cf, status, _, _ = solve_hull_distance(
            cached["candidate"], retained, cached["key"][2].split("-")
        )
        if status != "feasible":
            exclusion_rows.append({
                "transition_id": transition_id, "scope": "SINGLE_ENTRY", "stage": "single_removal",
                "reason": f"counterfactual_{status}", "unified_entry_id": uid, "resolved": False,
            })
            continue
        relation_outcomes.append({
            "transition_id": transition_id,
            "candidate_lineage_id": str(row.candidate_lineage_id_change),
            "unit_id": str(row.competitor_contextual_id),
            "provenance_class": str(row.provenance_class),
            "thermo_type": str(row.thermo_type_change),
            "phase_context_chemsys": str(row.phase_context_chemsys_change),
            "target_unified_entry_id": uid,
            "id0_contextual": row.id0_contextual,
            "id1_strict_entry": row.id1_strict_entry,
            "id2_lineage_thermo_workflow": row.id2_lineage_thermo_workflow,
            "id3_lineage_only": row.id3_lineage_only,
            "selected_active": bool(row.selected_active),
            "full_e_hull_eV_per_atom": cached["full"],
            "single_counterfactual_e_hull_eV_per_atom": cf,
            "single_effect_eV_per_atom": cached["full"] - cf,
            "single_reversal_exact": _threshold_reversal(cached["full"], cf, 0.0, exact_tolerance),
            "single_reversal_5meV": _threshold_reversal(cached["full"], cf, 0.005, exact_tolerance),
            "single_reversal_10meV": _threshold_reversal(cached["full"], cf, 0.010, exact_tolerance),
            "single_reversal_25meV": _threshold_reversal(cached["full"], cf, 0.025, exact_tolerance),
            "retained_values_unchanged": True,
            "method_version": METHOD_VERSION,
        })
    relation_frame = pd.DataFrame(relation_outcomes).sort_values(["unit_id", "transition_id"]).reset_index(drop=True)
    if relation_frame.duplicated(["unit_id", "transition_id"]).any():
        raise RuntimeError("duplicate matched relation outcome key")
    gnome_edges = relation_frame.loc[relation_frame["provenance_class"].isin(["A", "B"])].copy()

    matched_comparison = _bootstrap_matched(
        design, relation_frame, thresholds,
        int(config["matching"]["bootstrap_replicates"]), int(config["seed"]),
    )
    maximum_absolute_smd = max(
        (float(row.value) for row in overlap.itertuples(index=False) if str(row.diagnostic).startswith("absolute_smd:")),
        default=0.0,
    )
    matched_fraction_value = float(overlap.loc[overlap.diagnostic.eq("matched_fraction"), "value"].iloc[0])
    matched_design_quality_pass = (
        matched_fraction_value >= float(config["matching"]["minimum_matched_fraction"])
        and maximum_absolute_smd <= float(config["matching"]["maximum_absolute_smd"])
    )
    matched_comparison["design_quality_pass"] = matched_design_quality_pass
    matched_comparison["estimability_status"] = (
        "ESTIMABLE" if matched_design_quality_pass else "NONESTIMABLE_COVARIATE_IMBALANCE"
    )
    summary = transitions.groupby(["scope", "threshold_eV_per_atom"], sort=True).agg(
        primary_transitions=("transition_id", "nunique"),
        estimable_transitions=("estimable", "sum"),
        threshold_eligible_transitions=("threshold_eligible", "sum"),
        transitions_with_removals=("removed_entry_count", lambda values: int((values > 0).sum())),
        reversed_transitions=("reversed_below_threshold", "sum"),
    ).reset_index()
    summary["reversal_fraction_primary"] = summary["reversed_transitions"] / summary["primary_transitions"]
    summary["reversal_fraction_threshold_eligible"] = (
        summary["reversed_transitions"] / summary["threshold_eligible_transitions"].replace(0, np.nan)
    )
    summary["route"] = "ROUTE_BOUNDED_CANDIDATE_RELATIONS_ONLY"

    state_path = repo / config["output"]["state"]
    transition_path = repo / config["output"]["transition"]
    edge_path = repo / config["output"]["gnome_edges"]
    relation_path = repo / config["output"].get("matched_relation_outcomes", "data/processed/R3_4B_S/matched_relation_outcomes.parquet")
    entry_outcome_path = repo / config["output"]["matched_entry_outcomes"]
    _write_parquet(state_path, states)
    _write_parquet(transition_path, transitions)
    _write_parquet(edge_path, gnome_edges)
    _write_parquet(relation_path, relation_frame)
    _write_parquet(entry_outcome_path, matched_comparison)
    _write_csv(report_dir / "counterfactual_summary.csv", summary)
    _write_csv(report_dir / "matched_comparison.csv", matched_comparison)
    _write_csv(report_dir / "exclusion_ambiguity_ledger.csv", pd.DataFrame(exclusion_rows, columns=[
        "transition_id", "scope", "stage", "reason", "unified_entry_id", "resolved"
    ]))
    _write_csv(report_dir / "contextual_alias_audit.csv", pd.DataFrame(alias_rows))
    validation_sample = _deterministic_validation_sample(
        states, int(config["counterfactual"]["validation_sample_n"]), int(config["seed"])
    )
    _write_csv(report_dir / "from_scratch_validation_sample.csv", validation_sample)
    figure_paths = _make_figures(report_dir, summary, gnome_edges)

    public_summary_path = report_dir / "public_release_summary.json"
    _write_json(public_summary_path, {
        "task_id": TASK_ID,
        "interpretation": "deterministic database-state counterfactual; not physical causality",
        "route": "SCOPED",
        "primary_population": len(primary),
        "aggregate_results": summary.to_dict(orient="records"),
        "restricted_identifiers_included": False,
    })

    output_paths = [
        state_path, transition_path, edge_path, relation_path, entry_outcome_path,
        design_path, design_manifest_path, report_dir / "counterfactual_summary.csv",
        report_dir / "matched_comparison.csv", report_dir / "overlap_diagnostics.csv",
        report_dir / "exclusion_ambiguity_ledger.csv", report_dir / "contextual_alias_audit.csv",
        report_dir / "from_scratch_validation_sample.csv", public_summary_path, *figure_paths,
    ]
    release_inventory_path = report_dir / "license_safe_release_inventory.csv"
    release_inventory = _public_release_inventory(repo, report_dir, [*output_paths, release_inventory_path])
    _write_csv(release_inventory_path, release_inventory)
    output_paths.append(release_inventory_path)

    public_text_files = [path for path in output_paths if path.name in {
        "counterfactual_summary.csv", "matched_comparison.csv", "overlap_diagnostics.csv", "public_release_summary.json"
    }]
    raw_pattern = re.compile(str(config["license"]["public_identifier_pattern"]), re.IGNORECASE)
    public_identifier_hits = sum(
        len(raw_pattern.findall(path.read_text(encoding="utf-8"))) for path in public_text_files
    )
    if public_identifier_hits:
        raise RuntimeError("public aggregate output contains a restricted material identifier")
    if sha256_file(design_path) != frozen_design_hash:
        raise RuntimeError("matched design changed after outcome computation")

    manifest_path = repo / config["output"]["manifest"]
    manifest = {
        "task_id": TASK_ID, "method_version": METHOD_VERSION, "created_at_utc": _utc_now(),
        "config_sha256": sha256_file(config_file),
        "route": "SCOPED",
        "input_hashes": {row.input_name: row.observed_sha256 for row in hash_audit.itertuples(index=False)},
        "matched_design_sha256": frozen_design_hash,
        "primary_rows": len(primary),
        "classification_counts": classification["provenance_class"].value_counts().reindex(["A", "B", "C", "U"], fill_value=0).astype(int).to_dict(),
        "output_artifacts": [
            {
                "path": path.relative_to(repo).as_posix(), "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                **({"rows": pq.ParquetFile(path).metadata.num_rows} if path.suffix == ".parquet" else {}),
            }
            for path in output_paths
        ],
        "forbidden_outcome_reads": 0,
        "network_concentration_computed": False,
    }
    _write_json(manifest_path, manifest)
    (manifest_path.parent / "counterfactual_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="ascii"
    )

    build_summary = {
        "task_id": TASK_ID, "method_version": METHOD_VERSION,
        "started_at_utc": started, "ended_at_utc": _utc_now(),
        "primary_rows": len(primary), "state_rows": len(states), "transition_rows": len(transitions),
        "gnome_edge_rows": len(gnome_edges), "matched_relation_rows": len(relation_frame),
        "classification_counts": manifest["classification_counts"],
        "max_baseline_error_eV_per_atom": max_baseline_error,
        "retained_value_change_count": max_retained_change,
        "matched_fraction": matched_fraction_value, "maximum_absolute_smd": maximum_absolute_smd,
        "matched_design_quality_pass": matched_design_quality_pass,
        "public_identifier_hits": public_identifier_hits,
        "api_total_doc": prerequisite["api_total_doc"],
        "network_concentration_computed": False,
        "summary": summary.to_dict(orient="records"),
    }
    _write_json(report_dir / "build_summary.json", build_summary)
    print(json.dumps(build_summary, indent=2, sort_keys=True, default=_json_default), flush=True)
    return build_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/r3/r3_4b_s_gnome_counterfactual.yaml")
    args = parser.parse_args()
    build(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
