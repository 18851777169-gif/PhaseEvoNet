"""R3.5B fixed-prediction, fixed-candidate, changing-label benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .common import open_formal_input, sha256_file


TASK_ID = "R3.5B"
METHOD_VERSION = "PHASEEVONET_R3_5B_FIXED_VERSION_BENCHMARK_V1"
SOURCE_VERSION = "2023-11-01"
TARGET_VERSIONS = ("2024-12-18", "2025-09-25")
MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
THRESHOLDS = (("exact_zero", 1.0e-8), ("within_10meV", 0.010), ("within_25meV", 0.025))
BUDGETS: tuple[int | str, ...] = (100, 500, 1000, "top_1pct", "top_5pct")
BOOTSTRAP_REPLICATES = 2000
SEED = 42

FROZEN_HASHES = {
    "data/manifests/R3_5/protocol.json": "a681beb467fbe721b703e96cd93ad36ef76d551610e946abac0bb222ec2d9e7b",
    "data/processed/R3_5/candidate_panel.parquet": "0dd81cc6ff1bcc88c152a6c4371536014ccd279bbf96a36e5f7d46b0a18f19b3",
    "data/processed/R3_5/frozen_predictions.parquet": "bfe553c5fdd53f0e1202e60c096515cc63857ae23846574f9fa7e6df35c12102",
    "reports/R3_5A/report.json": "4200c65ba3cc1cad7036f284a36863c9653ffe92095350412363be778353d587",
    "reports/R3_5A/verification.json": "d71f594a03f049a97350e8abcc1864fc51d31b6b20286a2247a9560767761ee6",
    "data/manifests/P3_1/transition_manifest.json": "e6779641627850b5b3086a1fc1e37817ba149aad810d691d7779e8ab7278ffbc",
    "data/interim/P3_1/transition_label.parquet": "ad896e7f150a14c5d724402fee61127c8d5d997c62b52ab207b2bdf6f3319e25",
    "data/manifests/P3_2/phase_diagram_manifest.json": "ad464b2e2b19ef0d547a41f5d72842dc4bac6b29585339de773c3f20ff093fc7",
    "data/interim/P3_2/phase_entry_unified.parquet": "9236b0fdaeafe14b453afba818d26e4ca6d435812486b0924cb85b5fb26e1f7b",
    "data/manifests/R3_2/manifest.json": "3701db16613446767e35ae527b7b6ea2a76c8be29461d485ab536a82215b5aae",
    "data/interim/R3_2/source_signed_margin_cache.parquet": "5ca0618590c01fe09d4f4caaa6862461625568414968eecd5803f98f9040c942",
    "reports/R3_1/report.json": "4baeead9cd2d022d51b0de809f3742a066fa23395e93ab828b165068ec20692a",
    "reports/R3_2/report.json": "720580e839a44465d7c34d97b278c4955e231208f2855f815157b705b5ced6ab",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def artifact(repo: Path, path: Path, rows: int | None = None) -> dict[str, Any]:
    value = {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if rows is not None:
        value["rows"] = int(rows)
    return value


def transition_hex(value: Any) -> str:
    return bytes(value).hex() if isinstance(value, (bytes, bytearray, memoryview)) else str(value)


def prediction_vector_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("panel_unit_id", kind="mergesort").itertuples(index=False):
        digest.update(str(row.panel_unit_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(row.probability)))
    return digest.hexdigest()


def resolve_budget(value: int | str, n: int) -> int:
    if isinstance(value, int):
        return min(value, n)
    fractions = {"top_1pct": 0.01, "top_5pct": 0.05}
    if value not in fractions:
        raise ValueError(f"unknown budget {value}")
    return min(n, max(1, int(math.ceil(n * fractions[value]))))


def ece_equal_width(y: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    bins = np.minimum((probability * 10).astype(int), 9)
    value = 0.0
    for index in range(10):
        mask = bins == index
        mass = float(weights[mask].sum())
        if mass:
            value += mass / total * abs(float(np.average(probability[mask], weights=weights[mask])) - float(np.average(y[mask], weights=weights[mask])))
    return value


def point_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute every frozen point metric for one model/label/version frame."""
    y = frame["event"].to_numpy(dtype=bool)
    probability = frame["probability"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    base = {
        "model_id": str(frame["model_id"].iloc[0]),
        "label_definition": str(frame["label_definition"].iloc[0]),
        "label_version": str(frame["label_version"].iloc[0]),
        "panel_rows": len(frame),
        "positives": int(y.sum()),
        "negatives": int((~y).sum()),
    }
    estimable = bool(y.any() and (~y).any())
    values = {
        "average_precision": average_precision_score(y, probability) if estimable else np.nan,
        "brier_score": brier_score_loss(y, probability),
        "roc_auc": roc_auc_score(y, probability) if estimable else np.nan,
        "expected_calibration_error": ece_equal_width(y, probability),
        "spearman_continuous": (
            spearmanr(probability, frame["energy_change_eV_per_atom"].to_numpy(dtype=float)).statistic
            if np.ptp(probability) > 0 and np.ptp(frame["energy_change_eV_per_atom"].to_numpy(dtype=float)) > 0
            else np.nan
        ),
    }
    for metric, value in values.items():
        rows.append({**base, "metric": metric, "budget": "", "value": float(value), "estimable": bool(np.isfinite(value))})
    ranked = frame.sort_values(["probability", "panel_unit_id"], ascending=[False, True], kind="mergesort")
    total_positive = int(y.sum())
    for budget in BUDGETS:
        k = resolve_budget(budget, len(ranked))
        selected = ranked.head(k)["event"].to_numpy(dtype=bool)
        tp = int(selected.sum())
        rows.append({**base, "metric": "precision_at_budget", "budget": str(budget), "resolved_budget": k, "value": tp / k if k else np.nan, "estimable": bool(k)})
        rows.append({**base, "metric": "recall_at_budget", "budget": str(budget), "resolved_budget": k, "value": tp / total_positive if total_positive else np.nan, "estimable": bool(total_positive)})
    return rows


class FormalReader:
    def __init__(self, repo: Path, access_log: Path, scope_log: Path):
        self.repo = repo
        self.access_log = access_log
        self.scope_log = scope_log

    def _record_scope(self, relative: str, columns: Iterable[str], stage: str, outcome_values: bool) -> None:
        with self.scope_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"timestamp_utc": utc_now(), "path": relative, "columns": list(columns), "stage": stage, "outcome_values": outcome_values}, sort_keys=True) + "\n")

    def authenticate(self, relative: str, purpose: str) -> dict[str, Any]:
        with open_formal_input(self.repo / relative, FROZEN_HASHES[relative], task_id=TASK_ID, access_log=self.access_log, purpose=purpose, allowed_roots=(self.repo,)):
            pass
        path = self.repo / relative
        return {"path": relative, "expected_sha256": FROZEN_HASHES[relative], "observed_sha256": sha256_file(path), "bytes": path.stat().st_size, "hash_match": True}

    def json(self, relative: str, purpose: str) -> Any:
        with open_formal_input(self.repo / relative, FROZEN_HASHES[relative], task_id=TASK_ID, access_log=self.access_log, purpose=purpose, allowed_roots=(self.repo,)) as handle:
            return json.load(handle)

    def parquet(self, relative: str, columns: list[str], stage: str, outcome_values: bool = False) -> pd.DataFrame:
        self._record_scope(relative, columns, stage, outcome_values)
        with open_formal_input(self.repo / relative, FROZEN_HASHES[relative], task_id=TASK_ID, access_log=self.access_log, purpose=stage, allowed_roots=(self.repo,)) as handle:
            return pq.read_table(handle, columns=columns).to_pandas()


def freeze_common_panel(repo: Path, reader: FormalReader, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_columns = [
        "transition_id", "canonical_lineage_id", "source_snapshot", "target_snapshot", "source_material_id",
        "target_material_id", "thermo_type", "observation_status", "source_state_present", "target_state_present",
        "source_state_usable", "target_state_usable",
    ]
    transitions = reader.parquet("data/interim/P3_1/transition_label.parquet", transition_columns, "availability-only endpoint identity join before outcome access")
    transitions["transition_id_hex"] = transitions["transition_id"].map(transition_hex)
    panel_key = panel[["panel_unit_id", "transition_id", "canonical_lineage_id", "prediction_source_snapshot", "thermo_type", "phase_context_chemsys"]].copy()

    base = panel_key.merge(
        transitions.loc[(transitions["source_snapshot"] == SOURCE_VERSION) & (transitions["target_snapshot"] == TARGET_VERSIONS[0])],
        left_on="transition_id", right_on="transition_id_hex", how="left", suffixes=("", "_transition"), validate="one_to_one",
    )
    if base["source_material_id"].isna().any() or base["target_material_id"].isna().any():
        raise RuntimeError("frozen candidate transitions do not map uniquely to 2023/2024 endpoint identities")
    references = []
    for version, material_column, present_column, usable_column in (
        (SOURCE_VERSION, "source_material_id", "source_state_present", "source_state_usable"),
        (TARGET_VERSIONS[0], "target_material_id", "target_state_present", "target_state_usable"),
    ):
        item = base[["panel_unit_id", "canonical_lineage_id", "thermo_type", "phase_context_chemsys", material_column, present_column, usable_column, "transition_id_hex"]].copy()
        item.columns = ["panel_unit_id", "canonical_lineage_id", "thermo_type", "phase_context_chemsys", "material_id", "state_present", "state_usable", "state_transition_id"]
        item["snapshot_id"] = version
        references.append(item)
    future = transitions.loc[transitions["target_snapshot"] == TARGET_VERSIONS[1], ["transition_id_hex", "canonical_lineage_id", "thermo_type", "target_material_id", "target_state_present", "target_state_usable"]].copy()
    if future.duplicated(["canonical_lineage_id", "thermo_type"]).any():
        raise RuntimeError("2025 endpoint identity mapping is not one-to-one by lineage/workflow")
    future = panel_key.merge(future, on=["canonical_lineage_id", "thermo_type"], how="left", validate="one_to_one")
    future = future[["panel_unit_id", "canonical_lineage_id", "thermo_type", "phase_context_chemsys", "target_material_id", "target_state_present", "target_state_usable", "transition_id_hex"]]
    future.columns = ["panel_unit_id", "canonical_lineage_id", "thermo_type", "phase_context_chemsys", "material_id", "state_present", "state_usable", "state_transition_id"]
    future["snapshot_id"] = TARGET_VERSIONS[1]
    references.append(future)
    refs = pd.concat(references, ignore_index=True)

    entry_columns = ["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "material_id", "is_target", "phase_diagram_status"]
    entries = reader.parquet("data/interim/P3_2/phase_entry_unified.parquet", entry_columns, "availability-only phase-entry identity join before outcome access")
    entries = entries.loc[entries["is_target"] & entries["snapshot_id"].isin((SOURCE_VERSION, *TARGET_VERSIONS))].copy()
    # The full contextual phase-entry table can contain the same material key
    # outside this frozen candidate set.  Permit that table-level multiplicity,
    # then enforce the scientifically relevant panel/version uniqueness below.
    common_long = refs.merge(entries, on=["snapshot_id", "thermo_type", "phase_context_chemsys", "material_id"], how="left", validate="one_to_many")
    common_long["availability_reason"] = np.select(
        [~common_long["state_present"].fillna(False), ~common_long["state_usable"].fillna(False), common_long["unified_entry_id"].isna(), common_long["phase_diagram_status"].ne("computed")],
        ["state_not_present", "state_not_usable", "phase_entry_missing", "phase_diagram_not_computed"], default="estimable",
    )
    if common_long.duplicated(["panel_unit_id", "snapshot_id"]).any():
        raise RuntimeError("matched phase-entry identity is not unique per panel unit/version")
    counts = common_long.groupby("panel_unit_id")["availability_reason"].apply(lambda values: int((values == "estimable").sum()))
    accepted = counts[counts == 3].index
    common_long["primary_common_panel"] = common_long["panel_unit_id"].isin(accepted)
    common = common_long.loc[common_long["primary_common_panel"]].copy()
    if common.duplicated(["panel_unit_id", "snapshot_id"]).any():
        raise RuntimeError("common-panel version key is not unique")
    common["unified_entry_id"] = common["unified_entry_id"].map(transition_hex)
    common = common.sort_values(["panel_unit_id", "snapshot_id"], kind="mergesort").reset_index(drop=True)
    output = repo / "data/processed/R3_5/common_panel_index.parquet"
    common.to_parquet(output, index=False, compression="zstd")
    sidecar = repo / "data/manifests/R3_5/common_panel_index.sha256"
    sidecar.write_text(f"{sha256_file(output)}  ../../processed/R3_5/common_panel_index.parquet\n", encoding="ascii", newline="\n")
    exclusions = common_long.loc[~common_long["primary_common_panel"]].copy()
    return common, exclusions


def build_labels(repo: Path, reader: FormalReader, common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = ["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "material_id", "is_target", "energy_above_hull"]
    values = reader.parquet("data/interim/P3_2/phase_entry_unified.parquet", value_columns, "authorized non-locked version energy outcomes after common-panel freeze", outcome_values=True)
    values = values.loc[values["is_target"] & values["snapshot_id"].isin((SOURCE_VERSION, *TARGET_VERSIONS))].copy()
    values["unified_entry_id"] = values["unified_entry_id"].map(transition_hex)
    state = common.merge(values[["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "material_id", "energy_above_hull"]], on=["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "material_id"], how="left", validate="one_to_one")
    if not np.isfinite(state["energy_above_hull"]).all():
        raise RuntimeError("nonfinite energy on the frozen common panel")

    margin_columns = ["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id", "signed_margin_eV_per_atom", "solver_status"]
    margins = reader.parquet("data/interim/R3_2/source_signed_margin_cache.parquet", margin_columns, "authorized frozen signed-margin secondary outcomes after common-panel freeze", outcome_values=True)
    margins["unified_entry_id"] = margins["unified_entry_id"].map(transition_hex)
    state = state.merge(margins, on=["snapshot_id", "thermo_type", "phase_context_chemsys", "unified_entry_id"], how="left", validate="one_to_one")
    source = state.loc[state["snapshot_id"] == SOURCE_VERSION, ["panel_unit_id", "energy_above_hull", "signed_margin_eV_per_atom"]].rename(columns={"energy_above_hull": "source_energy_above_hull", "signed_margin_eV_per_atom": "source_signed_margin"})
    rows = []
    for version in TARGET_VERSIONS:
        target = state.loc[state["snapshot_id"] == version, ["panel_unit_id", "canonical_lineage_id", "thermo_type", "phase_context_chemsys", "energy_above_hull", "signed_margin_eV_per_atom"]].rename(columns={"energy_above_hull": "target_energy_above_hull", "signed_margin_eV_per_atom": "target_signed_margin"})
        joined = target.merge(source, on="panel_unit_id", how="inner", validate="one_to_one")
        joined["label_version"] = version
        joined["energy_change_eV_per_atom"] = joined["target_energy_above_hull"] - joined["source_energy_above_hull"]
        joined["signed_margin_change_eV_per_atom"] = joined["target_signed_margin"] - joined["source_signed_margin"]
        for name, threshold in THRESHOLDS:
            label = joined.copy()
            label["label_definition"] = name
            label["threshold_eV_per_atom"] = threshold
            label["source_at_risk"] = label["source_energy_above_hull"].le(threshold)
            label["target_stable"] = label["target_energy_above_hull"].le(threshold)
            label["event"] = label["source_at_risk"] & ~label["target_stable"]
            rows.append(label)
    labels = pd.concat(rows, ignore_index=True).sort_values(["label_definition", "label_version", "panel_unit_id"], kind="mergesort").reset_index(drop=True)
    if not labels["source_at_risk"].all():
        raise RuntimeError("frozen stable-to-unstable candidate panel contains a source row outside a frozen risk set")
    output = repo / "data/processed/R3_5/versioned_labels.parquet"
    labels.to_parquet(output, index=False, compression="zstd")
    return labels, state


def error_transition_rows(
    benchmark: pd.DataFrame,
    target_versions: tuple[str, str] = TARGET_VERSIONS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, label), group in benchmark.groupby(["model_id", "label_definition"], sort=True):
        wide = group.pivot(index="panel_unit_id", columns="label_version", values="event")
        scores = group.drop_duplicates("panel_unit_id").set_index("panel_unit_id")["probability"]
        rank = pd.DataFrame({"panel_unit_id": scores.index, "probability": scores.values}).sort_values(["probability", "panel_unit_id"], ascending=[False, True], kind="mergesort")
        for budget in BUDGETS:
            k = resolve_budget(budget, len(rank))
            selected = set(rank.head(k)["panel_unit_id"])
            y0 = wide[target_versions[0]].astype(bool)
            y1 = wide[target_versions[1]].astype(bool)
            sel = wide.index.to_series().isin(selected).set_axis(wide.index)
            transitions = {
                "TP_to_FP": sel & y0 & ~y1,
                "FP_to_TP": sel & ~y0 & y1,
                "FN_to_TN": ~sel & y0 & ~y1,
                "TN_to_FN": ~sel & ~y0 & y1,
            }
            for name, mask in transitions.items():
                rows.append({"model_id": model, "label_definition": label, "budget": str(budget), "resolved_budget": k, "error_transition": name, "count": int(mask.sum()), "fraction_of_panel": float(mask.mean())})
    return rows


def _weighted_ap(y: np.ndarray, score: np.ndarray, panel_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Average precision is defined on distinct score thresholds.  The lexical
    # rule is reserved for top-budget truncation; it must not split score ties.
    order = np.argsort(-score, kind="mergesort")
    wo = weights[:, order]
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    positive_group = np.add.reduceat(wo * y[order], starts, axis=1)
    total_group = np.add.reduceat(wo, starts, axis=1)
    cum_total = np.cumsum(total_group, axis=1)
    cum_positive = np.cumsum(positive_group, axis=1)
    precision = np.divide(cum_positive, cum_total, out=np.zeros_like(cum_positive), where=cum_total > 0)
    denominator = positive_group.sum(axis=1)
    return np.divide((precision * positive_group).sum(axis=1), denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def _weighted_auc(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    wo = weights[:, order]
    pgroup = np.add.reduceat(wo * y[order], starts, axis=1)
    ngroup = np.add.reduceat(wo * (~y[order]), starts, axis=1)
    before = np.cumsum(ngroup, axis=1) - ngroup
    positives = pgroup.sum(axis=1)
    negatives = ngroup.sum(axis=1)
    denominator = positives * negatives
    numerator = (pgroup * (before + 0.5 * ngroup)).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def _weighted_spearman(score: np.ndarray, continuous: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = rankdata(score, method="average")
    y = rankdata(continuous, method="average")
    total = weights.sum(axis=1)
    mx = weights @ x / total
    my = weights @ y / total
    cov = weights @ (x * y) / total - mx * my
    vx = weights @ (x * x) / total - mx * mx
    vy = weights @ (y * y) / total - my * my
    return np.divide(cov, np.sqrt(vx * vy), out=np.full(len(weights), np.nan), where=(vx > 0) & (vy > 0))


def bootstrap_metrics(
    benchmark: pd.DataFrame,
    output_audit: Path,
    target_versions: tuple[str, str] = TARGET_VERSIONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = benchmark[["panel_unit_id", "canonical_lineage_id"]].drop_duplicates().sort_values("panel_unit_id", kind="mergesort")
    panel_ids = panel["panel_unit_id"].to_numpy(dtype=str)
    lineages, codes = np.unique(panel["canonical_lineage_id"].to_numpy(dtype=str), return_inverse=True)
    base = benchmark.set_index(["panel_unit_id", "model_id", "label_definition", "label_version"]).sort_index()
    scores = {model: base.xs((model, THRESHOLDS[0][0], target_versions[0]), level=("model_id", "label_definition", "label_version")).reindex(panel_ids)["probability"].to_numpy(float) for model in MODELS}
    outcomes = {(label, version): base.xs((MODELS[0], label, version), level=("model_id", "label_definition", "label_version")).reindex(panel_ids)["event"].to_numpy(bool) for label, _ in THRESHOLDS for version in target_versions}
    continuous = {version: base.xs((MODELS[0], THRESHOLDS[0][0], version), level=("model_id", "label_definition", "label_version")).reindex(panel_ids)["energy_change_eV_per_atom"].to_numpy(float) for version in target_versions}
    result: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    transition_result: dict[tuple[str, str, str, str], np.ndarray] = {}
    audit_rows = []
    rng = np.random.default_rng(SEED)
    chunk_size = 25
    chunks: dict[tuple[str, str, str, str, str], list[np.ndarray]] = {}
    transition_chunks: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    for start in range(0, BOOTSTRAP_REPLICATES, chunk_size):
        size = min(chunk_size, BOOTSTRAP_REPLICATES - start)
        lineage_weights = rng.multinomial(len(lineages), np.full(len(lineages), 1.0 / len(lineages)), size=size).astype(float)
        weights = lineage_weights[:, codes]
        for offset in range(size):
            raw = lineage_weights[offset].astype(np.int16).tobytes()
            audit_rows.append({"replicate": start + offset, "draw_sha256": hashlib.sha256(raw).hexdigest(), "sampled_lineages": len(lineages), "unique_sampled_lineages": int((lineage_weights[offset] > 0).sum()), "expanded_panel_weight": int(weights[offset].sum())})
        denominator = weights.sum(axis=1)
        for label, _ in THRESHOLDS:
            for version in target_versions:
                y = outcomes[(label, version)]
                positives = weights @ y.astype(float)
                for model in MODELS:
                    probability = scores[model]
                    values = {
                        "average_precision": _weighted_ap(y, probability, panel_ids, weights),
                        "brier_score": (weights @ np.square(probability - y)) / denominator,
                        "roc_auc": _weighted_auc(y, probability, weights),
                        "spearman_continuous": _weighted_spearman(probability, continuous[version], weights),
                    }
                    bins = np.minimum((probability * 10).astype(int), 9)
                    ece = np.zeros(size)
                    for index in range(10):
                        mask = bins == index
                        mass = weights[:, mask].sum(axis=1)
                        if mask.any():
                            prob_sum = weights[:, mask] @ probability[mask]
                            y_sum = weights[:, mask] @ y[mask].astype(float)
                            ece += np.divide(mass, denominator, out=np.zeros(size), where=denominator > 0) * np.abs(np.divide(prob_sum, mass, out=np.zeros(size), where=mass > 0) - np.divide(y_sum, mass, out=np.zeros(size), where=mass > 0))
                    values["expected_calibration_error"] = ece
                    order = np.lexsort((panel_ids, -probability))
                    for budget in BUDGETS:
                        k = resolve_budget(budget, len(panel_ids))
                        selected = np.zeros(len(panel_ids), dtype=bool)
                        selected[order[:k]] = True
                        selected_mass = weights[:, selected].sum(axis=1)
                        tp = weights[:, selected] @ y[selected].astype(float)
                        values[f"precision_at_budget|{budget}"] = np.divide(tp, selected_mass, out=np.full(size, np.nan), where=selected_mass > 0)
                        values[f"recall_at_budget|{budget}"] = np.divide(tp, positives, out=np.full(size, np.nan), where=positives > 0)
                    for metric_key, array in values.items():
                        metric, _, budget = metric_key.partition("|")
                        chunks.setdefault((model, label, version, metric, budget), []).append(array)
            y0 = outcomes[(label, target_versions[0])]
            y1 = outcomes[(label, target_versions[1])]
            for model in MODELS:
                probability = scores[model]
                order = np.lexsort((panel_ids, -probability))
                for budget in BUDGETS:
                    k = resolve_budget(budget, len(panel_ids))
                    selected = np.zeros(len(panel_ids), dtype=bool)
                    selected[order[:k]] = True
                    masks = {"TP_to_FP": selected & y0 & ~y1, "FP_to_TP": selected & ~y0 & y1, "FN_to_TN": ~selected & y0 & ~y1, "TN_to_FN": ~selected & ~y0 & y1}
                    for name, mask in masks.items():
                        transition_chunks.setdefault((model, label, str(budget), name), []).append((weights @ mask.astype(float)) / denominator)
    pd.DataFrame(audit_rows).to_csv(output_audit, index=False)
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    transition_result = {key: np.concatenate(value) for key, value in transition_chunks.items()}
    def summarize(values: np.ndarray) -> tuple[float, float, float, float]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return (float("nan"),) * 4
        q = np.percentile(finite, [2.5, 50, 97.5])
        return float(finite.mean()), float(q[0]), float(q[1]), float(q[2])

    rows = []
    for key, values in sorted(result.items()):
        model, label, version, metric, budget = key
        mean, lower, median, upper = summarize(values)
        rows.append({"scope": "version_metric", "model_id": model, "label_definition": label, "label_version": version, "metric": metric, "budget": budget, "replicates": len(values), "finite_replicates": int(np.isfinite(values).sum()), "estimate": mean, "lower_2_5": lower, "median": median, "upper_97_5": upper})
    for model in MODELS:
        for label, _ in THRESHOLDS:
            keys = [key for key in result if key[0] == model and key[1] == label and key[2] == target_versions[0]]
            for key in keys:
                _, _, _, metric, budget = key
                delta = result[(model, label, target_versions[1], metric, budget)] - result[key]
                mean, lower, median, upper = summarize(delta)
                rows.append({"scope": "version_delta", "model_id": model, "label_definition": label, "label_version": f"{target_versions[1]}-minus-{target_versions[0]}", "metric": metric, "budget": budget, "replicates": len(delta), "finite_replicates": int(np.isfinite(delta).sum()), "estimate": mean, "lower_2_5": lower, "median": median, "upper_97_5": upper})
    for key, values in sorted(transition_result.items()):
        model, label, budget, name = key
        mean, lower, median, upper = summarize(values)
        rows.append({"scope": "error_transition", "model_id": model, "label_definition": label, "label_version": f"{target_versions[0]}->{target_versions[1]}", "metric": name, "budget": budget, "replicates": len(values), "finite_replicates": int(np.isfinite(values).sum()), "estimate": mean, "lower_2_5": lower, "median": median, "upper_97_5": upper})
    return pd.DataFrame(rows), {"lineages": len(lineages), "replicates": BOOTSTRAP_REPLICATES, "seed": SEED, "chunk_size": chunk_size, "binding": "all panel rows/models/labels/versions bound by canonical lineage", "spearman_bootstrap": "weighted Pearson correlation of fixed full-panel midranks under cluster multiplicity"}


def build(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    report_dir = repo / "reports/R3_5B"
    manifest_dir = repo / "data/manifests/R3_5"
    report_dir.mkdir(parents=True, exist_ok=True)
    access_log = report_dir / "input_access_log.jsonl"
    scope_log = report_dir / "read_scope_log.jsonl"
    access_log.unlink(missing_ok=True)
    scope_log.unlink(missing_ok=True)
    reader = FormalReader(repo, access_log, scope_log)
    input_audit = [reader.authenticate(relative, "R3.5B frozen formal-input hash preflight") for relative in FROZEN_HASHES]
    pd.DataFrame(input_audit).to_csv(report_dir / "input_hash_audit.csv", index=False)
    protocol = reader.json("data/manifests/R3_5/protocol.json", "read frozen R3.5A protocol")
    r35a = reader.json("reports/R3_5A/report.json", "verify R3.5A completion")
    if protocol["protocol_status"] != "FROZEN" or not r35a["acceptance_passed"]:
        raise RuntimeError("R3.5A protocol prerequisite is not valid")
    sidecar = (repo / "data/manifests/R3_5/protocol.sha256").read_text(encoding="ascii").split()[0]
    if sidecar != FROZEN_HASHES["data/manifests/R3_5/protocol.json"]:
        raise RuntimeError("protocol sidecar mismatch")
    panel = reader.parquet("data/processed/R3_5/candidate_panel.parquet", ["panel_unit_id", "transition_id", "canonical_lineage_id", "prediction_source_snapshot", "thermo_type", "phase_context_chemsys", "model_count"], "read frozen candidate panel")
    predictions = reader.parquet("data/processed/R3_5/frozen_predictions.parquet", ["panel_unit_id", "canonical_lineage_id", "prediction_source_snapshot", "thermo_type", "model_id", "probability", "evaluation_method"], "read immutable frozen prediction scores")
    if predictions.duplicated(["panel_unit_id", "model_id"]).any() or set(predictions["model_id"]) != set(MODELS):
        raise RuntimeError("frozen prediction uniqueness/model set failed")
    for model in MODELS:
        observed = prediction_vector_hash(predictions.loc[predictions["model_id"] == model, ["panel_unit_id", "probability"]])
        if observed != protocol["prediction_source"]["prediction_vector_sha256"][model]:
            raise RuntimeError(f"frozen prediction vector mismatch: {model}")

    common, exclusions = freeze_common_panel(repo, reader, panel)
    common_path = repo / "data/processed/R3_5/common_panel_index.parquet"
    frozen_common_hash = sha256_file(common_path)
    if len(common) // 3 < 1:
        raise RuntimeError("no usable common panel")
    exclusions.to_csv(report_dir / "exclusion_ambiguity_ledger.csv", index=False)
    labels, state = build_labels(repo, reader, common)
    benchmark = predictions.merge(labels, on=["panel_unit_id", "canonical_lineage_id", "thermo_type"], how="inner", validate="many_to_many")
    if len(benchmark) != len(panel) * len(MODELS) * len(TARGET_VERSIONS) * len(THRESHOLDS):
        raise RuntimeError("benchmark cross-product row count mismatch")
    benchmark_path = repo / "data/processed/R3_5/versioned_benchmark.parquet"
    benchmark.to_parquet(benchmark_path, index=False, compression="zstd")

    byte_audit = []
    for model in MODELS:
        expected = protocol["prediction_source"]["prediction_vector_sha256"][model]
        for label, _ in THRESHOLDS:
            for version in TARGET_VERSIONS:
                subset = benchmark.loc[(benchmark["model_id"] == model) & (benchmark["label_definition"] == label) & (benchmark["label_version"] == version), ["panel_unit_id", "probability"]]
                observed = prediction_vector_hash(subset)
                byte_audit.append({"model_id": model, "label_definition": label, "label_version": version, "expected_vector_sha256": expected, "observed_vector_sha256": observed, "byte_identical": expected == observed})
    byte_frame = pd.DataFrame(byte_audit)
    byte_frame.to_csv(report_dir / "prediction_byte_audit.csv", index=False)
    if not byte_frame["byte_identical"].all():
        raise RuntimeError("prediction bytes changed across label-version rows")

    metric_rows = []
    for _, group in benchmark.groupby(["model_id", "label_definition", "label_version"], sort=True):
        metric_rows.extend(point_metrics(group))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(report_dir / "metric_results.csv", index=False)
    index_cols = ["model_id", "label_definition", "metric", "budget"]
    earlier = metrics.loc[metrics["label_version"] == TARGET_VERSIONS[0]].set_index(index_cols)
    later = metrics.loc[metrics["label_version"] == TARGET_VERSIONS[1]].set_index(index_cols)
    deltas = later[["value"]].join(earlier[["value"]], lsuffix="_later", rsuffix="_earlier").reset_index()
    deltas["earlier_version"] = TARGET_VERSIONS[0]
    deltas["later_version"] = TARGET_VERSIONS[1]
    deltas["delta_later_minus_earlier"] = deltas["value_later"] - deltas["value_earlier"]
    deltas.to_csv(report_dir / "metric_deltas.csv", index=False)

    label_rows = []
    for label, _ in THRESHOLDS:
        wide = labels.loc[labels["label_definition"] == label].pivot(index="panel_unit_id", columns="label_version", values="event")
        for version in TARGET_VERSIONS:
            label_rows.append({"label_definition": label, "label_version": version, "panel_rows": len(wide), "events": int(wide[version].sum()), "prevalence": float(wide[version].mean()), "changed_from_earlier": "" if version == TARGET_VERSIONS[0] else int((wide[TARGET_VERSIONS[0]] != wide[version]).sum()), "zero_to_one": "" if version == TARGET_VERSIONS[0] else int((~wide[TARGET_VERSIONS[0]] & wide[version]).sum()), "one_to_zero": "" if version == TARGET_VERSIONS[0] else int((wide[TARGET_VERSIONS[0]] & ~wide[version]).sum())})
    pd.DataFrame(label_rows).to_csv(report_dir / "label_drift.csv", index=False)
    availability = []
    for version in (SOURCE_VERSION, *TARGET_VERSIONS):
        subset = common[common["snapshot_id"] == version]
        availability.append({"snapshot_id": version, "base_panel_rows": len(panel), "estimable_common_rows": len(subset), "excluded_rows": len(panel) - len(subset), "availability_fraction": len(subset) / len(panel)})
    pd.DataFrame(availability).to_csv(report_dir / "availability_drift.csv", index=False)
    workflow = common.groupby(["snapshot_id", "thermo_type"]).size().rename("rows").reset_index()
    workflow["fraction"] = workflow["rows"] / workflow.groupby("snapshot_id")["rows"].transform("sum")
    workflow.to_csv(report_dir / "workflow_composition_drift.csv", index=False)
    margin_coverage = state.groupby("snapshot_id").agg(panel_rows=("panel_unit_id", "size"), signed_margin_available=("signed_margin_eV_per_atom", "count")).reset_index()
    margin_coverage["signed_margin_fraction"] = margin_coverage["signed_margin_available"] / margin_coverage["panel_rows"]
    margin_coverage["secondary_status"] = np.where(margin_coverage["signed_margin_available"] == margin_coverage["panel_rows"], "ESTIMABLE", "NONESTIMABLE_MISSING_FROZEN_MARGIN")
    margin_coverage.to_csv(report_dir / "signed_margin_coverage.csv", index=False)

    errors = pd.DataFrame(error_transition_rows(benchmark))
    errors.to_csv(report_dir / "error_transitions.csv", index=False)
    ranking_rows = []
    ranking_metrics = metrics.loc[metrics["estimable"]].copy()
    for (label, metric, budget, version), group in ranking_metrics.groupby(["label_definition", "metric", "budget", "label_version"], dropna=False, sort=True):
        ascending = metric in {"brier_score", "expected_calibration_error"}
        ranked = group.copy()
        ranked["rank"] = ranked["value"].rank(method="min", ascending=ascending).astype(int)
        for row in ranked.itertuples():
            ranking_rows.append({"label_definition": label, "metric": metric, "budget": budget, "label_version": version, "model_id": row.model_id, "value": row.value, "rank": row.rank, "same_panel_rows": row.panel_rows})
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(report_dir / "model_rankings.csv", index=False)

    bootstrap, bootstrap_info = bootstrap_metrics(benchmark, report_dir / "bootstrap_replicate_audit.csv")
    bootstrap.to_csv(report_dir / "bootstrap_intervals.csv", index=False)
    core = metrics[metrics["metric"].isin(["average_precision", "precision_at_budget", "recall_at_budget", "brier_score"])]
    all_core_estimable = bool(core["estimable"].all())
    all_binary_classes = all((row["events"] > 0 and row["events"] < row["panel_rows"]) for row in label_rows)
    route = "GO" if all_core_estimable and all_binary_classes else "WORKED_DEMONSTRATION"

    outputs = {
        "common_panel": artifact(repo, common_path, len(common)),
        "versioned_labels": artifact(repo, repo / "data/processed/R3_5/versioned_labels.parquet", len(labels)),
        "versioned_benchmark": artifact(repo, benchmark_path, len(benchmark)),
    }
    manifest = {
        "task_id": TASK_ID, "method_version": METHOD_VERSION, "created_at_utc": utc_now(), "protocol_sha256": FROZEN_HASHES["data/manifests/R3_5/protocol.json"],
        "frozen_inputs": FROZEN_HASHES, "outputs": outputs, "panel": {"base_rows": len(panel), "common_rows": len(common) // 3, "common_long_rows": len(common), "common_panel_sha256": frozen_common_hash},
        "models": list(MODELS), "label_versions": list(TARGET_VERSIONS), "thresholds": dict(THRESHOLDS), "bootstrap": bootstrap_info,
        "prediction_byte_checks": len(byte_frame), "prediction_byte_checks_passed": int(byte_frame["byte_identical"].sum()), "route": route,
        "prohibited_actions": {"models_loaded": 0, "models_trained": 0, "predictions_regenerated": 0, "predictions_recalibrated": 0, "locked_test_reads": 0, "confirmation_outcome_reads": 0},
    }
    manifest_path = manifest_dir / "benchmark_manifest.json"
    write_json(manifest_path, manifest)
    (manifest_dir / "benchmark_manifest.sha256").write_text(f"{sha256_file(manifest_path)}  benchmark_manifest.json\n", encoding="ascii", newline="\n")
    summary = {
        "route": route, "base_panel_rows": len(panel), "common_panel_rows": len(common) // 3, "common_panel_fraction": (len(common) // 3) / len(panel),
        "label_rows": len(labels), "benchmark_rows": len(benchmark), "models": list(MODELS), "events": label_rows,
        "metric_rows": len(metrics), "metric_delta_rows": len(deltas), "error_transition_rows": len(errors), "bootstrap_interval_rows": len(bootstrap),
        "bootstrap": bootstrap_info, "signed_margin_coverage": margin_coverage.to_dict(orient="records"), "artifacts": outputs,
        "input_hashes_checked": len(input_audit), "formal_access_records": sum(1 for _ in access_log.open(encoding="utf-8")), "read_scope_records": sum(1 for _ in scope_log.open(encoding="utf-8")),
    }
    write_json(report_dir / "build_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    build(args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
