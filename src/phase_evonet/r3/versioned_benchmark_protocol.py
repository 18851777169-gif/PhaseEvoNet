"""R3.5A frozen-prediction inventory and versioned-benchmark protocol freeze.

This module deliberately does not load model objects or outcome columns.  It
only authenticates frozen artifacts, selects a label-free prediction panel,
and writes a prospective evaluation protocol for the separately authorized
R3.5B task.
"""

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
import yaml

from .common import open_formal_input, sha256_file


TASK_ID = "R3.5A"
METHOD_VERSION = "PHASEEVONET_R3_5A_PROTOCOL_FREEZE_V2"
PRIMARY_ESTIMAND = "stable_to_unstable_event"
PRIMARY_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
SOURCE_SNAPSHOT = "2023-11-01"
LABEL_VERSIONS = ("2024-12-18", "2025-09-25")
EXACT_LABEL_COLUMNS = {"event", "label", "target", "outcome", "rebuilt_unified_flip"}

FROZEN_HASHES = {
    "configs/r3/r3_5a_protocol_inventory.yaml": "cefc79fbb33b581023ab4bf38fe10fcdd45746b23797e30e515258d9fc0adaba",
    "configs/analysis/p5_1_baselines.yaml": "f9eb018f8ca8f96f4cd4fd74cfca778c347158ce8c58dbb662c47a980843f72d",
    "configs/analysis/v2_2_capacity_matched.yaml": "89b6042ac434a6367f17683f710e827af580679113deb631308a9f862cec4196",
    "configs/analysis/v2_1_model_matrix.yaml": "9e014a77466f9a3be6b2db9773c1ff7a0d2b5f34916cb3f057c8b45106733963",
    "data/manifests/P5_1/baseline_manifest.json": "04d680d5983c6702e0aaf02d9001c0f2ed889a231457a848e09b66f6edffe4b4",
    "data/manifests/V2_2/model_manifest.json": "77ba3b0e211a5b3cca588956b0855a0ad41201a9f07076ff9c662ff0d7d348a1",
    "data/manifests/V2_1/feature_schema.json": "a4618ad6510eed68892b773042cfdbbd08d976f5522af33aea187920edfe2307",
    "data/manifests/P4_2/split_manifest.json": "c9ced37dea21b0e7398daeae99a8c3a5274402960c14698de61baec960bc91a8",
    "data/manifests/P4_2/locked_thresholds.json": "ff08f03488ae7c8d708548041987a70509d623e2717646a11f9d0e3388d761ce",
    "data/processed/P5_1/baseline_features.parquet": "74128c104d698efbc7153cb39e7da9c4b2ed50f896da66268818cffc7361382a",
    "data/processed/P5_1/baseline_predictions.parquet": "c3a49b5c1ffc09a7a3cbe62c93b78025648cf33401aae7c8cb8832809bf5ae3d",
    "data/processed/V2_1/source_features.parquet": "d1569cbf2afcd749814672910a09dd2f6fba58781549122d0577c8bf60464aaf",
    "reports/V2_2/development_predictions.parquet": "4dc3014f75ec6f15be30bdd3a586b53d904536a51b198f8df53a64b933bdf8b0",
    "reports/R3_1/report.json": "4baeead9cd2d022d51b0de809f3742a066fa23395e93ab828b165068ec20692a",
    "reports/R3_1/verification.json": "c0bc362b025f7ed9f0c9ea5d53187224d84af12defa7a2c6efdb367bd630c932",
    "reports/R3_2/report.json": "720580e839a44465d7c34d97b278c4955e231208f2855f815157b705b5ced6ab",
    "reports/R3_2/verification.json": "94f80e914a64dbba1e307bedb067699669f8f06ad7c16205440245c6f32d870e",
    "configs/r3/r3_1_energy_amplitude.yaml": "d2a9a11203769e4ca9555d8e55fdedbe2c9e5ba6fd8f46e6b30298f5de41943f",
    "configs/r3/r3_2_standardized_survival.yaml": "e150f797a1c0feea6ec2e32cd0d6c707216eb1575a0c6116b89217bace7ac629",
    "LICENSE": "159325fcac59dc8f008ed82e939c78837361217a86f8be3d0485291cfe46a57d",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def canonical_panel_id(lineage: str, source_snapshot: str, thermo_type: str) -> str:
    material = f"{lineage}|{source_snapshot}|{thermo_type}".encode("utf-8")
    return "r35-" + hashlib.sha256(material).hexdigest()


def canonical_prediction_vector_hash(frame: pd.DataFrame) -> str:
    """Hash ordered panel identifiers and exact IEEE-754 score bytes."""

    required = {"panel_unit_id", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction vector lacks {sorted(required - set(frame.columns))}")
    ordered = frame.sort_values("panel_unit_id", kind="mergesort")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False):
        digest.update(str(row.panel_unit_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(row.probability)))
    return digest.hexdigest()


def deterministic_top_k(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Return a deterministic top-k using the frozen lexical tie break."""

    if budget < 0:
        raise ValueError("budget must be nonnegative")
    return frame.sort_values(
        ["probability", "panel_unit_id"], ascending=[False, True], kind="mergesort"
    ).head(min(budget, len(frame)))


def assert_label_free_columns(columns: Iterable[str]) -> None:
    lowered = [str(column).casefold() for column in columns]
    prohibited = [
        column
        for column in lowered
        if column in EXACT_LABEL_COLUMNS
        or column.startswith(("target_", "future_", "label_", "outcome_"))
        or column.endswith(("_label", "_outcome", "_event"))
    ]
    if prohibited:
        raise ValueError(f"label/outcome columns are prohibited in R3.5A: {sorted(prohibited)}")


def exact_model_intersection(frame: pd.DataFrame, models: Iterable[str]) -> pd.DataFrame:
    models = tuple(models)
    key = ["canonical_lineage_id", "source_snapshot", "thermo_type"]
    if frame.duplicated([*key, "model_name"]).any():
        raise ValueError("prediction rows are not unique per model-candidate")
    counts = frame.groupby(key, dropna=False)["model_name"].nunique()
    accepted = counts[counts == len(models)].index
    index = pd.MultiIndex.from_frame(frame[key])
    keep = index.isin(accepted)
    selected = frame.loc[keep].copy()
    observed = set(selected["model_name"].astype(str).unique())
    if observed != set(models):
        raise ValueError(f"model set mismatch: {sorted(observed)}")
    return selected


class FormalReader:
    def __init__(self, repo: Path, access_log: Path):
        self.repo = repo
        self.access_log = access_log

    def _handle(self, relative: str, purpose: str):
        return open_formal_input(
            self.repo / relative,
            FROZEN_HASHES[relative],
            task_id=TASK_ID,
            access_log=self.access_log,
            purpose=purpose,
            allowed_roots=(self.repo,),
        )

    def json(self, relative: str, purpose: str) -> Any:
        with self._handle(relative, purpose) as handle:
            return json.load(handle)

    def yaml(self, relative: str, purpose: str) -> Any:
        with self._handle(relative, purpose) as handle:
            return yaml.safe_load(handle.read().decode("utf-8"))

    def parquet(self, relative: str, columns: list[str], purpose: str) -> pd.DataFrame:
        assert_label_free_columns(columns)
        with self._handle(relative, purpose) as handle:
            return pq.read_table(handle, columns=columns).to_pandas()

    def authenticate(self, relative: str, purpose: str) -> dict[str, Any]:
        with self._handle(relative, purpose):
            pass
        path = self.repo / relative
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "expected_sha256": FROZEN_HASHES[relative],
            "observed_sha256": sha256_file(path),
            "hash_match": True,
        }

    def authenticate_expected(self, relative: str, expected: str, purpose: str) -> dict[str, Any]:
        with open_formal_input(
            self.repo / relative,
            expected,
            task_id=TASK_ID,
            access_log=self.access_log,
            purpose=purpose,
            allowed_roots=(self.repo,),
        ):
            pass
        path = self.repo / relative
        observed = sha256_file(path)
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "hash_match": observed == expected,
        }


def _artifact(repo: Path, path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _schema_hash(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def _model_rows(p5: dict[str, Any], v2: dict[str, Any], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    license_note = "Repository MIT; MP-derived training data remain subject to Materials Project Terms/BY-C conditions"
    rows: list[dict[str, Any]] = []
    p5_feature_counts = {
        "composition_complexity_logistic": len(p5["feature_sets"]["composition_complexity"]),
        "current_hull_distance_hgb": len(p5["feature_sets"]["hull_only"]),
        "static_phase_context_hgb": len(p5["feature_sets"]["phase_context"]),
        "train_prevalence": 0,
    }
    for artifact in [item for item in p5["outputs"] if item["path"].startswith("models/P5_1/")]:
        model_id = Path(artifact["path"]).stem
        rows.append({
            "source_task": "P5.1",
            "source_gate_status": p5["gate_status"],
            "estimand": "rebuilt_unified_flip",
            "prediction_direction": "any_direction_flip",
            "model_id": model_id,
            "mechanism": model_id,
            "artifact_path": artifact["path"],
            "artifact_sha256": artifact["sha256"],
            "training_cutoff": "2023-11-01",
            "feature_schema_sha256": p5["outputs"][0]["schema_sha256"],
            "feature_count": p5_feature_counts[model_id],
            "primary_r3_5_inclusion": False,
            "exclusion_reason": "target mismatch: any-direction flip cannot be relabeled as stable-to-unstable",
            "license": license_note,
            "license_status": "CONDITIONAL",
        })
    feature_count = {(row["estimand"], row["model_name"]): row.get("feature_count") for row in v2["training_records"]}
    for artifact in v2["models"]:
        stem = Path(artifact["path"]).stem
        estimand, model_id = stem.split("__", 1)
        included = estimand == PRIMARY_ESTIMAND and model_id in PRIMARY_MODELS
        mechanism = matrix["models"].get(model_id, {}).get("mechanism", model_id)
        rows.append({
            "source_task": "V2.2",
            "source_gate_status": v2["gate_status"],
            "estimand": estimand,
            "prediction_direction": estimand,
            "model_id": model_id,
            "mechanism": mechanism,
            "artifact_path": artifact["path"],
            "artifact_sha256": artifact["sha256"],
            "training_cutoff": "2023-11-01",
            "feature_schema_sha256": v2["inputs"]["feature_schema"]["sha256"],
            "feature_count": feature_count[(estimand, model_id)],
            "primary_r3_5_inclusion": included,
            "exclusion_reason": "" if included else "estimand mismatch with primary stable-to-unstable version benchmark",
            "license": license_note,
            "license_status": "CONDITIONAL",
        })
    return sorted(rows, key=lambda row: (row["source_task"], row["estimand"], row["model_id"]))


def _protocol(panel: pd.DataFrame, frozen_predictions: pd.DataFrame, vector_hashes: dict[str, str], artifacts: dict[str, Any]) -> dict[str, Any]:
    thresholds = [
        {"name": "exact_zero", "threshold_eV_per_atom": 1.0e-8},
        {"name": "within_10meV", "threshold_eV_per_atom": 0.010},
        {"name": "within_25meV", "threshold_eV_per_atom": 0.025},
    ]
    return {
        "task_id": TASK_ID,
        "method_version": METHOD_VERSION,
        "protocol_status": "FROZEN",
        "next_authorization_status": "PI_AUTHORIZATION_PENDING",
        "seed": 42,
        "scope": {
            "phase": "protocol_freeze_only",
            "formal_evaluation_performed": False,
            "models_loaded_or_trained": False,
            "outcome_columns_read": False,
            "locked_test_access": False,
            "confirmation_a_b_outcome_access": False,
        },
        "prediction_source": {
            "task": "V2.2",
            "source_gate_status_preserved": "NO_GO",
            "interpretation": "frozen development benchmark scores; not a successful predictive model claim",
            "estimand": PRIMARY_ESTIMAND,
            "direction": "loss of threshold-defined stability from fixed source snapshot to later frozen release",
            "training_interval": "2022-10-28->2023-11-01",
            "training_cutoff": "2023-11-01",
            "score_source_snapshot": SOURCE_SNAPSHOT,
            "models": list(PRIMARY_MODELS),
            "model_count": len(PRIMARY_MODELS),
            "prediction_regeneration_performed": False,
            "prediction_recalibration_allowed": False,
            "prediction_vector_sha256": vector_hashes,
        },
        "panel": {
            "identity": "canonical_lineage_id",
            "unit_key": ["canonical_lineage_id", "prediction_source_snapshot", "thermo_type"],
            "workflow_partitioned": True,
            "base_prediction_panel_rule": "exact intersection across all frozen primary models",
            "base_prediction_panel_rows": int(len(panel)),
            "base_prediction_unique_lineages": int(panel["canonical_lineage_id"].nunique()),
            "primary_evaluation_panel_rule": "exact intersection of base prediction units having one estimable workflow-matched outcome at every frozen label version; construct from identifiers and availability before reading label values",
            "expanded_panel_rule": "per-version estimable subset; sensitivity only and never a substitute for the primary common panel",
            "availability_drift_reporting": "report exclusions and counts separately from label drift",
            "candidate_panel_artifact": artifacts["candidate_panel"],
            "frozen_prediction_artifact": artifacts["frozen_predictions"],
        },
        "label_versions": {
            "source_snapshot": SOURCE_SNAPSHOT,
            "evaluation_versions": list(LABEL_VERSIONS),
            "definitions": [
                {
                    **item,
                    "event_definition": "source_energy_above_hull <= threshold AND target_version_energy_above_hull > threshold",
                    "comparison_operator": "same threshold on source and target",
                }
                for item in thresholds
            ],
            "continuous_secondary": {
                "energy_above_hull": "target-version workflow-matched energy_above_hull",
                "signed_margin": "target-version signed stability margin using the frozen R3.1 convention",
                "spearman_target": "target_version_energy_above_hull - source_energy_above_hull (greater means destabilization)",
            },
        },
        "metrics": {
            "primary": ["average_precision", "precision_at_budget", "recall_at_budget", "brier_score", "error_transition"],
            "secondary": ["roc_auc", "expected_calibration_error", "spearman_continuous"],
            "budgets": [100, 500, 1000, "top_1pct", "top_5pct"],
            "fractional_budget_rounding": "ceil(fraction * estimable panel rows), minimum one when panel nonempty",
            "ranking": "probability descending, then panel_unit_id ascending lexical tie-break",
            "ece": {"bins": 10, "edges": [round(index / 10, 1) for index in range(11)], "binning": "equal-width; left-closed/right-open except final bin closed"},
            "error_transition": "for each fixed budget selection, paired counts TP->FP, FP->TP, FN->TN, TN->FN between label versions",
            "delta_direction": "later version minus earlier version; report absolute and paired-bootstrap interval",
        },
        "bootstrap": {
            "unit": "canonical_lineage_id",
            "replicates": 2000,
            "seed": 42,
            "sampling": "lineages sampled with replacement from the primary common panel",
            "cluster_binding": "all workflow rows, models, label definitions, and label versions for a sampled lineage remain bound within each draw",
            "paired": True,
            "interval": "percentile 2.5%, 50%, 97.5%",
            "duplicate_lineage_weighting": "multiplicity in bootstrap draw",
        },
        "nonestimable_conditions": [
            "missing candidate lineage at any frozen label version on the primary panel",
            "missing or mismatched thermo workflow partition",
            "zero or multiple target states for a panel-unit/version key",
            "nonfinite required energy or signed margin",
            "duplicate model-candidate predictions or missing primary model prediction",
            "prediction bytes/vector hash differ across label versions",
            "no positive or no negative outcomes for a requested discrimination metric",
            "budget resolves to zero or exceeds usable rows (clip only to N and disclose)",
            "fewer than two canonical lineages in a bootstrap estimand",
        ],
        "execution_order_for_r3_5b": [
            "verify protocol sidecar and all frozen input/output hashes",
            "construct availability-only exact common panel without inspecting label values",
            "freeze and hash the common-panel index",
            "read only explicitly authorized non-locked version outcomes",
            "apply frozen label definitions and metrics without tuning or recalibration",
            "run paired lineage bootstrap and independent verification",
        ],
        "prohibitions": [
            "no model retraining, tuning, replacement, score recalibration, or score regeneration",
            "no locked-test, Confirmation A, or Confirmation B outcome access",
            "no metric, budget, panel, label, bin, or bootstrap-rule change after this freeze without a new signed protocol revision",
            "no interpretation of P5.1 or V2.2 as a passed model gate",
        ],
    }


def build(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    report_dir = repo / "reports/R3_5A"
    manifest_dir = repo / "data/manifests/R3_5"
    processed_dir = repo / "data/processed/R3_5"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    access_log = report_dir / "input_access_log.jsonl"
    access_log.unlink(missing_ok=True)
    reader = FormalReader(repo, access_log)

    input_audit = []
    for relative in FROZEN_HASHES:
        input_audit.append(reader.authenticate(relative, "R3.5A frozen input inventory"))

    p5 = reader.json("data/manifests/P5_1/baseline_manifest.json", "P5.1 artifact inventory")
    v2 = reader.json("data/manifests/V2_2/model_manifest.json", "V2.2 artifact inventory")
    matrix = reader.yaml("configs/analysis/v2_1_model_matrix.yaml", "V2.2 frozen model definitions")
    r31 = reader.json("reports/R3_1/report.json", "R3.5A prerequisite verification")
    r32 = reader.json("reports/R3_2/report.json", "R3.5A prerequisite verification")
    if r31.get("task_status") != "DONE" or r32.get("task_status") != "DONE":
        raise RuntimeError("R3.1 and R3.2 must both be DONE")
    for artifact in [item for item in p5["outputs"] if item["path"].startswith("models/P5_1/")]:
        input_audit.append(reader.authenticate_expected(artifact["path"], artifact["sha256"], "P5.1 frozen model hash audit; model object not loaded"))
    for artifact in v2["models"]:
        input_audit.append(reader.authenticate_expected(artifact["path"], artifact["sha256"], "V2.2 frozen model hash audit; model object not loaded"))

    p5_columns = ["transition_id", "canonical_lineage_id", "assigned_role", "model_name", "probability"]
    p5_predictions = reader.parquet(
        "data/processed/P5_1/baseline_predictions.parquet", p5_columns, "label-free P5.1 prediction inventory"
    )
    v2_columns = [
        "transition_id", "canonical_lineage_id", "assigned_role", "source_snapshot", "thermo_type",
        "phase_context_chemsys", "estimand", "model_name", "evaluation_method", "probability",
    ]
    predictions = reader.parquet(
        "reports/V2_2/development_predictions.parquet", v2_columns, "label-free V2.2 prediction panel selection"
    )
    eligible = predictions.loc[
        (predictions["estimand"] == PRIMARY_ESTIMAND)
        & (predictions["assigned_role"] == "validation")
        & (predictions["source_snapshot"] == SOURCE_SNAPSHOT)
        & predictions["model_name"].isin(PRIMARY_MODELS)
    ].copy()
    selected = exact_model_intersection(eligible, PRIMARY_MODELS)
    if selected.empty:
        raise RuntimeError("no usable exact-intersection prediction panel")
    if not np.isfinite(selected["probability"]).all() or not selected["probability"].between(0, 1).all():
        raise RuntimeError("predictions contain invalid probabilities")
    selected["panel_unit_id"] = [
        canonical_panel_id(lineage, snapshot, workflow)
        for lineage, snapshot, workflow in zip(
            selected["canonical_lineage_id"], selected["source_snapshot"], selected["thermo_type"]
        )
    ]
    if selected.duplicated(["panel_unit_id", "model_name"]).any():
        raise RuntimeError("panel unit/model uniqueness failed")

    transition_counts = selected.groupby("panel_unit_id")["transition_id"].nunique()
    if not (transition_counts == 1).all():
        raise RuntimeError("transition identity is inconsistent across models")
    panel = (
        selected.sort_values(["panel_unit_id", "model_name"], kind="mergesort")
        .drop_duplicates("panel_unit_id")
        [["panel_unit_id", "transition_id", "canonical_lineage_id", "source_snapshot", "thermo_type", "phase_context_chemsys"]]
        .rename(columns={"source_snapshot": "prediction_source_snapshot"})
        .sort_values("panel_unit_id", kind="mergesort")
        .reset_index(drop=True)
    )
    panel["transition_id"] = panel["transition_id"].map(lambda value: bytes(value).hex())
    panel["model_count"] = len(PRIMARY_MODELS)
    frozen = selected[
        ["panel_unit_id", "canonical_lineage_id", "source_snapshot", "thermo_type", "model_name", "probability", "evaluation_method"]
    ].rename(columns={"source_snapshot": "prediction_source_snapshot", "model_name": "model_id"})
    frozen = frozen.sort_values(["model_id", "panel_unit_id"], kind="mergesort").reset_index(drop=True)
    assert_label_free_columns(frozen.columns)

    panel_path = processed_dir / "candidate_panel.parquet"
    prediction_path = processed_dir / "frozen_predictions.parquet"
    panel.to_parquet(panel_path, index=False, compression="zstd")
    frozen.to_parquet(prediction_path, index=False, compression="zstd")
    artifacts = {
        "candidate_panel": _artifact(repo, panel_path, rows=len(panel)),
        "frozen_predictions": _artifact(repo, prediction_path, rows=len(frozen)),
    }
    vector_hashes = {
        model: canonical_prediction_vector_hash(
            frozen.loc[frozen["model_id"] == model, ["panel_unit_id", "probability"]]
        )
        for model in PRIMARY_MODELS
    }

    model_rows = _model_rows(p5, v2, matrix)
    pd.DataFrame(model_rows).to_csv(report_dir / "model_inventory.csv", index=False)
    artifact_rows = [
        {
            "source_task": "P5.1",
            "artifact_type": "predictions",
            "path": "data/processed/P5_1/baseline_predictions.parquet",
            "sha256": FROZEN_HASHES["data/processed/P5_1/baseline_predictions.parquet"],
            "rows": len(p5_predictions),
            "schema_sha256": _schema_hash(repo / "data/processed/P5_1/baseline_predictions.parquet"),
            "label_free_read": True,
            "primary_inclusion": False,
            "reason": "any-direction target mismatch",
        },
        {
            "source_task": "P5.1",
            "artifact_type": "source_features",
            "path": "data/processed/P5_1/baseline_features.parquet",
            "sha256": FROZEN_HASHES["data/processed/P5_1/baseline_features.parquet"],
            "rows": p5["outputs"][0]["rows"],
            "schema_sha256": p5["outputs"][0]["schema_sha256"],
            "label_free_read": "not_read_content_hash_authenticated_only",
            "primary_inclusion": False,
            "reason": "inventory only",
        },
        {
            "source_task": "V2.2",
            "artifact_type": "predictions",
            "path": "reports/V2_2/development_predictions.parquet",
            "sha256": FROZEN_HASHES["reports/V2_2/development_predictions.parquet"],
            "rows": len(predictions),
            "schema_sha256": _schema_hash(repo / "reports/V2_2/development_predictions.parquet"),
            "label_free_read": True,
            "primary_inclusion": True,
            "reason": "eligible stable-to-unstable frozen validation scores",
        },
        {
            "source_task": "V2.1",
            "artifact_type": "source_features",
            "path": "data/processed/V2_1/source_features.parquet",
            "sha256": FROZEN_HASHES["data/processed/V2_1/source_features.parquet"],
            "rows": v2["inputs"]["source_features"]["rows"],
            "schema_sha256": v2["inputs"]["source_features"]["schema_sha256"],
            "label_free_read": "not_read_content_hash_authenticated_only",
            "primary_inclusion": True,
            "reason": "source-only feature provenance; prediction regeneration not required",
        },
    ]
    pd.DataFrame(artifact_rows).to_csv(report_dir / "artifact_inventory.csv", index=False)
    pd.DataFrame(input_audit).to_csv(report_dir / "input_hash_audit.csv", index=False)

    coverage = []
    for model in PRIMARY_MODELS:
        model_frame = frozen[frozen["model_id"] == model]
        coverage.append({
            "model_id": model,
            "eligible_rows": int((eligible["model_name"] == model).sum()),
            "panel_rows": len(model_frame),
            "panel_unique_units": model_frame["panel_unit_id"].nunique(),
            "missing_from_exact_intersection": int((eligible["model_name"] == model).sum() - len(model_frame)),
            "duplicate_model_candidate_rows": int(model_frame.duplicated(["panel_unit_id", "model_id"]).sum()),
            "probability_missing": int(model_frame["probability"].isna().sum()),
            "prediction_vector_sha256": vector_hashes[model],
        })
    pd.DataFrame(coverage).to_csv(report_dir / "panel_coverage.csv", index=False)
    pd.DataFrame([
        {"source_task": "P5.1", "estimand": "rebuilt_unified_flip", "models": 4, "decision": "EXCLUDED_PRIMARY", "reason": "any-direction target mismatch; P5.1 STOPPED/FAIL preserved"},
        {"source_task": "V2.2", "estimand": "stable_to_unstable_event", "models": 6, "decision": "INCLUDED_PRIMARY", "reason": "direction matches versioned threshold-loss benchmark; V2.2 NO_GO preserved"},
        {"source_task": "V2.2", "estimand": "competitor_inventory_revision_event", "models": 5, "decision": "EXCLUDED_PRIMARY", "reason": "outcome mismatch"},
        {"source_task": "V2.2", "estimand": "candidate_displacement_event", "models": 5, "decision": "EXCLUDED_PRIMARY", "reason": "conditional mechanism outcome mismatch"},
    ]).to_csv(report_dir / "exclusion_inventory.csv", index=False)
    pd.DataFrame([
        {"audit": "prediction_unique_per_model_candidate", "value": int(not frozen.duplicated(["panel_unit_id", "model_id"]).any()), "status": "PASS"},
        {"audit": "exact_model_count_per_panel_unit", "value": int((frozen.groupby("panel_unit_id")["model_id"].nunique() == 6).all()), "status": "PASS"},
        {"audit": "label_columns_in_frozen_predictions", "value": 0, "status": "PASS"},
        {"audit": "prediction_regeneration_count", "value": 0, "status": "PASS"},
        {"audit": "model_training_count", "value": 0, "status": "PASS"},
        {"audit": "locked_or_confirmation_outcome_reads", "value": 0, "status": "PASS"},
    ]).to_csv(report_dir / "prediction_uniqueness_audit.csv", index=False)
    pd.DataFrame([
        {"model_id": model, "canonical_vector_sha256": digest, "label_version_count": len(LABEL_VERSIONS), "storage_rule": "single immutable vector cross-referenced by every label version", "status": "FROZEN"}
        for model, digest in vector_hashes.items()
    ]).to_csv(report_dir / "prediction_hash_audit.csv", index=False)

    protocol = _protocol(panel, frozen, vector_hashes, artifacts)
    protocol_path = manifest_dir / "protocol.json"
    write_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)
    (manifest_dir / "protocol.sha256").write_text(f"{protocol_hash}  protocol.json\n", encoding="ascii", newline="\n")
    return {
        "panel_rows": len(panel),
        "unique_lineages": panel["canonical_lineage_id"].nunique(),
        "prediction_rows": len(frozen),
        "models": list(PRIMARY_MODELS),
        "thermo_counts": panel["thermo_type"].value_counts().sort_index().to_dict(),
        "artifacts": artifacts,
        "protocol_sha256": protocol_hash,
        "input_count": len(input_audit),
        "input_access_records": sum(1 for _ in access_log.open(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    summary = build(args.repo)
    write_json(args.repo / "reports/R3_5A/build_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
