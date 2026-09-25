from __future__ import annotations

import hashlib
import base64
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pymatgen.core import Element
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation.metrics import expected_calibration_error, top_fraction_enrichment


PHASE_COLUMNS = [
    "snapshot_id",
    "thermo_type",
    "phase_context_chemsys",
    "is_target",
    "is_competitor",
    "thermo_id",
    "material_id",
    "composition_json",
    "nelements",
    "num_atoms",
    "correction",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "decomposition_component_count",
    "source_record_count",
]

LABEL_COLUMNS = [
    "transition_id",
    "canonical_lineage_id",
    "assigned_role",
    "identity_confidence",
    "source_snapshot",
    "target_snapshot",
    "source_material_id",
    "source_thermo_id",
    "thermo_type",
    "rebuilt_unified_flip",
]

IDENTIFIER_COLUMNS = [
    "transition_id",
    "canonical_lineage_id",
    "assigned_role",
    "identity_confidence",
    "source_snapshot",
    "source_material_id",
    "source_thermo_id",
    "thermo_type",
    "phase_context_chemsys",
]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(path: str | Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(value))
    os.replace(temporary, target)


def _write_csv_atomic(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, target)


def _write_parquet_atomic(
    path: str | Path, frame: pd.DataFrame, *, compression: str, row_group_size: int
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression=compression,
        row_group_size=row_group_size,
    )
    os.replace(temporary, target)


def _dump_joblib_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3, protocol=5)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _verify_p4_signature(manifest_path: str | Path, signature_path: str | Path, public_key_path: str | Path) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    signature = base64.b64decode(
        Path(signature_path).read_text(encoding="utf-8").strip()
    )
    public = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public, Ed25519PublicKey):
        raise TypeError("P4.2 manifest public key is not Ed25519")
    public.verify(signature, payload)


def _composition_statistics(value: str) -> tuple[float, float, float, float]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("composition_json must be a non-empty element-to-amount mapping")
    amounts = np.asarray([float(amount) for amount in parsed.values()], dtype=float)
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError("composition_json contains invalid amounts")
    fractions = amounts / amounts.sum()
    atomic_numbers = np.asarray([float(Element(symbol).Z) for symbol in parsed], dtype=float)
    entropy = float(-(fractions * np.log(fractions)).sum())
    maximum = float(fractions.max())
    mean_z = float(np.dot(fractions, atomic_numbers))
    std_z = float(np.sqrt(np.dot(fractions, (atomic_numbers - mean_z) ** 2)))
    return entropy, maximum, mean_z, std_z


def _context_aggregates(phase: pd.DataFrame, stable_tolerance: float) -> pd.DataFrame:
    context_key = ["snapshot_id", "thermo_type", "phase_context_chemsys"]
    work = phase.copy()
    work["_target"] = work["is_target"].astype(np.int64)
    work["_competitor"] = work["is_competitor"].astype(np.int64)
    work["_stable"] = (work["energy_above_hull"] <= stable_tolerance).astype(np.int64)
    for mev in (10, 25, 50):
        work[f"_near_{mev}"] = (
            work["energy_above_hull"] <= (mev / 1000.0 + stable_tolerance)
        ).astype(np.int64)
    work["_positive_hull"] = work["energy_above_hull"].where(
        work["energy_above_hull"] > stable_tolerance
    )
    grouped = work.groupby(context_key, sort=False, dropna=False)
    result = grouped.agg(
        context_entry_count=("energy_above_hull", "size"),
        context_target_count=("_target", "sum"),
        context_competitor_count=("_competitor", "sum"),
        context_stable_count=("_stable", "sum"),
        context_near_hull_10meV_count=("_near_10", "sum"),
        context_near_hull_25meV_count=("_near_25", "sum"),
        context_near_hull_50meV_count=("_near_50", "sum"),
        context_mean_energy_above_hull=("energy_above_hull", "mean"),
        context_std_energy_above_hull=("energy_above_hull", "std"),
        context_min_positive_energy_above_hull=("_positive_hull", "min"),
    ).reset_index()
    result["context_std_energy_above_hull"] = result[
        "context_std_energy_above_hull"
    ].fillna(0.0)
    result["context_min_positive_energy_above_hull"] = result[
        "context_min_positive_energy_above_hull"
    ].fillna(0.0)
    result["context_competitor_fraction"] = (
        result["context_competitor_count"] / result["context_entry_count"]
    )
    return result


def assemble_source_features(
    labels: pd.DataFrame, phase: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    stable_tolerance = float(
        config["features"]["stable_energy_tolerance_eV_per_atom"]
    )
    population = config["population"]
    allowed_roles = {population["train_role"], population["validation_role"]}
    if set(labels["assigned_role"].unique()) != allowed_roles:
        raise RuntimeError("development labels must contain exactly train and validation roles")
    expected_intervals = {
        population["train_role"]: (
            population["train_source_snapshot"],
            population["train_target_snapshot"],
        ),
        population["validation_role"]: (
            population["validation_source_snapshot"],
            population["validation_target_snapshot"],
        ),
    }
    wrong_interval = sum(
        int(
            (
                (labels.loc[labels["assigned_role"] == role, "source_snapshot"] != source)
                | (labels.loc[labels["assigned_role"] == role, "target_snapshot"] != target)
            ).sum()
        )
        for role, (source, target) in expected_intervals.items()
    )
    if wrong_interval:
        raise RuntimeError(f"{wrong_interval} development rows use a wrong temporal interval")
    train_lineages = set(
        labels.loc[
            labels["assigned_role"] == population["train_role"],
            "canonical_lineage_id",
        ]
    )
    validation_lineages = set(
        labels.loc[
            labels["assigned_role"] == population["validation_role"],
            "canonical_lineage_id",
        ]
    )
    overlap = train_lineages & validation_lineages
    if overlap:
        raise RuntimeError(f"{len(overlap)} lineages cross train/validation roles")

    source_snapshots = {source for source, _ in expected_intervals.values()}
    phase = phase[phase["snapshot_id"].isin(source_snapshots)].copy()
    context = _context_aggregates(phase, stable_tolerance)
    targets = phase[phase["is_target"]].copy()
    target_key = list(population["phase_target_key"])
    counts = (
        targets.groupby(target_key, dropna=False, sort=False)
        .size()
        .rename("_target_count")
        .reset_index()
    )
    targets = targets.merge(counts, on=target_key, how="left", validate="many_to_one")
    targets = targets[targets["_target_count"] == 1].drop(columns="_target_count")
    targets = targets.merge(
        context,
        on=["snapshot_id", "thermo_type", "phase_context_chemsys"],
        how="left",
        validate="many_to_one",
    )

    merged = labels.merge(
        targets,
        left_on=list(population["source_only_phase_join_key"]),
        right_on=target_key,
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_phase"),
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise RuntimeError(f"{missing} development rows lack one unique source P3.2 target")
    merged = merged.drop(columns="_merge")

    composition_rows = [
        _composition_statistics(value) for value in merged["composition_json"]
    ]
    composition = np.asarray(composition_rows, dtype=float)
    merged["source_energy_above_hull"] = merged["energy_above_hull"].clip(lower=0.0)
    merged["source_log1p_hull_distance_10meV"] = np.log1p(
        merged["source_energy_above_hull"] / 0.01
    )
    merged["source_is_stable"] = merged["is_stable"].astype(np.int8)
    merged["source_formation_energy_per_atom"] = merged[
        "formation_energy_per_atom"
    ]
    merged["source_nelements"] = merged["nelements"]
    merged["source_log1p_num_atoms"] = np.log1p(merged["num_atoms"])
    merged["source_composition_entropy"] = composition[:, 0]
    merged["source_max_element_fraction"] = composition[:, 1]
    merged["source_mean_atomic_number"] = composition[:, 2]
    merged["source_std_atomic_number"] = composition[:, 3]
    merged["source_abs_correction_per_atom"] = (
        merged["correction"].abs() / merged["num_atoms"]
    )
    merged["source_decomposition_component_count"] = merged[
        "decomposition_component_count"
    ]
    merged["source_record_count"] = merged["source_record_count"]
    for thermo_type in ("GGA_GGA+U", "GGA_GGA+U_R2SCAN", "R2SCAN"):
        merged[f"thermo_{thermo_type}"] = (merged["thermo_type"] == thermo_type).astype(
            np.int8
        )

    all_features = list(
        dict.fromkeys(
            config["features"]["hull_only"]
            + config["features"]["composition_complexity"]
            + config["features"]["phase_context"]
        )
    )
    missing_features = [name for name in all_features if name not in merged]
    if missing_features:
        raise RuntimeError(f"missing configured features: {missing_features}")
    feature_values = merged[all_features].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise RuntimeError("source feature matrix contains non-finite values")
    forbidden = tuple(config["features"]["forbidden_feature_patterns"])
    forbidden_selected = [
        name
        for name in all_features
        if any(
            (
                name.casefold().startswith(pattern.casefold())
                if pattern.endswith("_")
                else pattern.casefold() in name.casefold()
            )
            for pattern in forbidden
        )
    ]
    if forbidden_selected:
        raise RuntimeError(f"future/target feature names are forbidden: {forbidden_selected}")

    output_columns = IDENTIFIER_COLUMNS + ["rebuilt_unified_flip"] + all_features
    output = merged[output_columns].copy()
    output = output.sort_values(
        ["assigned_role", "canonical_lineage_id", "transition_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit = {
        "feature_rows": len(output),
        "source_join_missing_rows": missing,
        "train_validation_lineage_overlap": len(overlap),
        "wrong_interval_rows": wrong_interval,
        "source_phase_rows": len(phase),
        "source_phase_target_rows": int(phase["is_target"].sum()),
        "source_contexts": len(context),
        "configured_features": len(all_features),
    }
    return output, audit


def _correct_balanced_probability(probability: np.ndarray, prevalence: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped))
    prior_logit = math.log(prevalence / (1.0 - prevalence))
    corrected = 1.0 / (1.0 + np.exp(-(logit + prior_logit)))
    return np.clip(corrected, 0.0, 1.0)


def _build_estimator(model_name: str, config: dict[str, Any]) -> Any:
    model_config = config["models"][model_name]
    seed = int(config["seed"])
    if model_config["type"] == "logistic_regression":
        settings = config["models"]["logistic_regression"]
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(settings["C"]),
                        max_iter=int(settings["max_iter"]),
                        solver=str(settings["solver"]),
                        class_weight=str(settings["class_weight"]),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_config["type"] == "hist_gradient_boosting":
        settings = config["models"]["hist_gradient_boosting"]
        return HistGradientBoostingClassifier(
            learning_rate=float(settings["learning_rate"]),
            max_iter=int(settings["max_iter"]),
            max_leaf_nodes=int(settings["max_leaf_nodes"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            l2_regularization=float(settings["l2_regularization"]),
            max_bins=int(settings["max_bins"]),
            early_stopping=bool(settings["early_stopping"]),
            class_weight=str(settings["class_weight"]),
            random_state=seed,
        )
    raise ValueError(f"unsupported model type for {model_name}: {model_config['type']}")


def _classification_metrics(y_true: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "ece_10": float(expected_calibration_error(y_true, y_probability, n_bins=10)),
        "top_decile_enrichment": float(
            top_fraction_enrichment(y_true, y_probability, fraction=0.10)
        ),
    }


def _fit_and_predict(
    features: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_role = config["population"]["train_role"]
    validation_role = config["population"]["validation_role"]
    train_mask = features["assigned_role"].eq(train_role).to_numpy()
    validation_mask = features["assigned_role"].eq(validation_role).to_numpy()
    y = features[config["population"]["target"]].astype(np.int8).to_numpy()
    train_prevalence = float(y[train_mask].mean())
    if not 0.0 < train_prevalence < 1.0:
        raise RuntimeError("training target must contain both classes")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    model_artifacts: dict[str, Any] = {}
    probability_by_model: dict[str, np.ndarray] = {}
    candidate_names = [
        name
        for name, value in config["models"].items()
        if isinstance(value, dict) and "type" in value
    ]
    for model_name in candidate_names:
        model_config = config["models"][model_name]
        if model_config["type"] == "constant_train_prevalence":
            probability = np.full(len(features), train_prevalence, dtype=float)
            artifact = {
                "model_name": model_name,
                "model_type": model_config["type"],
                "training_prevalence": train_prevalence,
                "feature_names": [],
                "estimator": None,
            }
        else:
            feature_set_name = model_config["feature_set"]
            feature_names = list(config["features"][feature_set_name])
            estimator = _build_estimator(model_name, config)
            estimator.fit(features.loc[train_mask, feature_names], y[train_mask])
            balanced_probability = estimator.predict_proba(features[feature_names])[:, 1]
            probability = _correct_balanced_probability(
                balanced_probability, train_prevalence
            )
            artifact = {
                "model_name": model_name,
                "model_type": model_config["type"],
                "training_prevalence": train_prevalence,
                "probability_policy": config["models"]["balanced_probability_policy"],
                "feature_names": feature_names,
                "estimator": estimator,
            }
        probability_by_model[model_name] = probability
        model_artifacts[model_name] = artifact
        prediction_frames.append(
            pd.DataFrame(
                {
                    "transition_id": features["transition_id"],
                    "canonical_lineage_id": features["canonical_lineage_id"],
                    "assigned_role": features["assigned_role"],
                    "model_name": model_name,
                    "probability": probability,
                    "rebuilt_unified_flip": y.astype(bool),
                }
            )
        )
        for role, mask in ((train_role, train_mask), (validation_role, validation_mask)):
            values = _classification_metrics(y[mask], probability[mask])
            metric_rows.append(
                {
                    "model_name": model_name,
                    "role": role,
                    "rows": int(mask.sum()),
                    "positives": int(y[mask].sum()),
                    "prevalence": float(y[mask].mean()),
                    **values,
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["model_name", "assigned_role", "canonical_lineage_id", "transition_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["role", "model_name"], kind="mergesort"
    ).reset_index(drop=True)
    return predictions, metrics, model_artifacts, probability_by_model


def _weighted_average_precision_prepared(
    y: np.ndarray, score: np.ndarray, row_weight: np.ndarray
) -> float:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    sorted_weight = row_weight[order]
    group_start = np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    group = np.cumsum(group_start) - 1
    positive = np.bincount(group, weights=sorted_weight * sorted_y)
    total = np.bincount(group, weights=sorted_weight)
    cumulative_positive = np.cumsum(positive)
    cumulative_total = np.cumsum(total)
    denominator = cumulative_positive[-1]
    if denominator <= 0:
        return float("nan")
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    return float(np.sum(precision * positive) / denominator)


def _paired_lineage_bootstrap(
    features: pd.DataFrame,
    probability_by_model: dict[str, np.ndarray],
    config: dict[str, Any],
) -> pd.DataFrame:
    validation_role = config["population"]["validation_role"]
    mask = features["assigned_role"].eq(validation_role).to_numpy()
    validation = features.loc[mask].reset_index(drop=True)
    y = validation[config["population"]["target"]].astype(float).to_numpy()
    lineage_codes, lineage_values = pd.factorize(
        validation["canonical_lineage_id"], sort=True
    )
    n_lineages = len(lineage_values)
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["seed"]))
    reference = config["gate"]["hull_reference_model"]
    relation = config["gate"]["relation_model"]
    reference_score = probability_by_model[reference][mask]
    relation_score = probability_by_model[relation][mask]
    ap_difference = np.empty(replicates, dtype=float)
    brier_difference = np.empty(replicates, dtype=float)
    for index in range(replicates):
        lineage_weight = rng.multinomial(
            n_lineages, np.full(n_lineages, 1.0 / n_lineages)
        )
        row_weight = lineage_weight[lineage_codes].astype(float)
        total_weight = row_weight.sum()
        ap_reference = _weighted_average_precision_prepared(
            y, reference_score, row_weight
        )
        ap_relation = _weighted_average_precision_prepared(y, relation_score, row_weight)
        brier_reference = float(
            np.dot(row_weight, (reference_score - y) ** 2) / total_weight
        )
        brier_relation = float(
            np.dot(row_weight, (relation_score - y) ** 2) / total_weight
        )
        ap_difference[index] = ap_relation - ap_reference
        brier_difference[index] = brier_relation - brier_reference
    alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / 2.0
    rows = []
    for metric, values, direction in (
        ("average_precision_difference_relation_minus_hull", ap_difference, "greater_is_better"),
        ("brier_difference_relation_minus_hull", brier_difference, "less_or_equal_is_better"),
    ):
        rows.append(
            {
                "relation_model": relation,
                "reference_model": reference,
                "metric": metric,
                "direction": direction,
                "replicates": replicates,
                "mean": float(np.mean(values)),
                "standard_error": float(np.std(values, ddof=1)),
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "fraction_favorable": float(
                    np.mean(values > 0) if "average_precision" in metric else np.mean(values <= 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_and_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = metrics[
        metrics["role"] == config["population"]["validation_role"]
    ].copy()
    selected_row = validation.sort_values(
        ["average_precision", "brier", "ece_10", "model_name"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    selected = {
        "model_name": str(selected_row["model_name"]),
        "selection_role": str(selected_row["role"]),
        "selection_rule": config["evaluation"]["strongest_baseline_order"],
        "validation_metrics": {
            key: float(selected_row[key])
            for key in (
                "average_precision",
                "brier",
                "ece_10",
                "top_decile_enrichment",
            )
        },
    }
    relation_name = config["gate"]["relation_model"]
    reference_name = config["gate"]["hull_reference_model"]
    relation = validation[validation["model_name"] == relation_name].iloc[0]
    reference = validation[validation["model_name"] == reference_name].iloc[0]
    tolerance = float(config["gate"]["numerical_tolerance"])
    ap_difference = float(relation["average_precision"] - reference["average_precision"])
    brier_difference = float(relation["brier"] - reference["brier"])
    checks = {
        "validation_average_precision_strictly_greater": ap_difference > tolerance,
        "validation_brier_not_greater": brier_difference <= tolerance,
        "strongest_baseline_selected": bool(selected["model_name"]),
    }
    gate = {
        "relation_model": relation_name,
        "hull_reference_model": reference_name,
        "validation_average_precision_difference": ap_difference,
        "validation_brier_difference": brier_difference,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return selected, gate


def _artifact_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    record: dict[str, Any] = {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }
    if target.suffix == ".parquet":
        record["rows"] = pq.ParquetFile(target).metadata.num_rows
        record["schema_sha256"] = _schema_sha256(target)
    elif target.suffix == ".csv":
        record["rows"] = len(pd.read_csv(target))
    return record


def _data_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "rebuilt_unified_flip": "P4.2 development-only target; never used as a feature.",
        "phase_context_chemsys": "Exact P3.2 source-snapshot phase context.",
        "source_energy_above_hull": "Current P3.2 unified hull distance at the source snapshot.",
        "context_entry_count": "Number of P3.2 entries in the exact source phase context.",
        "context_competitor_count": "Number of source-context entries used as decomposition competitors.",
    }
    return pd.DataFrame(
        [
            {
                "table": "baseline_features",
                "field": column,
                "dtype": str(features[column].dtype),
                "description": descriptions.get(
                    column,
                    "Deterministic source-only identifier, target, candidate, composition, or phase-context field.",
                ),
            }
            for column in features.columns
        ]
    )


def build_baselines(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "P5.1":
        raise ValueError("baseline config task_id must be P5.1")
    inputs = config["input"]
    output = config["output"]
    p4_report = json.loads(Path(inputs["p4_2_report"]).read_text(encoding="utf-8"))
    p3_report = json.loads(Path(inputs["p3_2_report"]).read_text(encoding="utf-8"))
    if not (
        p4_report.get("task_status") == "DONE"
        and p4_report.get("status") == "PASS"
        and p4_report.get("gate_status") == "GO"
        and p3_report.get("task_status") == "DONE"
        and p3_report.get("status") == "PASS"
        and p3_report.get("gate_status") == "GO"
    ):
        raise RuntimeError("P3.2 and P4.2 DONE/PASS/GO prerequisites are required")
    _verify_p4_signature(
        inputs["p4_2_manifest"], inputs["p4_2_signature"], inputs["p4_2_public_key"]
    )
    p4_manifest = json.loads(Path(inputs["p4_2_manifest"]).read_text(encoding="utf-8"))
    p3_manifest = json.loads(Path(inputs["p3_2_manifest"]).read_text(encoding="utf-8"))
    p4_output_hashes = {item["path"]: item["sha256"] for item in p4_manifest["outputs"]}
    if p4_output_hashes.get(inputs["development_labels"]) != _sha256(
        inputs["development_labels"]
    ):
        raise RuntimeError("P4.2 development-label hash does not match signed manifest")
    p3_output_hashes = {item["path"]: item["sha256"] for item in p3_manifest["outputs"]}
    if p3_output_hashes.get(inputs["phase_entries"]) != _sha256(inputs["phase_entries"]):
        raise RuntimeError("P3.2 phase-entry hash does not match manifest")
    if _sha256(inputs["locked_thresholds"]) != p4_manifest["locked_thresholds_sha256"]:
        raise RuntimeError("locked-threshold hash changed after P4.2")

    labels = pq.read_table(inputs["development_labels"], columns=LABEL_COLUMNS).to_pandas()
    source_snapshots = [
        config["population"]["train_source_snapshot"],
        config["population"]["validation_source_snapshot"],
    ]
    phase = pq.read_table(
        inputs["phase_entries"],
        filters=[("snapshot_id", "in", source_snapshots)],
        columns=PHASE_COLUMNS,
    ).to_pandas()
    features, audit = assemble_source_features(labels, phase, config)
    predictions, metrics, model_artifacts, probability_by_model = _fit_and_predict(
        features, config
    )
    bootstrap = _paired_lineage_bootstrap(features, probability_by_model, config)
    selected, gate = _select_and_gate(metrics, config)

    parquet = config["parquet"]
    _write_parquet_atomic(
        output["features"],
        features,
        compression=parquet["compression"],
        row_group_size=int(parquet["row_group_size"]),
    )
    _write_parquet_atomic(
        output["predictions"],
        predictions,
        compression=parquet["compression"],
        row_group_size=int(parquet["row_group_size"]),
    )
    _write_csv_atomic(output["metrics"], metrics)
    _write_csv_atomic(output["bootstrap_comparison"], bootstrap)
    _write_json_atomic(output["selected_baseline"], selected)
    _write_csv_atomic(output["data_dictionary"], _data_dictionary(features))
    model_paths = []
    for model_name in sorted(model_artifacts):
        model_path = Path(output["models_root"]) / f"{model_name}.joblib"
        _dump_joblib_atomic(model_path, model_artifacts[model_name])
        model_paths.append(model_path)

    artifact_paths = [
        output["features"],
        output["predictions"],
        output["metrics"],
        output["bootstrap_comparison"],
        output["selected_baseline"],
        output["data_dictionary"],
        *model_paths,
    ]
    input_paths = [
        config_path,
        inputs["development_labels"],
        inputs["p4_2_manifest"],
        inputs["p4_2_signature"],
        inputs["p4_2_public_key"],
        inputs["p4_2_report"],
        inputs["locked_thresholds"],
        inputs["phase_entries"],
        inputs["p3_2_manifest"],
        inputs["p3_2_report"],
    ]
    manifest = {
        "task_id": "P5.1",
        "status": "PASS" if gate["passed"] else "FAIL",
        "gate_status": "GO" if gate["passed"] else "NO_GO",
        "method_version": config["method_version"],
        "seed": int(config["seed"]),
        "network_access_for_modeling": False,
        "locked_test_access": False,
        "config_path": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "input_hashes": {Path(path).as_posix(): _sha256(path) for path in input_paths},
        "audit": audit,
        "population": {
            "rows": len(features),
            "train_rows": int((features["assigned_role"] == config["population"]["train_role"]).sum()),
            "validation_rows": int((features["assigned_role"] == config["population"]["validation_role"]).sum()),
            "train_positives": int(features.loc[features["assigned_role"] == config["population"]["train_role"], config["population"]["target"]].sum()),
            "validation_positives": int(features.loc[features["assigned_role"] == config["population"]["validation_role"], config["population"]["target"]].sum()),
        },
        "feature_sets": {
            name: config["features"][name]
            for name in ("hull_only", "composition_complexity", "phase_context")
        },
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap_comparison": bootstrap.to_dict(orient="records"),
        "selected_baseline": selected,
        "gate": gate,
        "outputs": [_artifact_record(path) for path in artifact_paths],
        "warnings": [
            "P5.1 uses development labels only; locked-test labels remain sealed.",
            "Bootstrap intervals quantify validation uncertainty but do not redefine the frozen point-estimate gate.",
        ],
    }
    _write_json_atomic(output["manifest"], manifest)
    result = dict(manifest)
    result["manifest"] = _artifact_record(output["manifest"])
    return result


def verify_baselines(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = config["output"]
    manifest = json.loads(Path(output["manifest"]).read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("task_id") != "P5.1":
        failures.append("manifest task_id is not P5.1")
    if manifest.get("config_sha256") != _sha256(config_path):
        failures.append("config hash mismatch")
    for path, expected in manifest.get("input_hashes", {}).items():
        if not Path(path).exists() or _sha256(path) != expected:
            failures.append(f"input hash mismatch: {path}")
    for artifact in manifest.get("outputs", []):
        path = artifact["path"]
        if not Path(path).exists() or _sha256(path) != artifact["sha256"]:
            failures.append(f"output hash mismatch: {path}")
    features = pd.read_parquet(output["features"])
    predictions = pd.read_parquet(output["predictions"])
    metrics = pd.read_csv(output["metrics"])
    expected_models = {
        name
        for name, value in config["models"].items()
        if isinstance(value, dict) and "type" in value
    }
    if set(predictions["model_name"].unique()) != expected_models:
        failures.append("prediction model set mismatch")
    if len(predictions) != len(features) * len(expected_models):
        failures.append("prediction row count mismatch")
    if set(features["assigned_role"].unique()) != {
        config["population"]["train_role"],
        config["population"]["validation_role"],
    }:
        failures.append("feature roles are not development-only")
    train_lineages = set(
        features.loc[
            features["assigned_role"] == config["population"]["train_role"],
            "canonical_lineage_id",
        ]
    )
    validation_lineages = set(
        features.loc[
            features["assigned_role"] == config["population"]["validation_role"],
            "canonical_lineage_id",
        ]
    )
    if train_lineages & validation_lineages:
        failures.append("train/validation lineage overlap")
    recomputed_rows = []
    for (model_name, role), group in predictions.groupby(
        ["model_name", "assigned_role"], sort=True
    ):
        values = _classification_metrics(
            group["rebuilt_unified_flip"].astype(np.int8).to_numpy(),
            group["probability"].to_numpy(),
        )
        recomputed_rows.append({"model_name": model_name, "role": role, **values})
    recomputed = pd.DataFrame(recomputed_rows)
    merged = metrics.merge(
        recomputed, on=["model_name", "role"], suffixes=("_stored", "_recomputed")
    )
    for name in ("average_precision", "brier", "ece_10", "top_decile_enrichment"):
        if not np.allclose(
            merged[f"{name}_stored"], merged[f"{name}_recomputed"], atol=1e-15, rtol=1e-12
        ):
            failures.append(f"metric mismatch: {name}")
    selected, gate = _select_and_gate(metrics, config)
    stored_selected = json.loads(Path(output["selected_baseline"]).read_text(encoding="utf-8"))
    selected_metrics_match = all(
        np.isclose(
            selected["validation_metrics"][name],
            stored_selected.get("validation_metrics", {}).get(name, np.nan),
            atol=1e-15,
            rtol=1e-12,
        )
        for name in selected["validation_metrics"]
    )
    if (
        selected["model_name"] != stored_selected.get("model_name")
        or selected["selection_role"] != stored_selected.get("selection_role")
        or selected["selection_rule"] != stored_selected.get("selection_rule")
        or not selected_metrics_match
    ):
        failures.append("selected baseline mismatch")
    stored_gate = manifest.get("gate", {})
    gate_match = (
        gate["relation_model"] == stored_gate.get("relation_model")
        and gate["hull_reference_model"] == stored_gate.get("hull_reference_model")
        and gate["checks"] == stored_gate.get("checks")
        and gate["passed"] == stored_gate.get("passed")
        and np.isclose(
            gate["validation_average_precision_difference"],
            stored_gate.get("validation_average_precision_difference", np.nan),
            atol=1e-15,
            rtol=1e-12,
        )
        and np.isclose(
            gate["validation_brier_difference"],
            stored_gate.get("validation_brier_difference", np.nan),
            atol=1e-15,
            rtol=1e-12,
        )
    )
    if not gate_match:
        failures.append("gate reconstruction mismatch")
    temporary_files = [
        str(path)
        for root in (Path(output["root"]), Path(output["models_root"]), Path(output["manifest"]).parent, Path(output["metrics"]).parent)
        if root.exists()
        for path in root.rglob("*.tmp")
    ]
    if temporary_files:
        failures.append(f"temporary files remain: {temporary_files}")
    status = "PASS" if not failures and gate["passed"] else "FAIL"
    return {
        "task_id": "P5.1",
        "status": status,
        "gate_status": "GO" if status == "PASS" else "NO_GO",
        "failures": failures,
        "feature_rows": len(features),
        "prediction_rows": len(predictions),
        "models": len(expected_models),
        "selected_baseline": selected["model_name"],
        "relation_gate": gate,
        "lineage_overlap": len(train_lineages & validation_lineages),
        "locked_test_access": False,
        "temporary_files": len(temporary_files),
    }
