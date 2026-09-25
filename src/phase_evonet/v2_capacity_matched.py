from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation.metrics import expected_calibration_error, top_fraction_enrichment


IDENTIFIER_COLUMNS = [
    "transition_id",
    "canonical_lineage_id",
    "assigned_role",
    "source_snapshot",
    "thermo_type",
    "phase_context_chemsys",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(path: str | Path) -> str:
    return hashlib.sha256(
        str(pq.ParquetFile(path).schema_arrow).encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
    os.replace(temporary, target)


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


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(probability) != len(y):
        raise ValueError("metric inputs must be aligned and non-empty")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if y.sum() == 0:
        average_precision = float("nan")
        enrichment = float("nan")
    else:
        average_precision = float(average_precision_score(y, probability))
        enrichment = float(top_fraction_enrichment(y, probability, fraction=0.10))
    return {
        "average_precision": average_precision,
        "brier": float(brier_score_loss(y, probability)),
        "ece_10": float(expected_calibration_error(y, probability, n_bins=10)),
        "top_decile_enrichment": enrichment,
    }


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def exposure_rescale_probability(
    probability: np.ndarray, source_days: float, target_days: float
) -> np.ndarray:
    if source_days <= 0 or target_days <= 0:
        raise ValueError("exposure days must be positive")
    clipped = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0 - 1e-12)
    rate = -np.log1p(-clipped) / float(source_days)
    return np.clip(-np.expm1(-rate * float(target_days)), 0.0, 1.0)


def _build_estimator(config: dict[str, Any], max_iter: int) -> HistGradientBoostingClassifier:
    settings = config["shared_budget"]
    return HistGradientBoostingClassifier(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(max_iter),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        max_bins=int(settings["max_bins"]),
        early_stopping=bool(settings["early_stopping"]),
        class_weight=str(settings["class_weight"]),
        random_state=int(settings["execution_seed"]),
    )


def _matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    missing_indicator_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    values = frame[feature_names].to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise RuntimeError("feature matrix contains infinite values")
    if missing_indicator_names is None:
        missing_indicator_names = [
            name for index, name in enumerate(feature_names) if np.isnan(values[:, index]).any()
        ]
    indicator_indices = [feature_names.index(name) for name in missing_indicator_names]
    indicators = (
        np.isnan(values[:, indicator_indices]).astype(float)
        if indicator_indices
        else np.empty((len(frame), 0), dtype=float)
    )
    values = np.nan_to_num(values, nan=0.0)
    return np.column_stack([values, indicators]), missing_indicator_names


def _fold_assignment(
    y: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(len(y), -1, dtype=np.int16)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    placeholder = np.zeros((len(y), 1), dtype=float)
    for fold, (fit_index, holdout_index) in enumerate(
        splitter.split(placeholder, y, groups=groups)
    ):
        if len(np.unique(y[fit_index])) != 2 or len(np.unique(y[holdout_index])) != 2:
            raise RuntimeError(f"calibration fold {fold} lacks one target class")
        assignment[holdout_index] = fold
        splits.append((fit_index, holdout_index))
    if (assignment < 0).any():
        raise RuntimeError("incomplete lineage calibration fold assignment")
    return assignment, splits


def _fit_calibrated(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    target: str,
    feature_names: list[str],
    config: dict[str, Any],
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_train = train_frame[target].astype(np.int8).to_numpy()
    if len(np.unique(y_train)) != 2:
        raise RuntimeError(f"training target {target} must contain both classes")
    groups = train_frame[config["population"]["calibration_fold_unit"]].astype(str).to_numpy()
    X_train, missing_names = _matrix(train_frame, feature_names)
    X_predict, check_names = _matrix(predict_frame, feature_names, missing_names)
    if check_names != missing_names:
        raise RuntimeError("missing-indicator feature drift")
    assignment, splits = _fold_assignment(
        y_train,
        groups,
        int(config["shared_budget"]["calibration_folds"]),
        int(config["shared_budget"]["execution_seed"]),
    )
    raw_oof = np.full(len(train_frame), np.nan, dtype=float)
    for fit_index, holdout_index in splits:
        estimator = _build_estimator(config, max_iter)
        estimator.fit(X_train[fit_index], y_train[fit_index])
        raw_oof[holdout_index] = estimator.predict_proba(X_train[holdout_index])[:, 1]
    if not np.isfinite(raw_oof).all():
        raise RuntimeError("cross-fit probabilities are incomplete")
    calibrator = LogisticRegression(
        C=float(config["shared_budget"]["calibration_C"]),
        solver="lbfgs",
        max_iter=1000,
        random_state=int(config["shared_budget"]["execution_seed"]),
    )
    calibrator.fit(_logit(raw_oof).reshape(-1, 1), y_train)
    calibrated_oof = calibrator.predict_proba(_logit(raw_oof).reshape(-1, 1))[:, 1]
    final_estimator = _build_estimator(config, max_iter)
    final_estimator.fit(X_train, y_train)
    raw_predict = final_estimator.predict_proba(X_predict)[:, 1]
    calibrated_predict = calibrator.predict_proba(
        _logit(raw_predict).reshape(-1, 1)
    )[:, 1]
    fold_frame = pd.DataFrame(
        {
            "lineage": groups,
            "fold": assignment,
        }
    ).sort_values(["lineage", "fold"], kind="mergesort")
    fold_hash = hashlib.sha256(
        fold_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    artifact = {
        "algorithm_class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "calibration_class": "sklearn.linear_model.LogisticRegression",
        "calibration_method": config["shared_budget"]["calibration_method"],
        "calibration_fold_assignment_sha256": fold_hash,
        "calibration_folds": int(config["shared_budget"]["calibration_folds"]),
        "effective_boosting_iterations": int(max_iter),
        "feature_names": feature_names,
        "missing_indicator_features": missing_names,
        "training_rows": len(train_frame),
        "training_events": int(y_train.sum()),
        "training_lineages": int(pd.Series(groups).nunique()),
        "execution_seed": int(config["shared_budget"]["execution_seed"]),
        "estimator": final_estimator,
        "calibrator": calibrator,
    }
    return calibrated_oof, calibrated_predict, artifact


def _load_and_validate(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != "V2.2":
        raise ValueError("capacity-matched config task_id must be V2.2")
    if config.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("V2.2 execution config was not frozen before training")
    inputs = config["input"]
    expected = inputs["expected_sha256"]
    key_to_path = {
        "v2_1_report": inputs["v2_1_report"],
        "v2_1_model_matrix": inputs["v2_1_model_matrix"],
        "feature_schema": inputs["feature_schema"],
        "source_features": inputs["source_features"],
        "cause_specific_targets": inputs["cause_specific_targets"],
        "interval_exposure": inputs["interval_exposure"],
    }
    input_records: dict[str, Any] = {}
    for key, path in key_to_path.items():
        actual = _sha256(path)
        if actual != expected[key]:
            raise RuntimeError(f"frozen input hash mismatch for {key}: {actual}")
        input_records[key] = _artifact_record(path)
    sidecar_hash = Path(inputs["v2_1_report_sidecar"]).read_text(encoding="utf-8").split()[0]
    if sidecar_hash != expected["v2_1_report"]:
        raise RuntimeError("V2.1 report sidecar mismatch")
    v2_report = json.loads(Path(inputs["v2_1_report"]).read_text(encoding="utf-8"))
    if not (
        v2_report.get("task_status") == "DONE"
        and v2_report.get("status") == "PASS"
        and v2_report.get("gate_status") == "GO"
        and v2_report.get("locked_outcome_access_count") == 0
        and v2_report.get("model_training_performed") is False
    ):
        raise RuntimeError("V2.1 entry conditions are not satisfied")
    matrix = yaml.safe_load(Path(inputs["v2_1_model_matrix"]).read_text(encoding="utf-8"))
    schema = json.loads(Path(inputs["feature_schema"]).read_text(encoding="utf-8"))
    if matrix.get("training_authorized") is not False:
        raise RuntimeError("frozen V2.1 configuration unexpectedly authorizes training")
    if matrix.get("seed_schedule") != config.get("declared_seed_schedule"):
        raise RuntimeError("declared seed schedule differs from V2.1")
    for key in (
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_bins",
        "early_stopping",
        "class_weight",
        "calibration_folds",
    ):
        if matrix["shared_budget"].get(key) != config["shared_budget"].get(key):
            raise RuntimeError(f"V2.2 changed frozen shared budget field {key}")
    if int(matrix["shared_budget"]["effective_total_boosting_iterations"]) != int(
        config["shared_budget"]["effective_total_boosting_iterations"]
    ):
        raise RuntimeError("V2.2 changed the frozen boosting budget")
    features = pd.read_parquet(inputs["source_features"])
    targets = pd.read_parquet(inputs["cause_specific_targets"])
    exposure = pd.read_parquet(inputs["interval_exposure"])
    if not (len(features) == len(targets) == len(exposure) == 286357):
        raise RuntimeError("V2.1 aligned row count changed")
    if any(frame["transition_id"].duplicated().any() for frame in (features, targets, exposure)):
        raise RuntimeError("duplicate transition IDs in V2.1 input")
    target_columns = [
        column
        for column in targets.columns
        if column not in IDENTIFIER_COLUMNS and column not in {"target_snapshot"}
    ]
    exposure_columns = [
        column
        for column in exposure.columns
        if column
        not in {
            "transition_id",
            "canonical_lineage_id",
            "assigned_role",
            "source_snapshot",
            "target_snapshot",
        }
    ]
    joined = features.merge(
        targets[[*IDENTIFIER_COLUMNS, "target_snapshot", *target_columns]],
        on=IDENTIFIER_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    joined = joined.merge(
        exposure[
            [
                "transition_id",
                "canonical_lineage_id",
                "assigned_role",
                "source_snapshot",
                "target_snapshot",
                *exposure_columns,
            ]
        ],
        on=[
            "transition_id",
            "canonical_lineage_id",
            "assigned_role",
            "source_snapshot",
            "target_snapshot",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features):
        raise RuntimeError("V2.1 inputs are not one-to-one aligned")
    if set(joined["assigned_role"].unique()) != {
        config["population"]["train_role"],
        config["population"]["validation_role"],
    }:
        raise RuntimeError("non-development role encountered")
    train_lineages = set(
        joined.loc[joined["assigned_role"].eq(config["population"]["train_role"]), "canonical_lineage_id"]
    )
    validation_lineages = set(
        joined.loc[joined["assigned_role"].eq(config["population"]["validation_role"]), "canonical_lineage_id"]
    )
    overlap = train_lineages & validation_lineages
    all_feature_names = schema["feature_columns"]
    if not np.isfinite(joined[all_feature_names].to_numpy(dtype=float)).all():
        raise RuntimeError("V2.1 source feature table contains non-finite values")
    audit = {
        "input_rows": len(joined),
        "train_rows": int(joined["assigned_role"].eq(config["population"]["train_role"]).sum()),
        "validation_rows": int(joined["assigned_role"].eq(config["population"]["validation_role"]).sum()),
        "train_validation_lineage_overlap": len(overlap),
        "duplicate_transition_ids": int(joined["transition_id"].duplicated().sum()),
        "format_only_identifier_events": int(v2_report["identifier_audit"]["format_only_events"]),
        "locked_outcome_access_count": 0,
        "locked_or_external_outcome_files_read": [],
        "unledgered_errors": [],
        "source_feature_columns": len(all_feature_names),
    }
    return config, joined, {"matrix": matrix, "schema": schema}, {
        "inputs": input_records,
        "audit": audit,
    }


def _risk_frame(frame: pd.DataFrame, role: str, risk_column: str | None) -> pd.DataFrame:
    selected = frame[frame["assigned_role"].eq(role)]
    if risk_column:
        selected = selected[selected[risk_column].astype(bool)]
    return selected.reset_index(drop=True)


def _prediction_frame(
    frame: pd.DataFrame,
    estimand: str,
    model: str,
    target: str,
    probability: np.ndarray,
    evaluation_method: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transition_id": frame["transition_id"],
            "canonical_lineage_id": frame["canonical_lineage_id"],
            "assigned_role": frame["assigned_role"],
            "source_snapshot": frame["source_snapshot"],
            "thermo_type": frame["thermo_type"],
            "phase_context_chemsys": frame["phase_context_chemsys"],
            "source_chemsys_dimensionality": frame["source_chemsys_dimensionality"].astype(np.int8),
            "exposure_days": frame["exposure_days"].astype(np.int16),
            "estimand": estimand,
            "model_name": model,
            "evaluation_method": evaluation_method,
            "probability": np.asarray(probability, dtype=float),
            "event": frame[target].astype(bool),
        }
    )


def _metric_row(group: pd.DataFrame) -> dict[str, Any]:
    values = _classification_metrics(
        group["event"].astype(np.int8).to_numpy(), group["probability"].to_numpy()
    )
    return {
        "estimand": str(group["estimand"].iloc[0]),
        "model_name": str(group["model_name"].iloc[0]),
        "role": str(group["assigned_role"].iloc[0]),
        "evaluation_method": str(group["evaluation_method"].iloc[0]),
        "rows": len(group),
        "positives": int(group["event"].sum()),
        "prevalence": float(group["event"].mean()),
        **values,
    }


def _factorial_comparison(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validation_role = config["population"]["validation_role"]
    validation = metrics[metrics["role"].eq(validation_role)]
    rows: list[dict[str, Any]] = []
    for estimand, group in validation.groupby("estimand", sort=True):
        indexed = group.set_index("model_name")
        for increment, pair in config["comparison"]["primary_increments"].items():
            challenger, reference = pair
            if challenger not in indexed.index or reference not in indexed.index:
                continue
            c = indexed.loc[challenger]
            r = indexed.loc[reference]
            rows.append(
                {
                    "estimand": estimand,
                    "increment": increment,
                    "challenger_model": challenger,
                    "reference_model": reference,
                    "average_precision_difference": float(c["average_precision"] - r["average_precision"]),
                    "relative_average_precision_change": float(
                        (c["average_precision"] - r["average_precision"]) / r["average_precision"]
                    ),
                    "brier_difference": float(c["brier"] - r["brier"]),
                    "relative_brier_reduction": float((r["brier"] - c["brier"]) / r["brier"]),
                    "ece_difference": float(c["ece_10"] - r["ece_10"]),
                    "top_decile_enrichment_difference": float(
                        c["top_decile_enrichment"] - r["top_decile_enrichment"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["estimand", "increment"], kind="mergesort").reset_index(drop=True)


def _weighted_ap_preparation(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    groups = np.cumsum(starts) - 1
    return order, groups, int(groups[-1] + 1)


def _weighted_ap(
    y: np.ndarray,
    row_weight: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, int],
) -> float:
    order, groups, n_groups = prepared
    sorted_y = y[order]
    sorted_weight = row_weight[order]
    positives = np.bincount(groups, weights=sorted_weight * sorted_y, minlength=n_groups)
    totals = np.bincount(groups, weights=sorted_weight, minlength=n_groups)
    cumulative_positives = np.cumsum(positives)
    cumulative_totals = np.cumsum(totals)
    denominator = cumulative_positives[-1]
    if denominator <= 0:
        return float("nan")
    precision = np.divide(
        cumulative_positives,
        cumulative_totals,
        out=np.zeros_like(cumulative_positives),
        where=cumulative_totals > 0,
    )
    return float(np.dot(precision, positives) / denominator)


def _bootstrap_comparisons(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validation = predictions[
        predictions["assigned_role"].eq(config["population"]["validation_role"])
    ]
    requested = int(config["evaluation"]["bootstrap_replicates"])
    alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / 2.0
    rows: list[dict[str, Any]] = []
    for estimand, group in validation.groupby("estimand", sort=True):
        key_columns = ["transition_id", "canonical_lineage_id", "event"]
        pivot = group.pivot(index=key_columns, columns="model_name", values="probability").reset_index()
        for increment, pair in config["comparison"]["primary_increments"].items():
            challenger, reference = pair
            if challenger not in pivot or reference not in pivot:
                continue
            subset = pivot.dropna(subset=[challenger, reference]).reset_index(drop=True)
            y = subset["event"].astype(float).to_numpy()
            challenger_score = subset[challenger].to_numpy(dtype=float)
            reference_score = subset[reference].to_numpy(dtype=float)
            lineage_codes, lineage_values = pd.factorize(subset["canonical_lineage_id"], sort=True)
            n_lineages = len(lineage_values)
            seed_material = f"{config['seed']}|{estimand}|{increment}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            challenger_prepared = _weighted_ap_preparation(challenger_score)
            reference_prepared = _weighted_ap_preparation(reference_score)
            ap_differences = np.full(requested, np.nan, dtype=float)
            brier_differences = np.full(requested, np.nan, dtype=float)
            for replicate in range(requested):
                lineage_weight = rng.multinomial(
                    n_lineages, np.full(n_lineages, 1.0 / n_lineages)
                )
                row_weight = lineage_weight[lineage_codes].astype(float)
                total_weight = row_weight.sum()
                challenger_ap = _weighted_ap(y, row_weight, challenger_prepared)
                reference_ap = _weighted_ap(y, row_weight, reference_prepared)
                ap_differences[replicate] = challenger_ap - reference_ap
                challenger_brier = np.dot(row_weight, (challenger_score - y) ** 2) / total_weight
                reference_brier = np.dot(row_weight, (reference_score - y) ** 2) / total_weight
                brier_differences[replicate] = challenger_brier - reference_brier
            for metric, values, favorable in (
                ("average_precision_difference_challenger_minus_reference", ap_differences, "positive"),
                ("brier_difference_challenger_minus_reference", brier_differences, "nonpositive"),
            ):
                valid = values[np.isfinite(values)]
                rows.append(
                    {
                        "estimand": estimand,
                        "increment": increment,
                        "challenger_model": challenger,
                        "reference_model": reference,
                        "metric": metric,
                        "favorable_direction": favorable,
                        "replicates_requested": requested,
                        "replicates_valid": len(valid),
                        "lineages": n_lineages,
                        "mean": float(valid.mean()),
                        "standard_error": float(valid.std(ddof=1)),
                        "ci_lower": float(np.quantile(valid, alpha)),
                        "ci_upper": float(np.quantile(valid, 1.0 - alpha)),
                        "fraction_favorable": float(
                            np.mean(valid > 0) if favorable == "positive" else np.mean(valid <= 0)
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["estimand", "increment", "metric"], kind="mergesort").reset_index(drop=True)


def _select_non_relational(metrics: pd.DataFrame, estimand: str, config: dict[str, Any]) -> dict[str, Any]:
    candidates = config["comparison"]["non_relational_candidates"]
    rows = metrics[
        metrics["estimand"].eq(estimand)
        & metrics["role"].eq(config["population"]["validation_role"])
        & metrics["model_name"].isin(candidates)
    ].copy()
    selected = rows.sort_values(
        ["average_precision", "brier", "ece_10", "model_name"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return {key: (str(selected[key]) if key == "model_name" else float(selected[key])) for key in ["model_name", "average_precision", "brier", "ece_10", "top_decile_enrichment"]}


def _chemistry_strata(
    predictions: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    estimand = "stable_to_unstable_event"
    baseline = _select_non_relational(metrics, estimand, config)["model_name"]
    validation = predictions[
        predictions["assigned_role"].eq(config["population"]["validation_role"])
        & predictions["estimand"].eq(estimand)
        & predictions["model_name"].isin([baseline, config["comparison"]["flagship_relation_model"]])
    ]
    key = [
        "transition_id",
        "canonical_lineage_id",
        "event",
        "source_chemsys_dimensionality",
    ]
    pivot = validation.pivot(index=key, columns="model_name", values="probability").reset_index()
    challenger = config["comparison"]["flagship_relation_model"]
    tolerance = float(config["evaluation"]["numerical_tolerance"])
    rows: list[dict[str, Any]] = []
    for bucket in config["evaluation"]["chemistry_strata"]["buckets"]:
        lower = int(bucket["minimum"])
        upper = bucket["maximum"]
        mask = pivot["source_chemsys_dimensionality"].ge(lower)
        if upper is not None:
            mask &= pivot["source_chemsys_dimensionality"].le(int(upper))
        subset = pivot[mask]
        y = subset["event"].astype(np.int8).to_numpy()
        estimable = len(subset) > 0 and 0 < int(y.sum()) < len(subset)
        if estimable:
            c = _classification_metrics(y, subset[challenger].to_numpy())
            r = _classification_metrics(y, subset[baseline].to_numpy())
            ap_difference = c["average_precision"] - r["average_precision"]
            brier_difference = c["brier"] - r["brier"]
            relative_ap = ap_difference / r["average_precision"]
            relative_brier_reduction = (r["brier"] - c["brier"]) / r["brier"]
            favorable = ap_difference > tolerance and brier_difference <= tolerance
            material_adverse = relative_ap <= -0.20 or relative_brier_reduction <= -0.10
        else:
            c = {name: float("nan") for name in ["average_precision", "brier", "ece_10", "top_decile_enrichment"]}
            r = c.copy()
            ap_difference = brier_difference = relative_ap = relative_brier_reduction = float("nan")
            favorable = material_adverse = False
        rows.append(
            {
                "estimand": estimand,
                "stratum": bucket["name"],
                "challenger_model": challenger,
                "reference_model": baseline,
                "rows": len(subset),
                "events": int(y.sum()),
                "estimable": bool(estimable),
                "challenger_average_precision": c["average_precision"],
                "reference_average_precision": r["average_precision"],
                "average_precision_difference": ap_difference,
                "relative_average_precision_change": relative_ap,
                "challenger_brier": c["brier"],
                "reference_brier": r["brier"],
                "brier_difference": brier_difference,
                "relative_brier_reduction": relative_brier_reduction,
                "favorable": bool(favorable),
                "material_adverse": bool(material_adverse),
            }
        )
    return pd.DataFrame(rows)


def _calibration_table(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["estimand", "model_name", "assigned_role", "evaluation_method"], sort=True
    ):
        probability = group["probability"].to_numpy()
        event = group["event"].astype(float).to_numpy()
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (probability >= lower) & (
                probability < upper if upper < 1.0 else probability <= upper
            )
            rows.append(
                {
                    "estimand": keys[0],
                    "model_name": keys[1],
                    "role": keys[2],
                    "evaluation_method": keys[3],
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "rows": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()) if mask.any() else float("nan"),
                    "event_rate": float(event[mask].mean()) if mask.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _compute_gate(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    strata: pd.DataFrame,
    audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    estimand = "stable_to_unstable_event"
    baseline = _select_non_relational(metrics, estimand, config)
    validation = metrics[
        metrics["estimand"].eq(estimand)
        & metrics["role"].eq(config["population"]["validation_role"])
    ].set_index("model_name")
    relation_name = config["comparison"]["flagship_relation_model"]
    relation = validation.loc[relation_name]
    relative_ap = float(
        (relation["average_precision"] - baseline["average_precision"])
        / baseline["average_precision"]
    )
    relative_brier = float(
        (baseline["brier"] - relation["brier"]) / baseline["brier"]
    )
    dynamic = bootstrap[
        bootstrap["estimand"].eq(estimand)
        & bootstrap["increment"].eq("dynamic_relation_increment")
        & bootstrap["metric"].eq("average_precision_difference_challenger_minus_reference")
    ].iloc[0]
    favorable_strata = int(strata["favorable"].sum())
    material_adverse = int(strata["material_adverse"].sum())
    thresholds = config["development_gate"]
    checks = {
        "relative_average_precision_improvement_at_least_20pct": relative_ap >= float(thresholds["minimum_relative_average_precision_improvement"]),
        "relative_brier_reduction_at_least_10pct": relative_brier >= float(thresholds["minimum_relative_brier_reduction"]),
        "ece_at_most_0_04": float(relation["ece_10"]) <= float(thresholds["maximum_ece"]),
        "top_decile_enrichment_at_least_3": float(relation["top_decile_enrichment"]) >= float(thresholds["minimum_top_decile_enrichment"]),
        "four_coherent_chemistry_strata": favorable_strata >= int(thresholds["minimum_coherent_chemistry_strata"]),
        "M4_minus_M3_lineage_bootstrap_95pct_lower_bound_gt_zero": float(dynamic["ci_lower"]) > 0.0,
        "no_material_adverse_chemistry_stratum": material_adverse == 0,
        "zero_train_validation_lineage_overlap": audit["train_validation_lineage_overlap"] == 0,
        "zero_locked_outcome_reads": audit["locked_outcome_access_count"] == 0,
        "zero_format_only_events": audit["format_only_identifier_events"] == 0,
        "zero_unledgered_errors": not audit["unledgered_errors"],
    }
    return {
        "scope": "development_validation_only",
        "flagship_estimand": estimand,
        "flagship_relation_model": relation_name,
        "strongest_capacity_matched_non_relational_baseline": baseline,
        "relative_average_precision_improvement": relative_ap,
        "relative_brier_reduction": relative_brier,
        "relation_metrics": {
            name: float(relation[name])
            for name in ["average_precision", "brier", "ece_10", "top_decile_enrichment"]
        },
        "M4_minus_M3_average_precision_bootstrap_ci_lower": float(dynamic["ci_lower"]),
        "M4_minus_M3_average_precision_bootstrap_ci_upper": float(dynamic["ci_upper"]),
        "favorable_chemistry_strata": favorable_strata,
        "material_adverse_chemistry_strata": material_adverse,
        "checks": checks,
        "passed": all(checks.values()),
        "gate_status": "GO" if all(checks.values()) else "NO_GO",
        "deep_model_entry_gate_passed": bool(checks["M4_minus_M3_lineage_bootstrap_95pct_lower_bound_gt_zero"]),
        "confirmation_direction_gate": "DEFERRED_NOT_ASSESSABLE_IN_V2_2",
        "locked_outcome_access_count": audit["locked_outcome_access_count"],
    }


def _semantically_equal(left: Any, right: Any) -> bool:
    """Compare report structures while tolerating CSV float round trips."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _semantically_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=1e-10, equal_nan=True))
    return left == right


def build_v2_capacity_matched(config_path: str | Path) -> dict[str, Any]:
    started = _utc_now()
    config_path = Path(config_path)
    config, frame, frozen, validation = _load_and_validate(config_path)
    schema = frozen["schema"]
    output = config["output"]
    Path(output["root"]).mkdir(parents=True, exist_ok=True)
    Path(output["models_root"]).mkdir(parents=True, exist_ok=True)
    Path(output["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    train_role = config["population"]["train_role"]
    validation_role = config["population"]["validation_role"]
    prediction_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    for estimand, specification in config["estimands"].items():
        target = specification["target"]
        risk_column = specification["risk_column"]
        train = _risk_frame(frame, train_role, risk_column)
        validation_frame = _risk_frame(frame, validation_role, risk_column)
        for model_name in specification["models"]:
            if model_name == "M5":
                continue
            feature_names = list(schema["feature_sets"][model_name])
            train_probability, validation_probability, artifact = _fit_calibrated(
                train,
                validation_frame,
                target,
                feature_names,
                config,
                int(config["shared_budget"]["effective_total_boosting_iterations"]),
            )
            if specification["exposure_adjustment"] == "complementary_log_log_rate_rescaling":
                validation_probability = exposure_rescale_probability(
                    validation_probability,
                    config["population"]["reference_exposure_days"],
                    config["population"]["validation_exposure_days"],
                )
            artifact.update(
                {
                    "task_id": "V2.2",
                    "estimand": estimand,
                    "model_name": model_name,
                    "risk_column": risk_column,
                    "exposure_adjustment": specification["exposure_adjustment"],
                }
            )
            model_path = Path(output["models_root"]) / f"{estimand}__{model_name}.joblib"
            _dump_joblib_atomic(model_path, artifact)
            model_records.append(_artifact_record(model_path))
            training_records.append(
                {
                    "estimand": estimand,
                    "model_name": model_name,
                    "training_rows": len(train),
                    "training_events": int(train[target].sum()),
                    "validation_rows": len(validation_frame),
                    "validation_events": int(validation_frame[target].sum()),
                    "boosting_iterations": int(config["shared_budget"]["effective_total_boosting_iterations"]),
                    "feature_count": len(feature_names),
                }
            )
            prediction_frames.extend(
                [
                    _prediction_frame(
                        train,
                        estimand,
                        model_name,
                        target,
                        train_probability,
                        "lineage_group_crossfit_oof",
                    ),
                    _prediction_frame(
                        validation_frame,
                        estimand,
                        model_name,
                        target,
                        validation_probability,
                        "validation_final_train_only_calibration",
                    ),
                ]
            )
    stable_spec = config["estimands"]["stable_to_unstable_event"]
    stable_train = _risk_frame(frame, train_role, stable_spec["risk_column"])
    stable_validation = _risk_frame(frame, validation_role, stable_spec["risk_column"])
    arrival_spec = config["M5"]["competitor_arrival"]
    arrival_train_probability, arrival_validation_probability, arrival_artifact = _fit_calibrated(
        stable_train,
        stable_validation,
        arrival_spec["target"],
        list(schema["feature_sets"][arrival_spec["feature_set"]]),
        config,
        int(arrival_spec["boosting_iterations"]),
    )
    del arrival_train_probability
    arrival_validation_probability = exposure_rescale_probability(
        arrival_validation_probability,
        config["population"]["reference_exposure_days"],
        config["population"]["validation_exposure_days"],
    )
    displacement_spec = config["M5"]["displacement_vulnerability"]
    displacement_train = _risk_frame(frame, train_role, displacement_spec["risk_column"])
    displacement_train_probability, displacement_validation_probability, displacement_artifact = _fit_calibrated(
        displacement_train,
        stable_validation,
        displacement_spec["target"],
        list(schema["feature_sets"][displacement_spec["feature_set"]]),
        config,
        int(displacement_spec["boosting_iterations"]),
    )
    del displacement_train_probability
    M5_probability = np.clip(
        arrival_validation_probability * displacement_validation_probability, 0.0, 1.0
    )
    M5_artifact = {
        "task_id": "V2.2",
        "estimand": "stable_to_unstable_event",
        "model_name": "M5",
        "combination": config["M5"]["combination"],
        "effective_boosting_iterations": int(arrival_spec["boosting_iterations"])
        + int(displacement_spec["boosting_iterations"]),
        "competitor_arrival": arrival_artifact,
        "displacement_vulnerability": displacement_artifact,
    }
    M5_path = Path(output["models_root"]) / "stable_to_unstable_event__M5.joblib"
    _dump_joblib_atomic(M5_path, M5_artifact)
    model_records.append(_artifact_record(M5_path))
    training_records.append(
        {
            "estimand": "stable_to_unstable_event",
            "model_name": "M5",
            "training_rows": len(stable_train),
            "training_events": int(stable_train["stable_to_unstable_event"].sum()),
            "validation_rows": len(stable_validation),
            "validation_events": int(stable_validation["stable_to_unstable_event"].sum()),
            "arrival_training_rows": len(stable_train),
            "arrival_training_events": int(stable_train[arrival_spec["target"]].sum()),
            "displacement_training_rows": len(displacement_train),
            "displacement_training_events": int(displacement_train[displacement_spec["target"]].sum()),
            "boosting_iterations": int(arrival_spec["boosting_iterations"])
            + int(displacement_spec["boosting_iterations"]),
            "feature_count": len(set(arrival_artifact["feature_names"] + displacement_artifact["feature_names"])),
        }
    )
    prediction_frames.append(
        _prediction_frame(
            stable_validation,
            "stable_to_unstable_event",
            "M5",
            "stable_to_unstable_event",
            M5_probability,
            "validation_factorized_train_only_calibration",
        )
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["estimand", "model_name", "assigned_role", "canonical_lineage_id", "transition_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    metric_rows = [
        _metric_row(group)
        for _, group in predictions.groupby(
            ["estimand", "model_name", "assigned_role", "evaluation_method"], sort=True
        )
    ]
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["estimand", "role", "model_name"], kind="mergesort"
    ).reset_index(drop=True)
    factorial = _factorial_comparison(metrics, config)
    bootstrap = _bootstrap_comparisons(predictions, config)
    strata = _chemistry_strata(predictions, metrics, config)
    calibration = _calibration_table(predictions, int(config["evaluation"]["ece_bins"]))
    audit = validation["audit"]
    audit.update(
        {
            "prediction_rows": len(predictions),
            "model_artifacts": len(model_records),
            "training_runs": len(training_records),
            "hyperparameter_trials": int(config["shared_budget"]["hyperparameter_trials"]),
            "test_time_recalibration_count": 0,
            "target_prevalence_calibration_access_count": 0,
            "confirmation_evaluations": 0,
            "deep_models_trained": 0,
            "network_access_count": 0,
            "declared_seed_schedule": config["declared_seed_schedule"],
            "effective_execution_seed": int(config["shared_budget"]["execution_seed"]),
        }
    )
    gate = _compute_gate(metrics, bootstrap, strata, audit, config)
    parquet_settings = config["parquet"]
    _write_parquet_atomic(
        output["predictions"],
        predictions,
        compression=parquet_settings["compression"],
        row_group_size=int(parquet_settings["row_group_size"]),
    )
    _write_csv_atomic(output["metrics"], metrics)
    _write_csv_atomic(output["factorial_comparison"], factorial)
    _write_csv_atomic(output["bootstrap"], bootstrap)
    _write_csv_atomic(output["chemistry_strata"], strata)
    _write_csv_atomic(output["calibration"], calibration)
    _write_json_atomic(output["gate"], gate)
    _write_json_atomic(output["audit"], audit)
    artifact_paths = [
        output["predictions"],
        output["metrics"],
        output["factorial_comparison"],
        output["bootstrap"],
        output["chemistry_strata"],
        output["calibration"],
        output["gate"],
        output["audit"],
    ]
    manifest = {
        "task_id": "V2.2",
        "method_version": config["method_version"],
        "created_at_utc": _utc_now(),
        "config": _artifact_record(config_path),
        "inputs": validation["inputs"],
        "artifacts": [_artifact_record(path) for path in artifact_paths],
        "models": model_records,
        "training_records": training_records,
        "gate_status": gate["gate_status"],
        "locked_outcome_access_count": 0,
        "confirmation_evaluations": 0,
    }
    _write_json_atomic(output["manifest"], manifest)
    report = {
        "task_id": "V2.2",
        "task_status": "IN_PROGRESS",
        "status": "BUILD_PASS",
        "gate_status": gate["gate_status"],
        "scope": "V2.2 development only",
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "PyYAML", "phase-evonet"]
        },
        "git_state": {"repository": None, "commit": None, "status": "NO_GIT_REPOSITORY"},
        "config": _artifact_record(config_path),
        "inputs": validation["inputs"],
        "row_counts": {
            "aligned_input_rows": len(frame),
            "prediction_rows": len(predictions),
            "metric_rows": len(metrics),
            "factorial_rows": len(factorial),
            "bootstrap_summary_rows": len(bootstrap),
            "chemistry_strata_rows": len(strata),
            "calibration_rows": len(calibration),
        },
        "training_records": training_records,
        "development_gate": gate,
        "integrity_audit": audit,
        "model_artifacts": model_records,
        "generated_artifacts": [_artifact_record(path) for path in artifact_paths] + [_artifact_record(output["manifest"])],
        "model_training_performed": True,
        "locked_outcome_access_count": 0,
        "locked_or_external_outcome_files_read": [],
        "confirmation_A_accessed": False,
        "confirmation_B_accessed": False,
        "V2_3_executed": False,
        "warnings": [
            "V2.2 uses development outcomes only; Confirmation A/B direction agreement is not assessable here.",
            "The exact all-risk decomposition-set endpoint remains deferred because its formal V2.1 source table is unavailable locally.",
            "The dimensionality-1 chemistry stratum is retained even if sparse; non-estimability cannot be relabeled as favorable.",
        ],
        "tests": None,
        "verification": None,
        "commands": [],
    }
    _write_json_atomic(output["report"], report)
    report_hash = _sha256(output["report"])
    Path(output["report_sidecar"]).write_text(f"{report_hash}  report.json\n", encoding="utf-8")
    return {
        "task_id": "V2.2",
        "status": "BUILD_PASS",
        "gate_status": gate["gate_status"],
        "gate_passed": gate["passed"],
        "prediction_rows": len(predictions),
        "model_artifacts": len(model_records),
        "locked_outcome_access_count": 0,
        "confirmation_evaluations": 0,
    }


def verify_v2_capacity_matched(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config, source, _, validation = _load_and_validate(config_path)
    output = config["output"]
    manifest = json.loads(Path(output["manifest"]).read_text(encoding="utf-8"))
    predictions = pd.read_parquet(output["predictions"])
    metrics = pd.read_csv(output["metrics"])
    bootstrap = pd.read_csv(output["bootstrap"])
    strata = pd.read_csv(output["chemistry_strata"])
    stored_gate = json.loads(Path(output["gate"]).read_text(encoding="utf-8"))
    stored_audit = json.loads(Path(output["audit"]).read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in [*manifest["artifacts"], *manifest["models"]]:
        if _sha256(record["path"]) != record["sha256"]:
            failures.append(f"artifact hash mismatch: {record['path']}")
    if _sha256(config_path) != manifest["config"]["sha256"]:
        failures.append("config hash mismatch")
    recomputed_rows = [
        _metric_row(group)
        for _, group in predictions.groupby(
            ["estimand", "model_name", "assigned_role", "evaluation_method"], sort=True
        )
    ]
    recomputed = pd.DataFrame(recomputed_rows)
    merged_metrics = metrics.merge(
        recomputed,
        on=["estimand", "model_name", "role", "evaluation_method"],
        suffixes=("_stored", "_recomputed"),
        validate="one_to_one",
    )
    if len(merged_metrics) != len(metrics):
        failures.append("metric key mismatch")
    for name in ["rows", "positives", "prevalence", "average_precision", "brier", "ece_10", "top_decile_enrichment"]:
        stored = merged_metrics[f"{name}_stored"].to_numpy(dtype=float)
        recomputed_values = merged_metrics[f"{name}_recomputed"].to_numpy(dtype=float)
        if not np.allclose(stored, recomputed_values, atol=1e-12, rtol=1e-10, equal_nan=True):
            failures.append(f"metric mismatch: {name}")
    target_map = {
        name: specification["target"] for name, specification in config["estimands"].items()
    }
    for estimand, group in predictions.groupby("estimand", sort=True):
        keys = group[["transition_id", "event"]].drop_duplicates("transition_id")
        truth = source[["transition_id", target_map[estimand]]]
        checked = keys.merge(truth, on="transition_id", how="left", validate="one_to_one")
        if checked[target_map[estimand]].isna().any() or not np.array_equal(
            checked["event"].astype(bool).to_numpy(), checked[target_map[estimand]].astype(bool).to_numpy()
        ):
            failures.append(f"target mismatch: {estimand}")
    temporary_files = [
        str(path)
        for root in [Path(output["root"]), Path(output["models_root"]), Path(output["manifest"]).parent]
        if root.exists()
        for path in root.rglob("*.tmp")
    ]
    if temporary_files:
        failures.append(f"temporary files remain: {temporary_files}")
    recomputed_gate = _compute_gate(metrics, bootstrap, strata, stored_audit, config)
    if not _semantically_equal(recomputed_gate, stored_gate):
        failures.append("development gate reconstruction mismatch")
    if stored_audit["locked_outcome_access_count"] != 0:
        failures.append("locked outcome access was recorded")
    if stored_audit["train_validation_lineage_overlap"] != 0:
        failures.append("train/validation lineage overlap")
    if stored_audit["test_time_recalibration_count"] != 0:
        failures.append("test-time recalibration was recorded")
    if stored_audit["target_prevalence_calibration_access_count"] != 0:
        failures.append("target prevalence calibration access was recorded")
    if stored_audit["deep_models_trained"] != 0:
        failures.append("a prohibited deep model was trained")
    expected_models = 16
    if len(manifest["models"]) != expected_models:
        failures.append(f"expected {expected_models} model artifacts")
    for record in manifest["models"]:
        artifact = joblib.load(record["path"])
        if artifact["model_name"] == "M5":
            if artifact["effective_boosting_iterations"] != 250:
                failures.append("M5 capacity mismatch")
        elif artifact["effective_boosting_iterations"] != 250:
            failures.append(f"capacity mismatch: {record['path']}")
    result = {
        "task_id": "V2.2",
        "status": "PASS" if not failures else "FAIL",
        "integrity_gate_status": "PASS" if not failures else "FAIL",
        "scientific_gate_status": stored_gate["gate_status"],
        "failures": failures,
        "checks": {
            "input_hashes": True,
            "artifact_hashes": not any("hash mismatch" in failure for failure in failures),
            "metric_recomputation": not any("metric" in failure for failure in failures),
            "target_alignment": not any("target mismatch" in failure for failure in failures),
            "gate_reconstruction": "development gate reconstruction mismatch" not in failures,
            "capacity_matching": not any("capacity mismatch" in failure for failure in failures),
            "zero_lineage_overlap": stored_audit["train_validation_lineage_overlap"] == 0,
            "zero_locked_access": stored_audit["locked_outcome_access_count"] == 0,
            "zero_test_time_recalibration": stored_audit["test_time_recalibration_count"] == 0,
            "zero_target_prevalence_calibration": stored_audit["target_prevalence_calibration_access_count"] == 0,
            "zero_deep_models": stored_audit["deep_models_trained"] == 0,
            "zero_temporary_files": not temporary_files,
        },
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "model_artifacts": len(manifest["models"]),
        "locked_outcome_access_count": stored_audit["locked_outcome_access_count"],
        "confirmation_evaluations": stored_audit["confirmation_evaluations"],
    }
    return result


def finalize_v2_capacity_matched(
    config_path: str | Path,
    *,
    tests_passed: int,
    tests_failed: int,
    test_duration_seconds: float,
    command_log_path: str | Path,
    changed_files_path: str | Path,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = config["output"]
    report = json.loads(Path(output["report"]).read_text(encoding="utf-8"))
    verification = verify_v2_capacity_matched(config_path)
    gate = report["development_gate"]
    tasks_text = Path("TASKS_V2.md").read_text(encoding="utf-8")
    if "| V2.2 |" not in tasks_text:
        raise RuntimeError("V2.2 task row is missing")
    task_status = "DONE" if "| V2.2 | Development-only capacity-matched model fitting and factorial comparison | **DONE**" in tasks_text else "STOPPED" if "| V2.2 | Development-only capacity-matched model fitting and factorial comparison | **STOPPED**" in tasks_text else "IN_PROGRESS"
    commands = json.loads(Path(command_log_path).read_text(encoding="utf-8"))
    changed_files = [
        line.strip()
        for line in Path(changed_files_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    acceptance = {
        "V2_1_entry_verified": True,
        "capacity_matched_M0_M5_fitted": True,
        "factorial_increments_reported": True,
        "training_only_lineage_group_calibration": True,
        "lineage_bootstrap_complete": int(config["evaluation"]["bootstrap_replicates"]) == 1000,
        "chemistry_strata_reported": report["row_counts"]["chemistry_strata_rows"] == 4,
        "full_tests_pass": tests_failed == 0,
        "independent_verification_pass": verification["status"] == "PASS",
        "locked_outcome_access_zero": report["locked_outcome_access_count"] == 0,
        "V2_3_not_executed": report["V2_3_executed"] is False,
        "development_scientific_gate_pass": gate["passed"],
    }
    expected_status = "DONE" if gate["passed"] and all(acceptance.values()) else "STOPPED"
    if task_status != expected_status:
        raise RuntimeError(
            f"TASKS_V2 status {task_status} does not match required {expected_status}"
        )
    report.update(
        {
            "task_status": task_status,
            "status": "PASS" if task_status == "DONE" else "FAIL",
            "gate_status": gate["gate_status"],
            "finalized_at_utc": _utc_now(),
            "tests": {
                "command": "python -m pytest -q",
                "passed": int(tests_passed),
                "failed": int(tests_failed),
                "duration_seconds": float(test_duration_seconds),
            },
            "verification": verification,
            "commands": commands,
            "modified_files": changed_files,
            "generated_files": changed_files,
            "acceptance_criteria": acceptance,
            "next_task_eligibility": {
                "only_task": "V2.3",
                "eligible": bool(task_status == "DONE" and gate["passed"]),
                "authorized": False,
                "executed": False,
                "requires_separate_PI_instruction_and_named_custodian": True,
            },
        }
    )
    _write_json_atomic(output["report"], report)
    report_hash = _sha256(output["report"])
    Path(output["report_sidecar"]).write_text(f"{report_hash}  report.json\n", encoding="utf-8")
    return {
        "task_id": "V2.2",
        "task_status": task_status,
        "status": report["status"],
        "gate_status": report["gate_status"],
        "report_sha256": report_hash,
        "tests_passed": tests_passed,
        "verification_status": verification["status"],
        "locked_outcome_access_count": report["locked_outcome_access_count"],
        "V2_3_executed": False,
    }
