from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .identity_candidates import write_json_atomic
from .manifest import sha256_file


METHOD_VERSION = "P4_1_DESCRIPTIVE_ATLAS_V1"
CHANNELS = (
    "competitor_inventory",
    "uncorrected_energy",
    "compatibility_correction",
    "candidate_identity",
)


SURVIVAL_SCHEMA = pa.schema(
    [
        pa.field("survival_unit_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("baseline_snapshot", pa.string(), nullable=False),
        pa.field("baseline_material_id", pa.string(), nullable=False),
        pa.field("baseline_identity_confidence", pa.string(), nullable=False),
        pa.field("minimum_identity_confidence", pa.string(), nullable=False),
        pa.field("event_observed", pa.bool_(), nullable=False),
        pa.field("event_snapshot", pa.string()),
        pa.field("censor_snapshot", pa.string()),
        pa.field("last_observed_snapshot", pa.string(), nullable=False),
        pa.field("duration_days", pa.int32(), nullable=False),
        pa.field("duration_months", pa.float64(), nullable=False),
        pa.field("followup_snapshot_count", pa.int8(), nullable=False),
        pa.field("final_reported_is_stable", pa.bool_(), nullable=False),
        pa.field("observation_status", pa.string(), nullable=False),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


FRAGILITY_SCHEMA = pa.schema(
    [
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("source_composition_reduced_json", pa.string(), nullable=False),
        pa.field("source_chemsys", pa.string(), nullable=False),
        pa.field("source_elements_json", pa.string(), nullable=False),
        pa.field("source_element_count", pa.int8(), nullable=False),
        pa.field("duration_days", pa.int16(), nullable=False),
        pa.field("duration_months", pa.float64(), nullable=False),
        pa.field("exposure_years", pa.float64(), nullable=False),
        pa.field("destabilization_event", pa.bool_(), nullable=False),
        pa.field("target_reported_is_stable", pa.bool_(), nullable=False),
        pa.field("p3_3_attributed_flip", pa.bool_(), nullable=False),
        pa.field("p3_3_unified_label_transition", pa.string()),
        pa.field("p3_3_dominant_channel", pa.string()),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


TRANSITION_COLUMNS = [
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
    "source_is_stable",
    "target_is_stable",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    if value.get("task_id") != "P4.1":
        raise ValueError("P4.1 config must declare task_id: P4.1")
    return config_path, value


def _require_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    reports = {
        name: _read_json(Path(config["input"][name]))
        for name in ("p2_2_report", "p3_1_report", "p3_3_report")
    }
    failures: list[str] = []
    for name, task_id in (
        ("p2_2_report", "P2.2"),
        ("p3_1_report", "P3.1"),
        ("p3_3_report", "P3.3"),
    ):
        report = reports[name]
        if report.get("task_id") != task_id:
            failures.append(f"{name} task_id is not {task_id}")
        if report.get("task_status") not in (None, "DONE"):
            failures.append(f"{name} task_status is not DONE")
        if report.get("status") != "PASS":
            failures.append(f"{name} status is not PASS")
        gate_value = report.get("gate_status")
        if isinstance(gate_value, dict):
            gate_value = gate_value.get("status")
        gate_value = (
            gate_value
            or report.get("gate_decision")
            or (report.get("gate", {}).get("decision") if isinstance(report.get("gate"), dict) else None)
        )
        if gate_value not in (None, "GO"):
            failures.append(f"{name} gate_status is not GO")
    p3 = reports["p3_3_report"]
    important = int(p3.get("population", {}).get("important_high_confidence_flips", 0))
    coverage = float(p3.get("population", {}).get("attribution_coverage", 0.0))
    minimum_events = int(config["gate"]["minimum_important_high_confidence_flips"])
    minimum_coverage = float(config["gate"]["minimum_attribution_fraction"])
    if important < minimum_events:
        failures.append(f"P3 important flip count {important} is below {minimum_events}")
    if coverage < minimum_coverage:
        failures.append(f"P3 attribution coverage {coverage} is below {minimum_coverage}")
    if failures:
        raise RuntimeError("P4.1 precondition failure: " + "; ".join(failures))
    return {
        "p2_2": "DONE/PASS",
        "p3_1": "DONE/PASS",
        "p3_3": "DONE/PASS/GO",
        "important_high_confidence_flips": important,
        "attribution_coverage": coverage,
    }


def _date(value: str) -> date:
    return date.fromisoformat(str(value))


def _duration_days(source: str, target: str) -> int:
    days = (_date(target) - _date(source)).days
    if days <= 0:
        raise ValueError(f"Non-positive snapshot duration: {source} -> {target}")
    return days


def _optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _confidence_minimum(values: Iterable[str]) -> str:
    ranks = {"A1": 0, "A2": 1, "B": 2, "C": 3}
    items = [str(value) for value in values]
    return max(items, key=lambda value: ranks.get(value, 99))


def survival_unit_id(lineage_id: str, thermo_type: str, baseline: str) -> bytes:
    payload = f"{lineage_id}|{thermo_type}|{baseline}|{METHOD_VERSION}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def composition_elements(composition_json: str) -> list[str]:
    value = json.loads(str(composition_json))
    if not isinstance(value, dict) or not value:
        raise ValueError("Composition must be a nonempty JSON object")
    elements = []
    for element, amount in value.items():
        numeric = float(amount)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"Invalid composition amount for {element}: {amount}")
        elements.append(str(element))
    return sorted(elements)


def chemsys_from_composition(composition_json: str) -> str:
    return "-".join(composition_elements(composition_json))


def wilson_interval(events: int, total: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if total < 0 or events < 0 or events > total:
        raise ValueError("Wilson inputs must satisfy 0 <= events <= total")
    if total == 0:
        return math.nan, math.nan
    if not math.isclose(confidence_level, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("P4.1 freezes Wilson intervals at confidence_level=0.95")
    z = 1.959963984540054
    p = events / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_survival_cohort(
    transitions: pd.DataFrame,
    snapshot_order: list[str],
    baseline_snapshot: str,
    identity_confidences: set[str],
    days_per_month: float,
) -> pd.DataFrame:
    eligible = transitions[transitions["identity_confidence"].isin(identity_confidences)].copy()
    source_rows = eligible[eligible["source_state_present"].fillna(False)].copy()
    key_columns = ["canonical_lineage_id", "thermo_type", "source_snapshot"]
    duplicated = source_rows.duplicated(key_columns, keep=False)
    if bool(duplicated.any()):
        examples = source_rows.loc[duplicated, key_columns].head(5).to_dict("records")
        raise RuntimeError(f"Non-unique lineage/workflow source states: {examples}")
    lookup = {
        (str(row["canonical_lineage_id"]), str(row["thermo_type"]), str(row["source_snapshot"])): row
        for row in source_rows.to_dict("records")
    }
    baseline_rows = source_rows[
        source_rows["source_snapshot"].eq(baseline_snapshot)
        & source_rows["source_state_usable"].eq(True)
        & source_rows["source_is_stable"].eq(True)
    ].copy()
    baseline_rows = baseline_rows.sort_values(
        ["canonical_lineage_id", "thermo_type"], kind="mergesort"
    )
    if baseline_rows.empty:
        raise RuntimeError("P4.1 survival baseline risk set is empty")
    snapshot_index = {snapshot: index for index, snapshot in enumerate(snapshot_order)}
    if baseline_snapshot not in snapshot_index:
        raise ValueError("Baseline snapshot is not in snapshot_order")
    baseline_position = snapshot_index[baseline_snapshot]
    rows: list[dict[str, Any]] = []
    for baseline in baseline_rows.to_dict("records"):
        lineage_id = str(baseline["canonical_lineage_id"])
        thermo_type = str(baseline["thermo_type"])
        confidences = [str(baseline["identity_confidence"])]
        event_observed = False
        event_snapshot: str | None = None
        censor_snapshot: str | None = None
        last_observed_snapshot = baseline_snapshot
        final_stable = True
        observed_snapshots = 1
        current_stable = True
        for position in range(baseline_position, len(snapshot_order) - 1):
            source_snapshot = snapshot_order[position]
            expected_target = snapshot_order[position + 1]
            transition = lookup.get((lineage_id, thermo_type, source_snapshot))
            if transition is None:
                censor_snapshot = last_observed_snapshot
                break
            if str(transition["target_snapshot"]) != expected_target:
                raise RuntimeError(
                    f"Unexpected target snapshot for {lineage_id}/{thermo_type}: "
                    f"{transition['target_snapshot']} != {expected_target}"
                )
            source_usable = _optional_bool(transition["source_state_usable"])
            source_label = _optional_bool(transition["source_is_stable"])
            if not source_usable or source_label is None:
                censor_snapshot = last_observed_snapshot
                break
            if source_label != current_stable:
                raise RuntimeError(
                    f"Reported label discontinuity for {lineage_id}/{thermo_type} at {source_snapshot}"
                )
            confidences.append(str(transition["identity_confidence"]))
            target_usable = _optional_bool(transition["target_state_usable"])
            target_label = _optional_bool(transition["target_is_stable"])
            if not target_usable or target_label is None:
                censor_snapshot = source_snapshot
                break
            last_observed_snapshot = expected_target
            observed_snapshots += 1
            final_stable = target_label
            current_stable = target_label
            if not target_label:
                event_observed = True
                event_snapshot = expected_target
                break
        if not event_observed and censor_snapshot is None:
            censor_snapshot = last_observed_snapshot
        endpoint = event_snapshot if event_observed else censor_snapshot
        if endpoint is None:
            raise RuntimeError("Survival endpoint was not assigned")
        duration_days = (_date(endpoint) - _date(baseline_snapshot)).days
        rows.append(
            {
                "survival_unit_id": survival_unit_id(lineage_id, thermo_type, baseline_snapshot),
                "canonical_lineage_id": lineage_id,
                "thermo_type": thermo_type,
                "baseline_snapshot": baseline_snapshot,
                "baseline_material_id": str(baseline["source_material_id"]),
                "baseline_identity_confidence": str(baseline["identity_confidence"]),
                "minimum_identity_confidence": _confidence_minimum(confidences),
                "event_observed": event_observed,
                "event_snapshot": event_snapshot,
                "censor_snapshot": None if event_observed else censor_snapshot,
                "last_observed_snapshot": last_observed_snapshot,
                "duration_days": int(duration_days),
                "duration_months": float(duration_days / days_per_month),
                "followup_snapshot_count": int(observed_snapshots),
                "final_reported_is_stable": bool(final_stable),
                "observation_status": "event" if event_observed else "right_censored",
                "method_version": METHOD_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=[field.name for field in SURVIVAL_SCHEMA])


def kaplan_meier_curve(
    cohort: pd.DataFrame,
    snapshot_order: list[str],
    baseline_snapshot: str,
    days_per_month: float,
    stratum_type: str,
    stratum_value: str,
) -> pd.DataFrame:
    if cohort.empty:
        raise ValueError("Kaplan-Meier cohort cannot be empty")
    baseline = _date(baseline_snapshot)
    times = [(_date(snapshot) - baseline).days for snapshot in snapshot_order]
    survival = 1.0
    greenwood = 0.0
    output: list[dict[str, Any]] = []
    for index, (snapshot, time_days) in enumerate(zip(snapshot_order, times, strict=True)):
        at_risk = int((cohort["duration_days"] >= time_days).sum())
        events = int(
            (cohort["event_observed"] & cohort["duration_days"].eq(time_days)).sum()
        )
        censored = int(
            ((~cohort["event_observed"]) & cohort["duration_days"].eq(time_days)).sum()
        )
        if index > 0 and events:
            if at_risk <= 0 or events > at_risk:
                raise RuntimeError("Invalid Kaplan-Meier risk set")
            survival *= 1.0 - events / at_risk
            if at_risk > events:
                greenwood += events / (at_risk * (at_risk - events))
            else:
                survival = 0.0
                greenwood = math.inf
        if survival <= 0.0:
            lower, upper = 0.0, 0.0
        elif survival >= 1.0 or greenwood == 0.0:
            lower, upper = survival, survival
        else:
            theta = math.log(-math.log(survival))
            standard_error = math.sqrt(greenwood) / abs(math.log(survival))
            z = 1.959963984540054
            lower = math.exp(-math.exp(theta + z * standard_error))
            upper = math.exp(-math.exp(theta - z * standard_error))
        output.append(
            {
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "snapshot": snapshot,
                "time_days": int(time_days),
                "time_months": float(time_days / days_per_month),
                "n_at_risk": at_risk,
                "events_at_time": events,
                "censored_at_time": censored,
                "survival_probability": float(survival),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "greenwood_variance_factor": float(greenwood),
            }
        )
    result = pd.DataFrame(output)
    reached = result[result["survival_probability"] <= 0.5]
    half_life = float(reached.iloc[0]["time_months"]) if not reached.empty else math.nan
    result["half_life_reached"] = not reached.empty
    result["half_life_months"] = half_life
    result["half_life_lower_bound_months"] = (
        half_life if not reached.empty else float(result["time_months"].max())
    )
    result["method_version"] = METHOD_VERSION
    return result


def build_all_survival_curves(
    cohort: pd.DataFrame,
    snapshot_order: list[str],
    baseline_snapshot: str,
    days_per_month: float,
) -> pd.DataFrame:
    frames = [
        kaplan_meier_curve(
            cohort,
            snapshot_order,
            baseline_snapshot,
            days_per_month,
            "overall",
            "all_workflows",
        )
    ]
    for thermo_type in sorted(cohort["thermo_type"].unique()):
        group = cohort[cohort["thermo_type"].eq(thermo_type)]
        frames.append(
            kaplan_meier_curve(
                group,
                snapshot_order,
                baseline_snapshot,
                days_per_month,
                "thermo_type",
                str(thermo_type),
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_transition_fragility(
    transitions: pd.DataFrame,
    lineage: pd.DataFrame,
    attribution: pd.DataFrame,
    identity_confidences: set[str],
    days_per_month: float,
    days_per_year: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    risk = transitions[
        transitions["identity_confidence"].isin(identity_confidences)
        & transitions["observation_status"].eq("observed")
        & transitions["source_state_usable"].eq(True)
        & transitions["target_state_usable"].eq(True)
        & transitions["source_is_stable"].eq(True)
        & transitions["target_is_stable"].notna()
    ].copy()
    if risk.empty:
        raise RuntimeError("P4.1 rolling fragility risk set is empty")
    join_columns = ["canonical_lineage_id", "snapshot_id", "material_id"]
    if bool(lineage.duplicated(join_columns).any()):
        raise RuntimeError("P2.2 lineage source-composition join key is not unique")
    source_lineage = lineage[
        join_columns + ["composition_reduced_json", "lineage_confidence", "high_confidence"]
    ].rename(columns={"snapshot_id": "source_snapshot", "material_id": "source_material_id"})
    before = len(risk)
    risk = risk.merge(
        source_lineage,
        on=["canonical_lineage_id", "source_snapshot", "source_material_id"],
        how="left",
        validate="many_to_one",
    )
    missing_composition = int(risk["composition_reduced_json"].isna().sum())
    if missing_composition:
        raise RuntimeError(f"Missing P2.2 source composition for {missing_composition} risk rows")
    if len(risk) != before:
        raise RuntimeError("Lineage join changed rolling risk-set row count")
    risk["source_elements"] = risk["composition_reduced_json"].map(composition_elements)
    risk["source_chemsys"] = risk["source_elements"].map(lambda values: "-".join(values))
    risk["source_elements_json"] = risk["source_elements"].map(
        lambda values: json.dumps(values, separators=(",", ":"))
    )
    risk["source_element_count"] = risk["source_elements"].map(len)
    risk["duration_days"] = [
        _duration_days(source, target)
        for source, target in zip(risk["source_snapshot"], risk["target_snapshot"], strict=True)
    ]
    risk["duration_months"] = risk["duration_days"] / days_per_month
    risk["exposure_years"] = risk["duration_days"] / days_per_year
    risk["destabilization_event"] = ~risk["target_is_stable"].astype(bool)
    attr = attribution[
        ["transition_id", "phase_context_chemsys", "unified_label_transition", "dominant_channel"]
    ].copy()
    if bool(attr["transition_id"].duplicated().any()):
        raise RuntimeError("P3.3 attribution transition_id is not unique")
    attr = attr.rename(
        columns={
            "phase_context_chemsys": "p3_3_phase_context_chemsys",
            "unified_label_transition": "p3_3_unified_label_transition",
            "dominant_channel": "p3_3_dominant_channel",
        }
    )
    risk = risk.merge(attr, on="transition_id", how="left", validate="one_to_one")
    risk["p3_3_attributed_flip"] = risk["p3_3_unified_label_transition"].notna()
    matched = risk[risk["p3_3_attributed_flip"]]
    chemistry_mismatch = int(
        matched["source_chemsys"].ne(matched["p3_3_phase_context_chemsys"]).sum()
    )
    if chemistry_mismatch:
        raise RuntimeError(
            f"P3.3 phase context disagrees with P2.2 source chemistry for {chemistry_mismatch} rows"
        )
    risk["method_version"] = METHOD_VERSION
    output = pd.DataFrame(
        {
            "transition_id": risk["transition_id"],
            "canonical_lineage_id": risk["canonical_lineage_id"].astype(str),
            "identity_confidence": risk["identity_confidence"].astype(str),
            "source_snapshot": risk["source_snapshot"].astype(str),
            "target_snapshot": risk["target_snapshot"].astype(str),
            "source_material_id": risk["source_material_id"].astype(str),
            "target_material_id": risk["target_material_id"].astype(str),
            "thermo_type": risk["thermo_type"].astype(str),
            "source_composition_reduced_json": risk["composition_reduced_json"].astype(str),
            "source_chemsys": risk["source_chemsys"].astype(str),
            "source_elements_json": risk["source_elements_json"].astype(str),
            "source_element_count": risk["source_element_count"].astype(int),
            "duration_days": risk["duration_days"].astype(int),
            "duration_months": risk["duration_months"].astype(float),
            "exposure_years": risk["exposure_years"].astype(float),
            "destabilization_event": risk["destabilization_event"].astype(bool),
            "target_reported_is_stable": risk["target_is_stable"].astype(bool),
            "p3_3_attributed_flip": risk["p3_3_attributed_flip"].astype(bool),
            "p3_3_unified_label_transition": risk["p3_3_unified_label_transition"],
            "p3_3_dominant_channel": risk["p3_3_dominant_channel"],
            "method_version": METHOD_VERSION,
        },
        columns=[field.name for field in FRAGILITY_SCHEMA],
    )
    output = output.sort_values(
        ["source_snapshot", "thermo_type", "canonical_lineage_id"], kind="mergesort"
    ).reset_index(drop=True)
    return output, {
        "risk_rows_before_join": before,
        "missing_source_composition": missing_composition,
        "attribution_chemistry_mismatch": chemistry_mismatch,
    }


def _rate_record(events: int, total: int, exposure_years: float, confidence: float) -> dict[str, Any]:
    lower, upper = wilson_interval(events, total, confidence)
    return {
        "at_risk_transitions": int(total),
        "destabilization_events": int(events),
        "fragility_rate": float(events / total) if total else math.nan,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "exposure_years": float(exposure_years),
        "incidence_per_100_material_years": (
            float(100.0 * events / exposure_years) if exposure_years > 0 else math.nan
        ),
    }


def aggregate_interval_fragility(risk: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = sorted(set(zip(risk["source_snapshot"], risk["target_snapshot"], strict=True)))
    for source, target in pairs:
        pair = risk[risk["source_snapshot"].eq(source) & risk["target_snapshot"].eq(target)]
        groups = [("overall", "all_workflows", pair)]
        groups.extend(
            ("thermo_type", str(workflow), group)
            for workflow, group in pair.groupby("thermo_type", sort=True)
        )
        for stratum_type, stratum_value, group in groups:
            record = {
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "source_snapshot": source,
                "target_snapshot": target,
                "interval_days": _duration_days(source, target),
                **_rate_record(
                    int(group["destabilization_event"].sum()),
                    len(group),
                    float(group["exposure_years"].sum()),
                    confidence,
                ),
                "p3_3_attributed_flips_in_risk_set": int(group["p3_3_attributed_flip"].sum()),
                "method_version": METHOD_VERSION,
            }
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["stratum_type", "stratum_value", "source_snapshot"], kind="mergesort"
    ).reset_index(drop=True)


def aggregate_chemistry_fragility(
    risk: pd.DataFrame, confidence: float, minimum_display: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chemsys, group in risk.groupby("source_chemsys", sort=True):
        record = {
            "source_chemsys": str(chemsys),
            "element_count": int(group["source_element_count"].iloc[0]),
            **_rate_record(
                int(group["destabilization_event"].sum()),
                len(group),
                float(group["exposure_years"].sum()),
                confidence,
            ),
            "workflow_count": int(group["thermo_type"].nunique()),
            "p3_3_attributed_flips_in_risk_set": int(group["p3_3_attributed_flip"].sum()),
            "display_eligible": bool(len(group) >= minimum_display),
            "method_version": METHOD_VERSION,
        }
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["at_risk_transitions", "source_chemsys"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def aggregate_element_pair_fragility(
    risk: pd.DataFrame, confidence: float, minimum_display: int
) -> pd.DataFrame:
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for row in risk.itertuples(index=False):
        elements = json.loads(row.source_elements_json)
        memberships = [(element, element) for element in elements]
        memberships.extend(combinations(elements, 2))
        for element_a, element_b in memberships:
            key = tuple(sorted((str(element_a), str(element_b))))
            bucket = totals.setdefault(
                key, {"total": 0.0, "events": 0.0, "years": 0.0, "attributed": 0.0}
            )
            bucket["total"] += 1
            bucket["events"] += int(row.destabilization_event)
            bucket["years"] += float(row.exposure_years)
            bucket["attributed"] += int(row.p3_3_attributed_flip)
    rows: list[dict[str, Any]] = []
    for (element_a, element_b), values in totals.items():
        total = int(values["total"])
        events = int(values["events"])
        rows.append(
            {
                "element_a": element_a,
                "element_b": element_b,
                "pair_type": "element_membership" if element_a == element_b else "cooccurrence",
                **_rate_record(events, total, float(values["years"]), confidence),
                "p3_3_attributed_flips_in_risk_set": int(values["attributed"]),
                "display_eligible": bool(total >= minimum_display),
                "method_version": METHOD_VERSION,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["element_a", "element_b"], kind="mergesort"
    ).reset_index(drop=True)


def aggregate_attribution_by_chemistry(
    attribution: pd.DataFrame, top_count: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chemsys, group in attribution.groupby("phase_context_chemsys", sort=True):
        channel_counts = Counter(str(value) for value in group["dominant_channel"])
        record: dict[str, Any] = {
            "phase_context_chemsys": str(chemsys),
            "element_count": len(str(chemsys).split("-")),
            "attributed_flip_count": int(len(group)),
            "stable_to_unstable_count": int(
                group["unified_label_transition"].eq("stable_to_unstable").sum()
            ),
            "unstable_to_stable_count": int(
                group["unified_label_transition"].eq("unstable_to_stable").sum()
            ),
            "mean_absolute_hull_delta_eV_per_atom": float(
                group["delta_energy_above_hull"].abs().mean()
            ),
            "method_version": METHOD_VERSION,
        }
        for channel in CHANNELS:
            record[f"dominant_{channel}_count"] = int(channel_counts.get(channel, 0))
        rows.append(record)
    result = pd.DataFrame(rows).sort_values(
        ["attributed_flip_count", "phase_context_chemsys"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank_by_attributed_flip_count"] = np.arange(1, len(result) + 1)
    result["display_eligible"] = result["rank_by_attributed_flip_count"] <= top_count
    return result


def _write_parquet_atomic(
    path: Path, frame: pd.DataFrame, schema: pa.Schema, parquet_config: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    pq.write_table(
        table,
        temporary,
        compression=str(parquet_config["compression"]),
        row_group_size=int(parquet_config["row_group_size"]),
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schema_hash(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
    return hashlib.sha256(str(schema).encode()).hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        artifact["rows"] = int(parquet.metadata.num_rows)
        artifact["schema_sha256"] = _schema_hash(path)
    elif rows is not None:
        artifact["rows"] = int(rows)
    return artifact


def _figure_metadata(format_name: str) -> dict[str, Any]:
    if format_name == "pdf":
        return {
            "Title": "PhaseEvoNet P4.1 descriptive atlas",
            "Author": "PhaseEvoNet",
            "Creator": METHOD_VERSION,
            "CreationDate": None,
            "ModDate": None,
        }
    if format_name == "svg":
        return {"Date": None, "Creator": METHOD_VERSION}
    return {"Software": METHOD_VERSION}


def _save_figure(fig: plt.Figure, path: Path, format_name: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format=format_name,
        dpi=dpi,
        bbox_inches="tight",
        metadata=_figure_metadata(format_name),
    )
    os.replace(temporary, path)


def _plot_survival(
    curve: pd.DataFrame,
    interval: pd.DataFrame,
    figure_directory: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "PhaseEvoNet-P4.1",
        }
    )
    colors = {
        "all_workflows": "#111111",
        "GGA_GGA+U": "#4477AA",
        "GGA_GGA+U_R2SCAN": "#228833",
        "R2SCAN": "#CC6677",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax = axes[0]
    for value, group in curve.groupby("stratum_value", sort=False):
        group = group.sort_values("time_months")
        color = colors.get(str(value), "#888888")
        label = "Overall" if value == "all_workflows" else str(value)
        width = 2.2 if value == "all_workflows" else 1.5
        ax.step(
            group["time_months"],
            group["survival_probability"],
            where="post",
            color=color,
            linewidth=width,
            label=label,
        )
        ax.fill_between(
            group["time_months"].to_numpy(float),
            group["ci_lower"].to_numpy(float),
            group["ci_upper"].to_numpy(float),
            step="post",
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Months since 2022-10-28")
    ax.set_ylabel("Reported stable-label survival")
    ax.set_title("a  Earliest-snapshot stable cohort", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    interval = interval.copy()
    pairs = (
        interval[["source_snapshot", "target_snapshot"]]
        .drop_duplicates()
        .sort_values("source_snapshot")
    )
    labels = [f"{s[:4]}–{t[:4]}" for s, t in pairs.itertuples(index=False)]
    x = np.arange(len(labels), dtype=float)
    pair_positions = {
        (str(source), str(target)): position
        for position, (source, target) in enumerate(pairs.itertuples(index=False))
    }
    values = ["all_workflows"] + sorted(
        value for value in interval["stratum_value"].unique() if value != "all_workflows"
    )
    offsets = np.linspace(-0.24, 0.24, len(values))
    for offset, value in zip(offsets, values, strict=True):
        group = interval[interval["stratum_value"].eq(value)].sort_values("source_snapshot")
        color = colors.get(str(value), "#888888")
        label = "Overall" if value == "all_workflows" else str(value)
        group_x = np.asarray(
            [
                pair_positions[(str(source), str(target))]
                for source, target in zip(
                    group["source_snapshot"], group["target_snapshot"], strict=True
                )
            ],
            dtype=float,
        )
        y = 100.0 * group["fragility_rate"].to_numpy(float)
        lower = 100.0 * (group["fragility_rate"] - group["ci_lower"]).to_numpy(float)
        upper = 100.0 * (group["ci_upper"] - group["fragility_rate"]).to_numpy(float)
        ax.errorbar(
            group_x + offset,
            y,
            yerr=np.vstack([lower, upper]),
            fmt="o-" if value == "all_workflows" else "o",
            markersize=4,
            linewidth=1.4 if value == "all_workflows" else 0.8,
            capsize=2,
            color=color,
            label=label,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Reported stable→unstable rate (%)")
    ax.set_xlabel("Adjacent snapshot interval")
    ax.set_title("b  Rolling stable-source risk set", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    output_paths: list[Path] = []
    for format_name in formats:
        path = figure_directory / f"{stem}.{format_name}"
        _save_figure(fig, path, format_name, dpi)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def _plot_atlas(
    pairs: pd.DataFrame,
    attribution: pd.DataFrame,
    figure_directory: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    top_elements: int,
    minimum_pair_risk: int,
) -> list[Path]:
    diagonal = pairs[pairs["element_a"].eq(pairs["element_b"])].sort_values(
        ["at_risk_transitions", "element_a"], ascending=[False, True], kind="mergesort"
    )
    elements = diagonal.head(top_elements)["element_a"].tolist()
    size = len(elements)
    matrix = np.full((size, size), np.nan)
    index = {element: position for position, element in enumerate(elements)}
    for row in pairs.itertuples(index=False):
        if row.element_a not in index or row.element_b not in index:
            continue
        if int(row.at_risk_transitions) < minimum_pair_risk:
            continue
        i, j = index[row.element_a], index[row.element_b]
        matrix[i, j] = matrix[j, i] = 100.0 * float(row.fragility_rate)
    if not np.isfinite(matrix).any():
        raise RuntimeError("No element-pair atlas cell satisfies the frozen display threshold")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    ax = axes[0]
    image = ax.imshow(matrix, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(size), elements, rotation=90)
    ax.set_yticks(np.arange(size), elements)
    ax.set_title(
        f"a  Element-pair reported fragility (n≥{minimum_pair_risk})",
        loc="left",
        fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Stable→unstable rate (%)")

    ax = axes[1]
    display = attribution[attribution["display_eligible"]].copy()
    display = display.sort_values(
        ["attributed_flip_count", "phase_context_chemsys"],
        ascending=[True, False],
        kind="mergesort",
    )
    y = np.arange(len(display))
    left = np.zeros(len(display), dtype=float)
    channel_colors = {
        "competitor_inventory": "#4477AA",
        "uncorrected_energy": "#EE7733",
        "compatibility_correction": "#228833",
        "candidate_identity": "#CC6677",
    }
    for channel in CHANNELS:
        values = display[f"dominant_{channel}_count"].to_numpy(float)
        ax.barh(
            y,
            values,
            left=left,
            color=channel_colors[channel],
            label=channel.replace("_", " "),
        )
        left += values
    ax.set_yticks(y, display["phase_context_chemsys"])
    ax.set_xlabel("Attributed rebuilt-hull flips (count)")
    ax.set_title("b  Leading exact chemical systems by attribution count", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    output_paths: list[Path] = []
    for format_name in formats:
        path = figure_directory / f"{stem}.{format_name}"
        _save_figure(fig, path, format_name, dpi)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def _data_dictionary(paths: dict[str, Path]) -> pd.DataFrame:
    descriptions = {
        "survival_unit_id": "Deterministic lineage/workflow/baseline survival-unit identifier.",
        "event_observed": "True at the first reported stable-to-unstable target snapshot.",
        "survival_probability": "Discrete-snapshot Kaplan-Meier stable-label survival estimate.",
        "fragility_rate": "Observed reported stable-to-unstable events divided by at-risk transitions.",
        "display_eligible": "Meets the frozen outcome-independent exposure or rank display rule.",
        "p3_3_attributed_flip": "Transition also appears in the separate P3.3 rebuilt-hull attribution table.",
        "phase_context_chemsys": "Exact P3.3 phase context; not inferred from entry_id alone.",
    }
    rows: list[dict[str, Any]] = []
    for table_name, path in paths.items():
        if path.suffix == ".parquet":
            schema = pq.read_schema(path)
            columns = [(field.name, str(field.type), field.nullable) for field in schema]
        else:
            frame = pd.read_csv(path, nrows=50)
            columns = [(column, str(frame[column].dtype), bool(frame[column].isna().any())) for column in frame]
        for column, dtype, nullable in columns:
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "dtype": dtype,
                    "nullable": nullable,
                    "description": descriptions.get(column, column.replace("_", " ").capitalize() + "."),
                }
            )
    return pd.DataFrame(rows)


def _confidence_bounds_valid(frame: pd.DataFrame) -> bool:
    needed = {"ci_lower", "ci_upper"}
    if not needed.issubset(frame.columns):
        return False
    value_column = (
        "survival_probability"
        if "survival_probability" in frame.columns
        else "fragility_rate" if "fragility_rate" in frame.columns else None
    )
    if value_column is None:
        return False
    return bool(
        frame["ci_lower"].between(0, 1, inclusive="both").all()
        and frame["ci_upper"].between(0, 1, inclusive="both").all()
        and frame["ci_lower"].le(frame["ci_upper"]).all()
        and (frame["ci_lower"] - 1e-12).le(frame[value_column]).all()
        and frame[value_column].le(frame["ci_upper"] + 1e-12).all()
    )


def build_descriptive_atlas(config_path: str | Path) -> dict[str, Any]:
    started = utc_now()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    config_path, config = _load_config(config_path)
    preconditions = _require_preconditions(config)
    np.random.seed(int(config["seed"]))
    input_paths = {name: Path(path) for name, path in config["input"].items()}
    missing = [path.as_posix() for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing P4.1 inputs: " + ", ".join(missing))
    transitions = pd.read_parquet(input_paths["transition_labels"], columns=TRANSITION_COLUMNS)
    lineage = pd.read_parquet(
        input_paths["lineage"],
        columns=[
            "canonical_lineage_id",
            "snapshot_id",
            "material_id",
            "composition_reduced_json",
            "lineage_confidence",
            "high_confidence",
        ],
    )
    attribution = pd.read_parquet(
        input_paths["attribution"],
        columns=[
            "transition_id",
            "phase_context_chemsys",
            "unified_label_transition",
            "dominant_channel",
            "delta_energy_above_hull",
        ],
    )
    population = config["population"]
    statistics = config["statistics"]
    atlas_config = config["atlas"]
    snapshots = [str(value) for value in population["snapshot_order"]]
    baseline = str(population["baseline_snapshot"])
    confidences = {str(value) for value in population["identity_confidences"]}
    days_per_month = float(statistics["days_per_month"])
    days_per_year = float(statistics["days_per_year"])
    confidence = float(statistics["confidence_level"])
    cohort = build_survival_cohort(
        transitions, snapshots, baseline, confidences, days_per_month
    )
    curve = build_all_survival_curves(cohort, snapshots, baseline, days_per_month)
    risk, join_audit = build_transition_fragility(
        transitions,
        lineage,
        attribution,
        confidences,
        days_per_month,
        days_per_year,
    )
    interval = aggregate_interval_fragility(risk, confidence)
    chemistry = aggregate_chemistry_fragility(
        risk, confidence, int(atlas_config["exact_chemsys_minimum_at_risk_for_display"])
    )
    pairs = aggregate_element_pair_fragility(
        risk, confidence, int(atlas_config["element_pair_minimum_at_risk_for_display"])
    )
    attribution_chemistry = aggregate_attribution_by_chemistry(
        attribution, int(atlas_config["top_attribution_chemsys_by_flip_count"])
    )
    output = {name: Path(path) for name, path in config["output"].items() if name != "root"}
    _write_parquet_atomic(output["survival_cohort"], cohort, SURVIVAL_SCHEMA, config["parquet"])
    _write_parquet_atomic(output["transition_fragility"], risk, FRAGILITY_SCHEMA, config["parquet"])
    table_frames = {
        "survival_curve": curve,
        "interval_fragility": interval,
        "chemistry_fragility": chemistry,
        "element_pair_fragility": pairs,
        "attribution_by_chemistry": attribution_chemistry,
    }
    for name, frame in table_frames.items():
        _write_csv_atomic(output[name], frame)
    figure_config = config["figures"]
    formats = [str(value) for value in figure_config["formats"]]
    figure_directory = output["figure_directory"]
    survival_figures = _plot_survival(
        curve,
        interval,
        figure_directory,
        str(figure_config["survival_stem"]),
        formats,
        int(figure_config["dpi"]),
    )
    atlas_figures = _plot_atlas(
        pairs,
        attribution_chemistry,
        figure_directory,
        str(figure_config["atlas_stem"]),
        formats,
        int(figure_config["dpi"]),
        int(atlas_config["top_elements_by_exposure"]),
        int(atlas_config["element_pair_minimum_at_risk_for_display"]),
    )
    dictionary = _data_dictionary(
        {
            "survival_cohort": output["survival_cohort"],
            "survival_curve": output["survival_curve"],
            "interval_fragility": output["interval_fragility"],
            "transition_fragility": output["transition_fragility"],
            "chemistry_fragility": output["chemistry_fragility"],
            "element_pair_fragility": output["element_pair_fragility"],
            "attribution_by_chemistry": output["attribution_by_chemistry"],
        }
    )
    _write_csv_atomic(output["data_dictionary"], dictionary)
    figure_metadata = {
        "task_id": "P4.1",
        "method_version": METHOD_VERSION,
        "style_version": figure_config["style_version"],
        "network_access": False,
        "figures": [
            {
                "figure": "2",
                "title": "Stability-label survival curves and half-life",
                "machine_readable_sources": [
                    output["survival_cohort"].as_posix(),
                    output["survival_curve"].as_posix(),
                    output["interval_fragility"].as_posix(),
                ],
                "files": [_artifact(path) for path in survival_figures],
            },
            {
                "figure": "4",
                "title": "Chemical-space fragility atlas",
                "machine_readable_sources": [
                    output["transition_fragility"].as_posix(),
                    output["element_pair_fragility"].as_posix(),
                    output["attribution_by_chemistry"].as_posix(),
                ],
                "files": [_artifact(path) for path in atlas_figures],
            },
        ],
    }
    write_json_atomic(output["figure_metadata"], figure_metadata)
    curve_valid = _confidence_bounds_valid(curve)
    interval_valid = _confidence_bounds_valid(interval)
    chemistry_valid = _confidence_bounds_valid(chemistry)
    pair_valid = _confidence_bounds_valid(pairs)
    checks = {
        "p3_important_event_gate": preconditions["important_high_confidence_flips"]
        >= int(config["gate"]["minimum_important_high_confidence_flips"]),
        "p3_attribution_fraction_gate": preconditions["attribution_coverage"]
        >= float(config["gate"]["minimum_attribution_fraction"]),
        "nonempty_survival_risk_set": len(cohort) > 0,
        "at_least_one_survival_event": int(cohort["event_observed"].sum()) > 0,
        "nonempty_rolling_fragility_risk_set": len(risk) > 0,
        "valid_confidence_bounds": curve_valid and interval_valid and chemistry_valid and pair_valid,
        "displayable_atlas_cell": bool(pairs["display_eligible"].any()),
        "attribution_chemistry_complete": int(attribution_chemistry["attributed_flip_count"].sum())
        == len(attribution),
        "source_composition_join_complete": join_audit["missing_source_composition"] == 0,
        "no_phase_context_chemistry_mismatch": join_audit["attribution_chemistry_mismatch"] == 0,
        "all_preregistered_figure_files_present": len(survival_figures + atlas_figures)
        == 2 * len(formats),
    }
    gate_passed = all(checks.values())
    data_artifacts = [
        _artifact(output["survival_cohort"]),
        _artifact(output["survival_curve"], len(curve)),
        _artifact(output["interval_fragility"], len(interval)),
        _artifact(output["transition_fragility"]),
        _artifact(output["chemistry_fragility"], len(chemistry)),
        _artifact(output["element_pair_fragility"], len(pairs)),
        _artifact(output["attribution_by_chemistry"], len(attribution_chemistry)),
    ]
    manifest = {
        "task_id": "P4.1",
        "status": "PASS" if gate_passed else "FAIL",
        "gate_status": "GO" if gate_passed else "NO_GO",
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "seed": int(config["seed"]),
        "network_access": False,
        "method_version": METHOD_VERSION,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "preconditions": preconditions,
        "input_hashes": {path.as_posix(): sha256_file(path) for path in input_paths.values()},
        "population": {
            "transition_rows": int(len(transitions)),
            "survival_cohort_rows": int(len(cohort)),
            "survival_events": int(cohort["event_observed"].sum()),
            "rolling_at_risk_transitions": int(len(risk)),
            "rolling_destabilization_events": int(risk["destabilization_event"].sum()),
            "p3_3_attribution_rows": int(len(attribution)),
            "p3_3_attributed_rows_in_reported_stable_risk_set": int(
                risk["p3_3_attributed_flip"].sum()
            ),
        },
        "survival": {
            "strata": int(curve[["stratum_type", "stratum_value"]].drop_duplicates().shape[0]),
            "maximum_followup_months": float(curve["time_months"].max()),
            "half_life": {
                str(row.stratum_value): {
                    "reached": bool(row.half_life_reached),
                    "months": None if pd.isna(row.half_life_months) else float(row.half_life_months),
                    "lower_bound_months": float(row.half_life_lower_bound_months),
                }
                for row in curve.groupby("stratum_value", sort=True).tail(1).itertuples(index=False)
            },
        },
        "atlas": {
            "exact_chemsys_rows": int(len(chemistry)),
            "displayable_exact_chemsys_rows": int(chemistry["display_eligible"].sum()),
            "element_pair_rows": int(len(pairs)),
            "displayable_element_pair_rows": int(pairs["display_eligible"].sum()),
            "attribution_chemsys_rows": int(len(attribution_chemistry)),
        },
        "join_audit": join_audit,
        "outputs": data_artifacts,
        "supporting_files": {
            output["data_dictionary"].as_posix(): sha256_file(output["data_dictionary"]),
            output["figure_metadata"].as_posix(): sha256_file(output["figure_metadata"]),
        },
        "figures": [_artifact(path) for path in survival_figures + atlas_figures],
        "gate": {
            "checks": checks,
            "passed": gate_passed,
            "minimum_important_high_confidence_flips": int(
                config["gate"]["minimum_important_high_confidence_flips"]
            ),
            "minimum_attribution_fraction": float(config["gate"]["minimum_attribution_fraction"]),
        },
        "warnings": [
            "Reported-label survival and fragility are descriptive database-revision outcomes, not physical stability probabilities.",
            "P3.3 rebuilt-hull attribution is retained as a separate count layer and is not used as the reported-label risk denominator.",
            "Element and element-pair rows are overlapping multi-membership strata and must not be summed as independent populations.",
        ],
    }
    write_json_atomic(output["manifest"], manifest)
    return manifest


def _compare_csv_counts(
    expected: pd.DataFrame, observed: pd.DataFrame, keys: list[str], count_columns: list[str]
) -> list[str]:
    failures: list[str] = []
    merged = expected[keys + count_columns].merge(
        observed[keys + count_columns],
        on=keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        failures.append(f"Key mismatch for {keys}")
    for column in count_columns:
        left = merged[f"{column}_expected"]
        right = merged[f"{column}_observed"]
        if not left.fillna(-1).eq(right.fillna(-1)).all():
            failures.append(f"Count mismatch for {column}")
    return failures


def verify_descriptive_atlas(config_path: str | Path) -> dict[str, Any]:
    config_path, config = _load_config(config_path)
    preconditions = _require_preconditions(config)
    output = {name: Path(path) for name, path in config["output"].items() if name != "root"}
    manifest = _read_json(output["manifest"])
    failures: list[str] = []
    if manifest.get("task_id") != "P4.1":
        failures.append("manifest task_id mismatch")
    if manifest.get("config_sha256") != sha256_file(config_path):
        failures.append("config hash mismatch")
    for path_text, expected_hash in manifest.get("input_hashes", {}).items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"input hash mismatch: {path}")
    manifest_checks = manifest.get("gate", {}).get("checks", {})
    if not manifest_checks or not all(bool(value) for value in manifest_checks.values()):
        failures.append("manifest gate checks are incomplete or failed")
    for artifact in manifest.get("outputs", []) + manifest.get("figures", []):
        path = Path(artifact["path"])
        if not path.exists():
            failures.append(f"missing artifact: {path}")
            continue
        if sha256_file(path) != artifact["sha256"]:
            failures.append(f"artifact hash mismatch: {path}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if int(parquet.metadata.num_rows) != int(artifact["rows"]):
                failures.append(f"artifact row mismatch: {path}")
            if _schema_hash(path) != artifact["schema_sha256"]:
                failures.append(f"artifact schema mismatch: {path}")
    for path_text, expected_hash in manifest.get("supporting_files", {}).items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"supporting file mismatch: {path}")
    cohort = pd.read_parquet(output["survival_cohort"])
    curve = pd.read_csv(output["survival_curve"])
    risk = pd.read_parquet(output["transition_fragility"])
    interval = pd.read_csv(output["interval_fragility"])
    chemistry = pd.read_csv(output["chemistry_fragility"])
    pairs = pd.read_csv(output["element_pair_fragility"])
    attribution_chemistry = pd.read_csv(output["attribution_by_chemistry"])
    population = config["population"]
    statistics = config["statistics"]
    recomputed_curve = build_all_survival_curves(
        cohort,
        [str(value) for value in population["snapshot_order"]],
        str(population["baseline_snapshot"]),
        float(statistics["days_per_month"]),
    )
    curve_columns = [
        "stratum_type",
        "stratum_value",
        "snapshot",
        "n_at_risk",
        "events_at_time",
        "censored_at_time",
    ]
    if not recomputed_curve[curve_columns].reset_index(drop=True).equals(
        curve[curve_columns].reset_index(drop=True)
    ):
        failures.append("survival curve count reconstruction mismatch")
    recomputed_interval = aggregate_interval_fragility(
        risk, float(statistics["confidence_level"])
    )
    failures.extend(
        _compare_csv_counts(
            recomputed_interval,
            interval,
            ["stratum_type", "stratum_value", "source_snapshot", "target_snapshot"],
            ["at_risk_transitions", "destabilization_events"],
        )
    )
    recomputed_chemistry = aggregate_chemistry_fragility(
        risk,
        float(statistics["confidence_level"]),
        int(config["atlas"]["exact_chemsys_minimum_at_risk_for_display"]),
    )
    failures.extend(
        _compare_csv_counts(
            recomputed_chemistry,
            chemistry,
            ["source_chemsys"],
            ["at_risk_transitions", "destabilization_events"],
        )
    )
    recomputed_pairs = aggregate_element_pair_fragility(
        risk,
        float(statistics["confidence_level"]),
        int(config["atlas"]["element_pair_minimum_at_risk_for_display"]),
    )
    failures.extend(
        _compare_csv_counts(
            recomputed_pairs,
            pairs,
            ["element_a", "element_b"],
            ["at_risk_transitions", "destabilization_events"],
        )
    )
    if int(attribution_chemistry["attributed_flip_count"].sum()) != int(
        preconditions["important_high_confidence_flips"]
    ):
        failures.append("attribution chemistry total does not match P3.3 important flips")
    if not all(
        _confidence_bounds_valid(frame)
        for frame in (curve, interval, chemistry, pairs)
    ):
        failures.append("invalid confidence interval bounds")
    temporary_files = [
        path
        for root in (Path(config["output"]["root"]), Path("reports/P4_1"), Path("data/manifests/P4_1"))
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".tmp", ".partial", ".lock"}
    ]
    if temporary_files:
        failures.append(f"temporary files remain: {len(temporary_files)}")
    status = "PASS" if not failures else "FAIL"
    return {
        "task_id": "P4.1",
        "status": status,
        "gate_status": "GO" if not failures and manifest.get("gate_status") == "GO" else "NO_GO",
        "failures": failures,
        "survival_cohort_rows": int(len(cohort)),
        "survival_events": int(cohort["event_observed"].sum()),
        "rolling_at_risk_transitions": int(len(risk)),
        "rolling_destabilization_events": int(risk["destabilization_event"].sum()),
        "attribution_chemistry_rows": int(len(attribution_chemistry)),
        "temporary_files": len(temporary_files),
        "network_access": False,
        "verified_at_utc": utc_now(),
    }
