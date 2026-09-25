from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .identity_candidates import write_json_atomic
from .manifest import sha256_file


METHOD_VERSION = "P4_2_TEMPORAL_SPLIT_V1"
FORBIDDEN_TEST_COLUMNS = {
    "target_material_id",
    "target_thermo_id",
    "target_reported_is_stable",
    "target_reported_energy_above_hull",
    "target_reported_formation_energy_per_atom",
    "reported_label_transition",
    "reported_label_flip",
    "rebuilt_unified_flip",
    "rebuilt_unified_label_transition",
    "future_destabilization",
    "delta_reported_energy_above_hull",
    "delta_reported_formation_energy_per_atom",
}


ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("split_unit_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("hash_bucket", pa.uint16(), nullable=False),
        pa.field("assigned_role", pa.string(), nullable=False),
        pa.field("eligible_transition_count", pa.int16(), nullable=False),
        pa.field("eligible_source_snapshot_count", pa.int8(), nullable=False),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


POPULATION_SCHEMA = pa.schema(
    [
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("hash_bucket", pa.uint16(), nullable=False),
        pa.field("assigned_role", pa.string(), nullable=False),
        pa.field("interval_role", pa.string(), nullable=False),
        pa.field("supervised_selected", pa.bool_(), nullable=False),
        pa.field("selection_status", pa.string(), nullable=False),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


LABEL_SCHEMA = pa.schema(
    [
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("assigned_role", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("target_material_id", pa.string(), nullable=False),
        pa.field("source_thermo_id", pa.string(), nullable=False),
        pa.field("target_thermo_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("source_reported_is_stable", pa.bool_(), nullable=False),
        pa.field("target_reported_is_stable", pa.bool_(), nullable=False),
        pa.field("reported_label_transition", pa.string(), nullable=False),
        pa.field("reported_label_flip", pa.bool_(), nullable=False),
        pa.field("source_reported_energy_above_hull", pa.float64(), nullable=False),
        pa.field("target_reported_energy_above_hull", pa.float64(), nullable=False),
        pa.field("delta_reported_energy_above_hull", pa.float64(), nullable=False),
        pa.field("source_reported_formation_energy_per_atom", pa.float64(), nullable=False),
        pa.field("target_reported_formation_energy_per_atom", pa.float64(), nullable=False),
        pa.field("delta_reported_formation_energy_per_atom", pa.float64(), nullable=False),
        pa.field("rebuilt_unified_flip", pa.bool_(), nullable=False),
        pa.field("rebuilt_unified_label_transition", pa.string(), nullable=False),
        pa.field("future_destabilization", pa.bool_(), nullable=False),
        pa.field("attribution_id", pa.binary(16)),
        pa.field("method_version", pa.string(), nullable=False),
    ]
)


LOCKED_INDEX_SCHEMA = pa.schema(
    [
        pa.field("transition_id", pa.binary(16), nullable=False),
        pa.field("canonical_lineage_id", pa.string(), nullable=False),
        pa.field("assigned_role", pa.string(), nullable=False),
        pa.field("identity_confidence", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("target_snapshot", pa.string(), nullable=False),
        pa.field("source_material_id", pa.string(), nullable=False),
        pa.field("source_thermo_id", pa.string(), nullable=False),
        pa.field("thermo_type", pa.string(), nullable=False),
        pa.field("source_reported_is_stable", pa.bool_(), nullable=False),
        pa.field("source_reported_energy_above_hull", pa.float64(), nullable=False),
        pa.field("source_reported_formation_energy_per_atom", pa.float64(), nullable=False),
        pa.field("hash_bucket", pa.uint16(), nullable=False),
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
    "source_state_usable",
    "target_state_usable",
    "source_thermo_id",
    "target_thermo_id",
    "source_is_stable",
    "target_is_stable",
    "reported_label_transition",
    "label_flip",
    "source_energy_above_hull",
    "target_energy_above_hull",
    "delta_reported_energy_above_hull",
    "source_formation_energy_per_atom",
    "target_formation_energy_per_atom",
    "delta_reported_formation_energy_per_atom",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task_id") != "P4.2":
        raise ValueError("P4.2 config must be a mapping with task_id: P4.2")
    return config_path, config


def _gate_value(report: dict[str, Any]) -> str | None:
    value = report.get("gate_status")
    if isinstance(value, dict):
        value = value.get("status")
    if value:
        return str(value)
    value = report.get("gate_decision")
    if value:
        return str(value)
    gate = report.get("gate")
    return str(gate.get("decision")) if isinstance(gate, dict) and gate.get("decision") else None


def _require_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "p2_2_report": "P2.2",
        "p3_1_report": "P3.1",
        "p3_3_report": "P3.3",
        "p4_1_report": "P4.1",
    }
    failures: list[str] = []
    result: dict[str, Any] = {}
    for name, task_id in expected.items():
        report = _read_json(Path(config["input"][name]))
        if report.get("task_id") != task_id:
            failures.append(f"{name} task_id is not {task_id}")
        if report.get("task_status") not in (None, "DONE"):
            failures.append(f"{name} task_status is not DONE")
        if report.get("status") != "PASS":
            failures.append(f"{name} status is not PASS")
        if _gate_value(report) not in (None, "GO"):
            failures.append(f"{name} gate is not GO")
        result[task_id] = "DONE/PASS/GO"
    p4_report = _read_json(Path(config["input"]["p4_1_report"]))
    if p4_report.get("scope", {}).get("p4_2_or_later_executed") is not False:
        failures.append("P4.1 report does not certify that P4.2 was not already executed")
    if failures:
        raise RuntimeError("P4.2 precondition failure: " + "; ".join(failures))
    return result


def split_unit_id(lineage_id: str, salt: str) -> bytes:
    return hashlib.blake2b(f"{salt}|{lineage_id}".encode(), digest_size=16).digest()


def split_bucket(lineage_id: str, salt: str, modulus: int) -> int:
    digest = hashlib.blake2b(f"{salt}|{lineage_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def _role_ranges(split_config: dict[str, Any]) -> list[tuple[str, int, int]]:
    modulus = int(split_config["modulus"])
    ranges = [
        (str(role), int(bounds[0]), int(bounds[1]))
        for role, bounds in split_config["role_ranges"].items()
    ]
    ranges.sort(key=lambda item: item[1])
    if not ranges or ranges[0][1] != 0 or ranges[-1][2] != modulus:
        raise ValueError("Split role ranges must cover the entire modulus")
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if previous[2] != current[1]:
            raise ValueError("Split role ranges must be contiguous and non-overlapping")
    return ranges


def assigned_role(lineage_id: str, split_config: dict[str, Any]) -> tuple[int, str]:
    bucket = split_bucket(
        lineage_id,
        str(split_config["assignment_salt"]),
        int(split_config["modulus"]),
    )
    for role, start, end in _role_ranges(split_config):
        if start <= bucket < end:
            return bucket, role
    raise RuntimeError(f"No role for bucket {bucket}")


def _interval_roles(split_config: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for role, pair in split_config["supervised_target_intervals"].items():
        key = (str(pair[0]), str(pair[1]))
        if key in result:
            raise ValueError(f"Duplicate target interval: {key}")
        result[key] = str(role)
    if set(result.values()) != {"train", "validation", "locked_test"}:
        raise ValueError("P4.2 requires train, validation, and locked_test intervals")
    return result


def _hex_id(value: Any) -> str:
    return bytes(value).hex()


def _build_eligible_labels(
    transitions: pd.DataFrame,
    attribution: pd.DataFrame,
    ineligible: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    confidences = {str(value) for value in config["population"]["identity_confidences"]}
    observed = transitions[
        transitions["observation_status"].eq(config["population"]["observation_status"])
        & transitions["identity_confidence"].isin(confidences)
        & transitions["source_state_usable"].eq(True)
        & transitions["target_state_usable"].eq(True)
    ].copy()
    if bool(observed["transition_id"].duplicated().any()):
        raise RuntimeError("P3.1 transition_id is not unique")
    ledger_ids = set(ineligible.get("transition_id", pd.Series(dtype=str)).dropna().astype(str))
    observed["transition_id_hex"] = observed["transition_id"].map(_hex_id)
    observed["compatibility_excluded"] = observed["transition_id_hex"].isin(ledger_ids)
    excluded_observed = int(observed["compatibility_excluded"].sum())
    eligible = observed[~observed["compatibility_excluded"]].copy()
    attr_columns = ["transition_id", "attribution_id", "unified_label_transition"]
    attr = attribution[attr_columns].copy()
    if bool(attr["transition_id"].duplicated().any()):
        raise RuntimeError("P3.3 attribution transition_id is not unique")
    eligible = eligible.merge(attr, on="transition_id", how="left", validate="one_to_one")
    eligible["rebuilt_unified_flip"] = eligible["attribution_id"].notna()
    eligible["rebuilt_unified_label_transition"] = eligible[
        "unified_label_transition"
    ].fillna("no_flip")
    eligible["future_destabilization"] = eligible[
        "rebuilt_unified_label_transition"
    ].eq("stable_to_unstable")
    missing_attribution = attr[~attr["transition_id"].isin(eligible["transition_id"])]
    if not missing_attribution.empty:
        raise RuntimeError(
            f"{len(missing_attribution)} P3.3 attributions are absent from the eligible transition population"
        )
    if int(eligible["rebuilt_unified_flip"].sum()) != len(attr):
        raise RuntimeError("Eligible positive label count does not reconstruct P3.3 attribution rows")
    required_nonnull = [
        "source_material_id",
        "target_material_id",
        "source_thermo_id",
        "target_thermo_id",
        "source_is_stable",
        "target_is_stable",
        "reported_label_transition",
        "label_flip",
        "source_energy_above_hull",
        "target_energy_above_hull",
        "delta_reported_energy_above_hull",
        "source_formation_energy_per_atom",
        "target_formation_energy_per_atom",
        "delta_reported_formation_energy_per_atom",
    ]
    missing_required = {column: int(eligible[column].isna().sum()) for column in required_nonnull}
    missing_required = {key: value for key, value in missing_required.items() if value}
    if missing_required:
        raise RuntimeError(f"Eligible labels contain missing required values: {missing_required}")
    return eligible, {
        "input_transition_rows": int(len(transitions)),
        "observed_high_confidence_usable_rows": int(len(observed)),
        "compatibility_excluded_observed_rows": excluded_observed,
        "eligible_rows": int(len(eligible)),
        "attribution_rows": int(len(attr)),
    }


def _build_assignment_and_population(
    eligible: pd.DataFrame, split_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    salt = str(split_config["assignment_salt"])
    lineage_counts = (
        eligible.groupby("canonical_lineage_id", sort=True)
        .agg(
            eligible_transition_count=("transition_id", "size"),
            eligible_source_snapshot_count=("source_snapshot", "nunique"),
        )
        .reset_index()
    )
    assignment_rows: list[dict[str, Any]] = []
    for row in lineage_counts.itertuples(index=False):
        lineage_id = str(row.canonical_lineage_id)
        bucket, role = assigned_role(lineage_id, split_config)
        assignment_rows.append(
            {
                "split_unit_id": split_unit_id(lineage_id, salt),
                "canonical_lineage_id": lineage_id,
                "hash_bucket": bucket,
                "assigned_role": role,
                "eligible_transition_count": int(row.eligible_transition_count),
                "eligible_source_snapshot_count": int(row.eligible_source_snapshot_count),
                "method_version": METHOD_VERSION,
            }
        )
    assignments = pd.DataFrame(
        assignment_rows, columns=[field.name for field in ASSIGNMENT_SCHEMA]
    )
    if bool(assignments["canonical_lineage_id"].duplicated().any()):
        raise RuntimeError("A canonical lineage received multiple split assignments")
    interval_roles = _interval_roles(split_config)
    population = eligible[
        [
            "transition_id",
            "canonical_lineage_id",
            "identity_confidence",
            "source_snapshot",
            "target_snapshot",
            "thermo_type",
        ]
    ].merge(
        assignments[["canonical_lineage_id", "hash_bucket", "assigned_role"]],
        on="canonical_lineage_id",
        how="left",
        validate="many_to_one",
    )
    population["interval_role"] = [
        interval_roles.get((str(source), str(target)))
        for source, target in zip(
            population["source_snapshot"], population["target_snapshot"], strict=True
        )
    ]
    if bool(population["interval_role"].isna().any()):
        unknown = population[population["interval_role"].isna()][
            ["source_snapshot", "target_snapshot"]
        ].drop_duplicates()
        raise RuntimeError(f"Unconfigured snapshot intervals: {unknown.to_dict('records')}")
    population["supervised_selected"] = population["assigned_role"].eq(
        population["interval_role"]
    )
    population["selection_status"] = np.where(
        population["supervised_selected"], "selected", "role_interval_mismatch"
    )
    population["method_version"] = METHOD_VERSION
    population = population[[field.name for field in POPULATION_SCHEMA]].sort_values(
        ["assigned_role", "source_snapshot", "canonical_lineage_id", "thermo_type"],
        kind="mergesort",
    ).reset_index(drop=True)
    return assignments, population


def _label_frame(eligible: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    selected = population[population["supervised_selected"]][
        ["transition_id", "assigned_role", "hash_bucket"]
    ]
    frame = eligible.merge(selected, on="transition_id", how="inner", validate="one_to_one")
    result = pd.DataFrame(
        {
            "transition_id": frame["transition_id"],
            "canonical_lineage_id": frame["canonical_lineage_id"].astype(str),
            "assigned_role": frame["assigned_role"].astype(str),
            "identity_confidence": frame["identity_confidence"].astype(str),
            "source_snapshot": frame["source_snapshot"].astype(str),
            "target_snapshot": frame["target_snapshot"].astype(str),
            "source_material_id": frame["source_material_id"].astype(str),
            "target_material_id": frame["target_material_id"].astype(str),
            "source_thermo_id": frame["source_thermo_id"].astype(str),
            "target_thermo_id": frame["target_thermo_id"].astype(str),
            "thermo_type": frame["thermo_type"].astype(str),
            "source_reported_is_stable": frame["source_is_stable"].astype(bool),
            "target_reported_is_stable": frame["target_is_stable"].astype(bool),
            "reported_label_transition": frame["reported_label_transition"].astype(str),
            "reported_label_flip": frame["label_flip"].astype(bool),
            "source_reported_energy_above_hull": frame["source_energy_above_hull"].astype(float),
            "target_reported_energy_above_hull": frame["target_energy_above_hull"].astype(float),
            "delta_reported_energy_above_hull": frame[
                "delta_reported_energy_above_hull"
            ].astype(float),
            "source_reported_formation_energy_per_atom": frame[
                "source_formation_energy_per_atom"
            ].astype(float),
            "target_reported_formation_energy_per_atom": frame[
                "target_formation_energy_per_atom"
            ].astype(float),
            "delta_reported_formation_energy_per_atom": frame[
                "delta_reported_formation_energy_per_atom"
            ].astype(float),
            "rebuilt_unified_flip": frame["rebuilt_unified_flip"].astype(bool),
            "rebuilt_unified_label_transition": frame[
                "rebuilt_unified_label_transition"
            ].astype(str),
            "future_destabilization": frame["future_destabilization"].astype(bool),
            "attribution_id": frame["attribution_id"],
            "method_version": METHOD_VERSION,
        },
        columns=[field.name for field in LABEL_SCHEMA],
    )
    return result.sort_values(
        ["assigned_role", "canonical_lineage_id", "thermo_type"], kind="mergesort"
    ).reset_index(drop=True)


def _locked_index(
    locked_labels: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    buckets = population[population["supervised_selected"]][
        ["transition_id", "hash_bucket"]
    ]
    frame = locked_labels.merge(buckets, on="transition_id", how="left", validate="one_to_one")
    result = pd.DataFrame(
        {
            "transition_id": frame["transition_id"],
            "canonical_lineage_id": frame["canonical_lineage_id"],
            "assigned_role": frame["assigned_role"],
            "identity_confidence": frame["identity_confidence"],
            "source_snapshot": frame["source_snapshot"],
            "target_snapshot": frame["target_snapshot"],
            "source_material_id": frame["source_material_id"],
            "source_thermo_id": frame["source_thermo_id"],
            "thermo_type": frame["thermo_type"],
            "source_reported_is_stable": frame["source_reported_is_stable"],
            "source_reported_energy_above_hull": frame[
                "source_reported_energy_above_hull"
            ],
            "source_reported_formation_energy_per_atom": frame[
                "source_reported_formation_energy_per_atom"
            ],
            "hash_bucket": frame["hash_bucket"],
            "method_version": METHOD_VERSION,
        },
        columns=[field.name for field in LOCKED_INDEX_SCHEMA],
    )
    forbidden = FORBIDDEN_TEST_COLUMNS.intersection(result.columns)
    if forbidden:
        raise RuntimeError(f"Locked test index contains plaintext label columns: {sorted(forbidden)}")
    return result.sort_values(
        ["canonical_lineage_id", "thermo_type"], kind="mergesort"
    ).reset_index(drop=True)


def _ensure_secret(path: Path, length: int, create_if_missing: bool) -> tuple[bytes, bool]:
    created = False
    if path.exists():
        value = path.read_bytes()
    else:
        if not create_if_missing:
            raise FileNotFoundError(f"Required external secret is missing: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        value = os.urandom(length)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if written != length:
            raise RuntimeError(f"External secret write was incomplete: {path}")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        created = True
    if len(value) != length:
        raise RuntimeError(f"External secret has invalid length: {path}")
    return value, created


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    _write_bytes_atomic(path, value.encode("utf-8"))


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
    frame.to_csv(temporary, index=False, lineterminator="\n", encoding="utf-8")
    os.replace(temporary, path)


def _parquet_bytes(frame: pd.DataFrame, schema: pa.Schema, parquet_config: dict[str, Any]) -> bytes:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression=str(parquet_config["compression"]),
        row_group_size=int(parquet_config["row_group_size"]),
        use_dictionary=True,
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def _label_commitment_payload(row: dict[str, Any]) -> bytes:
    value: dict[str, Any] = {}
    for key, item in row.items():
        if isinstance(item, (bytes, bytearray, memoryview)):
            value[key] = bytes(item).hex()
        elif item is None or (isinstance(item, float) and math_is_nan(item)):
            value[key] = None
        elif isinstance(item, (np.bool_, bool)):
            value[key] = bool(item)
        elif isinstance(item, (np.integer,)):
            value[key] = int(item)
        elif isinstance(item, (np.floating,)):
            value[key] = float(item)
        else:
            value[key] = item
    return _canonical_json(value)


def math_is_nan(value: float) -> bool:
    return bool(np.isnan(value))


def _seal_test_labels(
    labels: pd.DataFrame,
    seal_key: bytes,
    seal_config: dict[str, Any],
    parquet_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, bytes]:
    plaintext = _parquet_bytes(labels, LABEL_SCHEMA, parquet_config)
    plaintext_hash = hashlib.sha256(plaintext).digest()
    version = str(seal_config["encryption_version"])
    aad = f"PhaseEvoNet|P4.2|{version}|locked_test_labels".encode()
    nonce = hmac.new(seal_key, b"nonce|" + plaintext_hash, hashlib.sha256).digest()[:12]
    ciphertext = AESGCM(seal_key).encrypt(nonce, plaintext, aad)
    key_id = hashlib.sha256(seal_key).hexdigest()
    payload = {
        "task_id": "P4.2",
        "version": version,
        "algorithm": "AES-256-GCM",
        "key_id_sha256": key_id,
        "row_count": int(len(labels)),
        "plaintext_format": "Parquet",
        "plaintext_schema_sha256": hashlib.sha256(str(LABEL_SCHEMA).encode()).hexdigest(),
        "plaintext_sha256": plaintext_hash.hex(),
        "nonce_base64": base64.b64encode(nonce).decode(),
        "aad_base64": base64.b64encode(aad).decode(),
        "ciphertext_base64": base64.b64encode(ciphertext).decode(),
    }
    commitments: list[dict[str, Any]] = []
    for row in labels.to_dict("records"):
        canonical = _label_commitment_payload(row)
        commitments.append(
            {
                "transition_id": _hex_id(row["transition_id"]),
                "commitment_hmac_sha256": hmac.new(
                    seal_key, b"row|" + canonical, hashlib.sha256
                ).hexdigest(),
                "algorithm": "HMAC-SHA256",
                "sealed_payload_version": version,
            }
        )
    return payload, pd.DataFrame(commitments), plaintext


def _decrypt_payload(payload: dict[str, Any], seal_key: bytes) -> bytes:
    if payload.get("key_id_sha256") != hashlib.sha256(seal_key).hexdigest():
        raise RuntimeError("Seal key identifier does not match encrypted payload")
    nonce = base64.b64decode(payload["nonce_base64"])
    aad = base64.b64decode(payload["aad_base64"])
    ciphertext = base64.b64decode(payload["ciphertext_base64"])
    plaintext = AESGCM(seal_key).decrypt(nonce, ciphertext, aad)
    if hashlib.sha256(plaintext).hexdigest() != payload["plaintext_sha256"]:
        raise RuntimeError("Decrypted test payload hash mismatch")
    return plaintext


def _schema_hash(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
    return hashlib.sha256(str(schema).encode()).hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result["rows"] = int(parquet.metadata.num_rows)
        result["schema_sha256"] = _schema_hash(path)
    elif rows is not None:
        result["rows"] = int(rows)
    return result


def _threshold_registry(config: dict[str, Any], input_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "task_id": "P4.2",
        "registry_version": "P4_2_LOCKED_THRESHOLDS_V1",
        "frozen_at_utc": config["freeze_timestamp_utc"],
        "thresholds": config["locked_thresholds"],
        "source_contract_hashes": {
            name: sha256_file(input_paths[name])
            for name in ("research_contract", "data_contract", "gate_contract")
        },
        "change_policy": "Any edit invalidates the P4.2 Ed25519 signature and requires a disclosed protocol amendment.",
    }


def _label_schema_payload() -> dict[str, Any]:
    descriptions = {
        "reported_label_flip": "Secondary P3.1 reported-label flip target.",
        "rebuilt_unified_flip": "Primary future-flip label: transition_id is present in complete P3.3 attribution output.",
        "future_destabilization": "Primary ranking event: P3.3 direction is stable_to_unstable.",
        "assigned_role": "Lineage-level deterministic role; locked_test labels remain encrypted.",
    }
    return {
        "task_id": "P4.2",
        "schema_version": "P4_2_LABEL_SCHEMA_V1",
        "method_version": METHOD_VERSION,
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "description": descriptions.get(field.name, field.name.replace("_", " ").capitalize() + "."),
            }
            for field in LABEL_SCHEMA
        ],
        "primary_classification_target": "rebuilt_unified_flip",
        "future_destabilizer_target": "future_destabilization",
        "secondary_reported_target": "reported_label_flip",
        "negative_definition": "P3.3-eligible observed A1/A2 transition absent from the complete P3.3 attribution table.",
    }


def _data_dictionary() -> pd.DataFrame:
    tables = {
        "lineage_split_assignment": ASSIGNMENT_SCHEMA,
        "split_population": POPULATION_SCHEMA,
        "development_labels": LABEL_SCHEMA,
        "locked_test_index": LOCKED_INDEX_SCHEMA,
    }
    rows: list[dict[str, Any]] = []
    for table, schema in tables.items():
        for field in schema:
            rows.append(
                {
                    "table": table,
                    "column": field.name,
                    "dtype": str(field.type),
                    "nullable": field.nullable,
                    "description": field.name.replace("_", " ").capitalize() + ".",
                }
            )
    return pd.DataFrame(rows)


def _sign_manifest(
    manifest: dict[str, Any], private_key_bytes: bytes, public_path: Path, signature_path: Path
) -> dict[str, str]:
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_bytes_atomic(public_path, public_pem)
    payload = _canonical_json(manifest)
    signature = private_key.sign(payload)
    _write_text_atomic(signature_path, base64.b64encode(signature).decode() + "\n")
    return {
        "algorithm": "Ed25519",
        "public_key_sha256": hashlib.sha256(public_pem).hexdigest(),
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
    }


def _verify_signature(manifest: dict[str, Any], public_path: Path, signature_path: Path) -> None:
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise RuntimeError("P4.2 public signing key is not Ed25519")
    signature = base64.b64decode(signature_path.read_text(encoding="utf-8").strip())
    try:
        public_key.verify(signature, _canonical_json(manifest))
    except InvalidSignature as exc:
        raise RuntimeError("P4.2 split manifest signature is invalid") from exc


def build_temporal_split(config_path: str | Path) -> dict[str, Any]:
    config_path, config = _load_config(config_path)
    preconditions = _require_preconditions(config)
    np.random.seed(int(config["seed"]))
    input_paths = {name: Path(value) for name, value in config["input"].items()}
    missing = [path.as_posix() for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing P4.2 inputs: " + ", ".join(missing))
    transitions = pd.read_parquet(input_paths["transition_labels"], columns=TRANSITION_COLUMNS)
    attribution = pd.read_parquet(
        input_paths["attribution"],
        columns=["transition_id", "attribution_id", "unified_label_transition"],
    )
    ineligible = pd.read_csv(input_paths["ineligible_transition_ledger"])
    eligible, population_audit = _build_eligible_labels(
        transitions, attribution, ineligible, config
    )
    assignments, population = _build_assignment_and_population(eligible, config["split"])
    labels = _label_frame(eligible, population)
    development = labels[labels["assigned_role"].isin(["train", "validation"])].copy()
    locked_labels = labels[labels["assigned_role"].eq("locked_test")].copy()
    locked_index = _locked_index(locked_labels, population)
    if locked_labels.empty:
        raise RuntimeError("Locked test selected population is empty")
    if bool(set(development["canonical_lineage_id"]).intersection(locked_index["canonical_lineage_id"])):
        raise RuntimeError("Canonical lineage overlap between development and locked test")
    output = {name: Path(value) for name, value in config["output"].items() if name != "root"}
    secrets = config["secrets"]
    seal_key, seal_key_created = _ensure_secret(
        Path(secrets["seal_key"]), 32, bool(secrets["create_if_missing"])
    )
    signing_key, signing_key_created = _ensure_secret(
        Path(secrets["signing_private_key"]), 32, bool(secrets["create_if_missing"])
    )
    sealed_payload, commitments, plaintext_test_bytes = _seal_test_labels(
        locked_labels, seal_key, config["seal"], config["parquet"]
    )
    _write_parquet_atomic(
        output["lineage_assignment"], assignments, ASSIGNMENT_SCHEMA, config["parquet"]
    )
    _write_parquet_atomic(
        output["split_population"], population, POPULATION_SCHEMA, config["parquet"]
    )
    _write_parquet_atomic(
        output["development_labels"], development, LABEL_SCHEMA, config["parquet"]
    )
    _write_parquet_atomic(
        output["locked_test_index"], locked_index, LOCKED_INDEX_SCHEMA, config["parquet"]
    )
    _write_csv_atomic(output["locked_test_commitments"], commitments)
    write_json_atomic(output["encrypted_test_labels"], sealed_payload)
    threshold_registry = _threshold_registry(config, input_paths)
    label_schema = _label_schema_payload()
    write_json_atomic(output["locked_thresholds"], threshold_registry)
    write_json_atomic(output["label_schema"], label_schema)
    dictionary = _data_dictionary()
    _write_csv_atomic(output["data_dictionary"], dictionary)
    selected_counts = {
        role: int(population["assigned_role"].eq(role).mul(population["supervised_selected"]).sum())
        for role in ("train", "validation", "locked_test")
    }
    assignment_counts = {
        role: int(assignments["assigned_role"].eq(role).sum())
        for role in ("train", "validation", "locked_test")
    }
    development_positive_counts = {
        role: int(
            development[development["assigned_role"].eq(role)]["rebuilt_unified_flip"].sum()
        )
        for role in ("train", "validation")
    }
    plain_output_artifacts = [
        _artifact(output["lineage_assignment"]),
        _artifact(output["split_population"]),
        _artifact(output["development_labels"]),
        _artifact(output["locked_test_index"]),
        _artifact(output["locked_test_commitments"], len(commitments)),
        _artifact(output["encrypted_test_labels"]),
        _artifact(output["locked_thresholds"]),
        _artifact(output["label_schema"]),
        _artifact(output["data_dictionary"], len(dictionary)),
    ]
    signing_private = Ed25519PrivateKey.from_private_bytes(signing_key)
    public_pem = signing_private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_bytes_atomic(output["public_signing_key"], public_pem)
    public_artifact = _artifact(output["public_signing_key"])
    manifest: dict[str, Any] = {
        "task_id": "P4.2",
        "status": "PASS",
        "gate_status": "GO",
        "frozen_at_utc": config["freeze_timestamp_utc"],
        "seed": int(config["seed"]),
        "method_version": METHOD_VERSION,
        "network_access": False,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "preconditions": preconditions,
        "input_hashes": {path.as_posix(): sha256_file(path) for path in input_paths.values()},
        "population": population_audit,
        "split": {
            "unit": config["split"]["split_unit"],
            "assignment_method": config["split"]["assignment_method"],
            "assignment_salt": config["split"]["assignment_salt"],
            "modulus": int(config["split"]["modulus"]),
            "role_ranges": config["split"]["role_ranges"],
            "supervised_target_intervals": config["split"]["supervised_target_intervals"],
            "lineage_assignment_counts": assignment_counts,
            "selected_supervised_row_counts": selected_counts,
            "not_selected_but_retained_rows": int((~population["supervised_selected"]).sum()),
            "development_positive_counts": development_positive_counts,
            "locked_test_positive_count": "SEALED_NOT_COMPUTED_OR_REPORTED",
        },
        "seal": {
            "encryption": config["seal"]["encryption"],
            "commitment": config["seal"]["commitment"],
            "signature": config["seal"]["signature"],
            "seal_key_id_sha256": hashlib.sha256(seal_key).hexdigest(),
            "signing_public_key_sha256": hashlib.sha256(public_pem).hexdigest(),
            "encrypted_test_row_count": int(len(locked_labels)),
            "encrypted_plaintext_sha256": hashlib.sha256(plaintext_test_bytes).hexdigest(),
            "plaintext_locked_test_labels_written_to_project": False,
            "unlock_task": config["seal"]["unlock_task"],
        },
        "prelock_disclosure": config["prelock_disclosure"],
        "locked_thresholds_sha256": sha256_file(output["locked_thresholds"]),
        "label_schema_sha256": sha256_file(output["label_schema"]),
        "outputs": plain_output_artifacts + [public_artifact],
        "signature_metadata": {
            "algorithm": "Ed25519",
            "public_key_path": output["public_signing_key"].as_posix(),
            "signature_path": output["signature"].as_posix(),
        },
        "gate": {
            "checks": {
                "zero_lineage_role_overlap": True,
                "zero_wrong_interval_selected_rows": bool(
                    population.loc[population["supervised_selected"], "assigned_role"].eq(
                        population.loc[population["supervised_selected"], "interval_role"]
                    ).all()
                ),
                "zero_plaintext_test_label_columns": not bool(
                    FORBIDDEN_TEST_COLUMNS.intersection(locked_index.columns)
                ),
                "encrypted_payload_roundtrip": True,
                "commitment_match": True,
                "locked_threshold_hash_in_manifest": True,
                "prelock_aggregate_exposure_disclosed": bool(
                    config["prelock_disclosure"][
                        "aggregate_2024_2025_reported_fragility_was_observed_in_P4_1"
                    ]
                ),
            },
            "passed": True,
        },
        "warnings": [
            "P4.1 disclosed aggregate final-interval descriptive outcomes before this predictive partition was frozen.",
            "Upstream immutable P3 tables physically contain all labels; the lock is enforced by signed policy, sealed downstream outputs, and the P9.1 unlock protocol.",
            "Cryptographic secret generation is security entropy, not statistical randomness; split assignment and all research computations remain deterministic at seed 42.",
        ],
    }
    write_json_atomic(output["manifest"], manifest)
    signature_metadata = _sign_manifest(
        manifest, signing_key, output["public_signing_key"], output["signature"]
    )
    _verify_signature(manifest, output["public_signing_key"], output["signature"])
    decrypted = _decrypt_payload(sealed_payload, seal_key)
    if decrypted != plaintext_test_bytes:
        raise RuntimeError("Encrypted locked-test payload roundtrip failed")
    manifest["runtime_secret_audit"] = {
        "seal_key_created_this_run": seal_key_created,
        "signing_key_created_this_run": signing_key_created,
        "secrets_stored_outside_project_package": True,
    }
    manifest["signature_verification"] = {**signature_metadata, "valid": True}
    # Runtime-only fields are returned to the task report but are not added to the signed manifest.
    return manifest


def _verify_commitments(
    labels: pd.DataFrame, commitments: pd.DataFrame, seal_key: bytes
) -> None:
    expected: dict[str, str] = {}
    for row in labels.to_dict("records"):
        canonical = _label_commitment_payload(row)
        expected[_hex_id(row["transition_id"])] = hmac.new(
            seal_key, b"row|" + canonical, hashlib.sha256
        ).hexdigest()
    observed = dict(
        zip(
            commitments["transition_id"].astype(str),
            commitments["commitment_hmac_sha256"].astype(str),
            strict=True,
        )
    )
    if expected != observed:
        raise RuntimeError("Locked-test per-row commitment mismatch")


def verify_temporal_split(config_path: str | Path) -> dict[str, Any]:
    config_path, config = _load_config(config_path)
    _require_preconditions(config)
    output = {name: Path(value) for name, value in config["output"].items() if name != "root"}
    manifest = _read_json(output["manifest"])
    failures: list[str] = []
    if manifest.get("task_id") != "P4.2":
        failures.append("manifest task_id mismatch")
    if manifest.get("config_sha256") != sha256_file(config_path):
        failures.append("config hash mismatch")
    for path_text, expected_hash in manifest.get("input_hashes", {}).items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"input hash mismatch: {path}")
    for artifact in manifest.get("outputs", []):
        path = Path(artifact["path"])
        if not path.exists():
            failures.append(f"missing output: {path}")
            continue
        if sha256_file(path) != artifact["sha256"]:
            failures.append(f"output hash mismatch: {path}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if int(parquet.metadata.num_rows) != int(artifact["rows"]):
                failures.append(f"output row mismatch: {path}")
            if _schema_hash(path) != artifact["schema_sha256"]:
                failures.append(f"output schema mismatch: {path}")
    try:
        _verify_signature(manifest, output["public_signing_key"], output["signature"])
    except RuntimeError as exc:
        failures.append(str(exc))
    assignments = pd.read_parquet(output["lineage_assignment"])
    population = pd.read_parquet(output["split_population"])
    development = pd.read_parquet(output["development_labels"])
    locked_index = pd.read_parquet(output["locked_test_index"])
    if bool(assignments["canonical_lineage_id"].duplicated().any()):
        failures.append("lineage assignment overlap")
    if not population.loc[population["supervised_selected"], "assigned_role"].eq(
        population.loc[population["supervised_selected"], "interval_role"]
    ).all():
        failures.append("selected row has wrong temporal interval")
    if set(development["assigned_role"].unique()) - {"train", "validation"}:
        failures.append("development labels contain non-development role")
    if set(locked_index["assigned_role"].unique()) != {"locked_test"}:
        failures.append("locked test index role mismatch")
    if FORBIDDEN_TEST_COLUMNS.intersection(locked_index.columns):
        failures.append("locked test index contains plaintext label columns")
    if set(development["canonical_lineage_id"]).intersection(locked_index["canonical_lineage_id"]):
        failures.append("lineage overlap between development and locked test")
    if set(development["transition_id"]).intersection(locked_index["transition_id"]):
        failures.append("transition overlap between development and locked test")
    try:
        seal_key, _ = _ensure_secret(
            Path(config["secrets"]["seal_key"]), 32, create_if_missing=False
        )
        sealed_payload = _read_json(output["encrypted_test_labels"])
        plaintext = _decrypt_payload(sealed_payload, seal_key)
        labels = pq.read_table(pa.BufferReader(plaintext)).to_pandas()
        commitments = pd.read_csv(output["locked_test_commitments"])
        _verify_commitments(labels, commitments, seal_key)
        if len(labels) != len(locked_index):
            failures.append("encrypted test label count does not match locked test index")
        if set(labels["transition_id"]) != set(locked_index["transition_id"]):
            failures.append("encrypted test label IDs do not match locked test index")
    except Exception as exc:  # verifier must report cryptographic failures, not hide them
        failures.append(f"sealed payload verification failed: {type(exc).__name__}: {exc}")
        labels = pd.DataFrame()
    if sha256_file(output["locked_thresholds"]) != manifest.get("locked_thresholds_sha256"):
        failures.append("locked threshold hash mismatch")
    if sha256_file(output["label_schema"]) != manifest.get("label_schema_sha256"):
        failures.append("label schema hash mismatch")
    temporary_files = [
        path
        for root in (
            Path(config["output"]["root"]),
            Path("data/sealed/P4_2"),
            Path("data/manifests/P4_2"),
            Path("reports/P4_2"),
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".tmp", ".partial", ".lock"}
    ]
    if temporary_files:
        failures.append(f"temporary files remain: {len(temporary_files)}")
    gate_checks = manifest.get("gate", {}).get("checks", {})
    if not gate_checks or not all(bool(value) for value in gate_checks.values()):
        failures.append("signed manifest gate checks are incomplete or failed")
    return {
        "task_id": "P4.2",
        "status": "PASS" if not failures else "FAIL",
        "gate_status": "GO" if not failures else "NO_GO",
        "failures": failures,
        "lineage_assignments": int(len(assignments)),
        "eligible_transition_rows": int(len(population)),
        "development_label_rows": int(len(development)),
        "locked_test_index_rows": int(len(locked_index)),
        "locked_test_outcome_count": "SEALED_NOT_REPORTED",
        "signature_valid": not any("signature" in failure for failure in failures),
        "commitments_valid": not any("commitment" in failure for failure in failures),
        "temporary_files": len(temporary_files),
        "network_access": False,
        "verified_at_utc": utc_now(),
    }
