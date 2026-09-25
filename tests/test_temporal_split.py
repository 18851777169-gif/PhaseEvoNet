from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from phase_evonet.manifest import sha256_file
from phase_evonet.temporal_split import (
    FORBIDDEN_TEST_COLUMNS,
    assigned_role,
    build_temporal_split,
    verify_temporal_split,
)


SNAPSHOTS = ["2022-10-28", "2023-11-01", "2024-12-18", "2025-09-25"]
PAIRS = list(zip(SNAPSHOTS[:-1], SNAPSHOTS[1:], strict=True))


def _base_split() -> dict[str, object]:
    return {
        "split_unit": "canonical_lineage_id",
        "assignment_method": "blake2b_64_modulo",
        "assignment_salt": "TEST_P4_2",
        "modulus": 10000,
        "role_ranges": {
            "train": [0, 7000],
            "validation": [7000, 8500],
            "locked_test": [8500, 10000],
        },
        "supervised_target_intervals": {
            "train": list(PAIRS[0]),
            "validation": list(PAIRS[1]),
            "locked_test": list(PAIRS[2]),
        },
    }


def _lineage_for_role(role: str, split: dict[str, object], start: int) -> tuple[str, int]:
    for number in range(start, start + 100000):
        lineage = f"lin-{number:06d}"
        if assigned_role(lineage, split)[1] == role:
            return lineage, number + 1
    raise AssertionError(f"Could not find lineage for {role}")


def _transition(number: int, lineage: str, pair_index: int) -> dict[str, object]:
    source, target = PAIRS[pair_index]
    source_stable = True
    target_stable = pair_index != 2
    return {
        "transition_id": number.to_bytes(16, "big"),
        "canonical_lineage_id": lineage,
        "identity_confidence": "A1",
        "source_snapshot": source,
        "target_snapshot": target,
        "source_material_id": f"{lineage}-m{pair_index}",
        "target_material_id": f"{lineage}-m{pair_index + 1}",
        "thermo_type": "GGA_GGA+U",
        "observation_status": "observed",
        "source_state_usable": True,
        "target_state_usable": True,
        "source_thermo_id": f"{lineage}-t{pair_index}",
        "target_thermo_id": f"{lineage}-t{pair_index + 1}",
        "source_is_stable": source_stable,
        "target_is_stable": target_stable,
        "reported_label_transition": (
            "stable_to_stable" if target_stable else "stable_to_unstable"
        ),
        "label_flip": not target_stable,
        "source_energy_above_hull": 0.0,
        "target_energy_above_hull": 0.0 if target_stable else 0.1,
        "delta_reported_energy_above_hull": 0.0 if target_stable else 0.1,
        "source_formation_energy_per_atom": -1.0,
        "target_formation_energy_per_atom": -0.9,
        "delta_reported_formation_energy_per_atom": 0.1,
    }


def _write_fixture(root: Path) -> Path:
    split = _base_split()
    cursor = 0
    role_lineages: dict[str, list[str]] = {}
    for role in ("train", "validation", "locked_test"):
        role_lineages[role] = []
        for _ in range(2):
            lineage, cursor = _lineage_for_role(role, split, cursor)
            role_lineages[role].append(lineage)
    rows: list[dict[str, object]] = []
    transition_number = 1
    transition_by_role_interval: dict[tuple[str, int], bytes] = {}
    for role, lineages in role_lineages.items():
        for lineage in lineages:
            for pair_index in range(3):
                row = _transition(transition_number, lineage, pair_index)
                rows.append(row)
                transition_by_role_interval.setdefault(
                    (role, pair_index), row["transition_id"]
                )
                transition_number += 1
    excluded_lineage, cursor = _lineage_for_role("train", split, cursor)
    excluded = _transition(transition_number, excluded_lineage, 0)
    rows.append(excluded)
    input_dir = root / "input"
    reports = root / "reports"
    output = root / "output"
    secrets = root / "secrets"
    input_dir.mkdir(parents=True)
    reports.mkdir(parents=True)
    transitions = pd.DataFrame(rows)
    transition_path = input_dir / "transitions.parquet"
    transitions.to_parquet(transition_path, index=False)
    positive_ids = [
        transition_by_role_interval[("train", 0)],
        transition_by_role_interval[("validation", 1)],
        transition_by_role_interval[("locked_test", 2)],
    ]
    attribution = pd.DataFrame(
        [
            {
                "transition_id": value,
                "attribution_id": (100 + index).to_bytes(16, "big"),
                "unified_label_transition": "stable_to_unstable",
            }
            for index, value in enumerate(positive_ids)
        ]
    )
    attribution_path = input_dir / "attribution.parquet"
    attribution.to_parquet(attribution_path, index=False)
    ledger_path = input_dir / "ineligible.csv"
    pd.DataFrame([{"transition_id": bytes(excluded["transition_id"]).hex()}]).to_csv(
        ledger_path, index=False
    )
    report_values = {
        "p2.json": {"task_id": "P2.2", "task_status": "DONE", "status": "PASS", "gate_status": "GO"},
        "p31.json": {"task_id": "P3.1", "task_status": "DONE", "status": "PASS", "gate_status": "GO"},
        "p33.json": {"task_id": "P3.3", "task_status": "DONE", "status": "PASS", "gate_status": "GO"},
        "p41.json": {
            "task_id": "P4.1",
            "task_status": "DONE",
            "status": "PASS",
            "gate_status": "GO",
            "scope": {"p4_2_or_later_executed": False},
        },
        "p41_manifest.json": {"task_id": "P4.1", "status": "PASS", "gate_status": "GO"},
    }
    for name, value in report_values.items():
        (reports / name).write_text(json.dumps(value), encoding="utf-8")
    for name in ("research.md", "data.md", "gates.md"):
        (reports / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    config = {
        "task_id": "P4.2",
        "seed": 42,
        "freeze_timestamp_utc": "2026-01-01T00:00:00Z",
        "input": {
            "transition_labels": str(transition_path),
            "attribution": str(attribution_path),
            "ineligible_transition_ledger": str(ledger_path),
            "p2_2_report": str(reports / "p2.json"),
            "p3_1_report": str(reports / "p31.json"),
            "p3_3_report": str(reports / "p33.json"),
            "p4_1_report": str(reports / "p41.json"),
            "p4_1_manifest": str(reports / "p41_manifest.json"),
            "research_contract": str(reports / "research.md"),
            "data_contract": str(reports / "data.md"),
            "gate_contract": str(reports / "gates.md"),
        },
        "output": {
            "root": str(output),
            "lineage_assignment": str(output / "assignment.parquet"),
            "split_population": str(output / "population.parquet"),
            "development_labels": str(output / "development.parquet"),
            "locked_test_index": str(output / "test_index.parquet"),
            "locked_test_commitments": str(output / "commitments.csv"),
            "encrypted_test_labels": str(output / "test_labels.aesgcm.json"),
            "locked_thresholds": str(output / "thresholds.json"),
            "label_schema": str(output / "label_schema.json"),
            "public_signing_key": str(output / "public.pem"),
            "manifest": str(output / "manifest.json"),
            "signature": str(output / "manifest.sig"),
            "data_dictionary": str(output / "dictionary.csv"),
        },
        "secrets": {
            "seal_key": str(secrets / "seal.key"),
            "signing_private_key": str(secrets / "signing.key"),
            "create_if_missing": True,
        },
        "population": {
            "observation_status": "observed",
            "identity_confidences": ["A1", "A2"],
        },
        "split": split,
        "seal": {
            "encryption": "AES-256-GCM",
            "encryption_version": "TEST_AESGCM_V1",
            "commitment": "HMAC-SHA256",
            "signature": "Ed25519",
            "unlock_task": "P9.1",
        },
        "locked_thresholds": {
            "p9_flagship": {"maximum_locked_test_ece": 0.04},
            "change_policy": "immutable",
        },
        "prelock_disclosure": {
            "p4_1_used_all_four_snapshots_descriptively": True,
            "individual_locked_partition_membership_or_labels_inspected": False,
            "aggregate_2024_2025_reported_fragility_was_observed_in_P4_1": True,
        },
        "gate": {},
        "parquet": {"compression": "zstd", "row_group_size": 1000},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_lineage_assignment_is_deterministic_and_bounded() -> None:
    split = _base_split()
    first = assigned_role("lin-example", split)
    second = assigned_role("lin-example", split)
    assert first == second
    assert 0 <= first[0] < 10000
    assert first[1] in {"train", "validation", "locked_test"}


def test_synthetic_split_is_signed_sealed_and_deterministic(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    first = build_temporal_split(config_path)
    assert first["status"] == "PASS"
    assert first["gate_status"] == "GO"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = config["output"]
    hashes = {
        name: sha256_file(output[name])
        for name in ("manifest", "signature", "encrypted_test_labels", "lineage_assignment")
    }
    second = build_temporal_split(config_path)
    assert second["status"] == "PASS"
    assert {name: sha256_file(output[name]) for name in hashes} == hashes
    verification = verify_temporal_split(config_path)
    assert verification["status"] == "PASS", verification["failures"]
    assert verification["gate_status"] == "GO"
    development = pd.read_parquet(output["development_labels"])
    locked = pd.read_parquet(output["locked_test_index"])
    assert set(development.assigned_role) == {"train", "validation"}
    assert set(locked.assigned_role) == {"locked_test"}
    assert not FORBIDDEN_TEST_COLUMNS.intersection(locked.columns)
    assert not set(development.canonical_lineage_id).intersection(locked.canonical_lineage_id)
    assert "rebuilt_unified_flip" not in Path(output["encrypted_test_labels"]).read_text()


def test_signature_tampering_is_detected(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    build_temporal_split(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["output"]["manifest"])
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["seed"] = 7
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    verification = verify_temporal_split(config_path)
    assert verification["status"] == "FAIL"
    assert any("signature" in failure for failure in verification["failures"])
    manifest_path.write_bytes(original)
    assert verify_temporal_split(config_path)["status"] == "PASS"
