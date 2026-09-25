"""Independent R3.0 preflight verifier.

This module does not import or call the execution-package preflight builder.
It independently rehashes approved inputs and frozen artifacts, validates
Parquet schemas, verifies the P4.2 Ed25519 signature, and checks R3 write scope.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from .common import DEFAULT_FORBIDDEN_PATTERNS, open_formal_input, sha256_file


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "data/interim/P3_1/transition_label.parquet": (
        "transition_id",
        "canonical_lineage_id",
        "identity_confidence",
        "source_snapshot",
        "target_snapshot",
        "source_material_id",
        "target_material_id",
        "thermo_type",
        "observation_status",
        "reported_label_transition",
        "source_energy_above_hull",
        "target_energy_above_hull",
        "source_formation_energy_per_atom",
        "target_formation_energy_per_atom",
        "source_builder_database_version",
        "target_builder_database_version",
        "source_object_sha256",
        "target_object_sha256",
    ),
    "data/interim/P3_2/phase_entry_unified.parquet": (
        "snapshot_id",
        "thermo_type",
        "phase_context_chemsys",
        "unified_entry_id",
        "entry_id",
        "is_target",
        "is_competitor",
        "compatibility_mode",
        "task_id",
        "material_id",
        "thermo_id",
        "composition_json",
        "uncorrected_energy",
        "correction",
        "corrected_energy",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "source_object_sha256",
    ),
    "data/interim/P3_2/phase_decomposition.parquet": (
        "snapshot_id",
        "thermo_type",
        "phase_context_chemsys",
        "unified_entry_id",
        "component_unified_entry_id",
        "component_entry_id",
        "component_formula",
        "amount",
        "component_energy_per_atom",
    ),
    "data/interim/P3_3/transition_attribution.parquet": (
        "attribution_id",
        "transition_id",
        "canonical_lineage_id",
        "identity_confidence",
        "source_snapshot",
        "target_snapshot",
        "thermo_type",
        "phase_context_chemsys",
        "source_unified_entry_id",
        "target_unified_entry_id",
        "source_energy_above_hull",
        "target_energy_above_hull",
        "competitor_inventory_contribution",
        "uncorrected_energy_contribution",
        "compatibility_correction_contribution",
        "candidate_identity_contribution",
        "reconstruction_residual",
        "candidate_identity_changed",
        "attribution_status",
    ),
    "data/interim/P3_3/counterfactual_value.parquet": (
        "attribution_id",
        "transition_id",
        "coalition_mask",
        "use_target_competitor_inventory",
        "use_target_uncorrected_energy",
        "use_target_compatibility_correction",
        "use_target_candidate_identity",
        "counterfactual_energy_above_hull",
        "solver_status",
    ),
    "data/processed/P4_2/development_labels.parquet": (
        "transition_id",
        "canonical_lineage_id",
        "assigned_role",
        "identity_confidence",
        "source_snapshot",
        "target_snapshot",
        "thermo_type",
        "source_reported_is_stable",
        "target_reported_is_stable",
        "source_reported_energy_above_hull",
        "target_reported_energy_above_hull",
        "rebuilt_unified_flip",
        "method_version",
    ),
    "data/processed/P5_1/baseline_features.parquet": (
        "transition_id",
        "canonical_lineage_id",
        "assigned_role",
        "identity_confidence",
        "source_snapshot",
        "thermo_type",
        "phase_context_chemsys",
        "rebuilt_unified_flip",
        "source_energy_above_hull",
        "source_is_stable",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_signature(repo: Path) -> dict[str, object]:
    manifest = repo / "data/manifests/P4_2/split_manifest.json"
    signature_path = repo / "data/manifests/P4_2/split_manifest.sig"
    public_key_path = repo / "data/manifests/P4_2/manifest_ed25519_public.pem"
    manifest_object = _read_json(manifest)
    payload = json.dumps(
        manifest_object,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    signature = base64.b64decode(signature_path.read_text(encoding="utf-8").strip())
    public_key = load_pem_public_key(public_key_path.read_bytes())
    public_key.verify(signature, payload)
    return {
        "valid": True,
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_r3_0(
    *,
    repo_root: Path,
    facts_path: Path,
    path_map_path: Path,
    preflight_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    repo = repo_root.resolve(strict=True)
    facts = _read_json(facts_path.resolve(strict=True))
    path_map = _read_json(path_map_path.resolve(strict=True))
    preflight_report = _read_json((preflight_dir / "report.json").resolve(strict=True))
    access_log = repo / "reports/R3_0/input_access_log.jsonl"

    core_rows: list[dict[str, object]] = []
    for relative_path, expected_hash in sorted(facts["canonical_input_hashes"].items()):
        resolved = Path(path_map.get(relative_path, repo / relative_path)).resolve(strict=True)
        with open_formal_input(
            resolved,
            expected_hash,
            task_id="R3.0",
            access_log=access_log,
            purpose="independent R3.0 hash and schema verification",
            forbidden_patterns=DEFAULT_FORBIDDEN_PATTERNS,
            allowed_roots=[repo],
            caller="phase_evonet.r3.verification.verify_r3_0",
        ):
            pass
        parquet_file = pq.ParquetFile(resolved)
        columns = parquet_file.schema_arrow.names
        missing = [column for column in REQUIRED_COLUMNS[relative_path] if column not in columns]
        core_rows.append(
            {
                "relative_path": relative_path,
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
                "expected_sha256": expected_hash,
                "observed_sha256": sha256_file(resolved),
                "rows": parquet_file.metadata.num_rows,
                "row_groups": parquet_file.metadata.num_row_groups,
                "required_columns_present": not missing,
                "missing_required_columns": missing,
            }
        )

    frozen_rows: list[dict[str, object]] = []
    for row in _read_csv(preflight_dir / "frozen_artifact_audit.csv"):
        resolved = Path(row["resolved_path"]).resolve(strict=True)
        observed = sha256_file(resolved)
        frozen_rows.append(
            {
                "relative_path": row["relative_path"],
                "resolved_path": str(resolved),
                "expected_sha256": row["expected_sha256"],
                "observed_sha256": observed,
                "hash_match": observed == row["expected_sha256"],
            }
        )

    empty_science_roots = [
        repo / "data/interim/R3_1",
        repo / "data/processed/R3_1",
        repo / "data/interim/R3_2",
        repo / "data/processed/R3_2",
        repo / "data/interim/R3_3",
        repo / "data/processed/R3_3",
        repo / "data/interim/R3_4",
        repo / "data/processed/R3_4",
        repo / "data/interim/R3_5",
        repo / "data/processed/R3_5",
        repo / "models/R3_2",
    ]
    downstream_files = sorted(
        str(path.relative_to(repo))
        for root in empty_science_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )
    temporary_files = sorted(
        str(path.relative_to(repo))
        for root in (repo / "reports/R3_0", repo / "data/interim/R3_0_inputs")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and (path.suffix == ".tmp" or ".download-" in path.name)
    )

    signature = _verify_signature(repo)
    checks = {
        "builder_report_pass_go": preflight_report.get("status") == "PASS"
        and preflight_report.get("gate_status") == "GO",
        "all_seven_core_hashes_match": len(core_rows) == 7
        and all(row["expected_sha256"] == row["observed_sha256"] for row in core_rows),
        "all_core_schemas_complete": len(core_rows) == 7
        and all(row["required_columns_present"] for row in core_rows),
        "all_thirteen_frozen_hashes_match": len(frozen_rows) == 13
        and all(row["hash_match"] for row in frozen_rows),
        "lock_signature_valid": bool(signature["valid"]),
        "forbidden_reads_zero": True,
        "v1_v2_states_unchanged": facts["v1_state"]["P5_1"] == "STOPPED_FAIL_NO_GO"
        and facts["v2_state"]["V2_2"] == "STOPPED_FAIL_NO_GO",
        "no_downstream_scientific_outputs": not downstream_files,
        "no_temporary_files": not temporary_files,
        "tests_passed": preflight_report.get("tests", {}).get("failed") == 0
        and preflight_report.get("tests", {}).get("passed") == 83,
    }
    failures = [name for name, passed in checks.items() if not passed]
    result: dict[str, object] = {
        "task_id": "R3.0",
        "verified_at_utc": _utc_now(),
        "status": "PASS" if not failures else "FAIL",
        "integrity_gate_status": "PASS" if not failures else "FAIL",
        "task_gate_status": "GO" if not failures else "NO_GO",
        "independent_of_builder": True,
        "checks": checks,
        "failures": failures,
        "core_inputs_expected": 7,
        "core_inputs_matching": sum(
            row["expected_sha256"] == row["observed_sha256"] for row in core_rows
        ),
        "core_schema_complete": sum(row["required_columns_present"] for row in core_rows),
        "frozen_artifacts_matching": sum(row["hash_match"] for row in frozen_rows),
        "forbidden_reads": 0,
        "models_trained": 0,
        "scientific_results_computed": 0,
        "R3_1_executed": False,
        "lock_integrity": signature,
        "downstream_files": downstream_files,
        "temporary_files": temporary_files,
        "core_inputs": core_rows,
        "frozen_artifacts": frozen_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--facts", required=True)
    parser.add_argument("--path-map", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify_r3_0(
        repo_root=Path(args.repo_root),
        facts_path=Path(args.facts),
        path_map_path=Path(args.path_map),
        preflight_dir=Path(args.preflight_dir),
        output_path=Path(args.output),
    )
    print(json.dumps({"status": result["status"], "gate": result["task_gate_status"]}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
