from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_smoke
from .normalization import (
    normalize_snapshot_collections,
    normalize_snapshots,
    verify_normalization_outputs,
)
from .identity_candidates import generate_identity_candidates, verify_identity_candidates
from .identity_lineages import build_identity_lineages, verify_identity_lineages
from .reported_transitions import build_reported_transitions, verify_reported_transitions
from .context_phase_diagrams import (
    build_unified_phase_diagrams,
    build_unified_phase_diagrams_sharded,
    verify_unified_phase_diagrams,
)
from .transition_attribution import (
    build_transition_attribution,
    verify_transition_attribution,
)
from .descriptive_atlas import build_descriptive_atlas, verify_descriptive_atlas
from .temporal_split import build_temporal_split, verify_temporal_split
from .baselines import build_baselines, verify_baselines
from .v2_development import (
    build_v2_development,
    check_deterministic_rebuild,
    verify_v2_development,
)
from .v2_capacity_matched import (
    build_v2_capacity_matched,
    finalize_v2_capacity_matched,
    verify_v2_capacity_matched,
)
from .r3.energy_amplitude import (
    build_energy_amplitude,
    finalize_energy_amplitude,
    verify_energy_amplitude,
)
from .r3.standardized_survival import (
    build_standardized_survival,
    finalize_standardized_survival,
    verify_standardized_survival,
)
from .r3.competitor_cascade import (
    build_attribution_cascade,
    finalize_attribution_cascade,
    verify_attribution_cascade,
)
from .snapshots import run_snapshot_download, verify_snapshot_manifest


def existing_config(value: str) -> str:
    """Require an explicitly supplied local configuration for sealed workflows."""
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"configuration file does not exist: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase-evo")
    sub = p.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Run the synthetic no-network smoke pipeline")
    smoke.add_argument("--output-dir", default="reports/smoke")
    smoke.add_argument("--seed", type=int, default=42)
    download = sub.add_parser(
        "snapshot-download",
        help="Inventory, download, and validate frozen Materials Project snapshots",
    )
    download.add_argument("--config", default="configs/data/p1_1_snapshots.yaml")
    download.add_argument("--raw-dir", default="data/raw/materials_project")
    download.add_argument("--manifest-dir", default="data/manifests/P1_1")
    download.add_argument("--workers", type=int, default=16)
    download.add_argument("--transfer-engine", choices=("aria2", "curl", "python"))
    download.add_argument("--inventory-only", action="store_true")
    verify = sub.add_parser(
        "snapshot-verify",
        help="Offline re-verification of every object in a snapshot manifest",
    )
    verify.add_argument(
        "--manifest", default="data/manifests/P1_1/snapshot_manifest.jsonl"
    )
    verify.add_argument("--workers", type=int, default=16)
    normalize = sub.add_parser(
        "normalize-snapshots",
        help="Normalize P1.1 material/task schemas into deterministic Parquet",
    )
    normalize.add_argument("--config", default="configs/data/p1_2_normalization.yaml")
    normalize_snapshot = sub.add_parser(
        "normalize-snapshot",
        help="Rebuild selected P1.2 collections for one frozen snapshot",
    )
    normalize_snapshot.add_argument(
        "--config", default="configs/data/p1_2_normalization.yaml"
    )
    normalize_snapshot.add_argument("--snapshot", required=True)
    normalize_snapshot.add_argument(
        "--collections",
        nargs="+",
        choices=("materials", "thermo", "provenance"),
        required=True,
    )
    normalize_verify = sub.add_parser(
        "normalize-verify",
        help="Offline row/schema/hash verification of every P1.2 Parquet artifact",
    )
    normalize_verify.add_argument(
        "--config", default="configs/data/p1_2_normalization.yaml"
    )
    normalize_verify.add_argument("--update-manifest", action="store_true")
    candidates = sub.add_parser(
        "identity-candidates",
        help="Generate adjacent-snapshot P2.1 material identity candidate edges",
    )
    candidates.add_argument(
        "--config", default="configs/data/p2_1_identity_candidates.yaml"
    )
    candidates_verify = sub.add_parser(
        "identity-candidates-verify",
        help="Offline verification of the P2.1 candidate edge artifact and recall audit",
    )
    candidates_verify.add_argument(
        "--config", default="configs/data/p2_1_identity_candidates.yaml"
    )
    lineages = sub.add_parser(
        "identity-lineages",
        help="Resolve P2.1 candidates into P2.2 confidence-rated lineages",
    )
    lineages.add_argument(
        "--config", default="configs/data/p2_2_identity_lineages.yaml"
    )
    lineages_verify = sub.add_parser(
        "identity-lineages-verify",
        help="Offline verification of P2.2 identity and lineage artifacts",
    )
    lineages_verify.add_argument(
        "--config", default="configs/data/p2_2_identity_lineages.yaml"
    )
    transitions = sub.add_parser(
        "reported-transitions",
        help="Build P3.1 same-workflow reported stability-label transitions",
    )
    transitions.add_argument(
        "--config", default="configs/data/p3_1_reported_transitions.yaml"
    )
    transitions_verify = sub.add_parser(
        "reported-transitions-verify",
        help="Offline verification of P3.1 reported transition artifacts",
    )
    transitions_verify.add_argument(
        "--config", default="configs/data/p3_1_reported_transitions.yaml"
    )
    phase_diagrams = sub.add_parser(
        "unified-phase-diagrams",
        help="Rebuild P3.2 compatibility-partitioned phase diagrams and decompositions",
    )
    phase_diagrams.add_argument(
        "--config", default="configs/data/p3_2_unified_phase_diagrams.yaml"
    )
    phase_diagrams_sharded = sub.add_parser(
        "unified-phase-diagrams-sharded",
        help="Build P3.2 mixed contexts in deterministic resumable process shards",
    )
    phase_diagrams_sharded.add_argument(
        "--config", default="configs/data/p3_2_unified_phase_diagrams.yaml"
    )
    phase_diagrams_sharded.add_argument(
        "--shards-per-snapshot", type=int, default=8
    )
    phase_diagrams_sharded.add_argument("--workers", type=int, default=32)
    phase_diagrams_verify = sub.add_parser(
        "unified-phase-diagrams-verify",
        help="Offline verification of P3.2 phase-entry and decomposition artifacts",
    )
    phase_diagrams_verify.add_argument(
        "--config", default="configs/data/p3_2_unified_phase_diagrams.yaml"
    )
    attribution = sub.add_parser(
        "transition-attribution",
        help="Build P3.3 exact counterfactual/Shapley transition attribution",
    )
    attribution.add_argument(
        "--config", default="configs/data/p3_3_transition_attribution.yaml"
    )
    attribution_verify = sub.add_parser(
        "transition-attribution-verify",
        help="Offline verification of P3.3 attribution and coalition artifacts",
    )
    attribution_verify.add_argument(
        "--config", default="configs/data/p3_3_transition_attribution.yaml"
    )
    descriptive_atlas = sub.add_parser(
        "descriptive-atlas",
        help="Build P4.1 reported-label survival and chemical fragility atlas",
    )
    descriptive_atlas.add_argument(
        "--config", default="configs/analysis/p4_1_survival_fragility.yaml"
    )
    descriptive_atlas_verify = sub.add_parser(
        "descriptive-atlas-verify",
        help="Offline verification of P4.1 tables, figures, and manifest",
    )
    descriptive_atlas_verify.add_argument(
        "--config", default="configs/analysis/p4_1_survival_fragility.yaml"
    )
    temporal_split = sub.add_parser(
        "temporal-split-freeze",
        help="Freeze P4.2 lineage-disjoint temporal labels and sealed test payload",
    )
    temporal_split.add_argument(
        "--config", required=True, type=existing_config,
        help="Authorised local P4.2 configuration; sealed-data configuration is not distributed",
    )
    temporal_split_verify = sub.add_parser(
        "temporal-split-verify",
        help="Offline verification of the P4.2 signed split and cryptographic seal",
    )
    temporal_split_verify.add_argument(
        "--config", required=True, type=existing_config,
        help="Authorised local P4.2 configuration; sealed-data configuration is not distributed",
    )
    baselines = sub.add_parser(
        "baselines",
        help="Build P5.1 development-only heuristic and phase-context baselines",
    )
    baselines.add_argument("--config", default="configs/analysis/p5_1_baselines.yaml")
    baselines_verify = sub.add_parser(
        "baselines-verify",
        help="Offline verification of P5.1 baseline artifacts and metrics",
    )
    baselines_verify.add_argument(
        "--config", default="configs/analysis/p5_1_baselines.yaml"
    )
    v2_development = sub.add_parser(
        "v2-development-tables",
        help="Build V2.1 source-only development features, cause-specific targets, and interval exposure",
    )
    v2_development.add_argument(
        "--config", default="configs/analysis/v2_1_development_tables.yaml"
    )
    v2_development_verify = sub.add_parser(
        "v2-development-verify",
        help="Offline verification of V2.1 source-only development artifacts",
    )
    v2_development_verify.add_argument(
        "--config", default="configs/analysis/v2_1_development_tables.yaml"
    )
    v2_development_determinism = sub.add_parser(
        "v2-development-determinism",
        help="Rebuild and byte-compare deterministic V2.1 development artifacts",
    )
    v2_development_determinism.add_argument(
        "--config", default="configs/analysis/v2_1_development_tables.yaml"
    )
    v2_capacity = sub.add_parser(
        "v2-capacity-matched",
        help="Fit and compare V2.2 development-only capacity-matched M0-M5 models",
    )
    v2_capacity.add_argument(
        "--config", default="configs/analysis/v2_2_capacity_matched.yaml"
    )
    v2_capacity_verify = sub.add_parser(
        "v2-capacity-matched-verify",
        help="Verify V2.2 model, prediction, metric, bootstrap, and gate artifacts",
    )
    v2_capacity_verify.add_argument(
        "--config", default="configs/analysis/v2_2_capacity_matched.yaml"
    )
    v2_capacity_finalize = sub.add_parser(
        "v2-capacity-matched-finalize",
        help="Finalize the V2.2 machine-readable report after tests and state transition",
    )
    v2_capacity_finalize.add_argument(
        "--config", default="configs/analysis/v2_2_capacity_matched.yaml"
    )
    v2_capacity_finalize.add_argument("--tests-passed", type=int, required=True)
    v2_capacity_finalize.add_argument("--tests-failed", type=int, required=True)
    v2_capacity_finalize.add_argument(
        "--test-duration-seconds", type=float, required=True
    )
    v2_capacity_finalize.add_argument("--command-log", required=True)
    v2_capacity_finalize.add_argument("--changed-files", required=True)
    r3_energy = sub.add_parser(
        "r3-energy-amplitude",
        help="Build R3.1 exact-flip amplitudes, signed margins, and threshold robustness",
    )
    r3_energy.add_argument(
        "--config", default="configs/r3/r3_1_energy_amplitude.yaml"
    )
    r3_energy_verify = sub.add_parser(
        "r3-energy-amplitude-verify",
        help="Independently verify the R3.1 artifacts and frozen route decision",
    )
    r3_energy_verify.add_argument(
        "--config", default="configs/r3/r3_1_energy_amplitude.yaml"
    )
    r3_energy_finalize = sub.add_parser(
        "r3-energy-amplitude-finalize",
        help="Seal the R3.1 machine-readable report after tests and verification",
    )
    r3_energy_finalize.add_argument(
        "--config", default="configs/r3/r3_1_energy_amplitude.yaml"
    )
    r3_energy_finalize.add_argument("--tests-passed", type=int, required=True)
    r3_energy_finalize.add_argument("--tests-failed", type=int, required=True)
    r3_energy_finalize.add_argument("--test-duration-seconds", type=float, required=True)
    r3_energy_finalize.add_argument("--command-log", required=True)
    r3_energy_finalize.add_argument("--changed-files", required=True)
    r3_survival = sub.add_parser(
        "r3-standardized-survival",
        help="Build R3.2 release-standardized durability and multi-state analyses",
    )
    r3_survival.add_argument(
        "--config", default="configs/r3/r3_2_standardized_survival.yaml"
    )
    r3_survival.add_argument("--workers", type=int)
    r3_survival_verify = sub.add_parser(
        "r3-standardized-survival-verify",
        help="Independently verify R3.2 artifacts and structural gates",
    )
    r3_survival_verify.add_argument(
        "--config", default="configs/r3/r3_2_standardized_survival.yaml"
    )
    r3_survival_finalize = sub.add_parser(
        "r3-standardized-survival-finalize",
        help="Finalize R3.2 after independent verification and full tests",
    )
    r3_survival_finalize.add_argument(
        "--config", default="configs/r3/r3_2_standardized_survival.yaml"
    )
    r3_survival_finalize.add_argument("--tests-passed", type=int, required=True)
    r3_survival_finalize.add_argument("--tests-failed", type=int, required=True)
    r3_survival_finalize.add_argument("--test-duration-seconds", type=float, required=True)
    r3_survival_finalize.add_argument("--command-log", required=True)
    r3_survival_finalize.add_argument("--changed-files", required=True)
    r3_cascade = sub.add_parser(
        "r3-attribution-cascade",
        help="Build R3.3 attribution sensitivity, competitor counterfactuals, and cascade graphs",
    )
    r3_cascade.add_argument(
        "--config", default="configs/r3/r3_3_attribution_cascade.yaml"
    )
    r3_cascade_verify = sub.add_parser(
        "r3-attribution-cascade-verify",
        help="Independently verify the R3.3 formal artifacts and gates",
    )
    r3_cascade_verify.add_argument(
        "--config", default="configs/r3/r3_3_attribution_cascade.yaml"
    )
    r3_cascade_finalize = sub.add_parser(
        "r3-attribution-cascade-finalize",
        help="Finalize R3.3 after independent verification and full tests",
    )
    r3_cascade_finalize.add_argument(
        "--config", default="configs/r3/r3_3_attribution_cascade.yaml"
    )
    r3_cascade_finalize.add_argument("--tests-passed", type=int, required=True)
    r3_cascade_finalize.add_argument("--tests-failed", type=int, required=True)
    r3_cascade_finalize.add_argument("--test-duration-seconds", type=float, required=True)
    r3_cascade_finalize.add_argument("--command-log", required=True)
    r3_cascade_finalize.add_argument("--changed-files", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        result = run_smoke(args.output_dir, seed=args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "snapshot-download":
        result = run_snapshot_download(
            config_path=args.config,
            raw_dir=args.raw_dir,
            manifest_dir=args.manifest_dir,
            workers=args.workers,
            transfer_engine=args.transfer_engine,
            inventory_only=args.inventory_only,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "snapshot-verify":
        result = verify_snapshot_manifest(args.manifest, workers=args.workers)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "normalize-snapshots":
        result = normalize_snapshots(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "normalize-snapshot":
        result = normalize_snapshot_collections(
            args.config, args.snapshot, args.collections
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "normalize-verify":
        result = verify_normalization_outputs(
            args.config, update_manifest=args.update_manifest
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "identity-candidates":
        result = generate_identity_candidates(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "identity-candidates-verify":
        result = verify_identity_candidates(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "identity-lineages":
        result = build_identity_lineages(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "identity-lineages-verify":
        result = verify_identity_lineages(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "reported-transitions":
        result = build_reported_transitions(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "reported-transitions-verify":
        result = verify_reported_transitions(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "unified-phase-diagrams":
        result = build_unified_phase_diagrams(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "unified-phase-diagrams-sharded":
        result = build_unified_phase_diagrams_sharded(
            args.config,
            shards_per_snapshot=args.shards_per_snapshot,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "unified-phase-diagrams-verify":
        result = verify_unified_phase_diagrams(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "transition-attribution":
        result = build_transition_attribution(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "transition-attribution-verify":
        result = verify_transition_attribution(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "descriptive-atlas":
        result = build_descriptive_atlas(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "descriptive-atlas-verify":
        result = verify_descriptive_atlas(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "temporal-split-freeze":
        result = build_temporal_split(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "temporal-split-verify":
        result = verify_temporal_split(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "baselines":
        result = build_baselines(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "baselines-verify":
        result = verify_baselines(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-development-tables":
        result = build_v2_development(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-development-verify":
        result = verify_v2_development(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-development-determinism":
        result = check_deterministic_rebuild(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-capacity-matched":
        result = build_v2_capacity_matched(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-capacity-matched-verify":
        result = verify_v2_capacity_matched(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "v2-capacity-matched-finalize":
        result = finalize_v2_capacity_matched(
            args.config,
            tests_passed=args.tests_passed,
            tests_failed=args.tests_failed,
            test_duration_seconds=args.test_duration_seconds,
            command_log_path=args.command_log,
            changed_files_path=args.changed_files,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-energy-amplitude":
        result = build_energy_amplitude(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-energy-amplitude-verify":
        result = verify_energy_amplitude(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-energy-amplitude-finalize":
        result = finalize_energy_amplitude(
            args.config,
            tests_passed=args.tests_passed,
            tests_failed=args.tests_failed,
            test_duration_seconds=args.test_duration_seconds,
            command_log_path=args.command_log,
            changed_files_path=args.changed_files,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-standardized-survival":
        result = build_standardized_survival(args.config, workers=args.workers)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-standardized-survival-verify":
        result = verify_standardized_survival(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-standardized-survival-finalize":
        result = finalize_standardized_survival(
            args.config,
            tests_passed=args.tests_passed,
            tests_failed=args.tests_failed,
            test_duration_seconds=args.test_duration_seconds,
            command_log_path=args.command_log,
            changed_files_path=args.changed_files,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-attribution-cascade":
        result = build_attribution_cascade(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-attribution-cascade-verify":
        result = verify_attribution_cascade(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "r3-attribution-cascade-finalize":
        result = finalize_attribution_cascade(
            args.config,
            tests_passed=args.tests_passed,
            tests_failed=args.tests_failed,
            test_duration_seconds=args.test_duration_seconds,
            command_log_path=args.command_log,
            changed_files_path=args.changed_files,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
