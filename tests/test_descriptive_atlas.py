from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from phase_evonet.descriptive_atlas import (
    METHOD_VERSION,
    build_descriptive_atlas,
    kaplan_meier_curve,
    verify_descriptive_atlas,
    wilson_interval,
)
from phase_evonet.manifest import sha256_file


SNAPSHOTS = ["2022-10-28", "2023-11-01", "2024-12-18", "2025-09-25"]


def _transition(
    number: int,
    lineage: str,
    workflow: str,
    source: str,
    target: str,
    source_material: str,
    target_material: str,
    source_stable: bool,
    target_stable: bool | None,
    observed: bool = True,
) -> dict[str, object]:
    return {
        "transition_id": number.to_bytes(16, "big"),
        "canonical_lineage_id": lineage,
        "identity_confidence": "A1",
        "source_snapshot": source,
        "target_snapshot": target,
        "source_material_id": source_material,
        "target_material_id": target_material,
        "thermo_type": workflow,
        "observation_status": "observed" if observed else "source_only",
        "source_state_present": True,
        "target_state_present": observed,
        "source_state_usable": True,
        "target_state_usable": observed,
        "source_is_stable": source_stable,
        "target_is_stable": target_stable,
    }


def _write_fixture(root: Path) -> Path:
    input_dir = root / "input"
    output_dir = root / "output"
    report_dir = root / "reports"
    input_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    transitions = pd.DataFrame(
        [
            _transition(1, "lin-1", "GGA_GGA+U", SNAPSHOTS[0], SNAPSHOTS[1], "m1a", "m1b", True, True),
            _transition(2, "lin-1", "GGA_GGA+U", SNAPSHOTS[1], SNAPSHOTS[2], "m1b", "m1c", True, False),
            _transition(3, "lin-2", "R2SCAN", SNAPSHOTS[0], SNAPSHOTS[1], "m2a", "m2b", True, True),
            _transition(4, "lin-2", "R2SCAN", SNAPSHOTS[1], SNAPSHOTS[2], "m2b", "m2c", True, True),
            _transition(5, "lin-2", "R2SCAN", SNAPSHOTS[2], SNAPSHOTS[3], "m2c", "m2d", True, True),
            _transition(6, "lin-3", "GGA_GGA+U", SNAPSHOTS[0], SNAPSHOTS[1], "m3a", "m3b", True, None, False),
        ]
    )
    transition_path = input_dir / "transition.parquet"
    transitions.to_parquet(transition_path, index=False)
    lineage = pd.DataFrame(
        [
            ["lin-1", SNAPSHOTS[0], "m1a", '{"Fe":1,"O":1}', "A1", True],
            ["lin-1", SNAPSHOTS[1], "m1b", '{"Fe":1,"O":1}', "A1", True],
            ["lin-2", SNAPSHOTS[0], "m2a", '{"O":2,"Si":1}', "A1", True],
            ["lin-2", SNAPSHOTS[1], "m2b", '{"O":2,"Si":1}', "A1", True],
            ["lin-2", SNAPSHOTS[2], "m2c", '{"O":2,"Si":1}', "A1", True],
            ["lin-3", SNAPSHOTS[0], "m3a", '{"C":1}', "A1", True],
        ],
        columns=[
            "canonical_lineage_id",
            "snapshot_id",
            "material_id",
            "composition_reduced_json",
            "lineage_confidence",
            "high_confidence",
        ],
    )
    lineage_path = input_dir / "lineage.parquet"
    lineage.to_parquet(lineage_path, index=False)
    attribution = pd.DataFrame(
        [
            {
                "transition_id": (2).to_bytes(16, "big"),
                "phase_context_chemsys": "Fe-O",
                "unified_label_transition": "stable_to_unstable",
                "dominant_channel": "competitor_inventory",
                "delta_energy_above_hull": 0.1,
            }
        ]
    )
    attribution_path = input_dir / "attribution.parquet"
    attribution.to_parquet(attribution_path, index=False)
    report_payloads = {
        "p2.json": {"task_id": "P2.2", "task_status": "DONE", "status": "PASS", "gate_status": "GO"},
        "p31.json": {"task_id": "P3.1", "task_status": "DONE", "status": "PASS", "gate_status": "GO"},
        "p33.json": {
            "task_id": "P3.3",
            "task_status": "DONE",
            "status": "PASS",
            "gate_status": "GO",
            "population": {"important_high_confidence_flips": 1, "attribution_coverage": 1.0},
        },
        "p33_manifest.json": {"task_id": "P3.3", "status": "PASS", "gate_status": "GO"},
    }
    for name, payload in report_payloads.items():
        (report_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "task_id": "P4.1",
        "seed": 42,
        "input": {
            "transition_labels": str(transition_path),
            "lineage": str(lineage_path),
            "attribution": str(attribution_path),
            "p2_2_report": str(report_dir / "p2.json"),
            "p3_1_report": str(report_dir / "p31.json"),
            "p3_3_manifest": str(report_dir / "p33_manifest.json"),
            "p3_3_report": str(report_dir / "p33.json"),
        },
        "output": {
            "root": str(output_dir),
            "survival_cohort": str(output_dir / "survival.parquet"),
            "survival_curve": str(output_dir / "survival_curve.csv"),
            "interval_fragility": str(output_dir / "interval.csv"),
            "transition_fragility": str(output_dir / "risk.parquet"),
            "chemistry_fragility": str(output_dir / "chemistry.csv"),
            "element_pair_fragility": str(output_dir / "pairs.csv"),
            "attribution_by_chemistry": str(output_dir / "attribution_chemistry.csv"),
            "manifest": str(output_dir / "manifest.json"),
            "data_dictionary": str(output_dir / "dictionary.csv"),
            "figure_metadata": str(output_dir / "figure_metadata.json"),
            "figure_directory": str(output_dir / "figures"),
        },
        "population": {
            "identity_confidences": ["A1", "A2"],
            "baseline_snapshot": SNAPSHOTS[0],
            "snapshot_order": SNAPSHOTS,
        },
        "statistics": {
            "confidence_level": 0.95,
            "days_per_month": 30.4375,
            "days_per_year": 365.25,
        },
        "atlas": {
            "exact_chemsys_minimum_at_risk_for_display": 1,
            "element_pair_minimum_at_risk_for_display": 1,
            "top_elements_by_exposure": 4,
            "top_attribution_chemsys_by_flip_count": 3,
        },
        "figures": {
            "formats": ["png", "svg", "pdf"],
            "dpi": 90,
            "style_version": "TEST",
            "survival_stem": "survival",
            "atlas_stem": "atlas",
        },
        "gate": {
            "minimum_important_high_confidence_flips": 1,
            "minimum_attribution_fraction": 0.6,
        },
        "parquet": {"compression": "zstd", "row_group_size": 1000},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_wilson_interval_is_bounded() -> None:
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert 0.0 < upper < 0.5
    lower, upper = wilson_interval(10, 10)
    assert 0.5 < lower < 1.0
    assert upper > 0.999999999999


def test_kaplan_meier_half_life() -> None:
    cohort = pd.DataFrame(
        {
            "duration_days": [365, 365, 782, 1063],
            "event_observed": [True, False, True, False],
        }
    )
    curve = kaplan_meier_curve(
        cohort,
        SNAPSHOTS,
        SNAPSHOTS[0],
        30.4375,
        "overall",
        "all_workflows",
    )
    assert curve.iloc[0].survival_probability == 1.0
    assert curve.iloc[-1].survival_probability <= 0.5
    assert bool(curve.iloc[-1].half_life_reached)
    assert curve.ci_lower.le(curve.survival_probability).all()
    assert curve.ci_upper.ge(curve.survival_probability).all()


def test_synthetic_build_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    first = build_descriptive_atlas(config_path)
    assert first["status"] == "PASS"
    assert first["gate_status"] == "GO"
    assert first["population"]["survival_cohort_rows"] == 3
    assert first["population"]["survival_events"] == 1
    assert first["population"]["rolling_destabilization_events"] == 1
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    curve_hash = sha256_file(config["output"]["survival_curve"])
    png_hash = sha256_file(Path(config["output"]["figure_directory"]) / "survival.png")
    second = build_descriptive_atlas(config_path)
    assert second["status"] == "PASS"
    assert sha256_file(config["output"]["survival_curve"]) == curve_hash
    assert sha256_file(Path(config["output"]["figure_directory"]) / "survival.png") == png_hash
    verification = verify_descriptive_atlas(config_path)
    assert verification["status"] == "PASS", verification["failures"]
    assert verification["gate_status"] == "GO"
    assert verification["temporary_files"] == 0
    assert json.loads(Path(config["output"]["manifest"]).read_text())["method_version"] == METHOD_VERSION
