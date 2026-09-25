import socket
from pathlib import Path

from phase_evonet.data.synthetic import make_synthetic_transitions
from phase_evonet.evaluation.metrics import expected_calibration_error, top_fraction_enrichment
from phase_evonet.pipeline import run_smoke


def test_synthetic_time_roles_do_not_overlap():
    df = make_synthetic_transitions(n_materials=50, n_snapshots=4, seed=7)
    assert set(df["transition"].unique()) == {0, 1, 2}
    assert df.groupby(["canonical_lineage_id", "transition"]).size().eq(1).all()


def test_metrics_are_finite():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.7, 0.9]
    assert 0 <= expected_calibration_error(y, p) <= 1
    assert top_fraction_enrichment(y, p, 0.5) >= 1


def test_smoke_pipeline(tmp_path: Path):
    report = run_smoke(tmp_path, seed=3)
    assert report["status"] == "PASS"
    assert report["n_train"] > 0 and report["n_test"] > 0
    assert (tmp_path / "report.json").exists()
    assert report["metrics"]["average_precision"] > report["prevalence_test"]


def test_smoke_pipeline_is_deterministic_and_offline(tmp_path: Path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("The synthetic smoke pipeline attempted network access")

    monkeypatch.setattr(socket, "socket", reject_network)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_smoke(first_dir, seed=11)
    second = run_smoke(second_dir, seed=11)

    assert first == second
    assert (
        first_dir / "predictions_preview.csv"
    ).read_bytes() == (
        second_dir / "predictions_preview.csv"
    ).read_bytes()
