"""Release regressions: source integrity, explicit inputs, and truthful QA."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

from phase_evonet.cli import build_parser
from phase_evonet.figure_sources import (
    SourceDataIntegrityError,
    load_manifest,
    verify_sources,
    verify_table,
)
from phase_evonet.r3.common import open_formal_input
from phase_evonet.r3.versioned_benchmark_protocol import FROZEN_HASHES

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path):
    shutil.copytree(ROOT / "source_data", tmp_path / "source_data")
    return tmp_path


@pytest.fixture
def figure_files(sources):
    for number, row in load_manifest(sources).items():
        target = sources / f"outputs/figures/Figure_{number}"
        target.mkdir(parents=True)
        shutil.copyfile(sources / row["path"], target / Path(row["path"]).name)
        Image.new("RGB", (1100, 800), "white").save(target / f"figure_{number}.png")
        Image.new("RGB", (1100, 800), "white").save(target / f"figure_{number}.tiff", dpi=(600, 600), compression="tiff_lzw")
        (target / f"figure_{number}.svg").write_text('<svg><text>synthetic test</text></svg>', encoding="utf-8")
        (target / f"figure_{number}.pdf").write_bytes(b"%PDF-1.4\nsynthetic signature fixture\n")
    return sources


def test_all_released_source_hashes_sizes_and_counts_match():
    records = verify_sources(ROOT)
    assert [row["rows"] for row in records] == [219, 26, 12, 12, 300, 64]


def test_all_shipped_protocol_inputs_keep_frozen_hashes(tmp_path):
    present = [path for path in FROZEN_HASHES if (ROOT / path).is_file()]
    assert "LICENSE" in present
    assert "configs/analysis/v2_1_model_matrix.yaml" in present
    for path in present:
        with open_formal_input(ROOT / path, FROZEN_HASHES[path], task_id="RELEASE_TEST", access_log=tmp_path / "access.jsonl", purpose="Public configuration and license integrity", allowed_roots=(ROOT,)):
            pass


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_git_preserves_published_source_and_frozen_input_bytes(tmp_path, autocrlf):
    if shutil.which("git") is None:
        pytest.skip("Git is required for the release-byte transport regression")
    paths = [".gitattributes", "LICENSE", "configs/analysis/v2_1_model_matrix.yaml", "configs/analysis/v2_1_development_tables.yaml"]
    paths += [row["path"] for row in load_manifest(ROOT).values()]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.autocrlf", autocrlf], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *paths], check=True, capture_output=True)
    for path in paths[1:]:
        blob = subprocess.check_output(["git", "-C", str(tmp_path), "show", ":" + path])
        assert blob == (ROOT / path).read_bytes(), path


def test_newline_normalisation_is_not_silently_accepted(sources):
    path = sources / "source_data/figures/figure_1_source_data.csv"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    with pytest.raises(SourceDataIntegrityError, match="Frozen source mismatch"):
        verify_sources(sources)


@pytest.mark.parametrize("field,value", [("rows", "220"), ("bytes", "98749")])
def test_manifest_count_and_size_are_enforced(sources, field, value):
    expected = load_manifest(sources)[1]
    expected[field] = int(value)
    with pytest.raises(SourceDataIntegrityError):
        verify_table(sources / expected["path"], expected)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "path_escape", "invalid_hash", "invalid_scope"])
def test_invalid_manifest_is_rejected(sources, mutation):
    path = sources / "source_data/source_data_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if mutation == "duplicate":
        rows[1] = rows[0].copy()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "path_escape":
        rows[0]["path"] = "../outside.csv"
    elif mutation == "invalid_hash":
        rows[0]["sha256"] = "not-a-digest"
    else:
        rows[0]["public_release_scope"] = "unapproved"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SourceDataIntegrityError):
        load_manifest(sources)


def test_qa_does_not_certify_visual_review_even_when_file_checks_pass(figure_files):
    result = script("qa_as_figure_revision").audit_figures(figure_files)
    assert result["automated_status"] == "PASS"
    assert result["status"] == "AUTOMATED_PASS_VISUAL_REVIEW_PENDING"
    assert result["manual_visual_qa"]["status"] == "PENDING"
    assert len(result["figures"]) == 6


def test_qa_rejects_identical_tampering_of_source_and_copy(figure_files):
    source = figure_files / "source_data/figures/figure_1_source_data.csv"
    copy = figure_files / "outputs/figures/Figure_1/figure_1_source_data.csv"
    changed = source.read_bytes() + b"\r\n"
    source.write_bytes(changed)
    copy.write_bytes(changed)
    result = script("qa_as_figure_revision").audit_figures(figure_files)
    assert result["automated_status"] == "FAIL"
    assert result["figures"][0]["status"] == "FAIL"
    assert "Frozen source mismatch" in result["figures"][0]["error"]


def test_qa_rejects_missing_output_and_returns_nonzero(figure_files, monkeypatch):
    (figure_files / "outputs/figures/Figure_1/figure_1.pdf").unlink()
    qa = script("qa_as_figure_revision")
    monkeypatch.setattr(sys, "argv", ["qa", "--root", str(figure_files)])
    with pytest.raises(SystemExit) as error:
        qa.main()
    assert error.value.code == 1
    result = json.loads((figure_files / "outputs/figure_revision_qa_report.json").read_text())
    assert result["status"] == "FAIL"
    assert result["manual_visual_qa"]["status"] == "PENDING"


def test_contact_sheet_font_is_bundled_not_system_arial():
    font = script("qa_as_figure_revision").contact_sheet_font()
    assert "DejaVu" in font.getname()[0]
    assert Path(font.path).is_relative_to(Path(matplotlib.get_data_path()))


@pytest.mark.parametrize("command", ["temporal-split-freeze", "temporal-split-verify"])
def test_sealed_workflows_require_explicit_existing_config(command, tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit) as missing:
        parser.parse_args([command])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as unavailable:
        parser.parse_args([command, "--config", str(tmp_path / "absent.yaml")])
    assert unavailable.value.code == 2
    config = tmp_path / "authorised.yaml"
    config.write_text("synthetic: true\n")
    args = parser.parse_args([command, "--config", str(config)])
    assert args.config == str(config)


def test_python_metadata_and_lock_agree():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["project"]["requires-python"] == ">=3.11"
    assert lock["requires-python"] == ">=3.11"


@pytest.mark.parametrize("font", ["Arial", "DejaVu Sans"])
def test_figure5_legend_is_above_points_and_panel_labels(font):
    render = script("render_as_figure_revision")
    with matplotlib.rc_context():
        render.configure_style()
        matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        fig = render.figure_5()
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axis = fig.axes[1]
            legend = axis.get_legend().get_window_extent(renderer)
            assert legend.y0 > axis.get_window_extent(renderer).y1
            for ax in fig.axes:
                for text in ax.texts:
                    if text.get_text() in ("a", "b", "c"):
                        assert not legend.overlaps(text.get_window_extent(renderer))
        finally:
            render.plt.close(fig)


@pytest.mark.parametrize("font", ["Arial", "DejaVu Sans"])
def test_figure6_count_labels_clear_categories_and_bars_keep_source_values(font):
    render = script("render_as_figure_revision")
    with matplotlib.rc_context():
        render.configure_style()
        matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        fig = render.figure_6()
        try:
            fig.canvas.draw()
            axis = fig.axes[1]
            data = render.read(6)
            values = data[data.panel == "state_label_evolution"].set_index("label_definition").loc[list(render.LABELS)]
            expected = values.stable_to_unstable.astype(int).tolist()
            np.testing.assert_array_equal([bar.get_width() for bar in axis.patches[:3]], -np.array(expected))
            numeric = [text for text in axis.texts if text.get_text() in {str(value) for value in expected}]
            assert len(numeric) == 3
            renderer = fig.canvas.get_renderer()
            assert all(not text.get_window_extent(renderer).overlaps(tick.get_window_extent(renderer)) for text in numeric for tick in axis.get_yticklabels())
        finally:
            render.plt.close(fig)
