# PhaseEvoNet

Research code for analysing how calculated phase-stability labels and fixed-score benchmarks change across frozen Materials Project database releases.

The central design keeps candidate identities, feature inputs, and prediction scores fixed while changing the database version that supplies the reference labels. The repository also contains lineage construction, phase-diagram reconstruction, transition accounting, survival analysis, provenance-defined omission analysis, and current-release extension code used by the study.

## Installation

PhaseEvoNet requires Python 3.11 or later. The release is tested with Python 3.11.

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows (PowerShell), use the environment's interpreter directly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

In the commands below, use `python` from the activated environment, or replace it
with `.\.venv\Scripts\python.exe` on Windows. For the recorded dependency
resolution, `uv sync --locked --extra dev --python 3.11` installs from `uv.lock`.

## Verification

Run the complete unit-test suite:

```bash
python -m pytest -q
```

Run the deterministic, network-free synthetic smoke workflow:

```bash
python -m phase_evonet.cli smoke --output-dir reports/smoke --seed 42
```

The smoke workflow generates synthetic versioned materials and does not access Materials Project.

## Repository layout

- `src/phase_evonet/`: analysis and reconstruction library.
- `configs/`: versioned analysis, data, experiment, and R3 configuration files.
- `scripts/`: current-release and figure-generation entry points.
- `source_data/`: identifier-free aggregate source tables for Figures 1–6.
- `tests/`: deterministic unit and regression tests based on synthetic fixtures.

Use `phase-evo --help` to inspect the supported command-line workflows. Detailed R3 analyses can also be invoked through the modules in `phase_evonet.r3`.

The self-contained public workflows are the synthetic smoke tests and aggregate
figure reproduction. Record-level study workflows additionally require the
authorised archival data, manifests, and prerequisite reports; this repository
alone is not a complete record-level replay bundle. The sealed temporal-split
commands require an explicit `--config` pointing to an authorised local
configuration. That configuration, encryption keys, and sealed data are not
distributed. Merely running `--help` or the figure workflow does not access them.

Regenerate the six manuscript figures from the released aggregate tables with:

```bash
python -m phase_evonet.figure_sources --root .
python scripts/render_as_figure_revision.py
python scripts/qa_as_figure_revision.py
```

Source integrity is checked against `source_data/source_data_manifest.csv`,
including SHA-256, byte size, and row count. Frozen CSV bytes, configurations,
and the original study license notice are preserved rather than renormalised.
Figure 5b and Figure 6b include the manuscript's approved legend/text repairs;
the numerical source tables are unchanged.

The QA command returns a nonzero exit code on automated failure. An automated
pass is reported as `AUTOMATED_PASS_VISUAL_REVIEW_PENDING`, **not** as a completed
manual review. Inspect the exported figures and contact sheets separately.
Contact-sheet fonts use Matplotlib's bundled DejaVu Sans Bold, so no system
Arial installation is required. Full figures use Arial when available, with
Liberation Sans and DejaVu Sans as fallbacks.

## Data availability and licensing boundary

The repository contains code, configuration files, and identifier-free aggregate source tables for the six manuscript figures. It does not redistribute Materials Project snapshots, restricted identifiers, encrypted test labels, or identifier-bearing GNoME-derived records. Reproducing the full record-level study requires separately obtained source data under the applicable provider terms. Generated data, reports, model files, and secrets are ignored by Git.

Set `MP_API_KEY` in the environment only when an authorised Materials Project operation requires it. Never commit credentials. The synthetic smoke workflow requires no API key and no network access.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

The software in this repository is released under the MIT License. Third-party data remain governed by their respective terms and licenses.
