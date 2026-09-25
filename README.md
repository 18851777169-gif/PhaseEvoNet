# PhaseEvoNet

Research code for analysing how calculated phase-stability labels and fixed-score benchmarks change across frozen Materials Project database releases.

The central design keeps candidate identities, feature inputs, and prediction scores fixed while changing the database version that supplies the reference labels. The repository also contains lineage construction, phase-diagram reconstruction, transition accounting, survival analysis, provenance-defined omission analysis, and current-release extension code used by the study.

## Installation

PhaseEvoNet requires Python 3.10 or later.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

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

Regenerate the six manuscript figures from the released aggregate tables with:

```bash
python scripts/render_as_figure_revision.py
python scripts/qa_as_figure_revision.py
```

## Data availability and licensing boundary

The repository contains code, configuration files, and identifier-free aggregate source tables for the six manuscript figures. It does not redistribute Materials Project snapshots, restricted identifiers, encrypted test labels, or identifier-bearing GNoME-derived records. Reproducing the full record-level study requires separately obtained source data under the applicable provider terms. Generated data, reports, model files, and secrets are ignored by Git.

Set `MP_API_KEY` in the environment only when an authorised Materials Project operation requires it. Never commit credentials. The synthetic smoke workflow requires no API key and no network access.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

The software in this repository is released under the MIT License. Third-party data remain governed by their respective terms and licenses.
