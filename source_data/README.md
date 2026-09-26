# Source Data for Figures 1–6

This directory contains the formal aggregate CSV tables used to render the six main figures. The files are copied from frozen authoring and R4.0 artifacts; their SHA-256 values are recorded in `source_data_manifest.csv`.

Run `python -m phase_evonet.figure_sources --root .` from the repository root to
check the registered hashes, byte sizes, and row counts. The CSV files are
byte-preserved by `.gitattributes`: do not change their line endings or rewrite
the manifest to accommodate an unapproved source change. Rendering and figure
QA both enforce these source checks.

The package contains no identifier-bearing GNoME-derived row-level data. GNoME-related public outputs are limited to approved aggregates. Raw Materials Project objects are not redistributed and remain governed by Materials Project terms and embedded licenses.

Figure 6 is a post-amendment descriptive comparison of the 2025 and v2026.04.13 label states under a fixed panel and byte-identical score artifacts. It is not a prospective or external model confirmation.

No values were read from PNG files. Vector and raster figure files are renderings of the registered source tables.
