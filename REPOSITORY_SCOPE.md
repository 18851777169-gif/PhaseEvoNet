# Repository scope

This repository is intentionally limited to the code required to inspect, test, and reproduce the study workflows when authorised source data are supplied separately.

Included:

- the installable `phase_evonet` Python package;
- frozen analysis and reconstruction configurations;
- deterministic synthetic fixtures and tests;
- identifier-free aggregate source tables for Figures 1–6;
- current-release and figure-generation entry points;
- dependency metadata, citation metadata, and the software license.

Not included:

- raw, intermediate, processed, sealed, or restricted datasets;
- credentials, encryption keys, access tokens, or machine-specific paths;
- identifier-bearing restricted records;
- generated reports, trained-model files, manuscript files, and rendered figures;
- internal task-state, review, command-log, and temporary working files;
- bundled third-party executables and caches.

These exclusions reduce duplication and prevent accidental redistribution of data or credentials. They do not alter the scientific calculations implemented by the released source code.
