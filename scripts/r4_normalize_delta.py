from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_evonet.r4_current_release import (
    build_current_release_terminal_supplement,
    finalize_existing_current_release_normalization,
    normalize_current_release,
    repair_missing_current_release_thermo_ids,
    repair_structured_current_release_entry_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/r4_current_release.yaml"),
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Strictly validate and finalize already atomically closed Parquet outputs.",
    )
    parser.add_argument(
        "--repair-missing-thermo-ids",
        action="store_true",
        help="Atomically reconstruct thermo_id when the official Delta schema omits it.",
    )
    parser.add_argument("--repair-structured-entry-ids", action="store_true")
    parser.add_argument("--build-terminal-supplement", action="store_true")
    args = parser.parse_args()
    selected_operations = sum(
        bool(item)
        for item in (
            args.finalize_existing,
            args.repair_missing_thermo_ids,
            args.repair_structured_entry_ids,
            args.build_terminal_supplement,
        )
    )
    if selected_operations > 1:
        parser.error("Choose only one recovery operation")
    if args.build_terminal_supplement:
        result = build_current_release_terminal_supplement(args.config)
    elif args.repair_structured_entry_ids:
        result = repair_structured_current_release_entry_ids(args.config)
    elif args.repair_missing_thermo_ids:
        result = repair_missing_current_release_thermo_ids(args.config)
    elif args.finalize_existing:
        result = finalize_existing_current_release_normalization(args.config)
    else:
        result = normalize_current_release(args.config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
