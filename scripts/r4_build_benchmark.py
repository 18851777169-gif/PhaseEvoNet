from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_evonet.r4_current_release import build_current_release_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/r4_current_release.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(build_current_release_benchmark(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
