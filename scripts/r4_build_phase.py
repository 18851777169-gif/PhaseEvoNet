from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_evonet.context_phase_diagrams import (
    build_unified_phase_diagrams_sharded,
    verify_unified_phase_diagrams,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/r4_phase_diagrams.yaml"),
    )
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build = build_unified_phase_diagrams_sharded(
        args.config, shards_per_snapshot=args.shards, workers=args.workers
    )
    verification = verify_unified_phase_diagrams(args.config)
    print(json.dumps({"build": build, "verification": verification}, sort_keys=True))


if __name__ == "__main__":
    main()
