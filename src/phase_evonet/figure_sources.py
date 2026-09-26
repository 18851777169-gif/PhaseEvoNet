"""Byte-exact verification of the released aggregate figure source tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


class SourceDataIntegrityError(ValueError):
    """A source table or its manifest violates the release contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[int, dict[str, object]]:
    path = root / "source_data/source_data_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    if len(rows) != 6:
        raise SourceDataIntegrityError("The manifest must contain exactly six figures")
    for number, row in enumerate(rows, start=1):
        expected_path = f"source_data/figures/figure_{number}_source_data.csv"
        if row.get("figure") != f"Figure {number}" or row.get("path") != expected_path:
            raise SourceDataIntegrityError(f"Unexpected or duplicate manifest entry for Figure {number}")
        digest = row.get("sha256", "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise SourceDataIntegrityError(f"Invalid SHA-256 for Figure {number}")
        if row.get("public_release_scope") != "aggregate_only":
            raise SourceDataIntegrityError(f"Unapproved release scope for Figure {number}")
        try:
            count, size = int(row["rows"]), int(row["bytes"])
        except (KeyError, ValueError) as exc:
            raise SourceDataIntegrityError(f"Invalid size or row count for Figure {number}") from exc
        if count <= 0 or size <= 0:
            raise SourceDataIntegrityError(f"Nonpositive size or row count for Figure {number}")
        records[number] = {**row, "rows": count, "bytes": size}
    return records


def verify_table(path: Path, expected: dict[str, object]) -> dict[str, object]:
    observed_hash = sha256(path)
    size = path.stat().st_size
    if observed_hash != expected["sha256"] or size != expected["bytes"]:
        raise SourceDataIntegrityError(
            f"Frozen source mismatch: {path}; SHA-256 {observed_hash} "
            f"(expected {expected['sha256']}), bytes {size} (expected {expected['bytes']})"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if not header:
            raise SourceDataIntegrityError(f"Missing CSV header: {path}")
        count = 0
        for row in rows:
            if len(row) != len(header):
                raise SourceDataIntegrityError(f"CSV field-count mismatch: {path}, row {count + 2}")
            count += 1
    if count != expected["rows"]:
        raise SourceDataIntegrityError(f"Frozen row-count mismatch: {path}: {count} != {expected['rows']}")
    return {"path": str(path), "sha256": observed_hash, "bytes": size, "rows": count}


def verify_sources(root: Path) -> list[dict[str, object]]:
    return [verify_table(root / row["path"], row) for row in load_manifest(root).values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        records = verify_sources(args.root.resolve())
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps({"status": "PASS", "figures": records}, indent=2))


if __name__ == "__main__":
    main()
