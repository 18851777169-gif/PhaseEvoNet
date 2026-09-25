from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests


DB_VERSION = "2026.04.13"
DB_VERSION_PATH = "2026-04-13"
COLLECTIONS = ("materials", "thermo", "provenance")
S3_BASE = "https://materialsproject-build.s3.us-east-1.amazonaws.com/collections"
HEARTBEAT = "https://api.materialsproject.org/heartbeat"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def request_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def prepare(repo: Path) -> dict[str, Any]:
    schema_path = repo / "reports/V2_0/source_2026_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    heartbeat_bytes = request_bytes(HEARTBEAT)
    heartbeat = json.loads(heartbeat_bytes)
    observed = str(heartbeat.get("db_version", "")).lstrip("v")
    if observed != DB_VERSION:
        raise RuntimeError(f"Heartbeat mismatch: expected {DB_VERSION}, observed {observed}")

    raw_root = repo / f"data/raw/materials_project/{DB_VERSION_PATH}"
    manifest_dir = repo / "data/manifests/R4_0"
    report_dir = repo / "reports/R4_0"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    heartbeat_path = raw_root / "heartbeat.json"
    heartbeat_path.write_bytes(heartbeat_bytes)
    objects: list[dict[str, Any]] = []
    aria_entries: list[str] = []
    total_bytes = 0
    total_rows = 0

    for collection in COLLECTIONS:
        collection_meta = schema["collections"][collection]
        for item in collection_meta["active_2026_files"]:
            path = str(item["path"])
            if item.get("partitionValues", {}).get("version") != DB_VERSION_PATH:
                raise RuntimeError(f"Unexpected active partition: {collection}/{path}")
            source_uri = f"{S3_BASE}/{collection}/{path}"
            local_path = raw_root / "collections" / collection / path
            stats = json.loads(item.get("stats") or "{}")
            row_count = int(stats.get("numRecords", 0))
            size = int(item["size"])
            total_bytes += size
            total_rows += row_count
            objects.append(
                {
                    "collection": collection,
                    "database_version": DB_VERSION,
                    "delta_partition": DB_VERSION_PATH,
                    "source_uri": source_uri,
                    "source_delta_path": path,
                    "local_path": local_path.relative_to(repo).as_posix(),
                    "expected_bytes": size,
                    "expected_rows": row_count,
                    "delta_modification_time_ms": int(item["modificationTime"]),
                    "delta_schema_sha256": collection_meta["schema_sha256"],
                    "downloaded": local_path.exists() and local_path.stat().st_size == size,
                    "sha256": None,
                    "schema_sha256": None,
                    "row_count": None,
                    "license_counts": None,
                }
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            aria_entries.extend(
                [
                    source_uri,
                    f"  dir={local_path.parent}",
                    f"  out={local_path.name}",
                    "  continue=true",
                ]
            )

    log_rows: list[dict[str, Any]] = []
    for collection in COLLECTIONS:
        log_count = len(schema["collections"][collection].get("active_2026_files", []))
        del log_count  # active data-file count is not the transaction-log count
        probe_payload = json.loads((repo / "reports/V2_0/minimal_probe_log.json").read_text(encoding="utf-8"))
        probe_rows = probe_payload["probes"]
        for probe in probe_rows:
            if probe.get("probe_id", "").startswith(f"s3_{collection}_") and probe.get("query_scope", "").endswith(".json"):
                url = probe["endpoint"]
                content = request_bytes(url)
                rel = str(probe["query_scope"])
                target = raw_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                observed_sha = hashlib.sha256(content).hexdigest()
                if observed_sha != probe["response_sha256"]:
                    raise RuntimeError(f"Delta log changed for {rel}")
                log_rows.append(
                    {
                        "collection": collection,
                        "source_uri": url,
                        "local_path": target.relative_to(repo).as_posix(),
                        "bytes": len(content),
                        "sha256": observed_sha,
                    }
                )

    (manifest_dir / "aria2_input.txt").write_text("\n".join(aria_entries) + "\n", encoding="utf-8")
    write_jsonl(manifest_dir / "source_objects.prepared.jsonl", objects)
    write_json(
        report_dir / "source_preflight.json",
        {
            "task_id": "R4.0",
            "status": "READY_FOR_DOWNLOAD",
            "verified_at_utc": utc_now(),
            "heartbeat": heartbeat,
            "heartbeat_sha256": hashlib.sha256(heartbeat_bytes).hexdigest(),
            "v2_schema_path": schema_path.relative_to(repo).as_posix(),
            "v2_schema_sha256": sha256_file(schema_path),
            "delta_logs": log_rows,
            "collections": list(COLLECTIONS),
            "object_count": len(objects),
            "expected_bytes": total_bytes,
            "expected_rows": total_rows,
            "summary_collection_downloaded": False,
            "summary_exclusion_reason": "not required for material lineage, provenance, thermo reconstruction, or fixed-panel labels",
            "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
            "confirmation_b_status": "RETIRED_UNEXECUTED",
        },
    )
    return {"objects": len(objects), "bytes": total_bytes, "rows": total_rows}


def license_counts(path: Path) -> dict[str, int]:
    parquet = pq.ParquetFile(path)
    field_names = set(parquet.schema_arrow.names)
    if "builder_meta" not in field_names:
        return {"UNAVAILABLE": parquet.metadata.num_rows}
    counts: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=["builder_meta"], batch_size=65536):
        for meta in batch.column(0).to_pylist():
            value = str((meta or {}).get("license") or "UNSPECIFIED")
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def finalize(repo: Path) -> dict[str, Any]:
    manifest_dir = repo / "data/manifests/R4_0"
    report_dir = repo / "reports/R4_0"
    rows = [
        json.loads(line)
        for line in (manifest_dir / "source_objects.prepared.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures: list[str] = []
    inventory: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        path = repo / row["local_path"]
        if not path.exists():
            failures.append(f"missing:{row['local_path']}")
            continue
        observed_bytes = path.stat().st_size
        parquet = pq.ParquetFile(path)
        observed_rows = int(parquet.metadata.num_rows)
        schema_sha = hashlib.sha256(str(parquet.schema_arrow.remove_metadata()).encode("utf-8")).hexdigest()
        observed_sha = sha256_file(path)
        if observed_bytes != int(row["expected_bytes"]):
            failures.append(f"size:{row['local_path']}")
        if observed_rows != int(row["expected_rows"]):
            failures.append(f"rows:{row['local_path']}")
        row.update(
            {
                "downloaded": True,
                "observed_bytes": observed_bytes,
                "row_count": observed_rows,
                "sha256": observed_sha,
                "schema_sha256": schema_sha,
                "license_counts": license_counts(path),
                "verified_at_utc": utc_now(),
            }
        )
        inventory.append(row)
        print(f"verified={index}/{len(rows)} {row['collection']} rows={observed_rows}", flush=True)

    if failures:
        raise RuntimeError("R4 source verification failed: " + ", ".join(failures))
    write_jsonl(manifest_dir / "source_objects.jsonl", inventory)
    write_csv(
        report_dir / "source_inventory.csv",
        [
            {
                "collection": row["collection"],
                "database_version": row["database_version"],
                "source_uri": row["source_uri"],
                "local_path": row["local_path"],
                "bytes": row["observed_bytes"],
                "rows": row["row_count"],
                "sha256": row["sha256"],
                "schema_sha256": row["schema_sha256"],
                "license_counts_json": json.dumps(row["license_counts"], sort_keys=True),
                "status": "PASS",
            }
            for row in inventory
        ],
    )
    manifest = {
        "task_id": "R4.0",
        "status": "PASS",
        "finalized_at_utc": utc_now(),
        "database_version": DB_VERSION,
        "delta_partition": DB_VERSION_PATH,
        "object_count": len(inventory),
        "bytes": sum(row["observed_bytes"] for row in inventory),
        "rows": sum(row["row_count"] for row in inventory),
        "source_objects": inventory,
        "raw_layer_immutable": True,
        "v2_2_status_preserved": "STOPPED/FAIL/NO_GO",
        "confirmation_b_status": "RETIRED_UNEXECUTED",
    }
    manifest_path = manifest_dir / "source_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (manifest_dir / "source_manifest.sha256").write_text(
        f"{manifest_sha256}  source_manifest.json\n", encoding="ascii", newline="\n"
    )
    result = {
        **manifest,
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar": "data/manifests/R4_0/source_manifest.sha256",
    }
    write_json(report_dir / "source_freeze_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    result = finalize(args.repo.resolve()) if args.finalize else prepare(args.repo.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
