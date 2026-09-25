from __future__ import annotations

import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
import requests

from .manifest import sha256_file

S3_XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
JSONL_SCHEMA_METHOD = "jsonl-top-level-fields-and-json-types-v1"
PARQUET_SCHEMA_METHOD = "arrow-schema-without-metadata-v1"
_HTTP_STATE = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_http_session() -> requests.Session:
    session = getattr(_HTTP_STATE, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "PhaseEvoNet-P1.1/1.0"})
        _HTTP_STATE.session = session
    return session


@contextlib.contextmanager
def exclusive_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if stream.tell() == stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another snapshot download is active: {path}") from exc
        else:  # pragma: no cover - Windows is the task execution platform
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another snapshot download is active: {path}") from exc
        yield
    finally:
        if os.name == "nt":
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:  # pragma: no cover
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Unsupported JSON value type: {type(value)!r}")


def parse_s3_listing(xml_body: bytes) -> tuple[list[dict[str, Any]], str | None]:
    root = ET.fromstring(xml_body)
    objects: list[dict[str, Any]] = []
    for node in root.findall("s3:Contents", S3_XML_NAMESPACE):
        key = urllib.parse.unquote(
            node.findtext("s3:Key", default="", namespaces=S3_XML_NAMESPACE)
        )
        objects.append(
            {
                "key": key,
                "content_length": int(
                    node.findtext("s3:Size", default="0", namespaces=S3_XML_NAMESPACE)
                ),
                "etag": node.findtext(
                    "s3:ETag", default="", namespaces=S3_XML_NAMESPACE
                ).strip('"'),
                "last_modified": node.findtext(
                    "s3:LastModified", default="", namespaces=S3_XML_NAMESPACE
                ),
                "storage_class": node.findtext(
                    "s3:StorageClass", default="", namespaces=S3_XML_NAMESPACE
                ),
            }
        )
    is_truncated = (
        root.findtext("s3:IsTruncated", default="false", namespaces=S3_XML_NAMESPACE)
        .strip()
        .lower()
        == "true"
    )
    token = root.findtext("s3:NextContinuationToken", namespaces=S3_XML_NAMESPACE)
    if is_truncated and not token:
        raise ValueError("Truncated S3 listing did not include a continuation token")
    return objects, token if is_truncated else None


def source_uri(endpoint: str, key: str) -> str:
    return endpoint.rstrip("/") + "/" + urllib.parse.quote(key, safe="/")


def list_s3_prefix(
    endpoint: str,
    prefix: str,
    *,
    timeout_seconds: int,
    retries: int,
) -> list[dict[str, Any]]:
    token: str | None = None
    results: list[dict[str, Any]] = []
    while True:
        query = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
            "encoding-type": "url",
        }
        if token:
            query["continuation-token"] = token
        url = endpoint.rstrip("/") + "/?" + urllib.parse.urlencode(query)
        body: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "PhaseEvoNet-P1.1/1.0"}
                )
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    body = response.read()
                break
            except Exception as exc:  # pragma: no cover - exercised against live S3
                last_error = exc
                if attempt < retries:
                    time.sleep(2**attempt)
        if body is None:
            raise RuntimeError(f"Could not list {prefix}: {last_error}")
        page, token = parse_s3_listing(body)
        results.extend(page)
        if token is None:
            break
    return results


def object_format(key: str) -> str:
    if key.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if key.endswith(".parquet"):
        return "parquet"
    raise ValueError(f"Unsupported raw object format: {key}")


def local_object_path(raw_dir: Path, version: str, collection: str, key: str) -> Path:
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    suffix = ".jsonl.gz" if key.endswith(".jsonl.gz") else ".parquet"
    return raw_dir / version / collection / key_hash[:2] / f"{key_hash}{suffix}"


def canonical_hash(payload: Any) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def validate_jsonl_gzip(path: Path) -> dict[str, Any]:
    field_types: dict[str, set[str]] = defaultdict(set)
    root_types: set[str] = set()
    licenses: set[str] = set()
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows += 1
            root_types.add(json_type(record))
            if isinstance(record, dict):
                for key, value in record.items():
                    field_types[str(key)].add(json_type(value))
                builder_meta = record.get("builder_meta")
                if isinstance(builder_meta, dict) and builder_meta.get("license"):
                    licenses.add(str(builder_meta["license"]))
    signature = {
        "root_types": sorted(root_types),
        "fields": {key: sorted(values) for key, values in sorted(field_types.items())},
    }
    return {
        "row_count": rows,
        "schema_hash": canonical_hash(signature),
        "schema_method": JSONL_SCHEMA_METHOD,
        "embedded_licenses": sorted(licenses),
    }


def validate_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("pyarrow is required to validate Parquet objects") from exc
    parquet_file = pq.ParquetFile(path)
    schema_text = str(parquet_file.schema_arrow.remove_metadata())
    return {
        "row_count": int(parquet_file.metadata.num_rows),
        "schema_hash": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "schema_method": PARQUET_SCHEMA_METHOD,
        "embedded_licenses": [],
    }


def validate_local_object(path: Path, file_format: str) -> dict[str, Any]:
    if file_format == "jsonl.gz":
        return validate_jsonl_gzip(path)
    if file_format == "parquet":
        return validate_parquet(path)
    raise ValueError(f"Unsupported file format: {file_format}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_curl_config(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    raw_dir: Path,
    parallel_max: int,
    retries: int,
    max_time_seconds: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("parallel\n")
        stream.write(f"parallel-max = {parallel_max}\n")
        stream.write("fail\n")
        stream.write(f"retry = {retries}\n")
        stream.write("retry-all-errors\n")
        stream.write("retry-delay = 1\n")
        stream.write("continue-at = -\n")
        stream.write("connect-timeout = 10\n")
        stream.write("speed-limit = 1024\n")
        stream.write("speed-time = 30\n")
        stream.write(f"max-time = {max_time_seconds}\n")
        stream.write("silent\n")
        stream.write("show-error\n")
        stream.write("create-dirs\n")
        for row in rows:
            final_path = local_object_path(
                raw_dir,
                row["database_version"],
                row["collection"],
                row["key"],
            )
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            part_path.parent.mkdir(parents=True, exist_ok=True)
            stream.write(
                "url = " + json.dumps(row.get("retrieval_uri", row["source_uri"])) + "\n"
            )
            stream.write("output = " + json.dumps(part_path.as_posix()) + "\n")


def write_aria2_input(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    raw_dir: Path,
    transfer_endpoints: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            offset = int(
                hashlib.sha256(row["key"].encode("utf-8")).hexdigest()[:8], 16
            ) % len(transfer_endpoints)
            ordered = transfer_endpoints[offset:] + transfer_endpoints[:offset]
            urls = [source_uri(endpoint, row["key"]) for endpoint in ordered]
            final_path = local_object_path(
                raw_dir,
                row["database_version"],
                row["collection"],
                row["key"],
            )
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            part_path.parent.mkdir(parents=True, exist_ok=True)
            stream.write("\t".join(urls) + "\n")
            stream.write("  dir=" + part_path.parent.resolve().as_posix() + "\n")
            stream.write("  out=" + part_path.name + "\n")


def validate_and_promote_curl_object(
    record: dict[str, Any], *, raw_dir: Path
) -> dict[str, Any]:
    final_path = local_object_path(
        raw_dir, record["database_version"], record["collection"], record["key"]
    )
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        if final_path.stat().st_size != record["content_length"]:
            raise RuntimeError(f"Immutable raw-object collision with wrong size: {final_path}")
        validation_path = final_path
        action = "validated_existing"
    else:
        if not part_path.exists() or part_path.stat().st_size != record["content_length"]:
            raise FileNotFoundError(f"Complete curl transfer not available: {record['key']}")
        validation_path = part_path
        action = "downloaded_curl"
    sha256 = sha256_file(validation_path)
    validation = validate_local_object(validation_path, record["file_format"])
    if validation_path == part_path:
        os.replace(part_path, final_path)
    return {
        **record,
        "local_path": final_path.as_posix(),
        "retrieval_timestamp_utc": utc_now(),
        "sha256": sha256,
        **validation,
        "download_action": action,
        "validation_status": "PASS",
        "error": None,
    }


def run_curl_transfer(
    rows: list[dict[str, Any]],
    *,
    raw_dir: Path,
    manifest_dir: Path,
    workers: int,
    config: dict[str, Any],
    on_result,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    transfer_log: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    transfer_endpoints = list(
        map(
            str,
            config["source"].get(
                "transfer_endpoints", [config["source"]["endpoint"]]
            ),
        )
    )
    remaining = []
    for row in rows:
        endpoint_index = int(
            hashlib.sha256(row["key"].encode("utf-8")).hexdigest()[:8], 16
        ) % len(transfer_endpoints)
        remaining.append(
            {
                **row,
                "retrieval_uri": source_uri(
                    transfer_endpoints[endpoint_index], row["key"]
                ),
            }
        )
    max_passes = int(config["download"].get("curl_passes", 3))
    parallel_max = int(config["download"].get("curl_parallel_max", workers))
    max_time = int(config["download"].get("curl_max_time_seconds", 180))

    for pass_number in range(0, max_passes + 1):
        candidates: list[dict[str, Any]] = []
        still_missing: list[dict[str, Any]] = []
        for row in remaining:
            final_path = local_object_path(
                raw_dir, row["database_version"], row["collection"], row["key"]
            )
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            if final_path.exists() and final_path.stat().st_size == row["content_length"]:
                candidates.append(row)
            elif part_path.exists() and part_path.stat().st_size == row["content_length"]:
                candidates.append(row)
            else:
                if part_path.exists() and part_path.stat().st_size > row["content_length"]:
                    part_path.unlink()
                still_missing.append(row)

        if candidates:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(validate_and_promote_curl_object, row, raw_dir=raw_dir): row
                    for row in candidates
                }
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    row = futures[future]
                    try:
                        on_result(future.result())
                    except Exception as exc:
                        validation_errors.append(
                            {"key": row["key"], "error": repr(exc)}
                        )
                    if index % 100 == 0 or index == len(futures):
                        print(
                            f"curl_validation={index}/{len(futures)} "
                            f"errors={len(validation_errors)}",
                            flush=True,
                        )
        remaining = still_missing
        if validation_errors or not remaining:
            break
        if pass_number == max_passes:
            break

        plan_path = manifest_dir / f"curl_transfer_pass_{pass_number + 1}.cfg"
        write_curl_config(
            plan_path,
            remaining,
            raw_dir=raw_dir,
            parallel_max=parallel_max,
            retries=int(config["download"].get("curl_retries", config["download"]["retries"])),
            max_time_seconds=max_time,
        )
        print(
            f"curl_pass={pass_number + 1} objects={len(remaining)} "
            f"parallel_max={parallel_max}",
            flush=True,
        )
        started = time.monotonic()
        completed = subprocess.run(
            ["curl.exe", "--config", str(plan_path)],
            check=False,
        )
        transfer_log.append(
            {
                "pass": pass_number + 1,
                "object_count": len(remaining),
                "parallel_max": parallel_max,
                "exit_code": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
                "config_path": plan_path.as_posix(),
            }
        )

    transfer_errors = list(validation_errors)
    transfer_errors.extend(
        {"key": row["key"], "error": "curl transfer incomplete after all passes"}
        for row in remaining
    )
    return transfer_log, transfer_errors


def aria2_object_is_complete(
    final_path: Path, part_path: Path, expected_size: int
) -> bool:
    if final_path.exists() and final_path.stat().st_size == expected_size:
        return True
    control_path = Path(str(part_path) + ".aria2")
    return (
        part_path.exists()
        and part_path.stat().st_size == expected_size
        and not control_path.exists()
    )


def run_aria2_transfer(
    rows: list[dict[str, Any]],
    *,
    raw_dir: Path,
    manifest_dir: Path,
    workers: int,
    config: dict[str, Any],
    on_result,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    endpoints = list(map(str, config["source"]["transfer_endpoints"]))
    threshold = int(config["download"].get("aria2_large_threshold_bytes", 1_000_000))
    classes = [
        (
            "large",
            [row for row in rows if row["content_length"] >= threshold],
            int(config["download"].get("aria2_large_concurrent", 2)),
            int(config["download"].get("aria2_large_splits", len(endpoints))),
        ),
        (
            "small",
            [row for row in rows if row["content_length"] < threshold],
            int(config["download"].get("aria2_small_concurrent", 14)),
            1,
        ),
    ]
    aria2_executable = Path(str(config["download"]["aria2_executable"]))
    if not aria2_executable.is_file():
        raise FileNotFoundError(f"aria2 executable not found: {aria2_executable}")
    transfer_log: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for class_name, class_rows, concurrent_downloads, splits in classes:
        if not class_rows:
            continue
        input_path = manifest_dir / f"aria2_{class_name}_input.txt"
        write_aria2_input(
            input_path,
            class_rows,
            raw_dir=raw_dir,
            transfer_endpoints=endpoints,
        )
        command = [
            str(aria2_executable),
            f"--input-file={input_path}",
            "--continue=true",
            f"--max-concurrent-downloads={concurrent_downloads}",
            f"--split={splits}",
            "--max-connection-per-server=1",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--connect-timeout=10",
            "--timeout="
            + str(config["download"].get("aria2_timeout_seconds", 60)),
            "--lowest-speed-limit="
            + str(config["download"].get("aria2_lowest_speed_limit", "1K")),
            f"--max-tries={int(config['download'].get('aria2_max_tries', 20))}",
            "--retry-wait=1",
            "--summary-interval="
            + str(config["download"].get("aria2_summary_interval_seconds", 60)),
            "--console-log-level="
            + str(config["download"].get("aria2_console_log_level", "notice")),
            "--show-console-readout=false",
            "--quiet="
            + str(bool(config["download"].get("aria2_quiet", False))).lower(),
        ]
        print(
            f"aria2_class={class_name} objects={len(class_rows)} "
            f"concurrent={concurrent_downloads} splits={splits}",
            flush=True,
        )
        started = time.monotonic()
        result = subprocess.run(command, check=False)
        transfer_log.append(
            {
                "class": class_name,
                "object_count": len(class_rows),
                "concurrent_downloads": concurrent_downloads,
                "splits": splits,
                "exit_code": result.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
                "input_path": input_path.as_posix(),
            }
        )

        candidates = []
        missing = []
        for row in class_rows:
            final_path = local_object_path(
                raw_dir, row["database_version"], row["collection"], row["key"]
            )
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            if aria2_object_is_complete(
                final_path, part_path, row["content_length"]
            ):
                candidates.append(row)
            else:
                missing.append(row)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(validate_and_promote_curl_object, row, raw_dir=raw_dir): row
                for row in candidates
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = futures[future]
                try:
                    validated = future.result()
                    validated["download_action"] = "downloaded_aria2"
                    validated["retrieval_uris"] = [
                        source_uri(endpoint, row["key"]) for endpoint in endpoints
                    ]
                    on_result(validated)
                except Exception as exc:
                    errors.append({"key": row["key"], "error": repr(exc)})
                if index % 100 == 0 or index == len(futures):
                    print(
                        f"aria2_validation={index}/{len(futures)} errors={len(errors)}",
                        flush=True,
                    )
        errors.extend(
            {"key": row["key"], "error": f"aria2 {class_name} transfer incomplete"}
            for row in missing
        )
        if errors:
            break
    return transfer_log, errors


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                latest[row["key"]] = row
    return latest


def download_one(
    record: dict[str, Any],
    *,
    raw_dir: Path,
    timeout_seconds: int,
    retries: int,
) -> dict[str, Any]:
    path = local_object_path(
        raw_dir, record["database_version"], record["collection"], record["key"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    action = "downloaded"
    if path.exists():
        if path.stat().st_size != record["content_length"]:
            raise RuntimeError(
                f"Immutable raw-object collision with wrong size: {path} "
                f"({path.stat().st_size} != {record['content_length']})"
            )
        action = "validated_existing"
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                session = thread_http_session()
                with session.get(
                    record["source_uri"],
                    stream=True,
                    timeout=(15, timeout_seconds),
                ) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if temporary.stat().st_size != record["content_length"]:
                    raise IOError(
                        f"Size mismatch for {record['key']}: "
                        f"{temporary.stat().st_size} != {record['content_length']}"
                    )
                os.replace(temporary, path)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - exercised against live S3
                last_error = exc
                if temporary.exists():
                    temporary.unlink()
                if attempt < retries:
                    time.sleep(2**attempt)
        if last_error is not None:
            raise RuntimeError(f"Could not download {record['key']}: {last_error}")
    sha256 = sha256_file(path)
    validation = validate_local_object(path, record["file_format"])
    return {
        **record,
        "local_path": path.as_posix(),
        "retrieval_timestamp_utc": utc_now(),
        "sha256": sha256,
        **validation,
        "download_action": action,
        "validation_status": "PASS",
        "error": None,
    }


def inventory_snapshots(config: dict[str, Any], workers: int) -> list[dict[str, Any]]:
    endpoint = str(config["source"]["endpoint"])
    timeout_seconds = int(config["download"]["timeout_seconds"])
    retries = int(config["download"]["retries"])
    specs = [
        (str(version), str(collection))
        for version in config["snapshots"]
        for collection in config["collections"]
    ]

    def inventory_spec(spec: tuple[str, str]) -> list[dict[str, Any]]:
        version, collection = spec
        prefix = f"collections/{version}/{collection}/"
        objects = list_s3_prefix(
            endpoint,
            prefix,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        rows = []
        for item in objects:
            rows.append(
                {
                    "database_version": version,
                    "collection": collection,
                    **item,
                    "source_uri": source_uri(endpoint, item["key"]),
                    "license": str(config["source"]["license"]),
                    "file_format": object_format(item["key"]),
                }
            )
        return rows

    inventory: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        for rows in pool.map(inventory_spec, specs):
            inventory.extend(rows)
    inventory.sort(key=lambda row: row["key"])
    keys = [row["key"] for row in inventory]
    if len(keys) != len(set(keys)):
        raise RuntimeError("S3 inventory contains duplicate object keys")
    return inventory


def inventory_fingerprint(rows: list[dict[str, Any]]) -> str:
    stable = [
        {
            key: row[key]
            for key in ("key", "content_length", "etag", "last_modified")
        }
        for row in rows
    ]
    return canonical_hash(stable)


def run_snapshot_download(
    *,
    config_path: str | Path,
    raw_dir: str | Path,
    manifest_dir: str | Path,
    workers: int,
    transfer_engine: str | None = None,
    inventory_only: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_dir)
    with exclusive_run_lock(manifest_path / ".snapshot_download.lock"):
        return _run_snapshot_download_locked(
            config_path=Path(config_path),
            raw_path=Path(raw_dir),
            manifest_path=manifest_path,
            workers=workers,
            transfer_engine=transfer_engine,
            inventory_only=inventory_only,
        )


def _run_snapshot_download_locked(
    *,
    config_path: Path,
    raw_path: Path,
    manifest_path: Path,
    workers: int,
    transfer_engine: str | None,
    inventory_only: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    config_file = config_path
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    manifest_path.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    inventory = inventory_snapshots(config, workers)
    inventory_hash = inventory_fingerprint(inventory)
    write_jsonl_atomic(manifest_path / "source_inventory.jsonl", inventory)
    inventory_summary = {
        "created_at_utc": utc_now(),
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "inventory_fingerprint": inventory_hash,
        "object_count": len(inventory),
        "total_bytes": sum(row["content_length"] for row in inventory),
        "snapshots": list(map(str, config["snapshots"])),
        "collections": list(map(str, config["collections"])),
    }
    write_json_atomic(manifest_path / "source_inventory_summary.json", inventory_summary)
    if inventory_only:
        return {"status": "PASS", "inventory_only": True, **inventory_summary}

    checkpoint_path = manifest_path / ".download_checkpoint.jsonl"
    completed = load_checkpoint(checkpoint_path)
    pending: list[dict[str, Any]] = []
    for row in inventory:
        old = completed.get(row["key"])
        if old and all(
            old.get(field) == row.get(field)
            for field in ("content_length", "etag", "last_modified")
        ):
            local = Path(old["local_path"])
            if local.exists() and local.stat().st_size == row["content_length"]:
                if sha256_file(local) != old.get("sha256"):
                    raise RuntimeError(
                        f"Immutable raw object changed after checkpoint: {local}"
                    )
                continue
        pending.append(row)

    errors: list[dict[str, str]] = []
    transfer_log: list[dict[str, Any]] = []
    with checkpoint_path.open("a", encoding="utf-8", newline="\n") as checkpoint:
        def record_result(result: dict[str, Any]) -> None:
            completed[result["key"]] = result
            checkpoint.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            checkpoint.flush()

        engine = transfer_engine or str(config["download"].get("transfer_engine", "python"))
        if engine == "aria2":
            transfer_log, errors = run_aria2_transfer(
                pending,
                raw_dir=raw_path,
                manifest_dir=manifest_path,
                workers=workers,
                config=config,
                on_result=record_result,
            )
        elif engine == "curl":
            transfer_log, errors = run_curl_transfer(
                pending,
                raw_dir=raw_path,
                manifest_dir=manifest_path,
                workers=workers,
                config=config,
                on_result=record_result,
            )
        elif engine == "python":
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_row = {
                    pool.submit(
                        download_one,
                        row,
                        raw_dir=raw_path,
                        timeout_seconds=int(config["download"]["timeout_seconds"]),
                        retries=int(config["download"]["retries"]),
                    ): row
                    for row in pending
                }
                total = len(future_to_row)
                for index, future in enumerate(
                    concurrent.futures.as_completed(future_to_row), 1
                ):
                    row = future_to_row[future]
                    try:
                        record_result(future.result())
                    except Exception as exc:
                        errors.append({"key": row["key"], "error": repr(exc)})
                    if index % 100 == 0 or index == total:
                        print(
                            f"processed={index}/{total} "
                            f"completed={len(completed)}/{len(inventory)} "
                            f"errors={len(errors)}",
                            flush=True,
                        )
        else:
            raise ValueError(f"Unsupported transfer engine: {engine}")

    error_path = manifest_path / "download_errors.json"
    if errors:
        write_json_atomic(error_path, errors)
        raise RuntimeError(f"{len(errors)} raw objects failed; see download_errors.json")
    if error_path.exists():
        error_path.unlink()
    if set(completed) != {row["key"] for row in inventory}:
        raise RuntimeError("Checkpoint does not cover the complete source inventory")

    final_rows = [completed[row["key"]] for row in inventory]
    for row in final_rows:
        row.setdefault("retrieval_uri", row["source_uri"])
    write_jsonl_atomic(manifest_path / "snapshot_manifest.jsonl", final_rows)

    reverified = inventory_snapshots(config, workers)
    reverified_hash = inventory_fingerprint(reverified)
    source_unchanged = reverified_hash == inventory_hash
    if not source_unchanged:
        raise RuntimeError("Official S3 inventory changed during P1.1 download")

    by_snapshot: dict[str, dict[str, Any]] = {}
    for version in map(str, config["snapshots"]):
        rows = [row for row in final_rows if row["database_version"] == version]
        by_snapshot[version] = {
            "object_count": len(rows),
            "compressed_bytes": sum(row["content_length"] for row in rows),
            "row_count": sum(row["row_count"] for row in rows),
            "collections": sorted({row["collection"] for row in rows}),
            "embedded_licenses": sorted(
                {license_name for row in rows for license_name in row["embedded_licenses"]}
            ),
            "all_objects_valid": all(
                row["validation_status"] == "PASS" for row in rows
            ),
        }
    summary = {
        "status": "PASS",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "inventory_fingerprint_before": inventory_hash,
        "inventory_fingerprint_after": reverified_hash,
        "source_inventory_unchanged": source_unchanged,
        "object_count": len(final_rows),
        "compressed_bytes": sum(row["content_length"] for row in final_rows),
        "row_count": sum(row["row_count"] for row in final_rows),
        "validation_failures": 0,
        "transfer_engine": engine,
        "transfer_log": transfer_log,
        "snapshots": by_snapshot,
        "manifest_sha256": sha256_file(manifest_path / "snapshot_manifest.jsonl"),
        "raw_layer_immutable": True,
        "schema_methods": [JSONL_SCHEMA_METHOD, PARQUET_SCHEMA_METHOD],
    }
    write_json_atomic(manifest_path / "snapshot_summary.json", summary)
    return summary


def verify_snapshot_manifest(
    manifest_path: str | Path, *, workers: int = 16
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    path = Path(manifest_path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "key",
        "source_uri",
        "retrieval_timestamp_utc",
        "database_version",
        "license",
        "content_length",
        "local_path",
        "sha256",
        "schema_hash",
        "row_count",
        "file_format",
        "validation_status",
    }
    keys = [row.get("key") for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Snapshot manifest contains duplicate object keys")

    def verify_one(row: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"{row.get('key', '<unknown>')}: missing fields {missing}")
        local = Path(row["local_path"])
        if not local.is_file():
            raise FileNotFoundError(f"{row['key']}: local object missing: {local}")
        if local.stat().st_size != row["content_length"]:
            raise RuntimeError(f"{row['key']}: content length mismatch")
        if sha256_file(local) != row["sha256"]:
            raise RuntimeError(f"{row['key']}: SHA256 mismatch")
        validation = validate_local_object(local, row["file_format"])
        for field in ("row_count", "schema_hash", "embedded_licenses"):
            if validation[field] != row[field]:
                raise RuntimeError(f"{row['key']}: {field} mismatch")
        if row["validation_status"] != "PASS":
            raise RuntimeError(f"{row['key']}: manifest validation status is not PASS")
        if not all(
            isinstance(row[field], str) and row[field].strip()
            for field in (
                "source_uri",
                "retrieval_timestamp_utc",
                "database_version",
                "license",
                "sha256",
                "schema_hash",
            )
        ):
            raise RuntimeError(f"{row['key']}: empty data-contract metadata")
        return {
            "database_version": row["database_version"],
            "content_length": row["content_length"],
            "row_count": row["row_count"],
        }

    verified: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_row = {pool.submit(verify_one, row): row for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_row), 1):
            row = future_to_row[future]
            try:
                verified.append(future.result())
            except Exception as exc:
                errors.append({"key": str(row.get("key")), "error": repr(exc)})
            if index % 1000 == 0 or index == len(rows):
                print(
                    f"manifest_verify={index}/{len(rows)} errors={len(errors)}",
                    flush=True,
                )
    if errors:
        raise RuntimeError(
            f"Snapshot manifest verification failed for {len(errors)} objects; "
            f"first error: {errors[0]}"
        )
    by_snapshot: dict[str, dict[str, int]] = defaultdict(
        lambda: {"object_count": 0, "compressed_bytes": 0, "row_count": 0}
    )
    for result in verified:
        summary = by_snapshot[result["database_version"]]
        summary["object_count"] += 1
        summary["compressed_bytes"] += result["content_length"]
        summary["row_count"] += result["row_count"]
    return {
        "status": "PASS",
        "verified_at_utc": utc_now(),
        "manifest_path": path.as_posix(),
        "manifest_sha256": sha256_file(path),
        "object_count": len(verified),
        "compressed_bytes": sum(item["content_length"] for item in verified),
        "row_count": sum(item["row_count"] for item in verified),
        "validation_failures": 0,
        "snapshots": dict(sorted(by_snapshot.items())),
        "network_access": False,
    }
