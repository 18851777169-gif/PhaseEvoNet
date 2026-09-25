import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from phase_evonet.snapshots import (
    aria2_object_is_complete,
    inventory_fingerprint,
    local_object_path,
    parse_s3_listing,
    source_uri,
    validate_jsonl_gzip,
    validate_parquet,
    verify_snapshot_manifest,
    write_aria2_input,
    write_curl_config,
    write_jsonl_atomic,
)


def test_parse_s3_listing_and_continuation_token():
    body = b"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>true</IsTruncated>
      <Contents>
        <Key>collections%2F2022-10-28%2Fthermo%2Fthermo_type%3DGGA%2BGGA%2BU%2Fa.jsonl.gz</Key>
        <LastModified>2024-01-01T00:00:00.000Z</LastModified>
        <ETag>&quot;abc&quot;</ETag><Size>123</Size><StorageClass>STANDARD</StorageClass>
      </Contents>
      <NextContinuationToken>opaque-token</NextContinuationToken>
    </ListBucketResult>"""
    rows, token = parse_s3_listing(body)
    assert token == "opaque-token"
    assert rows == [
        {
            "key": "collections/2022-10-28/thermo/thermo_type=GGA+GGA+U/a.jsonl.gz",
            "content_length": 123,
            "etag": "abc",
            "last_modified": "2024-01-01T00:00:00.000Z",
            "storage_class": "STANDARD",
        }
    ]
    assert "%2B" in source_uri("https://example.test", rows[0]["key"])


def test_jsonl_gzip_validation_is_deterministic(tmp_path: Path):
    path = tmp_path / "sample.jsonl.gz"
    records = [
        {"material_id": "mp-1", "value": 1, "builder_meta": {"license": "BY-C"}},
        {"material_id": "mp-2", "value": None, "tags": ["x"]},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    first = validate_jsonl_gzip(path)
    second = validate_jsonl_gzip(path)
    assert first == second
    assert first["row_count"] == 2
    assert first["embedded_licenses"] == ["BY-C"]
    assert len(first["schema_hash"]) == 64


def test_parquet_validation_and_hashed_local_path(tmp_path: Path):
    path = tmp_path / "manifest.parquet"
    pq.write_table(pa.table({"key": ["a", "b"], "size": [1, 2]}), path)
    result = validate_parquet(path)
    assert result["row_count"] == 2
    assert len(result["schema_hash"]) == 64
    key = "collections/2025-09-25/thermo/manifest.parquet"
    local = local_object_path(tmp_path, "2025-09-25", "thermo", key)
    assert local.name == hashlib.sha256(key.encode()).hexdigest() + ".parquet"


def test_inventory_fingerprint_and_jsonl_output_are_order_stable(tmp_path: Path):
    rows = [
        {"key": "a", "content_length": 1, "etag": "x", "last_modified": "t"},
        {"key": "b", "content_length": 2, "etag": "y", "last_modified": "t"},
    ]
    assert inventory_fingerprint(rows) == inventory_fingerprint(list(rows))
    output = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(output, rows)
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows


def test_curl_config_pairs_encoded_url_with_hashed_part_path(tmp_path: Path):
    row = {
        "database_version": "2023-11-01",
        "collection": "thermo",
        "key": "collections/2023-11-01/thermo/thermo_type=GGA+U/a.jsonl.gz",
        "source_uri": "https://example.test/thermo_type%3DGGA%2BU/a.jsonl.gz",
    }
    path = tmp_path / "transfer.cfg"
    write_curl_config(
        path,
        [row],
        raw_dir=tmp_path / "raw",
        parallel_max=4,
        retries=2,
        max_time_seconds=30,
    )
    body = path.read_text(encoding="utf-8")
    assert "parallel-max = 4" in body
    assert "continue-at = -" in body
    assert "speed-time = 30" in body
    assert "%2B" in body
    assert ".jsonl.gz.part" in body


def test_curl_config_uses_explicit_retrieval_uri(tmp_path: Path):
    row = {
        "database_version": "2025-09-25",
        "collection": "materials",
        "key": "collections/2025-09-25/materials/a.jsonl.gz",
        "source_uri": "https://canonical.example/a.jsonl.gz",
        "retrieval_uri": "https://equivalent.example/a.jsonl.gz",
    }
    path = tmp_path / "transfer.cfg"
    write_curl_config(
        path,
        [row],
        raw_dir=tmp_path / "raw",
        parallel_max=1,
        retries=1,
        max_time_seconds=30,
    )
    body = path.read_text(encoding="utf-8")
    assert "equivalent.example" in body
    assert "canonical.example" not in body


def test_aria2_input_rotates_official_mirrors_and_targets_part_file(tmp_path: Path):
    row = {
        "database_version": "2025-09-25",
        "collection": "thermo",
        "key": "collections/2025-09-25/thermo/a.jsonl.gz",
    }
    path = tmp_path / "aria2.txt"
    endpoints = ["https://one.example", "https://two.example"]
    write_aria2_input(
        path,
        [row],
        raw_dir=tmp_path / "raw",
        transfer_endpoints=endpoints,
    )
    body = path.read_text(encoding="utf-8")
    assert "one.example" in body and "two.example" in body
    assert "\t" in body
    assert "  dir=" in body
    assert "  out=" in body and ".jsonl.gz.part" in body


def test_aria2_control_file_marks_preallocated_part_as_incomplete(tmp_path: Path):
    part_path = tmp_path / "object.jsonl.gz.part"
    final_path = tmp_path / "object.jsonl.gz"
    part_path.write_bytes(b"1234")
    control_path = Path(str(part_path) + ".aria2")
    control_path.write_bytes(b"state")
    assert not aria2_object_is_complete(final_path, part_path, 4)
    control_path.unlink()
    assert aria2_object_is_complete(final_path, part_path, 4)


def test_snapshot_manifest_offline_reverification(tmp_path: Path):
    raw = tmp_path / "sample.jsonl.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"material_id": "mp-1"}) + "\n")
    validation = validate_jsonl_gzip(raw)
    row = {
        "key": "collections/2025-09-25/materials/sample.jsonl.gz",
        "source_uri": "https://example.test/sample.jsonl.gz",
        "retrieval_timestamp_utc": "2026-08-21T00:00:00+00:00",
        "database_version": "2025-09-25",
        "license": "test license",
        "content_length": raw.stat().st_size,
        "local_path": raw.as_posix(),
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "file_format": "jsonl.gz",
        "validation_status": "PASS",
        **validation,
    }
    manifest = tmp_path / "snapshot_manifest.jsonl"
    write_jsonl_atomic(manifest, [row])
    result = verify_snapshot_manifest(manifest, workers=1)
    assert result["status"] == "PASS"
    assert result["object_count"] == 1
    assert result["row_count"] == 1
    assert result["network_access"] is False
