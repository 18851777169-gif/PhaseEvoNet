from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phase_evonet.r3.common import (
    ForbiddenFormalInputError,
    FormalInputError,
    HashMismatchError,
    open_formal_input,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_absolute_and_relative_approved_paths_are_read_only_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic-r3-fixture\n"
    source = tmp_path / "approved.bin"
    source.write_bytes(payload)
    log = tmp_path / "audit.jsonl"

    with open_formal_input(
        source,
        _digest(payload),
        task_id="R3.0",
        access_log=log,
        purpose="absolute-form test",
        allowed_roots=[tmp_path],
    ) as handle:
        assert handle.read() == payload
        assert handle.writable() is False

    monkeypatch.chdir(tmp_path)
    with open_formal_input(
        "approved.bin",
        _digest(payload),
        task_id="R3.0",
        access_log=log,
        purpose="relative-form test",
        allowed_roots=[tmp_path],
    ) as handle:
        assert handle.read() == payload

    records = _records(log)
    assert [record["status"] for record in records] == [
        "AUTHORIZED_READ_ONLY",
        "AUTHORIZED_READ_ONLY",
    ]
    assert all(record["observed_sha256"] == _digest(payload) for record in records)


@pytest.mark.parametrize(
    "name",
    [
        "locked_test_outcomes.parquet",
        "LoCkEd_TeSt_OutcomeS.PARQUET",
        "locked_predictions.json",
        "sealed_payload.bin",
    ],
)
def test_forbidden_names_are_rejected_case_insensitively_before_open(
    tmp_path: Path, name: str
) -> None:
    payload = b"synthetic-only"
    source = tmp_path / name
    source.write_bytes(payload)
    log = tmp_path / "audit.jsonl"

    with pytest.raises(ForbiddenFormalInputError):
        open_formal_input(
            source,
            _digest(payload),
            task_id="R3.0",
            access_log=log,
            purpose="forbidden-pattern test",
            allowed_roots=[tmp_path],
        )

    assert _records(log)[0]["status"] == "REJECTED_FORBIDDEN_PATTERN"


def test_confirmation_directory_is_rejected(tmp_path: Path) -> None:
    payload = b"synthetic-only"
    directory = tmp_path / "Confirmation_A" / "nested"
    directory.mkdir(parents=True)
    source = directory / "outcome.parquet"
    source.write_bytes(payload)

    with pytest.raises(ForbiddenFormalInputError):
        open_formal_input(
            source,
            _digest(payload),
            task_id="R3.0",
            access_log=tmp_path / "audit.jsonl",
            purpose="confirmation-directory test",
            allowed_roots=[tmp_path],
        )


def test_parent_traversal_is_rejected_before_resolution(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"synthetic-only")
    traversal = approved / ".." / "outside.bin"
    log = tmp_path / "audit.jsonl"

    with pytest.raises(ForbiddenFormalInputError):
        open_formal_input(
            traversal,
            _digest(b"synthetic-only"),
            task_id="R3.0",
            access_log=log,
            purpose="traversal test",
            allowed_roots=[approved],
        )

    assert _records(log)[0]["status"] == "REJECTED_PATH_TRAVERSAL"


def test_symlink_escape_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"synthetic-only")
    link = approved / "alias.bin"
    outside_resolved = outside.resolve()
    original_resolve = Path.resolve
    try:
        link.symlink_to(outside)
    except OSError:
        # Windows commonly denies symlink creation without Developer Mode or
        # elevated privilege. Simulate the one filesystem fact the guard uses:
        # the benign alias resolves outside the approved root.
        def resolve_with_escape(self: Path, strict: bool = False) -> Path:
            if self == link:
                return outside_resolved
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", resolve_with_escape)

    log = tmp_path / "audit.jsonl"
    with pytest.raises(ForbiddenFormalInputError):
        open_formal_input(
            link,
            _digest(b"synthetic-only"),
            task_id="R3.0",
            access_log=log,
            purpose="symlink-escape test",
            allowed_roots=[approved],
        )

    assert _records(log)[0]["status"] == "REJECTED_ROOT_ESCAPE"


def test_hash_mismatch_is_rejected_and_recorded(tmp_path: Path) -> None:
    source = tmp_path / "approved.bin"
    source.write_bytes(b"actual")
    log = tmp_path / "audit.jsonl"

    with pytest.raises(HashMismatchError):
        open_formal_input(
            source,
            _digest(b"expected"),
            task_id="R3.0",
            access_log=log,
            purpose="hash-mismatch test",
            allowed_roots=[tmp_path],
        )

    assert _records(log)[0]["status"] == "REJECTED_HASH_MISMATCH"


def test_invalid_expected_digest_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "approved.bin"
    source.write_bytes(b"synthetic-only")

    with pytest.raises(FormalInputError):
        open_formal_input(
            source,
            "not-a-sha256",
            task_id="R3.0",
            access_log=tmp_path / "audit.jsonl",
            purpose="invalid-hash test",
            allowed_roots=[tmp_path],
        )
