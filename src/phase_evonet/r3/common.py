"""Deterministic, hash-checked, audited access to R3 formal inputs."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


DEFAULT_FORBIDDEN_PATTERNS = (
    "**/locked_test*.parquet",
    "**/confirmation_a/**/*",
    "**/confirmation_b/**/*",
    "**/locked_predictions*",
    "**/sealed_payload*",
)


class FormalInputError(RuntimeError):
    """Base exception for rejected R3 formal-input access."""


class ForbiddenFormalInputError(FormalInputError):
    """Raised before opening a prohibited or out-of-scope path."""


class HashMismatchError(FormalInputError):
    """Raised when a formal input does not match its frozen SHA-256."""


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path* using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.casefold()


def _has_parent_traversal(value: str | os.PathLike[str]) -> bool:
    return ".." in PurePosixPath(_normalized(value)).parts


def _matches_forbidden(
    value: str | os.PathLike[str], patterns: Iterable[str]
) -> bool:
    candidate = _normalized(value)
    candidates = {
        candidate,
        candidate.lstrip("./"),
        PurePosixPath(candidate).name,
    }
    for raw_pattern in patterns:
        pattern = _normalized(raw_pattern)
        variants = {pattern}
        if pattern.startswith("**/"):
            variants.add(pattern[3:])
        for item in candidates:
            if any(fnmatch.fnmatchcase(item, variant) for variant in variants):
                return True
    return False


def _within_any_root(path: Path, roots: Iterable[str | os.PathLike[str]]) -> bool:
    resolved_text = os.path.normcase(str(path))
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve(strict=True)
        try:
            if os.path.commonpath((resolved_text, os.path.normcase(str(root)))) == os.path.normcase(
                str(root)
            ):
                return True
        except ValueError:
            continue
    return False


def _append_access_log(access_log: Path, record: dict[str, object]) -> None:
    access_log.parent.mkdir(parents=True, exist_ok=True)
    with access_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def open_formal_input(
    path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    task_id: str,
    access_log: str | os.PathLike[str],
    purpose: str,
    forbidden_patterns: Iterable[str] = DEFAULT_FORBIDDEN_PATTERNS,
    allowed_roots: Iterable[str | os.PathLike[str]] | None = None,
    caller: str | None = None,
) -> BinaryIO:
    """Validate and open one formal input in binary read-only mode.

    The raw path is screened before resolution. The resolved path is screened
    again to catch symlink aliases. When ``allowed_roots`` is supplied, the
    resolved file must remain inside at least one root. Every authorization or
    rejection is appended to the R3 JSONL access log; prohibited files are
    never opened.
    """

    raw_path = os.fspath(path)
    patterns = tuple(forbidden_patterns)
    log_path = Path(access_log)
    caller_value = caller or inspect.stack()[1].function
    base_record: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "purpose": purpose,
        "caller": caller_value,
        "requested_path": raw_path,
    }

    if _has_parent_traversal(raw_path):
        _append_access_log(log_path, {**base_record, "status": "REJECTED_PATH_TRAVERSAL"})
        raise ForbiddenFormalInputError(f"parent traversal is not allowed: {raw_path}")
    if _matches_forbidden(raw_path, patterns):
        _append_access_log(log_path, {**base_record, "status": "REJECTED_FORBIDDEN_PATTERN"})
        raise ForbiddenFormalInputError(f"forbidden formal-input path: {raw_path}")

    requested = Path(raw_path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _append_access_log(
            log_path,
            {**base_record, "status": "REJECTED_UNAVAILABLE", "error": type(exc).__name__},
        )
        raise FormalInputError(f"formal input is unavailable: {raw_path}") from exc

    resolved_text = str(resolved)
    if _matches_forbidden(resolved_text, patterns):
        _append_access_log(
            log_path,
            {**base_record, "resolved_path": resolved_text, "status": "REJECTED_FORBIDDEN_RESOLVED_PATH"},
        )
        raise ForbiddenFormalInputError(f"resolved path is forbidden: {resolved_text}")
    if allowed_roots is not None and not _within_any_root(resolved, allowed_roots):
        _append_access_log(
            log_path,
            {**base_record, "resolved_path": resolved_text, "status": "REJECTED_ROOT_ESCAPE"},
        )
        raise ForbiddenFormalInputError(f"resolved path escapes approved roots: {resolved_text}")

    expected = expected_sha256.casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        _append_access_log(
            log_path,
            {**base_record, "resolved_path": resolved_text, "status": "REJECTED_INVALID_EXPECTED_HASH"},
        )
        raise FormalInputError("expected_sha256 must be a 64-character hexadecimal digest")

    stat_before = resolved.stat()
    observed = sha256_file(resolved)
    stat_after = resolved.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        _append_access_log(
            log_path,
            {
                **base_record,
                "resolved_path": resolved_text,
                "observed_sha256": observed,
                "status": "REJECTED_CHANGED_DURING_HASH",
            },
        )
        raise FormalInputError(f"formal input changed while hashing: {resolved_text}")
    if observed != expected:
        _append_access_log(
            log_path,
            {
                **base_record,
                "resolved_path": resolved_text,
                "bytes": stat_after.st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "REJECTED_HASH_MISMATCH",
            },
        )
        raise HashMismatchError(
            f"formal input hash mismatch for {resolved_text}: expected {expected}, observed {observed}"
        )

    _append_access_log(
        log_path,
        {
            **base_record,
            "resolved_path": resolved_text,
            "bytes": stat_after.st_size,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "status": "AUTHORIZED_READ_ONLY",
        },
    )
    return resolved.open("rb")
