"""File hashes, canonical JSON identities and atomic JSON I/O."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def hash_array(values) -> str:
    a = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(list(a.shape)).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def read_json(path: str | Path):
    from .atomic_io import read_text_snapshot

    return json.loads(read_text_snapshot(path))


def write_json(path: str | Path, value) -> None:
    from .atomic_io import atomic_write_json

    atomic_write_json(path, value)


def assert_identity(actual: dict, expected: dict) -> None:
    """Compare every requested field. Missing fields never silently match."""
    mismatches = [
        key
        for key, value in expected.items()
        if key not in actual or actual[key] != value
    ]
    if mismatches:
        raise ValueError("Artifact identity mismatch: " + ", ".join(mismatches))
