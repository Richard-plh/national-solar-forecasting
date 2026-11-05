"""Bounded, verified file publication; no model, data or experiment policy.

Only a fully written, flushed, closed and validated same-directory temporary
file may replace a target. The target is never deleted first. Failed candidates
and best-effort diagnostics remain available for audit, never as final results.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import BinaryIO, Callable
from contextlib import contextmanager

IO_VERSION = "windows_atomic_io_v1"
RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


@contextmanager
def _path_mutex(path: Path):
    """Coordinate project readers/publication across Windows processes.

    A kernel named mutex leaves no lock file in historical run directories.
    It is held only for reading bytes, or replacement + byte verification,
    never JSON parsing, model computation, serialization or backoff preparation.
    An abandoned mutex is safely acquired; atomic publication is still checked.
    """
    if os.name != "nt":
        yield
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateMutexW
    create.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    create.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    normalized = os.path.normcase(str(path.resolve()))
    name = (
        "Local\\solar_forecasting_io_"
        + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    )
    handle = create(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 10000)
        if result not in (0, 0x80):
            if result == 258:
                raise TimeoutError(
                    "Timed out after 10s waiting for the project file I/O mutex"
                )
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = True
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def _shared_read_bytes(path: Path) -> bytes:
    """Read a snapshot with FILE_SHARE_DELETE; close before parsing or return."""
    if os.name != "nt":
        return path.read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(str(path.resolve()), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except Exception:
        close(handle)
        raise
    # open_osfhandle transfers ownership: the Python stream closes the OS handle.
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def read_text_snapshot(path: str | Path) -> str:
    """Snapshot read with bounded retries only for explicit sharing/lock errors."""
    for number in range(len(RETRY_DELAYS) + 1):
        try:
            with _path_mutex(Path(path)):
                encoded = _shared_read_bytes(Path(path))
            return encoded.decode("utf-8")
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or number == len(
                RETRY_DELAYS
            ):
                raise
            time.sleep(RETRY_DELAYS[number])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _delete_access_probe(path: Path) -> int | None:
    """Read-only Windows DELETE-access probe; it does not delete/change a file.

    0: the requested access is currently available; 32/33: current sharing/lock
    conflict; 5: access still denied, so it must NOT be treated as transient.
    Unknown errors are fail-closed. No ACL, privilege or security policy changes.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(str(path.resolve()), 0x00010000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        return ctypes.get_last_error()
    close(handle)
    return 0


def _retry_reason(exc: OSError, temporary: Path, target: Path) -> str | None:
    code = getattr(exc, "winerror", None)
    if code in (32, 33):
        return f"Windows sharing/lock conflict {code}"
    if code != 5 or os.name != "nt":
        return None
    # Generic ACCESS_DENIED is ambiguous. Retry it only if the existing regular
    # target is not read-only and metadata-only probes support a sharing/lock
    # conflict or show DELETE access is now available for both files. These are
    # bounded retry candidates, not proof of the historical external lock owner.
    try:
        if (
            not temporary.is_file()
            or not target.is_file()
            or target.is_symlink()
            or not (target.stat().st_mode & stat.S_IWRITE)
        ):
            return None
        source_evidence = _delete_access_probe(temporary)
        evidence = _delete_access_probe(target)
    except OSError:
        return None
    if source_evidence not in (0, 32, 33) or evidence not in (0, 32, 33):
        return None
    if source_evidence in (32, 33):
        return f"ACCESS_DENIED with observed candidate sharing/lock conflict {source_evidence}"
    if evidence in (32, 33):
        return f"ACCESS_DENIED with observed sharing/lock conflict {evidence}"
    if source_evidence == evidence == 0:
        return "ACCESS_DENIED but both candidate and target DELETE-access probes now succeed"
    return None


class AtomicWriteError(OSError):
    """Publication failed. A candidate file is evidence, never success proof."""

    def __init__(self, message: str, temporary: Path, evidence: dict):
        super().__init__(message)
        self.temporary_path = temporary
        self.evidence = evidence
        self.failure_path = temporary.with_name(temporary.name + ".failure.json")


def _failure(
    temporary: Path,
    target: Path,
    exc: Exception,
    phase: str,
    attempts: list,
    expected_hash: str | None,
) -> AtomicWriteError:
    evidence = {
        "schema": "atomic_write_failure_v1",
        "io_version": IO_VERSION,
        "status": "WRITE_FAILED",
        "target": str(target),
        "temporary": str(temporary),
        "phase": phase,
        "replace_attempts": attempts,
        "candidate_sha256": expected_hash,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "winerror": getattr(exc, "winerror", None),
        "errno": getattr(exc, "errno", None),
        "candidate_is_not_a_final_artifact": True,
        "candidate_retained": temporary.exists(),
    }
    location = (
        f"candidate retained at {temporary}"
        if temporary.exists()
        else "candidate no longer present after publication"
    )
    error = AtomicWriteError(
        f"Atomic publication failed in {phase}: {exc}; {location}", temporary, evidence
    )
    # Evidence is a distinct create-only file, not another replacement of the
    # failed target. Failure to record diagnostics never hides the original error.
    try:
        with error.failure_path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as diagnostic_error:
        error.evidence["diagnostic_write_error"] = str(diagnostic_error)
    return error


def atomic_write(
    path: str | Path,
    writer: Callable[[BinaryIO], None],
    *,
    validator: Callable[[Path], None] | None = None,
) -> dict:
    """Publish a validated file, with at most 7 replace calls / 2.55s backoff.

    Serialization/validation errors and nonclassified OS errors are not retried.
    No Python/NumPy/Torch random stream is used by this I/O policy.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix="." + target.name + ".", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    attempts, expected_hash = [], None
    phase = "serialize"
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # The fd/handle is closed before validators, replace calls or retries.
        phase = "validate_candidate"
        if validator is not None:
            validator(temporary)
        expected_hash = _digest(temporary)
        expected_size = temporary.stat().st_size
        phase = "replace"
        with _path_mutex(target):
            for number in range(len(RETRY_DELAYS) + 1):
                try:
                    os.replace(temporary, target)
                    attempts.append({"attempt": number + 1, "outcome": "SUCCESS"})
                    break
                except OSError as exc:
                    reason = _retry_reason(exc, temporary, target)
                    attempts.append(
                        {
                            "attempt": number + 1,
                            "outcome": "ERROR",
                            "winerror": getattr(exc, "winerror", None),
                            "errno": exc.errno,
                            "retry_reason": reason,
                        }
                    )
                    if reason is None or number == len(RETRY_DELAYS):
                        raise
                    time.sleep(RETRY_DELAYS[number])
            phase = "verify_published"
            if (
                target.stat().st_size != expected_size
                or _digest(target) != expected_hash
            ):
                raise OSError("Published bytes differ from the validated candidate")
        return {
            "io_version": IO_VERSION,
            "sha256": expected_hash,
            "bytes": expected_size,
            "replace_attempts": len(attempts),
            "retry_count": len(attempts) - 1,
        }
    except Exception as exc:
        raise _failure(temporary, target, exc, phase, attempts, expected_hash) from exc


def atomic_write_json(path: str | Path, value) -> dict:
    # Reject invalid JSON before any write.
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode(
        "utf-8"
    )
    expected = hashlib.sha256(encoded).hexdigest()

    def validate(candidate: Path) -> None:
        json.loads(candidate.read_text(encoding="utf-8"))
        if _digest(candidate) != expected:
            raise ValueError("JSON candidate differs from the serialized payload")

    return atomic_write(path, lambda stream: stream.write(encoded), validator=validate)


def validate_checkpoint_archive(path: Path) -> None:
    """Verify torch.save's complete ZIP/CRC without loading model parameters."""
    import zipfile

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if not any(n.endswith("/data.pkl") for n in names) or not any(
            n.endswith("/version") for n in names
        ):
            raise ValueError("Incomplete Torch checkpoint archive")
        if archive.testzip() is not None:
            raise ValueError("Torch checkpoint archive CRC mismatch")
