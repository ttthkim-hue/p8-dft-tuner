"""File-lock slot management and disjoint CPU-mask verification."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import tempfile
import time
from typing import Any, Iterator

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows msvcrt fallback
    _fcntl = None  # type: ignore[assignment]

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX hosts
    _msvcrt = None  # type: ignore[assignment]

from .topology import parse_cpulist


class SlotAllocationError(RuntimeError):
    """Raised when no concurrency lock slot can be acquired within timeout."""


def default_lock_dir() -> pathlib.Path:
    """Resolve runtime lock directory cleanly via environment variable or OS temp directory."""
    env = os.environ.get("P8_DFT_TUNER_LOCK_DIR", "").strip()
    if env:
        return pathlib.Path(env)
    return pathlib.Path(tempfile.gettempdir()) / "p8_dft_tuner_locks"


def verify_non_overlapping_masks(gpu_masks: list[str], cpu_mask: str) -> list[int]:
    """Verify zero CPU index intersection across GPU feeder slots and CPU lane.

    Returns sorted overlap list (empty list when strictly disjoint).
    Raises ValueError if any GPU slot overlaps another GPU slot or the CPU lane.
    """
    seen_gpu: set[int] = set()
    overlaps: set[int] = set()
    for mask in gpu_masks:
        cpus = set(parse_cpulist(mask))
        overlaps.update(seen_gpu & cpus)
        seen_gpu.update(cpus)
    cpu_set = set(parse_cpulist(cpu_mask)) if cpu_mask.strip() else set()
    overlaps.update(seen_gpu & cpu_set)
    overlap_sorted = sorted(overlaps)
    if overlap_sorted:
        raise ValueError(f"Non-disjoint CPU core assignment detected: {overlap_sorted}")
    return overlap_sorted


@dataclasses.dataclass
class SlotLease:
    """Metadata for an acquired concurrency slot lock."""

    slot_id: int
    lock_path: pathlib.Path
    pid: int
    acquired_at_epoch: float
    metadata: dict[str, Any]


_WIN_LOCK_OFFSET = 4096


class SlotLockManager:
    """Advisory `flock`-backed GPU concurrency slot manager with non-blocking/timed acquisition."""

    def __init__(self, lock_dir: pathlib.Path | str | None = None) -> None:
        self.lock_dir = pathlib.Path(lock_dir) if lock_dir is not None else default_lock_dir()
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def _slot_path(self, slot_id: int) -> pathlib.Path:
        if slot_id < 0:
            raise ValueError(f"slot_id must be >= 0, got {slot_id}")
        return self.lock_dir / f"gpu_slot_{slot_id}.lock"

    def _try_lock_fd(self, fd: int) -> bool:
        if _fcntl is not None:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False
        if _msvcrt is not None:  # pragma: no cover - Windows host fallback
            try:
                # Lock byte offset past JSON header so concurrent readers can inspect lock metadata.
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        raise SlotAllocationError("No file locking backend available (neither fcntl nor msvcrt)")

    def _unlock_fd(self, fd: int) -> None:
        if _fcntl is not None:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:
                pass
            return
        if _msvcrt is not None:  # pragma: no cover - Windows host fallback
            try:
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    @contextlib.contextmanager
    def acquire_slot(
        self,
        slot_id: int,
        *,
        timeout_s: float = 0.0,
        poll_interval_s: float = 0.05,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[SlotLease]:
        """Acquire an exclusive lock on `slot_id`, releasing cleanly on exit."""
        lpath = self._slot_path(slot_id)
        fd = os.open(lpath, os.O_CREAT | os.O_RDWR, 0o664)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        locked = False
        try:
            while True:
                if self._try_lock_fd(fd):
                    locked = True
                    break
                if time.monotonic() >= deadline:
                    raise SlotAllocationError(f"GPU slot {slot_id} is busy ({lpath})")
                time.sleep(max(0.005, float(poll_interval_s)))

            payload = {
                "slot_id": slot_id,
                "pid": os.getpid(),
                "acquired_at_epoch": round(time.time(), 3),
                **(metadata or {}),
            }
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
            os.fsync(fd)
            yield SlotLease(
                slot_id=slot_id,
                lock_path=lpath,
                pid=os.getpid(),
                acquired_at_epoch=payload["acquired_at_epoch"],
                metadata=dict(metadata or {}),
            )
        finally:
            if locked:
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                self._unlock_fd(fd)
            os.close(fd)

    @contextlib.contextmanager
    def acquire_first_free_slot(
        self,
        num_slots: int = 3,
        *,
        timeout_s: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[SlotLease]:
        """Acquire the lowest-indexed available slot in `[0, num_slots - 1]`."""
        if int(num_slots) < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            for sid in range(int(num_slots)):
                try:
                    with self.acquire_slot(sid, timeout_s=0.0, metadata=metadata) as lease:
                        yield lease
                        return
                except SlotAllocationError:
                    continue
            if time.monotonic() >= deadline:
                raise SlotAllocationError(f"All {num_slots} GPU slots in {self.lock_dir} are busy")
            time.sleep(0.05)

    def slot_status(self, num_slots: int = 3) -> list[dict[str, Any]]:
        """Probe lock availability across slots `[0, num_slots - 1]` without creating missing files."""
        if int(num_slots) < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        statuses: list[dict[str, Any]] = []
        for sid in range(int(num_slots)):
            lpath = self._slot_path(sid)
            busy = False
            holder: dict[str, Any] = {}
            if lpath.is_file():
                try:
                    fd = os.open(lpath, os.O_RDWR)
                except FileNotFoundError:
                    statuses.append({"slot_id": sid, "busy": False, "holder": {}})
                    continue
                try:
                    if self._try_lock_fd(fd):
                        self._unlock_fd(fd)
                        busy = False
                    else:
                        busy = True
                        raw = lpath.read_text(encoding="utf-8", errors="ignore").strip()
                        if raw:
                            try:
                                holder = json.loads(raw)
                            except json.JSONDecodeError:
                                holder = {"raw": raw}
                finally:
                    os.close(fd)
            statuses.append({"slot_id": sid, "busy": busy, "holder": holder})
        return statuses
