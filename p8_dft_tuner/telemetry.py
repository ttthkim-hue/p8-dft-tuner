"""Maintenance gate check, live profile hot-reload, and fail-safe JSONL telemetry logger."""
from __future__ import annotations

import copy
import datetime
import json
import os
import pathlib
import tempfile
from typing import Any

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None  # type: ignore[assignment]


DEFAULT_SAFE_PROFILE: dict[str, Any] = {
    "profile_version": "portable-default-v1",
    "concurrent_gpu_workers": 3,
    "cores_per_gpu_slot": 4,
    "omp_num_threads": 1,
    "omp_proc_bind_multi_rank": "false",
    "diagonalization": "david",
    "ozaki_mode": "off",
}


def default_gate_path() -> pathlib.Path:
    env = os.environ.get("P8_DFT_TUNER_GATE_FILE", "").strip()
    if env:
        return pathlib.Path(env)
    return pathlib.Path(tempfile.gettempdir()) / "p8_dft_tuner_maintenance.flag"


def default_profile_path() -> pathlib.Path:
    env = os.environ.get("P8_DFT_TUNER_PROFILE_FILE", "").strip()
    if env:
        return pathlib.Path(env)
    return pathlib.Path(tempfile.gettempdir()) / "p8_dft_tuner_profile.json"


def default_telemetry_path() -> pathlib.Path:
    env = os.environ.get("P8_DFT_TUNER_TELEMETRY_FILE", "").strip()
    if env:
        return pathlib.Path(env)
    return pathlib.Path(tempfile.gettempdir()) / "p8_dft_tuner_runs.jsonl"


def check_maintenance_gate(gate_path: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Check maintenance gate file fail-closed: missing file is inactive; unreadable/corrupt file reports active."""
    gp = pathlib.Path(gate_path) if gate_path is not None else default_gate_path()
    if not gp.is_file():
        return {"active": False, "reason": None, "gate_path": str(gp)}
    try:
        raw = gp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw.startswith("{"):
            data = json.loads(raw)
            return {
                "active": bool(data.get("active", True)),
                "reason": data.get("reason") or "maintenance_active",
                "gate_path": str(gp),
            }
        return {"active": True, "reason": raw or "maintenance_flag_present", "gate_path": str(gp)}
    except (OSError, json.JSONDecodeError):
        return {"active": True, "reason": "unreadable_gate_file", "gate_path": str(gp)}


def load_profile_with_fallback(
    profile_path: pathlib.Path | str | None = None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hot-reload JSON tuning profile while always falling back cleanly to safe defaults."""
    base = copy.deepcopy(defaults if defaults is not None else DEFAULT_SAFE_PROFILE)
    pp = pathlib.Path(profile_path) if profile_path is not None else default_profile_path()
    if not pp.is_file():
        base["profile_source"] = "defaults"
        return base
    try:
        loaded = json.loads(pp.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            base.update(loaded)
            base["profile_source"] = str(pp)
            return base
    except (OSError, json.JSONDecodeError):
        pass
    base["profile_source"] = "defaults_fallback_after_parse_error"
    return base


def record_run(
    record: dict[str, Any],
    *,
    log_path: pathlib.Path | str | None = None,
) -> bool:
    """Append one JSONL telemetry record atomically and fail-safely without interrupting callers."""
    lp = pathlib.Path(log_path) if log_path is not None else default_telemetry_path()
    payload = dict(record)
    payload.setdefault("recorded_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
    line = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lp, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o664)
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            os.write(fd, line)
        finally:
            if _fcntl is not None:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
        return True
    except OSError:
        return False
