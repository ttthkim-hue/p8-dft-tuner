"""Portability test ensuring zero hardcoded private/local paths exist anywhere in the package."""
from __future__ import annotations

import pathlib
import re


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Construct pattern fragments dynamically so this test file stays clean under its own scan
FORBIDDEN_PATTERNS = [
    re.compile(r"(?<![A-Za-z])[C-Zc-z]:" + r"(?:\\\\|/)" + r"(?![\s\n])"),
    re.compile(r"/mn" + r"t/" + r"[a-z]/"),
    re.compile(r"/op" + r"t/" + r"p8-"),
    re.compile(r"ttth" + r"k(?!im)"),
    re.compile(r"rahu" + r"l", re.IGNORECASE),
    re.compile(r"p8-kmo" + r"u", re.IGNORECASE),
    re.compile(r"GPU-" + r"[0-9a-f]{8}"),
]


def test_zero_hardcoded_local_paths_in_package() -> None:
    offenders: list[str] = []
    for p in sorted(PACKAGE_ROOT.rglob("*")):
        if p.is_dir():
            continue
        if any(part in {"__pycache__", ".pytest_cache", "dist", "build"} or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.suffix not in {".py", ".toml", ".md", ".cff", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN_PATTERNS:
            m = pat.search(text)
            if m:
                offenders.append(f"{p.relative_to(PACKAGE_ROOT)} matched {pat.pattern!r} -> {m.group(0)!r}")
    assert offenders == [], "Hardcoded local machine references detected - " + "; ".join(offenders)
