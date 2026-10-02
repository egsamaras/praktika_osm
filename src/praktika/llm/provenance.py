"""Provenance helpers recorded on every draft: the repository's git SHA and
one hash per model role from the model register."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from praktika.models import ModelRegister


def git_sha() -> str:
    """Short git SHA of the repository containing this package, or ``"unknown"``."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, resolved binary, no shell
            [git, "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def model_hashes(register: ModelRegister | dict[str, str] | None) -> dict[str, str]:
    """One hash string per model role from a register (or a ready-made mapping)."""
    if register is None:
        return {}
    if isinstance(register, dict):
        return dict(register)
    return {
        m.role: ",".join(sorted(set(m.files_sha256.values())))[:64] or "unknown"
        for m in register.models
    }
