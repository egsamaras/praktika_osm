"""Code-signature check of the ScreenCaptureKit capture helper (C-04).

``helper_is_signed`` verifies the helper with ``/usr/bin/codesign --verify --strict --deep``
(integrity: a binary modified after signing fails), then reads ``-dv --verbose=4`` for a
``Developer ID Application`` authority and, when a Team ID is pinned, the matching
``TeamIdentifier``. Everything else yields ``None``, so Praktika never launches a helper that
the deploying organisation did not sign.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from praktika.logging import get_logger

log = get_logger(__name__)

CODESIGN = "/usr/bin/codesign"


def _codesign(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        # Fixed absolute executable and argument list, no shell; codesign is part of macOS.
        return subprocess.run(  # noqa: S603
            [CODESIGN, *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (FileNotFoundError, PermissionError, subprocess.SubprocessError):
        return None


def helper_is_signed(helper: Path, *, team_id: str | None = None) -> str | None:
    """Return the Developer ID signing identity of ``helper`` or ``None``.

    Two steps, both with ``/usr/bin/codesign``: ``--verify --strict --deep`` must exit 0 (a
    binary modified after signing fails here), then ``-dv --verbose=4`` must show an
    ``Authority=Developer ID Application:`` line and, when ``team_id`` is given, a
    ``TeamIdentifier=`` equal to it. Ad-hoc signatures, unsigned or tampered files, another
    team's signature, missing files and hosts without ``codesign`` all yield ``None``, so
    Praktika can never launch a helper that the deploying organisation did not sign.
    """
    helper = Path(helper)
    if not helper.is_file():
        return None
    verified = _codesign(["--verify", "--strict", "--deep", str(helper)])
    if verified is None or verified.returncode != 0:
        return None
    shown = _codesign(["-dv", "--verbose=4", str(helper)])
    if shown is None or shown.returncode != 0:
        return None
    identity, team = None, None
    for line in (shown.stderr + shown.stdout).splitlines():
        if line.startswith("Authority=Developer ID Application:"):
            identity = identity or line.split("=", 1)[1].strip()
        elif line.startswith("TeamIdentifier="):
            team = line.split("=", 1)[1].strip()
    if identity is None:
        return None
    if team_id is not None and team != team_id:
        log.warning("capture.helper_team_mismatch", expected=team_id, found=team)
        return None
    return identity
