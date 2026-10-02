"""Verify local model weights against the register before any backend loads them (C-13).

``ensure_verified`` runs ``manage.verify`` once per process for a given register (keyed by
``models_dir`` and the register file's mtime and size), raises ``ModelRegisterMismatch`` on any
difference and emits ``models.verified`` on the audit log. Configurations that load no local
weights (``fake`` / ``http`` / ``none`` STT, no pyannote) are exempt: there is nothing to verify.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praktika.config import Settings
from praktika.errors import ModelRegisterMismatch
from praktika.logging import get_logger
from praktika.models_registry import manage

log = get_logger(__name__)

_REMOTE = frozenset({"fake", "http", "none"})
_verified: dict[tuple[str, float, int], list[str]] = {}


def loads_local_weights(settings: Settings) -> bool:
    """True when the configured STT or diarisation backends read weights from ``models_dir``."""
    return (
        settings.stt_en not in _REMOTE
        or settings.stt_ar not in _REMOTE
        or (settings.diarize and settings.diarize_backend == "pyannote")
    )


def ensure_verified(
    settings: Settings, audit: Any | None, *, meeting_id: str | None = None
) -> None:
    """Verify the register once per process; refuse (``ModelRegisterMismatch``) on mismatch.

    A missing register is a mismatch too when local weights are configured: nothing may be
    loaded that the register does not describe.
    """
    if not loads_local_weights(settings):
        return
    path = manage.register_path(settings)
    if not path.exists():
        raise ModelRegisterMismatch(
            f"no model register at {path}; run `praktika models pull`/`register` before ingest"
        )
    st = path.stat()
    key = (str(Path(settings.models_dir).resolve()), st.st_mtime, st.st_size)
    if key in _verified:
        return
    roles = [r.role for r in manage.verify(settings)]
    _verified[key] = roles
    if audit is not None:
        audit.append("models.verified", meeting_id, roles=roles, register=str(path))
    log.info("models.verified_before_load", roles=roles)


def reset_cache() -> None:
    """Forget verification results (tests)."""
    _verified.clear()
