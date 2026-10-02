"""Opportunistic retention run at the start of every user command (C-05).

``sweep_retention`` executes the retention timers (and the orphan-audio sweep) as the
``system`` actor and never raises: a failed sweep is logged and the command goes on. It is
what makes "audio deleted automatically" true even when nobody runs ``praktika retention run``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from praktika.audit import AuditLog
from praktika.logging import get_logger

if TYPE_CHECKING:
    from praktika.cli.context import Runtime

log = get_logger(__name__)


def sweep_retention(rt: Runtime) -> int:
    """Best-effort retention run recorded as ``system``; returns deletions, never raises."""
    from praktika import retention
    from praktika.cli.context import audit_sink

    system_log = AuditLog(audit_sink(rt.settings), rt.store, None)
    try:
        policy = retention.RetentionPolicy.from_settings(rt.settings)
        deletions = retention.run(
            rt.store,
            system_log,
            datetime.now(UTC),
            policy=policy,
            audio_root=Path(rt.settings.data_dir) / "audio",
        )
    except retention.RetentionRunError as exc:  # some items failed; the rest were deleted
        log.warning("retention.sweep_partial", deleted=len(exc.done), failed=len(exc.failed))
        return len(exc.done)
    except Exception as exc:  # noqa: BLE001 - a failed sweep must not block the command
        log.warning("retention.sweep_failed", error=str(exc))
        return 0
    if deletions:
        log.info("retention.sweep", deleted=len(deletions))
    return len(deletions)
