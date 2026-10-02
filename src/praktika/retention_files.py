"""File-level helpers of the retention job: secure wipe and the orphan-audio sweep (C-05).

``wipe_file`` overwrites a file with zeros, fsyncs and unlinks it, so a retained recording is
not recoverable from free blocks after deletion. ``orphan_audio`` lists WAV files under the
audio root that no live media row references (left by a crash between conversion and
registration) once they are older than the shortest non-zero ``audio_hours``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praktika.retention import Deletion, RetentionPolicy

_WIPE_CHUNK = 1 << 20


def wipe_file(path: Path) -> bool:
    """Overwrite ``path`` with zeros, fsync and unlink; True if a file was removed."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    with path.open("r+b") as fh:
        remaining = size
        while remaining > 0:
            n = min(_WIPE_CHUNK, remaining)
            fh.write(b"\0" * n)
            remaining -= n
        fh.flush()
        os.fsync(fh.fileno())
    path.unlink()
    return True


def orphan_audio(
    audio_root: Path | None,
    live: set[Path],
    now: datetime,
    p: RetentionPolicy,
) -> list[Deletion]:
    """WAV files under ``audio_root/<meeting_id>/`` that no live media row references and that
    are older than the shortest non-zero ``audio_hours`` (24 h by default)."""
    if audio_root is None or not Path(audio_root).is_dir():
        return []
    hours = min((h for h in p.audio_hours.values() if h > 0), default=24)
    aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    cutoff = aware - timedelta(hours=hours)
    from praktika.retention import Deletion  # local: the two modules import each other

    out: list[Deletion] = []
    for wav in sorted(Path(audio_root).glob("*/*.wav")):
        if wav.resolve() in live or not wav.is_file():
            continue
        if datetime.fromtimestamp(wav.stat().st_mtime, tz=UTC) > cutoff:
            continue
        out.append(
            Deletion(
                kind="audio",
                meeting_id=wav.parent.name,
                path=wav,
                reason=f"orphan file with no media row older than {hours} h",
            )
        )
    return out
