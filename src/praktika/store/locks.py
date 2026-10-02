"""Per-meeting run locks: one run at a time works on a meeting.

Contract: every command that works on a meeting (``start``, ``ingest``, ``transcribe`` and
``generate`` from the command line, ``regenerate`` from the review page) takes the meeting's
lock before it reads or moves anything and releases it when it ends, on an error and on Ctrl-C
too. The ``run_locks`` row records the command, process id, host and start time. It is taken
atomically (``BEGIN IMMEDIATE``), so of two runs started together exactly one gets it and the
other is refused with ``RunLockedError``, which names the command holding it. A lock whose
process no longer exists on this host (the run crashed or was killed) is taken over:
``take_run_lock`` returns the lock it replaced so that the caller audits the takeover
(``run.lock_taken_over``). A lock recorded on another host is never taken over, because its
process cannot be checked from here.

While a run holds the lock, review-page writes on the meeting are refused (``transition`` with
``unlocked=True`` checks the lock in the same transaction as the state). ``praktika abort``
never waits for the lock: it moves the meeting on and records in ``stopped_by`` what did so
(``abort``, ``discard`` from the review page, ``purge`` from a DSAR deletion), so the run can
name it when it stops. ``reopened_from`` records that a ``generate --reopen`` run started from
an approved meeting, so an abort during that run puts the approved record back instead of
discarding it.
"""

from __future__ import annotations

import os
import secrets
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from praktika.errors import PraktikaError
from praktika.models import MeetingState
from praktika.store.records import iso, parse_dt

#: What may be recorded in ``stopped_by``.
STOPPED_BY = ("abort", "discard", "purge")
#: A process that started this long after the lock was taken is not the one that took it (its
#: id has been reused). Well above clock rounding, far below any realistic reuse of an id.
PID_REUSE_MARGIN = timedelta(seconds=60)

RUN_LOCKS_SQL = """
CREATE TABLE IF NOT EXISTS run_locks (
    meeting_id     TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    command        TEXT    NOT NULL,
    pid            INTEGER NOT NULL,
    host           TEXT    NOT NULL,
    started_at     TEXT    NOT NULL,
    token          TEXT    NOT NULL,
    reopened_from  TEXT,
    stopped_by     TEXT
);
"""


class RunLock(BaseModel):
    """One ``run_locks`` row: which run is working on a meeting."""

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    command: str
    pid: int
    host: str
    started_at: datetime
    token: str
    reopened_from: MeetingState | None = None
    stopped_by: str | None = None

    @property
    def busy(self) -> str:
        """The short refusal the review page shows (HTTP 409)."""
        article = "an" if self.command[:1] in ("a", "e", "i", "o", "u") else "a"
        return f"{article} {self.command} run is working on this meeting"

    def describe(self) -> str:
        """Who holds it, for the command-line refusal."""
        return (
            f"process {self.pid} on {self.host}, started "
            f"{self.started_at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC"
        )


class RunLockedError(PraktikaError):
    """Another run holds the meeting's lock; ``lock`` says which."""

    def __init__(self, lock: RunLock) -> None:
        self.lock = lock
        super().__init__(
            f"{lock.meeting_id}: {lock.busy} ({lock.describe()}); wait for it to finish, "
            "then try again"
        )


def this_host() -> str:
    """The name recorded as a lock's host."""
    return socket.gethostname() or "localhost"


def new_run_lock(meeting_id: str, command: str) -> RunLock:
    """A lock for ``command`` on ``meeting_id``, held by this process."""
    return RunLock(
        meeting_id=meeting_id,
        command=command,
        pid=os.getpid(),
        host=this_host(),
        started_at=datetime.now(UTC),
        token=secrets.token_hex(8),
    )


def takeover_detail(lock: RunLock, replaced: RunLock) -> dict[str, Any]:
    """The detail of a ``run.lock_taken_over`` audit event: the new run and the dead one."""
    return {
        "command": lock.command,
        "previous_command": replaced.command,
        "previous_pid": replaced.pid,
        "previous_host": replaced.host,
        "previous_started_at": iso(replaced.started_at),
    }


def _process_started(pid: int) -> datetime | None:
    """When process ``pid`` started, where the host says (``/proc``); ``None`` otherwise."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        ticks = int(stat[stat.rindex(")") + 2 :].split()[19])  # field 22, starttime
        boot = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
            if line.startswith("btime ")
        )
        return datetime.fromtimestamp(boot + ticks / os.sysconf("SC_CLK_TCK"), UTC)
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def holder_alive(lock: RunLock) -> bool:
    """Whether the process that took ``lock`` may still be running.

    A lock taken on another host is always treated as alive (it cannot be checked from here).
    On this host the process must exist and, where the start time of a process can be read,
    must have started before the lock was taken (not a later process that reused its id).
    """
    if lock.host != this_host():
        return True
    if lock.pid == os.getpid():
        return True
    if lock.pid <= 0:
        return False
    try:
        os.kill(lock.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # it exists, under another account
    except OSError:
        return True
    started = _process_started(lock.pid)
    return started is None or started <= lock.started_at + PID_REUSE_MARGIN


class LockMixin:
    """The ``run_locks`` rows of ``SqliteStore``."""

    conn: sqlite3.Connection

    if TYPE_CHECKING:  # provided by SqliteStore; never defined at runtime here

        def _one(self, sql: str, *params: Any) -> sqlite3.Row | None: ...

    def _lock_row(self, meeting_id: str) -> RunLock | None:
        row = self.conn.execute(
            "SELECT * FROM run_locks WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()
        if row is None:
            return None
        return RunLock(
            meeting_id=row["meeting_id"],
            command=row["command"],
            pid=int(row["pid"]),
            host=row["host"],
            started_at=parse_dt(row["started_at"]) or datetime.now(UTC),
            token=row["token"],
            reopened_from=MeetingState(row["reopened_from"]) if row["reopened_from"] else None,
            stopped_by=row["stopped_by"],
        )

    def _live_lock(self, meeting_id: str, exempt: str | None = None) -> RunLock | None:
        lock = self._lock_row(meeting_id)
        if lock is None or lock.token == exempt or not holder_alive(lock):
            return None
        return lock

    def take_run_lock(self, lock: RunLock) -> RunLock | None:
        """Record ``lock`` as the run working on its meeting, atomically.

        Returns ``None``, or the lock of a run whose process no longer exists on this host,
        which ``lock`` has replaced (the caller audits the takeover). Raises
        ``RunLockedError`` while another run's process is alive (nothing written) and
        ``KeyError`` for an unknown meeting.
        """
        conn = self.conn
        if conn.in_transaction:  # by contract every method commits; this is a failed leftover
            conn.rollback()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if self._one("SELECT 1 FROM meetings WHERE id = ?", lock.meeting_id) is None:
                raise KeyError(lock.meeting_id)
            held = self._lock_row(lock.meeting_id)
            if held is not None and holder_alive(held):
                raise RunLockedError(held)
            conn.execute(
                "INSERT OR REPLACE INTO run_locks (meeting_id, command, pid, host, started_at,"
                " token, reopened_from, stopped_by) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    lock.meeting_id,
                    lock.command,
                    lock.pid,
                    lock.host,
                    iso(lock.started_at),
                    lock.token,
                    lock.reopened_from.value if lock.reopened_from else None,
                ),
            )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        return held

    def release_run_lock(self, meeting_id: str, token: str) -> None:
        """Remove the lock ``token`` names (a lock another run took over is left alone)."""
        self.conn.execute(
            "DELETE FROM run_locks WHERE meeting_id = ? AND token = ?", (meeting_id, token)
        )
        self.conn.commit()

    def run_lock(self, meeting_id: str) -> RunLock | None:
        """The meeting's lock row, whether or not its process is still alive."""
        return self._lock_row(meeting_id)

    def live_run_lock(self, meeting_id: str, *, exempt: str | None = None) -> RunLock | None:
        """The meeting's lock while its process may still be running (not the one whose token
        is ``exempt``: the caller's own), else ``None``."""
        return self._live_lock(meeting_id, exempt)

    def mark_run_reopened(self, meeting_id: str, token: str, state: MeetingState) -> None:
        """Record on the lock ``token`` names that its run reopened a meeting at ``state``."""
        self.conn.execute(
            "UPDATE run_locks SET reopened_from = ? WHERE meeting_id = ? AND token = ?",
            (MeetingState(state).value, meeting_id, token),
        )
        self.conn.commit()

    def _mark_stopped(self, meeting_id: str, by: str) -> None:
        """Record what moved the meeting on under a run's lock (no commit: the caller's
        transaction carries it)."""
        if by not in STOPPED_BY:
            raise ValueError(f"unknown stopped_by {by!r}")
        self.conn.execute(
            "UPDATE run_locks SET stopped_by = ? WHERE meeting_id = ?", (by, meeting_id)
        )
