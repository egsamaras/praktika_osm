"""SQLite persistence.

``SqliteStore`` is the laptop implementation of the ``Store`` seam (``store/records.py``) over
stdlib ``sqlite3``. Domain objects round-trip through their Pydantic JSON so every column is a
projection, never the source of truth.

Threading: the connection is opened with ``check_same_thread=False`` (``store/db.py``) and every
public method runs under one re-entrant lock, so the review server's worker threads, the CLI
thread that opened the store and a retention job may share one ``SqliteStore``; multi-statement
methods (read the next version, then insert) are atomic with respect to each other.
"""

from __future__ import annotations

import functools
import inspect
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from praktika.models import (
    AuditEvent,
    ConsentRecord,
    LanguageMode,
    Meeting,
    MeetingState,
    Minutes,
    ReviewItem,
)
from praktika.store.artefacts import ArtefactMixin
from praktika.store.db import connect, json_dumps, utcnow_iso
from praktika.store.locks import LockMixin
from praktika.store.queries import QueryMixin
from praktika.store.records import MediaRecord, OpenAction, Store, iso, minutes_from_row, parse_dt

__all__ = ["GENESIS_HASH", "MediaRecord", "OpenAction", "SqliteStore", "Store"]

GENESIS_HASH = "0" * 64


def _synchronised[F: Callable[..., Any]](fn: F) -> F:
    """Run ``fn`` under the store's re-entrant lock (methods may call each other)."""

    @functools.wraps(fn)
    def wrapper(self: SqliteStore, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return fn(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class SqliteStore(QueryMixin, ArtefactMixin, LockMixin):
    """``Store`` over one SQLite file (WAL, 0600). Use ``":memory:"`` in tests.

    Safe to share between threads: see the module docstring.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.conn: sqlite3.Connection = connect(path)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def _insert(
        self, table: str, replace: bool = False, *, upsert_on: str | None = None, **cols: Any
    ) -> int:
        """Insert a row. ``replace`` is ``INSERT OR REPLACE`` (only for tables without
        children: REPLACE deletes the old row, which would cascade). ``upsert_on`` names a key
        column and updates the other columns in place instead."""
        marks = ",".join("?" for _ in cols)
        verb = "INSERT OR REPLACE" if replace and not upsert_on else "INSERT"
        sql = f"{verb} INTO {table} ({','.join(cols)}) VALUES ({marks})"  # noqa: S608 - names fixed
        if upsert_on:
            sets = ", ".join(f"{c} = excluded.{c}" for c in cols if c != upsert_on)
            sql += f" ON CONFLICT({upsert_on}) DO UPDATE SET {sets}"
        cur = self.conn.execute(sql, tuple(cols.values()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def _one(self, sql: str, *params: Any) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ meetings / consent
    def save_meeting(self, meeting: Meeting) -> None:
        """Insert or update the meeting in place (never REPLACE: that would cascade-delete the
        meeting's media, transcripts, consent, vault, minutes and holds); ``updated_at``
        advances, ``created_at`` is kept."""
        now = utcnow_iso()
        row = self._one("SELECT created_at FROM meetings WHERE id = ?", meeting.id)
        self._insert(
            "meetings",
            upsert_on="id",
            id=meeting.id,
            title=meeting.title,
            meeting_type=meeting.meeting_type.value,
            classification=meeting.classification.value,
            language_mode=meeting.language_mode.value,
            platform=meeting.platform.value,
            started_at=iso(meeting.started_at),
            ended_at=iso(meeting.ended_at),
            organiser=meeting.organiser,
            private=int(meeting.private),
            legal_hold=int(meeting.legal_hold),
            state=meeting.state.value,
            meeting_json=json_dumps(meeting.model_dump(mode="json")),
            created_at=row["created_at"] if row else now,
            updated_at=now,
        )

    def get_meeting(self, meeting_id: str) -> Meeting | None:
        row = self._one("SELECT meeting_json FROM meetings WHERE id = ?", meeting_id)
        return Meeting.model_validate_json(row["meeting_json"]) if row else None

    def set_state(self, meeting_id: str, state: MeetingState) -> None:
        """Update the state column and the stored JSON; raises ``KeyError`` if unknown.

        Unconditional (``abort``, approval, DSAR purge); a pipeline step uses ``transition``.
        """
        self._update_meeting(meeting_id, None, state=MeetingState(state))

    def transition(
        self,
        meeting_id: str,
        state: MeetingState,
        *,
        expected: Iterable[MeetingState],
        language_mode: LanguageMode | None = None,
        unlocked: bool = False,
        own_lock: str | None = None,
        stopped_by: str | None = None,
    ) -> bool:
        """Compare-and-set: move the meeting to ``state`` (and ``language_mode``, when given)
        only while its stored state is one of ``expected``. Returns ``True`` when it moved and
        ``False``, writing nothing, when the state was anything else, so a pipeline step can
        never overwrite an ``abort`` or a discard made by another process meanwhile. Raises
        ``KeyError`` for an unknown meeting.

        ``unlocked`` also refuses (``False``) while a run holds the meeting's lock and its
        process is alive, checked in the same transaction (review-page writes); ``own_lock``
        names the caller's own lock, which does not count. ``stopped_by`` (``abort``,
        ``discard``, ``purge``) is recorded on the lock of a run working on the meeting, in the
        same transaction as the move, so that run can say what stopped it.
        """
        update: dict[str, Any] = {"state": MeetingState(state)}
        if language_mode is not None:
            update["language_mode"] = language_mode
        moved = self._update_meeting(
            meeting_id,
            frozenset(expected),
            unlocked=unlocked,
            own_lock=own_lock,
            stopped_by=stopped_by,
            **update,
        )
        return moved is not None

    def _update_meeting(
        self,
        meeting_id: str,
        expected: frozenset[MeetingState] | None,
        *,
        unlocked: bool = False,
        own_lock: str | None = None,
        stopped_by: str | None = None,
        **update: Any,
    ) -> Meeting | None:
        """Read, check and rewrite one meeting row in a single ``BEGIN IMMEDIATE`` transaction.

        The write lock is taken before the read, so no other process can change the row between
        the check and the write (``append_audit`` uses the same pattern). ``expected=None``
        skips the state check; ``unlocked`` adds the run-lock check and ``stopped_by`` the lock
        note of ``transition``. Returns the stored meeting, or ``None`` when a check failed
        (nothing written); raises ``KeyError`` for an unknown meeting.
        """
        conn = self.conn
        if conn.in_transaction:  # by contract every method commits; this is a failed leftover
            conn.rollback()
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.get_meeting(meeting_id)
            if current is None:
                raise KeyError(meeting_id)
            refused = expected is not None and current.state not in expected
            if refused or (unlocked and self._live_lock(meeting_id, own_lock) is not None):
                conn.rollback()
                return None
            updated = current.model_copy(update=update)
            if stopped_by is not None:
                self._mark_stopped(meeting_id, stopped_by)
            self.save_meeting(updated)  # commits, which ends the transaction
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        return updated

    def save_consent(self, record: ConsentRecord) -> None:
        self._insert(
            "consent",
            replace=True,
            meeting_id=record.meeting_id,
            record_json=json_dumps(record.model_dump(mode="json")),
            recorded_at=iso(record.recorded_at),
        )

    def get_consent(self, meeting_id: str) -> ConsentRecord | None:
        row = self._one("SELECT record_json FROM consent WHERE meeting_id = ?", meeting_id)
        return ConsentRecord.model_validate_json(row["record_json"]) if row else None

    # ------------------------------------------------------------------ minutes / review
    def save_minutes(self, minutes: Minutes) -> int:
        """Store a minutes version and return it.

        The version is ``minutes.version`` when that number is free for the meeting; otherwise
        the next number after the highest stored version, so callers never overwrite history.
        """
        top = self._one(
            "SELECT MAX(version) AS v FROM minutes WHERE meeting_id = ?", minutes.meeting_id
        )
        taken = self._one(
            "SELECT 1 FROM minutes WHERE meeting_id = ? AND version = ?",
            minutes.meeting_id,
            minutes.version,
        )
        version = int(top["v"] or 0) + 1 if taken else minutes.version
        stored = minutes.model_copy(update={"version": version})
        self._insert(
            "minutes",
            meeting_id=stored.meeting_id,
            version=version,
            template=stored.meeting_type.value,
            status=stored.review.status,
            minutes_json=json_dumps(stored.model_dump(mode="json")),
            provenance_json=json_dumps(stored.provenance.model_dump(mode="json")),
            created_at=utcnow_iso(),
        )
        return version

    def latest_minutes(self, meeting_id: str) -> Minutes | None:
        return minutes_from_row(
            self._one(
                "SELECT template, minutes_json FROM minutes WHERE meeting_id = ? "
                "ORDER BY version DESC LIMIT 1",
                meeting_id,
            )
        )

    def get_minutes(self, meeting_id: str, version: int) -> Minutes | None:
        return minutes_from_row(
            self._one(
                "SELECT template, minutes_json FROM minutes WHERE meeting_id = ? AND version = ?",
                meeting_id,
                version,
            )
        )

    def save_review_item(self, meeting_id: str, version: int, item: ReviewItem) -> None:
        self._insert(
            "review_items",
            meeting_id=meeting_id,
            version=version,
            item_id=item.item_id,
            action=item.action,
            reason_code=item.reason_code,
            before=item.before,
            after=item.after,
            by=item.by,
            at=iso(item.at),
        )

    def list_review_items(self, meeting_id: str, version: int) -> list[ReviewItem]:
        rows = self.conn.execute(
            "SELECT * FROM review_items WHERE meeting_id = ? AND version = ? ORDER BY id",
            (meeting_id, version),
        ).fetchall()
        return [
            ReviewItem(
                item_id=r["item_id"],
                action=r["action"],
                reason_code=r["reason_code"],
                before=r["before"],
                after=r["after"],
                by=r["by"],
                at=parse_dt(r["at"]),
            )
            for r in rows
        ]

    def set_review_status(self, minutes: Minutes) -> None:
        """Rewrite the stored JSON and status column for ``minutes`` (same meeting and version)."""
        cur = self.conn.execute(
            "UPDATE minutes SET status = ?, minutes_json = ? WHERE meeting_id = ? AND version = ?",
            (
                minutes.review.status,
                json_dumps(minutes.model_dump(mode="json")),
                minutes.meeting_id,
                minutes.version,
            ),
        )
        if cur.rowcount == 0:
            raise KeyError(f"{minutes.meeting_id} v{minutes.version}")
        self.conn.commit()

    # ------------------------------------------------------------------ audit
    def append_audit(self, event: AuditEvent) -> str:
        """Append a sealed event whose ``prev_hash`` equals ``last_audit_hash()``; return its hash.

        The check and the insert run in one ``BEGIN IMMEDIATE`` transaction, which takes the
        database write lock before the last hash is read, so another process (the review server,
        the retention timer, an ingest run) cannot insert between the two and fork the chain;
        a writer that finds the lock held waits up to the connection's busy timeout (5 s). A
        mismatched ``prev_hash`` raises ``ValueError`` and nothing is stored, so a broken chain
        is never written; ``AuditLog.append`` re-reads the head and retries in that case. Any
        other failure rolls the transaction back and re-raises.
        """
        e = event.sealed()
        conn = self.conn
        if conn.in_transaction:  # by contract every method commits; this is a failed leftover
            conn.rollback()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            last = row["hash"] if row else GENESIS_HASH
            if e.prev_hash != last:
                raise ValueError(
                    "audit chain broken: prev_hash does not match the last stored hash"
                )
            conn.execute(
                "INSERT INTO audit (ts, actor, actor_source, event, meeting_id, classification,"
                " object, detail_json, model, prompt_sha, prev_hash, hash)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    iso(e.ts),
                    e.actor,
                    e.actor_source,
                    e.event,
                    e.meeting_id,
                    e.classification,
                    e.object,
                    json_dumps(e.detail),
                    e.model,
                    e.prompt_sha,
                    e.prev_hash,
                    e.hash,
                ),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return e.hash

    def last_audit_hash(self) -> str:
        """Hash of the last committed audit row, or ``GENESIS_HASH`` when there is none.

        A plain read: it sees every row other processes have committed (WAL), but the value can
        be stale by the time the caller writes, which is why ``append_audit`` re-checks it inside
        its write transaction."""
        row = self._one("SELECT hash FROM audit ORDER BY id DESC LIMIT 1")
        return row["hash"] if row else GENESIS_HASH

    def audit_count(self) -> int:
        """Number of audit rows, for cross-checking the JSONL chain (``audit.chain_report``)."""
        row = self._one("SELECT COUNT(*) AS n FROM audit")
        return int(row["n"]) if row else 0


# Every public method of the store (own and mixed-in) runs under ``self.lock``.
for _name in sorted({n for c in SqliteStore.__mro__ for n in vars(c) if not n.startswith("_")}):
    _member = getattr(SqliteStore, _name)
    if inspect.isfunction(_member) and _name != "close":
        setattr(SqliteStore, _name, _synchronised(_member))
