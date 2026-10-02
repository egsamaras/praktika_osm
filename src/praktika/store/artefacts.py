"""Artefact rows of ``SqliteStore``: media, transcripts, vault, retention seam, DSAR purge.

``ArtefactMixin`` holds everything that is subject to a retention timer plus the two methods
of the ``retention.RetentionStore`` seam (one ``RetentionSubject`` per meeting with retained
artefacts, and ``record_deletion``) and ``purge_meeting`` for the DSAR route. Deleting here
means erasing content: audio files are wiped by the caller (``retention.run``) or by
``purge_meeting`` itself, transcript rows keep their hash but lose their segments, drafts and
their review items are removed, the vault row is dropped and the search index row is cleared.
Approved minutes are the record and are only removed by ``purge_meeting``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from praktika.errors import PraktikaError
from praktika.models import Meeting, MeetingState, Minutes, Transcript
from praktika.retention import AudioArtefact, Deletion, RetentionSubject, wipe_file
from praktika.store.db import json_dumps, utcnow_iso
from praktika.store.records import MediaRecord, iso, parse_dt

_DELETABLE = {"media": "media", "transcript": "transcripts"}


class ArtefactMixin:
    """Media, transcript and vault rows; retention subjects, receipts and DSAR purge."""

    conn: sqlite3.Connection

    if TYPE_CHECKING:  # provided by SqliteStore / QueryMixin; never defined at runtime here

        def _one(self, sql: str, *params: Any) -> sqlite3.Row | None: ...
        def _insert(self, table: str, replace: bool = False, **cols: Any) -> int: ...
        def list_meetings(self, filters: dict[str, Any] | None = None) -> list[Meeting]: ...
        def latest_minutes(self, meeting_id: str) -> Minutes | None: ...
        def clear_index(self, meeting_id: str) -> None: ...
        def set_state(self, meeting_id: str, state: MeetingState) -> None: ...
        def _mark_stopped(self, meeting_id: str, by: str) -> None: ...

    # ------------------------------------------------------------------ media / transcripts
    def save_media(
        self, meeting_id: str, path: Path, sha256: str, *, kind: str, delete_after: datetime | None
    ) -> int:
        return self._insert(
            "media",
            meeting_id=meeting_id,
            kind=kind,
            path=str(path),
            sha256=sha256,
            created_at=utcnow_iso(),
            delete_after=iso(delete_after),
        )

    def list_media(self, meeting_id: str) -> list[MediaRecord]:
        rows = self.conn.execute(
            "SELECT * FROM media WHERE meeting_id = ? ORDER BY id", (meeting_id,)
        ).fetchall()
        return [
            MediaRecord(
                id=r["id"],
                meeting_id=r["meeting_id"],
                kind=r["kind"],
                path=Path(r["path"]),
                sha256=r["sha256"],
                delete_after=parse_dt(r["delete_after"]),
                deleted_at=parse_dt(r["deleted_at"]),
            )
            for r in rows
        ]

    def mark_deleted(self, kind: str, row_id: int, when: datetime) -> None:
        """Record deletion of a ``media`` or ``transcript`` row (callers unlink the file first)."""
        table = _DELETABLE[kind]
        sql = f"UPDATE {table} SET deleted_at = ? WHERE id = ?"  # noqa: S608 - fixed names
        self.conn.execute(sql, (iso(when), row_id))
        self.conn.commit()

    def save_transcript(
        self, transcript: Transcript, *, delete_after: datetime | None, vault: bytes | None = None
    ) -> int:
        """Store a transcript row and return its id. With ``vault``, the meeting's encrypted
        vault is replaced in the same transaction, so a failure (or Ctrl-C) between the two
        writes can never leave a vault that belongs to a transcript that was not stored, nor
        a redacted transcript without the vault that restores it."""
        cols = {
            "meeting_id": transcript.meeting_id,
            "source": transcript.source,
            "engines_json": json_dumps(transcript.engines),
            "segments_json": json_dumps([s.model_dump(mode="json") for s in transcript.segments]),
            "sha256": transcript.sha256(),
            "redacted": int(transcript.redacted),
            "created_at": utcnow_iso(),
            "delete_after": iso(delete_after),
        }
        if vault is None:
            return self._insert("transcripts", **cols)
        conn = self.conn
        if conn.in_transaction:  # by contract every method commits; this is a failed leftover
            conn.rollback()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO vault (meeting_id, blob, updated_at) VALUES (?, ?, ?)",
                (transcript.meeting_id, vault, cols["created_at"]),
            )
            row = self._insert_transcript_row(cols)
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        return row

    def _insert_transcript_row(self, cols: dict[str, Any]) -> int:
        """The transcript insert inside ``save_transcript``'s transaction (no commit)."""
        names = ",".join(cols)
        marks = ",".join("?" for _ in cols)
        sql = f"INSERT INTO transcripts ({names}) VALUES ({marks})"  # noqa: S608 - fixed names
        return int(self.conn.execute(sql, tuple(cols.values())).lastrowid or 0)

    def get_transcript(self, meeting_id: str) -> Transcript | None:
        """The newest transcript that has not been deleted, or ``None``."""
        row = self._one(
            "SELECT * FROM transcripts WHERE meeting_id = ? AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            meeting_id,
        )
        return self._transcript_from_row(row) if row is not None else None

    @staticmethod
    def _transcript_from_row(row: sqlite3.Row) -> Transcript:
        return Transcript(
            meeting_id=row["meeting_id"],
            source=row["source"],
            engines=json.loads(row["engines_json"]),
            segments=json.loads(row["segments_json"]),
            redacted=bool(row["redacted"]),
        )

    def transcript_delete_after(self, meeting_id: str) -> datetime | None:
        """The retention deadline of the newest live transcript (``None`` if none or unset)."""
        row = self._one(
            "SELECT delete_after FROM transcripts WHERE meeting_id = ? AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            meeting_id,
        )
        return parse_dt(row["delete_after"]) if row else None

    def transcript_with_sha256(self, meeting_id: str, sha256: str) -> Transcript | None:
        """The newest live transcript row of the meeting whose stored hash is ``sha256`` (the
        transcript a minutes version names in its provenance), or ``None``."""
        row = self._one(
            "SELECT * FROM transcripts WHERE meeting_id = ? AND sha256 = ? AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            meeting_id,
            sha256,
        )
        return self._transcript_from_row(row) if row is not None else None

    def erase_transcript(self, row_id: int, when: datetime) -> None:
        """Erase one transcript row (its hash stays, its segments go): a run that stopped for
        an abort removes the transcript it had stored, and only that one."""
        self.conn.execute(
            "UPDATE transcripts SET deleted_at = ?, segments_json = '[]' "
            "WHERE id = ? AND deleted_at IS NULL",
            (iso(when), row_id),
        )
        self.conn.commit()

    def delete_minutes_version(self, meeting_id: str, version: int) -> None:
        """Delete one unapproved minutes version and its review items (a run that stopped for
        an abort removes the draft it had stored); an approved version is never touched."""
        self._delete_minutes(meeting_id, f"version = {int(version)} AND status != 'approved'")
        self.conn.commit()

    def delete_vault(self, meeting_id: str) -> None:
        self.conn.execute("DELETE FROM vault WHERE meeting_id = ?", (meeting_id,))
        self.conn.commit()

    def save_vault(self, meeting_id: str, blob: bytes) -> None:
        self._insert(
            "vault", replace=True, meeting_id=meeting_id, blob=blob, updated_at=utcnow_iso()
        )

    def get_vault(self, meeting_id: str) -> bytes | None:
        row = self._one("SELECT blob FROM vault WHERE meeting_id = ?", meeting_id)
        return bytes(row["blob"]) if row else None

    # ------------------------------------------------------------------ retention seam
    def retention_candidates(self, now: datetime) -> list[RetentionSubject]:
        """One subject per meeting that still holds audio, a transcript, a vault or a draft.

        Meetings under legal hold are included: ``retention.plan`` skips them and logs the
        hold, so the receipt trail shows what was withheld. ``now`` is accepted for the seam
        and unused; timers are computed by the retention module.
        """
        out: list[RetentionSubject] = []
        for meeting in self.list_meetings():
            audio = [
                AudioArtefact(path=Path(r["path"]), created_at=parse_dt(r["created_at"]))
                for r in self.conn.execute(
                    "SELECT path, created_at FROM media WHERE meeting_id = ? "
                    "AND deleted_at IS NULL ORDER BY id",
                    (meeting.id,),
                ).fetchall()
            ]
            # The earliest live row anchors the hard maximum (C-05): a speaker mapping or a
            # re-transcription inserts a newer row and must not restart the clock. The stored
            # deadline is the one written with that first row.
            t = self._one(
                "SELECT created_at, delete_after FROM transcripts WHERE meeting_id = ? "
                "AND deleted_at IS NULL ORDER BY id ASC LIMIT 1",
                meeting.id,
            )
            d = self._one(
                "SELECT MAX(created_at) AS c FROM minutes WHERE meeting_id = ? "
                "AND status != 'approved'",
                meeting.id,
            )
            draft_at = parse_dt(d["c"]) if d is not None else None
            latest = self.latest_minutes(meeting.id)
            approved_at = (
                latest.review.reviewed_at
                if latest is not None and latest.review.status == "approved"
                else None
            )
            vault_present = self.get_vault(meeting.id) is not None
            if not audio and t is None and draft_at is None and not vault_present:
                continue
            out.append(
                RetentionSubject(
                    meeting=meeting,
                    audio=audio,
                    transcript_created_at=parse_dt(t["created_at"]) if t else None,
                    transcript_delete_after=parse_dt(t["delete_after"]) if t else None,
                    vault_present=vault_present,
                    approved_at=approved_at,
                    draft_created_at=draft_at,
                )
            )
        return out

    def record_deletion(self, deletion: Deletion, deleted_at: datetime) -> None:
        """Retire the rows behind one ``Deletion`` (the caller has already wiped any file)."""
        when, mid = iso(deleted_at), deletion.meeting_id
        if deletion.kind == "audio":
            self.conn.execute(
                "UPDATE media SET deleted_at = ? WHERE meeting_id = ? AND path = ? "
                "AND deleted_at IS NULL",
                (when, mid, str(deletion.path)),
            )
        elif deletion.kind == "transcript":
            self._erase_transcripts(mid, when)
        elif deletion.kind == "vault":
            self.conn.execute("DELETE FROM vault WHERE meeting_id = ?", (mid,))
        elif deletion.kind == "draft":
            self._delete_minutes(mid, "status != 'approved'")
            if self._one("SELECT 1 FROM minutes WHERE meeting_id = ?", mid) is None:
                self.clear_index(mid)
        self.conn.commit()

    def _erase_transcripts(self, meeting_id: str, when: str | None) -> None:
        self.conn.execute(
            "UPDATE transcripts SET deleted_at = ?, segments_json = '[]' "
            "WHERE meeting_id = ? AND deleted_at IS NULL",
            (when, meeting_id),
        )

    def _delete_minutes(self, meeting_id: str, where: str) -> None:
        """Delete minutes rows matching ``where`` and the review items of those versions."""
        self.conn.execute(
            f"DELETE FROM review_items WHERE meeting_id = ? AND version IN "  # noqa: S608
            f"(SELECT version FROM minutes WHERE meeting_id = ? AND {where})",
            (meeting_id, meeting_id),
        )
        sql = f"DELETE FROM minutes WHERE meeting_id = ? AND {where}"  # noqa: S608 - fixed
        self.conn.execute(sql, (meeting_id,))

    # ------------------------------------------------------------------ DSAR erasure
    def purge_meeting(self, meeting_id: str, when: datetime) -> int:
        """Erase every artefact of a meeting and mark it ``purged``; return files removed.

        Audio files are zero-overwritten and unlinked before their rows are retired; the
        transcript, vault, all minutes versions (approved included), review items and the
        search-index row go too. The meeting row, consent record and audit trail remain as the
        proof that the erasure happened. Raises ``KeyError`` for an unknown meeting and
        ``PraktikaError`` when the meeting is under legal hold.
        """
        row = self._one("SELECT legal_hold FROM meetings WHERE id = ?", meeting_id)
        if row is None:
            raise KeyError(meeting_id)
        if row["legal_hold"]:
            raise PraktikaError(f"{meeting_id} is under legal hold; deletion refused")
        self._mark_stopped(meeting_id, "purge")  # a run working on it stops and says why
        removed = 0
        for m in self.list_media(meeting_id):
            if m.deleted_at is None:
                removed += int(wipe_file(m.path))
                self.mark_deleted("media", m.id, when)
        self._erase_transcripts(meeting_id, iso(when))
        self.conn.execute("DELETE FROM vault WHERE meeting_id = ?", (meeting_id,))
        self._delete_minutes(meeting_id, "1 = 1")
        self.conn.commit()
        self.clear_index(meeting_id)
        self.set_state(meeting_id, MeetingState.purged)
        return removed
