"""Read-side queries and the search-index rows of ``SqliteStore``.

Split out of ``store/repo.py`` to keep both files within the house line budget. ``QueryMixin``
expects the host class to provide ``conn``, ``_one``, ``_insert``, ``get_meeting``,
``save_meeting``, ``_update_meeting``, ``get_transcript`` and ``list_review_items``
(``SqliteStore`` does).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from praktika.models import Meeting, MeetingState, ReviewItem, Transcript
from praktika.store.db import utcnow_iso
from praktika.store.records import OpenAction, iso, minutes_from_row


class QueryMixin:
    """Cross-meeting queries, legal hold, DSAR lookup and FTS rows."""

    conn: sqlite3.Connection

    if TYPE_CHECKING:  # provided by SqliteStore / ArtefactMixin; never defined at runtime here

        def _one(self, sql: str, *params: Any) -> sqlite3.Row | None: ...
        def _insert(self, table: str, replace: bool = False, **cols: Any) -> int: ...
        def get_meeting(self, meeting_id: str) -> Meeting | None: ...
        def save_meeting(self, meeting: Meeting) -> None: ...
        def _update_meeting(
            self, meeting_id: str, expected: frozenset[MeetingState] | None, **update: Any
        ) -> Meeting | None: ...
        def get_transcript(self, meeting_id: str) -> Transcript | None: ...
        def list_review_items(self, meeting_id: str, version: int) -> list[ReviewItem]: ...

    def list_meetings(self, filters: dict[str, Any] | None = None) -> list[Meeting]:
        """Meetings newest first. Filters: state, classification, meeting_type, organiser,
        since (datetime, inclusive on ``started_at``)."""
        f = filters or {}
        where = [
            f"{k} = ?"
            for k in ("state", "classification", "meeting_type", "organiser")
            if f.get(k) is not None
        ]
        params = [
            str(getattr(f[k], "value", f[k]))
            for k in ("state", "classification", "meeting_type", "organiser")
            if f.get(k) is not None
        ]
        if f.get("since") is not None:
            where.append("started_at >= ?")
            params.append(iso(f["since"]))
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        sql = f"SELECT meeting_json FROM meetings{clause} ORDER BY started_at DESC"  # noqa: S608
        rows = self.conn.execute(sql, params).fetchall()
        return [Meeting.model_validate_json(r["meeting_json"]) for r in rows]

    def open_actions(self, owner: str | None = None) -> list[OpenAction]:
        """Actions from the latest *approved* minutes of every non-private meeting, minus those
        a reviewer rejected; optionally filtered by owner (case-insensitive). Drafts and
        one-to-one (private) minutes never reach the cross-meeting register."""
        rows = self.conn.execute(
            "SELECT m.meeting_id, m.version, m.template, m.minutes_json, g.title FROM minutes m "
            "JOIN meetings g ON g.id = m.meeting_id WHERE m.version = (SELECT MAX(version) FROM "
            "minutes WHERE meeting_id = m.meeting_id) AND m.status = 'approved' "
            "AND g.private = 0 ORDER BY g.started_at DESC"
        ).fetchall()
        out: list[OpenAction] = []
        want = owner.strip().lower() if owner else None
        for r in rows:
            minutes = minutes_from_row(r)
            rejected = {
                i.item_id
                for i in self.list_review_items(r["meeting_id"], r["version"])
                if i.action == "reject"
            }
            for a in minutes.actions if minutes else []:
                if a.id in rejected or (want and (a.owner or "").strip().lower() != want):
                    continue
                out.append(
                    OpenAction(
                        meeting_id=r["meeting_id"],
                        meeting_title=r["title"],
                        version=r["version"],
                        action=a,
                    )
                )
        return out

    def set_hold(self, meeting_id: str, on: bool, reason: str, by: str) -> None:
        """Set or release a legal hold; mirrored into ``meetings.legal_hold``."""
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            raise KeyError(meeting_id)
        self._insert(
            "holds",
            replace=True,
            meeting_id=meeting_id,
            active=int(on),
            reason=reason,
            set_by=by,
            set_at=utcnow_iso(),
        )
        # Only the hold flag changes, atomically: a whole-row rewrite of the meeting read above
        # could put back a state another process has moved on meanwhile.
        self._update_meeting(meeting_id, None, legal_hold=on)

    def dsar_find(self, participant: str) -> list[str]:
        """Meeting ids in which ``participant`` appears (see ``dsar_matches``)."""
        return [mid for mid, _ in self.dsar_matches(participant)]

    def dsar_matches(self, participant: str) -> list[tuple[str, str]]:
        """``(meeting id, why)`` for every meeting whose organiser, roster (name, alias or UPN),
        transcript speakers or title match ``participant`` case-insensitively; ``why`` is
        ``organiser``, ``roster``, ``speaker`` or ``title``, the first that matched. Roster names
        match as substrings; aliases and UPNs exactly (the organiser is a UPN, so a meeting
        declared on someone's behalf is found too); a title only for a full name, as whole words
        (``search.mentions``), since a title often names the people the meeting is about."""
        from praktika.store.search import mentions

        needle = participant.strip().lower()
        found: list[tuple[str, str]] = []
        for meeting in self.list_meetings() if needle else []:
            why = ""
            if meeting.organiser.lower() == needle:
                why = "organiser"
            elif any(
                needle in a.name.lower()
                or needle in {x.lower() for x in a.aliases}
                or (a.upn or "").lower() == needle
                for a in meeting.roster
            ):
                why = "roster"
            else:
                t = self.get_transcript(meeting.id)
                if t is not None and any(needle == s.speaker.lower() for s in t.segments):
                    why = "speaker"
                elif mentions(meeting.title, participant):
                    why = "title"
            if why:
                found.append((meeting.id, why))
        return found

    # ------------------------------------------------------------------ search index
    def index_minutes(
        self, meeting_id: str, version: int, title: str, summary: str, body: str
    ) -> None:
        self.clear_index(meeting_id)
        self._insert(
            "minutes_fts",
            meeting_id=meeting_id,
            version=version,
            title=title,
            summary=summary,
            body=body,
        )

    def clear_index(self, meeting_id: str) -> None:
        self.conn.execute("DELETE FROM minutes_fts WHERE meeting_id = ?", (meeting_id,))
        self.conn.commit()

    def search_index(self, match: str, *, include_private: bool, limit: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT f.meeting_id, f.version, f.title, snippet(minutes_fts, -1, '[', ']', '…', 12) "
            "AS snippet, bm25(minutes_fts) AS rank FROM minutes_fts f JOIN meetings g ON "
            "g.id = f.meeting_id WHERE minutes_fts MATCH ? AND (? = 1 OR g.private = 0) "
            "ORDER BY rank LIMIT ?",
            (match, int(include_private), limit),
        ).fetchall()
        return [dict(r) for r in rows]
