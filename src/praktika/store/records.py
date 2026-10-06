"""Row records and the ``Store`` seam.

Split out of ``repo.py`` to keep that file within the house line budget. ``Store`` is the
Protocol every persistence backend implements; ``SqliteStore`` (``store/repo.py``) is the only
implementation. A ``PostgresStore`` for multi-user service deployments does not exist yet.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict

from praktika.models import (
    ActionItem,
    AuditEvent,
    ConsentRecord,
    LanguageMode,
    MancomMinutes,
    Meeting,
    MeetingState,
    Minutes,
    OneToOneMinutes,
    ReviewItem,
    Transcript,
)

if TYPE_CHECKING:
    from praktika.retention import Deletion, RetentionSubject
    from praktika.store.locks import RunLock

MINUTES_CLASSES: dict[str, type[Minutes]] = {
    "general": Minutes,
    "mancom": MancomMinutes,
    "one_to_one": OneToOneMinutes,
}


def iso(value: datetime | str | None) -> str | None:
    """ISO-8601 text for a datetime (``None`` passes through; strings are returned as-is)."""
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def parse_dt(value: str | None) -> datetime | None:
    """Inverse of ``iso``: ``None`` or empty text gives ``None``."""
    return datetime.fromisoformat(value) if value else None


def minutes_from_row(row: sqlite3.Row | None) -> Minutes | None:
    """Rebuild the right ``Minutes`` subclass from a row with ``template`` and ``minutes_json``."""
    if row is None:
        return None
    return MINUTES_CLASSES.get(row["template"], Minutes).model_validate_json(row["minutes_json"])


class MediaRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    meeting_id: str
    kind: str
    path: Path
    sha256: str
    delete_after: datetime | None
    deleted_at: datetime | None


class OpenAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meeting_id: str
    meeting_title: str
    version: int
    action: ActionItem


class Store(Protocol):
    """The persistence seam. Every method is synchronous and commits before returning.

    ``retention_candidates`` / ``record_deletion`` are the ``praktika.retention.RetentionStore``
    seam; ``purge_meeting`` is the DSAR erasure route; the ``*run_lock*`` methods are the
    per-meeting run locks (``store/locks.py``).
    """

    def save_meeting(self, meeting: Meeting) -> None: ...
    def get_meeting(self, meeting_id: str) -> Meeting | None: ...
    def set_state(self, meeting_id: str, state: MeetingState) -> None: ...
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
    ) -> bool: ...
    def save_consent(self, record: ConsentRecord) -> None: ...
    def get_consent(self, meeting_id: str) -> ConsentRecord | None: ...
    def save_media(
        self, meeting_id: str, path: Path, sha256: str, *, kind: str, delete_after: datetime | None
    ) -> int: ...
    def list_media(self, meeting_id: str) -> list[MediaRecord]: ...
    def mark_deleted(self, kind: str, row_id: int, when: datetime) -> None: ...
    def save_transcript(
        self, transcript: Transcript, *, delete_after: datetime | None, vault: bytes | None = None
    ) -> int: ...
    def get_transcript(self, meeting_id: str) -> Transcript | None: ...
    def transcript_delete_after(self, meeting_id: str) -> datetime | None: ...
    def transcript_with_sha256(self, meeting_id: str, sha256: str) -> Transcript | None: ...
    def erase_transcript(self, row_id: int, when: datetime) -> None: ...
    def delete_minutes_version(self, meeting_id: str, version: int) -> None: ...
    def delete_vault(self, meeting_id: str) -> None: ...
    def save_vault(self, meeting_id: str, blob: bytes) -> None: ...
    def get_vault(self, meeting_id: str) -> bytes | None: ...
    def save_minutes(self, minutes: Minutes) -> int: ...
    def latest_minutes(self, meeting_id: str) -> Minutes | None: ...
    def get_minutes(self, meeting_id: str, version: int) -> Minutes | None: ...
    def save_review_item(self, meeting_id: str, version: int, item: ReviewItem) -> None: ...
    def list_review_items(self, meeting_id: str, version: int) -> list[ReviewItem]: ...
    def set_review_status(self, minutes: Minutes) -> None: ...
    def append_audit(self, event: AuditEvent) -> str: ...
    def last_audit_hash(self) -> str: ...
    def list_meetings(self, filters: dict[str, Any] | None = None) -> list[Meeting]: ...
    def open_actions(self, owner: str | None = None) -> list[OpenAction]: ...
    def retention_candidates(self, now: datetime) -> list[RetentionSubject]: ...
    def record_deletion(self, deletion: Deletion, deleted_at: datetime) -> None: ...
    def purge_meeting(self, meeting_id: str, when: datetime) -> int: ...
    def set_hold(self, meeting_id: str, on: bool, reason: str, by: str) -> None: ...
    def dsar_find(self, participant: str) -> list[str]: ...
    def dsar_matches(self, participant: str) -> list[tuple[str, str]]: ...
    def index_minutes(
        self, meeting_id: str, version: int, title: str, summary: str, body: str
    ) -> None: ...
    def clear_index(self, meeting_id: str) -> None: ...
    def search_index(self, match: str, *, include_private: bool, limit: int) -> list[dict]: ...
    def take_run_lock(self, lock: RunLock) -> RunLock | None: ...
    def release_run_lock(self, meeting_id: str, token: str) -> None: ...
    def run_lock(self, meeting_id: str) -> RunLock | None: ...
    def live_run_lock(self, meeting_id: str, *, exempt: str | None = None) -> RunLock | None: ...
    def mark_run_reopened(self, meeting_id: str, token: str, state: MeetingState) -> None: ...
