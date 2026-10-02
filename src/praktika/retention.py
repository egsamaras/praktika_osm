"""Retention timers by classification with legal hold and deletion receipts (C-05, C-14).

Contract: ``plan`` is pure (no side effects) and lists what is due at ``now``; ``run`` executes
a plan: audio files are overwritten with zeros and unlinked *before* the store row is updated,
every deletion is audited as ``retention.deleted``, and a second run at the same instant deletes
nothing. ``legal_hold`` on a meeting blocks every timer. The store supplies one
``RetentionSubject`` per meeting with retained artefacts and records deletions.

Rules (docs/CONTROLS.md): audio is deleted when the minutes are approved or discarded, and in
any case after ``audio_hours``; restricted audio (``audio_hours == 0``) is deleted as soon as
transcription has finished. Transcript and vault go ``transcript_days`` after approval, or at
the hard maximum after creation — anchored on the *first* transcript row of the meeting (or
the ``delete_after`` stored with it, whichever is earlier), so re-saving the transcript for a
speaker mapping never restarts the clock. Unapproved drafts go ``draft_days`` after approval,
discard or the last draft (abandonment).

A file that cannot be wiped (permissions, an unmounted volume) does not stop the run: the
failure is logged, audited as ``retention.failed`` and the remaining deletions proceed; once
they are done ``run`` raises ``RetentionRunError`` naming every failed item so the CLI exits
non-zero and the launchd log shows it, while the row stays live for the next run to retry.

``run`` additionally sweeps ``audio_root`` for WAV files that no live media row knows about
(orphans left by a crash between conversion and registration) once they are older than the
shortest non-zero ``audio_hours``.

The timers are executed by ``praktika retention run``, by every user command on start-up
(``cli.context.sweep_retention``) and by the launchd agent ``praktika retention install`` sets
up, so no artefact outlives its window because nobody typed the command.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from praktika.audit import AuditLog
from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.logging import get_logger
from praktika.models import Classification, Meeting, MeetingState
from praktika.retention_files import orphan_audio, wipe_file

__all__ = [
    "AFTER_TRANSCRIPTION",
    "CLOSED",
    "TRANSCRIPT_MAX_DAYS",
    "AudioArtefact",
    "Deletion",
    "RetentionPolicy",
    "RetentionRunError",
    "RetentionStore",
    "RetentionSubject",
    "orphan_audio",
    "plan",
    "run",
    "wipe_file",
]

log = get_logger(__name__)

TRANSCRIPT_MAX_DAYS: dict[str, int] = {"internal": 60, "confidential": 60, "restricted": 30}
AFTER_TRANSCRIPTION: frozenset[MeetingState] = frozenset(
    {
        MeetingState.drafting,
        MeetingState.draft_ready,
        MeetingState.in_review,
        MeetingState.approved,
        MeetingState.discarded,
        MeetingState.purged,
    }
)
CLOSED: frozenset[MeetingState] = frozenset(
    {MeetingState.approved, MeetingState.discarded, MeetingState.purged}
)
Kind = Literal["audio", "transcript", "draft", "vault"]


class RetentionPolicy(BaseModel):
    """Timers per classification key (``internal`` / ``confidential`` / ``restricted``)."""

    model_config = ConfigDict(extra="forbid")

    audio_hours: dict[str, int]
    transcript_days: dict[str, int]
    draft_days: dict[str, int]
    transcript_max_days: dict[str, int] = TRANSCRIPT_MAX_DAYS

    @classmethod
    def from_settings(cls, settings: Settings) -> RetentionPolicy:
        return cls(
            audio_hours=settings.retention_audio_hours,
            transcript_days=settings.retention_transcript_days,
            draft_days=settings.retention_draft_days,
        )


class Deletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Kind
    meeting_id: str
    path: Path | None
    reason: str


class RetentionRunError(PraktikaError):
    """Raised by ``run`` after the plan has been executed when at least one artefact could
    not be deleted. ``done`` lists the deletions that succeeded, ``failed`` each deletion that
    did not with the OS error text."""

    def __init__(self, done: list[Deletion], failed: list[tuple[Deletion, str]]) -> None:
        self.done, self.failed = done, failed
        items = "; ".join(f"{d.meeting_id} {d.kind} {d.path or ''}: {err}" for d, err in failed)
        super().__init__(f"{len(failed)} retention deletion(s) failed: {items}")


class AudioArtefact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    created_at: datetime


class RetentionSubject(BaseModel):
    """Everything still retained for one meeting, as reported by the store."""

    model_config = ConfigDict(extra="forbid")

    meeting: Meeting
    audio: list[AudioArtefact] = []
    #: creation time of the *earliest* live transcript row (a speaker mapping re-saves the
    #: transcript as a new row; the hard maximum counts from the first one).
    transcript_created_at: datetime | None = None
    #: the deadline stored with the transcript when it was first saved, if any.
    transcript_delete_after: datetime | None = None
    vault_present: bool = False
    approved_at: datetime | None = None
    draft_created_at: datetime | None = None


class RetentionStore(Protocol):
    """The slice of the store the retention job needs (implemented by ``SqliteStore``)."""

    def retention_candidates(self, now: datetime) -> list[RetentionSubject]: ...

    def record_deletion(self, deletion: Deletion, deleted_at: datetime) -> None: ...


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _cls(meeting: Meeting) -> str:
    return Classification(meeting.classification).value


def _audio_due(
    subject: RetentionSubject, art: AudioArtefact, now: datetime, p: RetentionPolicy
) -> str | None:
    m = subject.meeting
    hours = p.audio_hours[_cls(m)]
    if hours == 0:
        if m.state in AFTER_TRANSCRIPTION:
            return "restricted: audio deleted at transcription"
        return None
    if m.state in CLOSED:
        return f"minutes {m.state.value}"
    if now >= _aware(art.created_at) + timedelta(hours=hours):
        return f"hard maximum {hours} h reached"
    return None


def _transcript_due(s: RetentionSubject, now: datetime, p: RetentionPolicy) -> str | None:
    if s.transcript_created_at is None:
        return None
    cls = _cls(s.meeting)
    if s.approved_at is not None:
        days = p.transcript_days[cls]
        if now >= _aware(s.approved_at) + timedelta(days=days):
            return f"approval + {days} days"
    max_days = p.transcript_max_days[cls]
    deadline = _aware(s.transcript_created_at) + timedelta(days=max_days)
    if s.transcript_delete_after is not None:
        deadline = min(deadline, _aware(s.transcript_delete_after))
    if now >= deadline:
        return f"hard maximum {max_days} days reached"
    return None


def _draft_due(s: RetentionSubject, now: datetime, p: RetentionPolicy) -> str | None:
    if s.draft_created_at is None:
        return None
    days = p.draft_days[_cls(s.meeting)]
    anchor = s.approved_at if s.approved_at is not None else s.draft_created_at
    if now >= _aware(anchor) + timedelta(days=days):
        what = "approval" if s.approved_at is not None else s.meeting.state.value
        return f"{what} + {days} days"
    return None


def plan(
    store: RetentionStore, now: datetime, policy: RetentionPolicy | None = None
) -> list[Deletion]:
    """List every artefact due for deletion at ``now``; meetings under legal hold are skipped."""
    p = policy or RetentionPolicy.from_settings(Settings())
    now = _aware(now)
    out: list[Deletion] = []
    for s in sorted(store.retention_candidates(now), key=lambda x: x.meeting.id):
        m = s.meeting
        if m.legal_hold:
            log.info("retention.legal_hold", meeting_id=m.id)
            continue
        for art in s.audio:
            reason = _audio_due(s, art, now, p)
            if reason:
                out.append(Deletion(kind="audio", meeting_id=m.id, path=art.path, reason=reason))
        reason = _transcript_due(s, now, p)
        if reason:
            out.append(Deletion(kind="transcript", meeting_id=m.id, path=None, reason=reason))
            if s.vault_present:
                out.append(Deletion(kind="vault", meeting_id=m.id, path=None, reason=reason))
        reason = _draft_due(s, now, p)
        if reason:
            out.append(Deletion(kind="draft", meeting_id=m.id, path=None, reason=reason))
    return out


def run(
    store: RetentionStore,
    audit: AuditLog,
    now: datetime,
    dry_run: bool = False,
    policy: RetentionPolicy | None = None,
    *,
    audio_root: Path | None = None,
) -> list[Deletion]:
    """Execute ``plan`` (plus the orphan sweep of ``audio_root``): wipe files, update rows,
    audit receipts. ``dry_run`` only plans. Returns the deletions performed; raises
    ``RetentionRunError`` at the end when any file could not be wiped (the other deletions
    are still carried out and the failed rows stay live for the next run)."""
    now = _aware(now)
    p = policy or RetentionPolicy.from_settings(Settings())
    candidates = store.retention_candidates(now)
    live = {a.path.resolve() for s in candidates for a in s.audio}
    deletions = plan(store, now, p) + orphan_audio(audio_root, live, now, p)
    if dry_run:
        return deletions
    classes = {s.meeting.id: _cls(s.meeting) for s in candidates}
    done: list[Deletion] = []
    failed: list[tuple[Deletion, str]] = []
    for d in deletions:
        target = str(d.path) if d.path is not None else d.kind
        try:
            removed = wipe_file(d.path) if d.path is not None else False
        except OSError as exc:
            error = exc.strerror or type(exc).__name__
            failed.append((d, error))
            log.warning("retention.failed", meeting_id=d.meeting_id, kind=d.kind, error=error)
            audit.append(
                "retention.failed",
                d.meeting_id,
                classification=classes.get(d.meeting_id),
                object=target,
                kind=d.kind,
                reason=d.reason,
                error=error,
            )
            continue
        store.record_deletion(d, now)
        audit.append(
            "retention.deleted",
            d.meeting_id,
            classification=classes.get(d.meeting_id),
            object=target,
            kind=d.kind,
            reason=d.reason,
            file_removed=removed,
        )
        log.info("retention.deleted", meeting_id=d.meeting_id, kind=d.kind, reason=d.reason)
        done.append(d)
    if failed:
        raise RetentionRunError(done, failed)
    return done
