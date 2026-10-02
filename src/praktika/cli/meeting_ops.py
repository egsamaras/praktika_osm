"""Meeting re-run commands: ``abort``, ``transcribe``, ``generate``.

Split from ``meetings`` (which holds ``start`` and ``ingest``) to keep both within the house
line budget. ``abort`` is the kill switch (C-04): it validates the meeting id before touching
the audio directory, refuses a meeting that does not exist, is approved or purged, or is under
legal hold (unless it is being captured), then discards the meeting first (so a run working on
it stops at its next step and removes what it had stored), signals a running capture,
overwrites and unlinks every WAV and audits it; it is also how an operator removes a meeting a
failed ingest left at ``created``. It never waits for a run's lock: an abort always wins. An
abort during a ``generate --reopen`` of approved minutes stops that run and puts the meeting
back to approved instead. ``transcribe`` and ``generate`` take the meeting's run lock first, so
a second run on the same meeting is refused with the name of the command that holds it;
``transcribe`` leaves the meeting at ``transcribing`` ("new transcript, awaiting
``generate``"); ``generate`` refuses closed meetings unless ``--reopen`` is given (audited).
"""

from __future__ import annotations

import os
import re
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from praktika.audio.capture import purge_file
from praktika.cli import context as ctx
from praktika.cli import steps
from praktika.errors import PraktikaError
from praktika.ingest.audio_file import build_diarizer
from praktika.logging import get_logger
from praktika.models import LanguageMode, Meeting, MeetingState, MeetingType
from praktika.server_support import CLOSED_STATES
from praktika.store.locks import holder_alive
from praktika.stt.router import require_language

log = get_logger(__name__)

PID_FILE = "capture.pid"
MEETING_ID_RE = re.compile(r"^M-\d{8}-[0-9a-f]{4}$")
#: ``abort`` discards a meeting in any state but these (the approved record stays approved).
ABORTABLE = frozenset(MeetingState) - {MeetingState.approved, MeetingState.purged}


def abort_refusal(meeting: Meeting) -> str | None:
    """Why ``abort`` leaves ``meeting`` exactly as it is, or ``None`` when it may proceed."""
    mid = meeting.id
    if meeting.state is MeetingState.approved:
        return f"{mid} is approved; abort never discards approved minutes, so nothing was changed"
    if meeting.state is MeetingState.purged:
        return f"{mid} is already purged; there is nothing to abort"
    if meeting.legal_hold and meeting.state is not MeetingState.capturing:
        return (
            f"{mid} is under legal hold; abort would overwrite its retained audio, so nothing "
            f"was changed (the hold must be released first: `praktika hold clear {mid}`)"
        )
    return None


def _reopened_approved(rt: ctx.Runtime, meeting: Meeting) -> bool:
    """Whether a ``generate --reopen`` run took this meeting from approved and is still
    re-drafting it (its process alive, or no new draft stored yet): an abort then stops the
    run and puts the approved record back instead of discarding it."""
    lock = rt.store.run_lock(meeting.id)
    if lock is None or lock.reopened_from is not MeetingState.approved:
        return False
    latest = rt.store.latest_minutes(meeting.id)
    return holder_alive(lock) or (latest is not None and latest.review.status == "approved")


def _discard_or_restore(rt: ctx.Runtime, meeting: Meeting) -> MeetingState:
    """Move the meeting for an abort, recording ``abort`` on a run's lock in the same
    transaction; returns the state it moved to. Raises ``PraktikaError`` when it can no longer
    be aborted (it was approved or purged meanwhile)."""
    mid = meeting.id
    if meeting.state is MeetingState.drafting and _reopened_approved(rt, meeting):
        back = MeetingState.approved
        if rt.store.transition(mid, back, expected=(MeetingState.drafting,), stopped_by="abort"):
            return back
    if rt.store.transition(mid, MeetingState.discarded, expected=ABORTABLE, stopped_by="abort"):
        return MeetingState.discarded
    now = rt.require_meeting(mid)
    raise PraktikaError(abort_refusal(now) or f"{mid} is {now.state.value}; nothing was changed")


@ctx.guarded
def abort(meeting_id: Annotated[str, typer.Argument(help="Meeting id.")]) -> None:
    """Kill switch: stop a running capture or run, overwrite and unlink its audio, audit it.

    Refused, with nothing changed, for a meeting that does not exist, is approved or purged,
    or is under legal hold (unless it is being captured). A run working on the meeting stops
    at its next step and removes what it had stored. During a `generate --reopen` of approved
    minutes, the run is stopped and the meeting goes back to approved.
    """
    if not MEETING_ID_RE.match(meeting_id):
        raise PraktikaError(f"{meeting_id!r} is not a meeting id (M-YYYYMMDD-xxxx)")
    rt = ctx.open_runtime()
    meeting = rt.store.get_meeting(meeting_id)
    if meeting is None:
        raise PraktikaError(f"no meeting {meeting_id} in this data directory; nothing was aborted")
    refused = abort_refusal(meeting)
    if refused is not None:
        raise PraktikaError(refused)
    state = _discard_or_restore(rt, meeting)  # first, so a run stops at its next step
    out_dir = Path(rt.settings.data_dir) / "audio" / meeting_id
    pid_file = out_dir / PID_FILE
    if pid_file.is_file():
        try:
            os.kill(int(pid_file.read_text(encoding="utf-8").strip()), signal.SIGTERM)
            time.sleep(1.0)
        except (ValueError, ProcessLookupError, PermissionError) as exc:
            log.warning("abort.signal_failed", meeting_id=meeting_id, error=str(exc))
        pid_file.unlink(missing_ok=True)
    removed = 0
    for wav in sorted(out_dir.glob("*.wav")) if out_dir.is_dir() else []:
        purge_file(wav)
        removed += 1
    for m in rt.store.list_media(meeting_id):
        if m.deleted_at is None:
            purge_file(m.path)
            rt.store.mark_deleted("media", m.id, datetime.now(UTC))
    if state is MeetingState.discarded:
        rt.store.clear_index(meeting_id)  # discarded content is never searchable
    rt.audit.append(
        "capture.aborted",
        meeting_id,
        classification=meeting.classification.value,
        files_removed=removed,
        **({"restored": state.value} if state is MeetingState.approved else {}),
    )
    if state is MeetingState.approved:
        ctx.console.print(
            f"Aborted the re-draft of {meeting_id}: the meeting is back to approved with its "
            f"approved minutes; {removed} audio file(s) overwritten and removed."
        )
        return
    ctx.console.print(f"Aborted {meeting_id}: {removed} audio file(s) overwritten and removed.")


@ctx.guarded
def transcribe(
    meeting_id: Annotated[str, typer.Argument(help="Meeting id.")],
    lang: Annotated[LanguageMode | None, typer.Option("--lang")] = None,
    diarize: Annotated[bool | None, typer.Option("--diarize/--no-diarize")] = None,
) -> None:
    """Re-run STT (and optionally diarisation) over the meeting's retained audio.

    The language mode it will run under (``--lang``, else the stored one) is checked before
    anything changes, so a meeting stored as ``ar-mixed`` while the Arabic path is off is
    refused with its audio and state untouched. A run that succeeds leaves the meeting at
    ``transcribing`` until ``praktika generate`` drafts from the new transcript (``approve``
    refuses the older draft meanwhile). A run that fails keeps the retained audio (it belongs
    to the media rows) and puts the meeting's state and language mode back. Refused for an
    approved, discarded or purged meeting, and while another run works on the meeting; an
    abort during the run wins.
    """
    rt = ctx.open_runtime()
    meeting = rt.require_meeting(meeting_id)
    if meeting.state in CLOSED_STATES:
        raise PraktikaError(f"{meeting_id} is {meeting.state.value}; nothing to transcribe")
    mode = lang if lang is not None else meeting.language_mode
    require_language(rt.settings, mode)
    with steps.run_lock(rt, meeting, "transcribe"):
        meeting = rt.require_meeting(meeting_id)  # as it is now that no other run can move it
        if meeting.state in CLOSED_STATES:
            raise PraktikaError(f"{meeting_id} is {meeting.state.value}; nothing to transcribe")
        meeting = meeting.model_copy(update={"language_mode": mode})
        media = [m for m in rt.store.list_media(meeting_id) if m.deleted_at is None]
        if not media:
            raise PraktikaError(f"no retained audio for {meeting_id}; it may have been deleted")
        settings = rt.settings
        if diarize is not None:
            settings = settings.model_copy(update={"diarize": diarize})
        diarizer = build_diarizer(settings)
        tracks = {m.kind if m.kind in ("mic", "system") else "file": m.path for m in media}
        with steps.honouring_abort(rt, meeting) as stored, steps.transcribing(rt, meeting):
            result = steps.transcribe_audio(  # the media rows keep their original delete_after
                rt,
                meeting,
                stored,
                tracks,
                settings=settings,
                diarizer=diarizer,
                register=False,
            )
            redacted = steps.redact_and_store(rt, meeting, result.transcript, stored=stored)
    ctx.console.print(
        f"Transcribed {meeting_id}: {len(redacted.segments)} segments stored. "
        f"Run `praktika generate {meeting_id}` to draft from them; until then the meeting "
        "stays at transcribing and approval is refused."
    )


@ctx.guarded
def generate(
    meeting_id: Annotated[str, typer.Argument(help="Meeting id.")],
    template: Annotated[MeetingType | None, typer.Option("--template")] = None,
    prompt_version: Annotated[str | None, typer.Option("--prompt-version")] = None,
    reopen: Annotated[
        bool, typer.Option("--reopen", help="Draft again after approval/discard (audited).")
    ] = False,
) -> None:
    """Re-draft the minutes as a new version from the stored redacted transcript.

    Refused once the meeting is approved, discarded or purged unless ``--reopen`` is given;
    reopening is audited and returns the meeting to ``draft_ready`` with an unapproved draft.
    Refused while another run works on the meeting. An abort while it re-drafts approved
    minutes stops it and leaves the meeting approved.
    """
    rt = ctx.open_runtime()
    meeting = rt.require_meeting(meeting_id)
    if meeting.state in CLOSED_STATES and not reopen:
        raise PraktikaError(
            f"{meeting_id} is {meeting.state.value}; pass --reopen to draft a new version"
        )
    with steps.run_lock(rt, meeting, "generate") as lock:
        meeting = rt.require_meeting(meeting_id)  # as it is now that no other run can move it
        if meeting.state in CLOSED_STATES and not reopen:
            raise PraktikaError(
                f"{meeting_id} is {meeting.state.value}; pass --reopen to draft a new version"
            )
        transcript = rt.store.get_transcript(meeting_id)
        if transcript is None:
            raise PraktikaError(f"no transcript stored for {meeting_id}")
        if meeting.state in CLOSED_STATES:
            if meeting.state is MeetingState.approved:
                rt.store.mark_run_reopened(meeting_id, lock.token, meeting.state)
            rt.audit.append(
                "review.reopened",
                meeting_id,
                classification=meeting.classification.value,
                previous_state=meeting.state.value,
            )
        with steps.honouring_abort(rt, meeting) as stored:
            minutes = steps.draft_minutes(
                rt,
                meeting,
                transcript,
                expected=meeting.state,
                template=template,
                prompt_version=prompt_version,
                stored=stored,
            )
    steps.announce_draft(rt, meeting, minutes)
