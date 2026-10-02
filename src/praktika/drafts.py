"""Which draft may be approved as the record of a meeting.

Shared by ``praktika approve`` and the review page's approve route, so both refuse the same
drafts: one approved while a run is still working on the meeting (its run lock is held, or the
meeting is at a running state), one built from an older transcript than the latest (after
``praktika transcribe``, until ``praktika generate`` has run), and one whose transcript can no
longer be checked because it is not stored any more.
"""

from __future__ import annotations

from typing import Any

from praktika.models import Meeting, MeetingState, Minutes, Transcript
from praktika.store.records import Store

#: States in which a capture, transcription or drafting run owns the meeting.
RUNNING_STATES = (MeetingState.capturing, MeetingState.transcribing, MeetingState.drafting)


def _spoken(transcript: Transcript) -> list[tuple[Any, ...]]:
    """What was said and when, without the speaker names a reviewer may map afterwards."""
    return [(s.id, s.start, s.end, s.text) for s in transcript.segments]


def stale_draft_reason(store: Store, meeting: Meeting, minutes: Minutes) -> str | None:
    """Why ``minutes`` cannot be approved as the record of ``meeting``, or ``None``.

    Refused while a run holds the meeting's lock, while the meeting is at ``capturing``,
    ``transcribing`` (a new transcript awaits ``praktika generate``) or ``drafting``, when no
    transcript is stored or the one the draft was built from is gone (nothing to check the
    draft against), and whenever the latest transcript says something other than the one the
    draft was built from (a speaker mapping alone does not count). Only the state and the
    stored rows are read.
    """
    mid, version = meeting.id, minutes.version
    lock = store.live_run_lock(mid)
    if lock is not None:
        return f"{mid}: {lock.busy}; wait for it to finish, then review the draft it leaves"
    if meeting.state is MeetingState.transcribing:
        return (
            f"{mid} has a newer transcript than draft v{version}, or its transcription was "
            f"interrupted; run `praktika generate {mid}` and review the new draft before approving "
            f"(if no transcript is stored, `praktika abort {mid}` discards it)"
        )
    if meeting.state in RUNNING_STATES:  # no live lock: the run was killed or the host restarted
        return (
            f"{mid} is {meeting.state.value}, but no run is working on it (the run was stopped "
            f"before it finished); run `praktika generate {mid}` to draft it again, or "
            f"`praktika abort {mid}` to discard it"
        )
    latest = store.get_transcript(mid)
    if latest is None:
        return (
            f"no transcript of {mid} is stored, so draft v{version} cannot be checked against "
            "what was said; it cannot be approved"
        )
    sha = minutes.provenance.transcript_sha256
    if latest.sha256() == sha:
        return None
    source = store.transcript_with_sha256(mid, sha)
    if source is None:
        return (
            f"the transcript draft v{version} of {mid} was built from is no longer stored; run "
            f"`praktika generate {mid}` and review the new draft before approving"
        )
    if _spoken(source) != _spoken(latest):
        return (
            f"draft v{version} of {mid} was built from an older transcript than the latest "
            f"one; run `praktika generate {mid}` and review the new draft before approving"
        )
    return None
