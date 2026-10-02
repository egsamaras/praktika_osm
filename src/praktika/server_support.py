"""Shared pieces of the review server: request models, pure helpers and ``ReviewContext``.

``ReviewContext`` bundles the collaborators every route needs (settings, store, identity
provider, audit log, optional LLM client, clock) with the small lookups that turn a missing row
into the right HTTP status. Every state change a write route makes is a compare-and-set
(``claim``): it moves the meeting only from the state the request read and only while no run
holds the meeting's lock, and answers 409 otherwise, before anything is written. Route modules
(``server_review``, ``server_export``) register their handlers against it;
``server.create_app`` assembles the application.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from praktika.audit import AuditLog
from praktika.config import Settings
from praktika.identity import Identity, IdentityProvider
from praktika.llm import pipeline
from praktika.logging import get_logger
from praktika.redact.tokenise import Tokeniser, TokenVault, tokens_in
from praktika.server_auth import LOOPBACK, AccessDeniedError, require_access, require_roles

__all__ = [
    "CLOSED_STATES",
    "OPEN_STATES",
    "ITEM_LISTS",
    "ITEM_MODELS",
    "LOOPBACK",
    "MAX_SLICE_S",
    "REASON_CODES",
    "STATIC_DIR",
    "ReasonRequest",
    "RegenerateRequest",
    "ReviewContext",
    "ReviewItemRequest",
    "item_kind",
    "locate_item",
    "refuse_identifiers",
    "rename_ref_speakers",
    "sorted_flags",
    "wav_slice",
]
from praktika.llm.base import LLMClient
from praktika.models import (
    ActionItem,
    Attendee,
    Decision,
    Meeting,
    MeetingState,
    Minutes,
    OpenQuestion,
    Risk,
    Transcript,
)
from praktika.store.locks import RunLock, RunLockedError, new_run_lock, takeover_detail
from praktika.store.records import Store

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
MAX_SLICE_S = 120.0
REASON_CODES = (
    "accurate",
    "wording",
    "wrong_owner",
    "wrong_date",
    "not_a_decision",
    "not_said",
    "duplicate",
    "sensitive",
    "cited_elsewhere",
    "other",
)
CLOSED_STATES = (MeetingState.approved, MeetingState.discarded, MeetingState.purged)
OPEN_STATES = tuple(s for s in MeetingState if s not in CLOSED_STATES)
#: How the page names what moved a meeting while its own run held the lock.
_MOVED_BY = {
    "abort": "by `praktika abort`",
    "discard": "on the review page",
    "purge": "by a DSAR deletion",
}
#: item list -> the editable text field, in body order.
ITEM_LISTS: dict[str, str] = {
    "decisions": "statement",
    "actions": "description",
    "open_questions": "question",
    "risks": "description",
}
ITEM_MODELS: dict[str, type[BaseModel]] = {
    "decisions": Decision,
    "actions": ActionItem,
    "open_questions": OpenQuestion,
    "risks": Risk,
}
_KIND_BY_KEY = (
    ("severity", "risks"),
    ("owner_confidence", "actions"),
    ("statement", "decisions"),
    ("question", "open_questions"),
)


class ReviewItemRequest(BaseModel):
    """Body of ``POST /api/minutes/{id}/items/{item_id}``: the reviewer's verdict on one item."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["accept", "modify", "reject", "restore"]
    reason_code: str = Field(min_length=1, max_length=40)
    before: str | None = None
    after: str | None = Field(default=None, max_length=4000)


class RegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: pipeline.Section
    instruction: str = Field(min_length=3, max_length=500)


class ReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason_code: str = Field(default="reviewed", min_length=1, max_length=40)


def locate_item(minutes: Minutes, item_id: str) -> tuple[str, int] | None:
    """``(list_name, index)`` of the body item with ``id == item_id``; ``None`` when absent."""
    for name in ITEM_LISTS:
        for i, item in enumerate(getattr(minutes, name)):
            if item.id == item_id:
                return name, i
    return None


def item_kind(item: dict[str, Any]) -> str:
    """Which minutes list a removed item's JSON belongs to, judged by its distinctive keys."""
    for key, name in _KIND_BY_KEY:
        if key in item:
            return name
    raise ValueError("item JSON matches no minutes list")


def rename_ref_speakers(minutes: Minutes, mapping: dict[str, str]) -> Minutes:
    """Return ``minutes`` with every ``Ref.speaker`` label found in ``mapping`` renamed."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "segment_id" in node and node.get("speaker") in mapping:
                node = {**node, "speaker": mapping[node["speaker"]]}
            return {k: walk(v) for k, v in node.items()}
        return [walk(v) for v in node] if isinstance(node, list) else node

    return type(minutes).model_validate(walk(minutes.model_dump(mode="json")))


def refuse_identifiers(text: str, roster: list[Attendee], field: str) -> str:
    """Return ``text`` unchanged, or raise ``HTTPException(422)`` when the redaction patterns
    find an identifier in it (IBAN, card, CPR, iqama, phone, e-mail, account, person-linked
    amount).

    Reviewer-typed text — a regenerate instruction or a modified item — is stored in the
    minutes and fed to the model on later regeneration, so it is held to the same rule as
    the transcript (C-06): no raw identifier reaches the model or the record. Refusing is
    safer than tokenising here: a throw-away vault would mint ``«IBAN_1»`` tokens that collide
    with the meeting's stored vault and de-tokenise to the wrong value on export. The
    response names the kinds found, never the values.
    """
    vault = TokenVault(meeting_id="M-00000000-0000")
    redacted = Tokeniser(roster).apply_text(text, vault)
    if not vault.entries:
        return text
    kinds = sorted({t.strip("«»").rsplit("_", 1)[0] for t in tokens_in(redacted)})
    log.warning("server.identifier_refused", field=field, kinds=kinds)
    raise HTTPException(
        422,
        f"{field} contains an identifier ({', '.join(kinds)}); minutes never carry raw "
        "identifiers and none may be sent to the model — describe it without the value",
    )


def sorted_flags(minutes: Minutes) -> list[dict[str, Any]]:
    """Flags with their original index, open ones first and priority 1 at the top."""
    rows = [
        {"n": n, **f.model_dump(mode="json"), "cleared": f.cleared}
        for n, f in enumerate(minutes.flags)
    ]
    return sorted(rows, key=lambda r: (r["cleared"], r["priority"], r["n"]))


def wav_slice(path: Path, start: float, end: float) -> bytes:
    """PCM-16 WAV bytes for ``[start, end)`` seconds of ``path`` (clamped to the file)."""
    with sf.SoundFile(str(path)) as fh:
        rate = fh.samplerate
        first = min(int(start * rate), fh.frames)
        fh.seek(first)
        data = fh.read(max(0, min(int(end * rate), fh.frames) - first), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class ReviewContext:
    """Collaborators shared by every route, plus the lookups that map absence to 404/409."""

    settings: Settings
    store: Store
    identity: IdentityProvider
    audit: AuditLog
    llm_client: LLMClient | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def who(self, request: Request) -> Identity:
        """The identity behind ``request``; raises ``IdentityError`` (mapped to 401) when the
        token is missing or invalid or when the identity carries no Praktika role."""
        return require_roles(self.identity.current(request))

    @staticmethod
    def authorise(user: Identity, meeting: Meeting, *, write: bool) -> None:
        """403 unless ``user`` may read (or change, with ``write``) ``meeting``."""
        require_access(user, meeting, write=write)

    @staticmethod
    def may_write(user: Identity, meeting: Meeting) -> bool:
        try:
            require_access(user, meeting, write=True)
        except AccessDeniedError:
            return False
        return True

    def deny(
        self,
        request: Request,
        user: Identity | None,
        meeting_id: str | None,
        *,
        reason: str,
        detail: str,
        status: int = 403,
    ) -> None:
        """Audit an authentication or authorisation refusal as ``auth.denied`` and log it, so
        an identity probing meeting ids or a token without a Praktika role is visible to
        whoever monitors the audit log (the event vocabulary is in docs/DEPLOYMENT.md)."""
        route = f"{request.method} {request.url.path}"
        self.audit.append(
            "auth.denied", meeting_id, actor=user, reason=reason, route=route, status=status
        )
        log.warning(
            "server.access_denied",
            actor=user.user if user else None,
            meeting_id=meeting_id,
            reason=reason,
            route=route,
            status=status,
            detail=detail,
        )

    def meeting_or_404(self, meeting_id: str) -> Meeting:
        meeting = self.store.get_meeting(meeting_id)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        return meeting

    def minutes_or_404(self, meeting_id: str) -> Minutes:
        minutes = self.store.latest_minutes(meeting_id)
        if minutes is None:
            raise HTTPException(404, "no minutes for this meeting")
        return minutes

    def transcript_or_404(self, meeting_id: str) -> Transcript:
        transcript = self.store.get_transcript(meeting_id)
        if transcript is None:
            raise HTTPException(404, "no transcript for this meeting")
        return transcript

    def move(self, meeting: Meeting, state: MeetingState) -> bool:
        """Best-effort move for a page read (a reviewer opening a draft moves it to
        ``in_review``): only while the meeting is still at the state read and no run holds its
        lock; otherwise it is left as it is and the page still loads. Returns whether it
        moved."""
        if meeting.state == state:
            return True
        moved = self.store.transition(meeting.id, state, expected=(meeting.state,), unlocked=True)
        if moved:
            meeting.state = state
        return moved

    def refuse_running(self, meeting_id: str, *, own_lock: str | None = None) -> None:
        """409 while a run (other than the caller's own, ``own_lock``) holds the meeting's
        lock: "a <command> run is working on this meeting"."""
        lock = self.store.live_run_lock(meeting_id, exempt=own_lock)
        if lock is not None:
            raise HTTPException(409, lock.busy)

    def claim(
        self,
        meeting: Meeting,
        state: MeetingState,
        *,
        expected: Iterable[MeetingState] | None = None,
        during_runs: bool = False,
        own_lock: str | None = None,
        stopped_by: str | None = None,
    ) -> None:
        """Compare-and-set for a review-page write, made before the write itself: move the
        meeting to ``state`` only while it is still at the state this request read (or one of
        ``expected``) and, unless ``during_runs``, no run other than ``own_lock`` holds its
        lock. 409 otherwise, naming the run or the state it is at now. ``stopped_by`` is
        recorded on a run's lock (a discard)."""
        allowed = tuple(expected) if expected is not None else (meeting.state,)
        if self.store.transition(
            meeting.id,
            state,
            expected=allowed,
            unlocked=not during_runs,
            own_lock=own_lock,
            stopped_by=stopped_by,
        ):
            meeting.state = state
            return
        if not during_runs:
            self.refuse_running(meeting.id, own_lock=own_lock)
        raise HTTPException(409, self.moved_message(meeting.id, own_lock=own_lock))

    def moved_message(self, meeting_id: str, *, own_lock: str | None = None) -> str:
        """Why a write found the meeting moved on: what moved it, when the caller's own lock
        recorded it, else the state it is at now."""
        now = self.store.get_meeting(meeting_id)
        state = now.state.value if now is not None else "gone"
        lock = self.store.run_lock(meeting_id) if own_lock is not None else None
        if lock is not None and lock.token == own_lock and lock.stopped_by in _MOVED_BY:
            return (
                f"the meeting was {state} {_MOVED_BY[lock.stopped_by]} meanwhile; nothing was kept"
            )
        return f"the meeting is now {state}; reload it before changing anything"

    @contextmanager
    def running(self, meeting: Meeting, command: str, user: Identity) -> Iterator[RunLock]:
        """Hold the meeting's run lock for a review-page run (``regenerate``): 409 while
        another run holds it; a lock whose process no longer exists on this host is taken over
        and the takeover audited as ``run.lock_taken_over``. Released on every exit."""
        lock = new_run_lock(meeting.id, command)
        try:
            replaced = self.store.take_run_lock(lock)
        except RunLockedError as exc:
            raise HTTPException(409, exc.lock.busy) from exc
        try:
            if replaced is not None:
                self.audit.append(
                    "run.lock_taken_over",
                    meeting.id,
                    actor=user,
                    classification=meeting.classification.value,
                    **takeover_detail(lock, replaced),
                )
            yield lock
        finally:
            self.store.release_run_lock(meeting.id, lock.token)

    @staticmethod
    def require_open(meeting: Meeting) -> None:
        """409 once the meeting is approved, discarded or purged: its minutes are closed."""
        if meeting.state in CLOSED_STATES:
            raise HTTPException(409, f"meeting is {meeting.state.value}; minutes are closed")

    def commit(self, meeting: Meeting, minutes: Minutes, *, claimed: bool = False) -> Minutes:
        """Persist edited minutes as in-review and move the meeting with them; derived
        sections (one-to-one commitments) are recomputed from the edited body first. The move
        to ``in_review`` is claimed before anything is written (409 while a run holds the
        meeting or when it has moved on), unless the route has ``claimed`` it already."""
        if minutes.review.status == "draft":
            minutes = minutes.model_copy(
                update={"review": minutes.review.model_copy(update={"status": "in_review"})}
            )
        minutes = pipeline.sync_derived(minutes, self.store.get_transcript(meeting.id))
        if not claimed:
            self.claim(meeting, MeetingState.in_review)
        self.store.set_review_status(minutes)
        return minutes
