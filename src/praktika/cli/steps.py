"""Pipeline steps shared by ``start``, ``ingest``, ``transcribe`` and ``generate``.

``parse_source`` turns a file into an unredacted ``Transcript`` (VTT, DOCX or audio via
``ingest.audio_file``, whose retained WAVs are registered inside its purge-on-error window);
``redact_and_store`` tokenises identifiers, encrypts the vault and persists the transcript
(C-06); ``draft_minutes`` runs the LLM pipeline, stores the version and moves the meeting to
``draft_ready`` (drafts are indexed for search only on approval). ``run_pipeline`` chains the
three for ``ingest`` and ``start``.

Meeting states during a run:

* Every run first takes the meeting's run lock (``run_lock``; ``store/locks.py``), so only one
  run works on a meeting at a time: a second one is refused with the name of the command
  holding it, and review-page writes are refused meanwhile. ``transcribing`` without a lock
  therefore means "new transcript, awaiting ``praktika generate``", never a running
  transcription.
* ``transcribing`` holds a meeting for a transcription and puts its state and language mode
  back if the run fails, so a failed run never leaves it there. A failed new ``ingest`` (no
  speech and an empty VTT included) returns the meeting to ``created``; a failed live capture
  whose audio was purged ends at ``discarded`` (the purge is audited as ``ingest.failed``).
  A *successful* ``praktika transcribe`` leaves the meeting at ``transcribing``, which then
  means "new transcript, awaiting ``praktika generate``": ``approve`` refuses the older draft
  until a new one is drafted (``refuse_stale_draft``).
* Every move is a compare-and-set in the store (``SqliteStore.transition``): a step moves the
  meeting only from the state it expects, so an ``abort``, a review-page discard or a DSAR
  purge always wins. A run that finds its meeting moved on raises ``RunStoppedError`` naming
  what moved it (recorded on its lock); so does any other failure once the lock records such a
  move (the audio vanishing mid-read, say). Inside ``honouring_abort`` the run first removes
  what it had stored (audio purged, transcript, vault, draft; never anything another run
  stored) and audits the stop once as ``ingest.failed``, so nothing it wrote outlives the
  abort. Any other failure of a first ingest is audited as ``ingest.failed`` too.

Every step audits with the event vocabulary listed in docs/DEPLOYMENT.md, and none of them can
run before the consent gate has stored the meeting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praktika.audio.capture import purge_file
from praktika.cli import context as ctx
from praktika.cli.gate_prompts import transcript_delete_after
from praktika.config import Settings
from praktika.drafts import stale_draft_reason
from praktika.errors import PraktikaError
from praktika.ingest.audio_file import (
    AUDIO_SUFFIXES,
    IngestedAudio,
    RetainedMedia,
    TrackName,
    build_diarizer,
    ingest_tracks,
    load_glossary,
)
from praktika.ingest.docx_transcript import parse_teams_docx
from praktika.ingest.vtt import parse_teams_vtt
from praktika.llm import pipeline
from praktika.llm import prompts as pr
from praktika.logging import get_logger
from praktika.models import (
    LanguageMode,
    Meeting,
    MeetingState,
    MeetingType,
    Minutes,
    Transcript,
)
from praktika.models_registry.manage import hashes_for_provenance
from praktika.redact.tokenise import Tokeniser, encrypt_vault
from praktika.store.locks import RunLock, new_run_lock, takeover_detail

log = get_logger(__name__)

CLOSED = (MeetingState.approved, MeetingState.discarded, MeetingState.purged)


class RunStoppedError(PraktikaError):
    """The meeting was moved on while this run held its lock, so the run stopped:
    ``praktika abort`` discarded it (or put a reopened meeting back to approved), a reviewer
    discarded it on the review page, or a DSAR deletion purged it. ``stopped_by`` says which
    (``abort``, ``discard``, ``purge``) when the lock recorded it."""

    def __init__(self, message: str, *, stopped_by: str | None = None) -> None:
        super().__init__(message)
        self.stopped_by = stopped_by


@dataclass
class Stored:
    """What one run has stored for its meeting, so a stop for an abort can remove it, and
    whether its failure has been audited already (``ingest_tracks`` audits its own)."""

    stage: str = "transcribe"
    media: list[tuple[int, Path]] = field(default_factory=list)
    transcripts: list[int] = field(default_factory=list)
    minutes: list[int] = field(default_factory=list)
    vault_written: bool = False
    vault_before: bytes | None = None
    audited: bool = False


# --------------------------------------------------------------------------- run lock


@contextmanager
def run_lock(rt: ctx.Runtime, meeting: Meeting, command: str) -> Iterator[RunLock]:
    """Hold ``meeting``'s run lock for the block: taken before the run reads or moves
    anything, released at the end, on an error and on Ctrl-C too.

    Raises ``RunLockedError`` (a ``PraktikaError`` naming the command that holds it) while
    another run's process is alive. A lock whose process no longer exists on this host is
    taken over, and the takeover is audited as ``run.lock_taken_over``.
    """
    lock = new_run_lock(meeting.id, command)
    replaced = rt.store.take_run_lock(lock)
    try:
        if replaced is not None:
            rt.audit.append(
                "run.lock_taken_over",
                meeting.id,
                classification=meeting.classification.value,
                **takeover_detail(lock, replaced),
            )
            log.warning(
                "run.lock_taken_over",
                meeting_id=meeting.id,
                previous=replaced.command,
                pid=replaced.pid,
            )
        yield lock
    finally:
        try:
            rt.store.release_run_lock(meeting.id, lock.token)
        except Exception as exc:  # noqa: BLE001 - the run's own outcome is the one to report
            log.warning("run.lock_release_failed", meeting_id=meeting.id, error=type(exc).__name__)


# --------------------------------------------------------------------------- state moves

#: How a stopped run names what moved its meeting (``RunLock.stopped_by``).
_MOVED_BY = {
    "abort": "by `praktika abort`",
    "discard": "on the review page",
    "purge": "by a DSAR deletion",
}


def _lock_of(rt: ctx.Runtime, meeting_id: str) -> RunLock | None:
    try:
        return rt.store.run_lock(meeting_id)
    except Exception:  # noqa: BLE001 - only the wording of a message depends on it
        return None


def _stopped(rt: ctx.Runtime, meeting_id: str, state: MeetingState) -> RunStoppedError:
    """The ``RunStoppedError`` for a meeting found at ``state``: it names what moved the
    meeting, as recorded on the run's lock, instead of guessing."""
    lock = _lock_of(rt, meeting_id)
    by = lock.stopped_by if lock is not None else None
    run = f"this {lock.command} run" if lock is not None else "this run"
    if by in _MOVED_BY:
        what = "purged" if by == "purge" else state.value
        if state is MeetingState.approved:
            what = "put back to approved"
        head = f"{meeting_id} was {what} {_MOVED_BY[by]} while {run} was working on it"
    else:
        head = f"{meeting_id} is now {state.value}: it changed while {run} was working on it"
    return RunStoppedError(f"{head}, so the run stopped and stores nothing further", stopped_by=by)


def _moved_on(rt: ctx.Runtime, meeting_id: str) -> RunStoppedError | None:
    """The ``RunStoppedError`` to report when the run's lock records that the meeting was
    moved on (``stopped_by``), else ``None``: a failure caused by an abort (its audio
    vanishing mid-read, say) then ends the run as the stop it is."""
    lock = _lock_of(rt, meeting_id)
    if lock is None or lock.stopped_by is None:
        return None
    try:
        meeting = rt.store.get_meeting(meeting_id)
    except Exception:  # noqa: BLE001 - the recorded stop is enough to name it
        meeting = None
    return _stopped(rt, meeting_id, meeting.state if meeting else MeetingState.discarded)


def advance(
    rt: ctx.Runtime,
    meeting_id: str,
    to: MeetingState,
    *,
    expected: Iterable[MeetingState],
    language_mode: LanguageMode | None = None,
) -> None:
    """Move the meeting to ``to`` only from one of ``expected`` (compare-and-set); raise
    ``RunStoppedError`` when another command has moved it meanwhile."""
    if not rt.store.transition(meeting_id, to, expected=expected, language_mode=language_mode):
        raise _stopped(rt, meeting_id, rt.require_meeting(meeting_id).state)


def _still(rt: ctx.Runtime, meeting_id: str, state: MeetingState) -> None:
    """Raise ``RunStoppedError`` unless the meeting is still at ``state`` (checked before a write;
    the compare-and-set after it is what guarantees the outcome)."""
    current = rt.require_meeting(meeting_id).state
    if current is not state:
        raise _stopped(rt, meeting_id, current)


def _put_back(
    rt: ctx.Runtime,
    meeting_id: str,
    current: MeetingState,
    back: MeetingState,
    language_mode: LanguageMode | None = None,
) -> None:
    """After a failure, move the meeting from ``current`` back to ``back``, and only if it is
    still at ``current``: a state another command set meanwhile (an abort) is kept."""
    try:
        rt.store.transition(meeting_id, back, expected=(current,), language_mode=language_mode)
    except Exception as exc:  # the original failure is the one to report
        log.warning("meeting.restore_failed", meeting_id=meeting_id, error=type(exc).__name__)


@contextmanager
def transcribing(
    rt: ctx.Runtime, meeting: Meeting, *, live_capture: bool = False
) -> Iterator[None]:
    """Hold the meeting at ``transcribing`` with ``meeting``'s language mode for the block.

    Refused for a meeting that is already approved, discarded or purged. If the block fails,
    the state and language mode it had before are put back (a new ingest returns to
    ``created``); for a ``live_capture`` the meeting goes to ``created`` when its audio is
    still registered and to ``discarded`` when the failure purged it. Only those fields
    change, over the row as it is at the time (a hold set meanwhile stays), and only while the
    meeting is still at ``transcribing``: an ``abort`` from another shell wins. On success the
    meeting stays at ``transcribing`` for the caller's next step.
    """
    if meeting.state in CLOSED:
        raise PraktikaError(f"{meeting.id} is {meeting.state.value}; nothing to transcribe")
    before = rt.require_meeting(meeting.id)
    if before.state in CLOSED:
        raise _stopped(rt, meeting.id, before.state)
    advance(
        rt,
        meeting.id,
        MeetingState.transcribing,
        expected=(before.state,),
        language_mode=meeting.language_mode,
    )
    try:
        yield
    except BaseException:
        back = _failed_capture_state(rt, meeting.id) if live_capture else before.state
        _put_back(rt, meeting.id, MeetingState.transcribing, back, before.language_mode)
        raise


def _failed_capture_state(rt: ctx.Runtime, meeting_id: str) -> MeetingState:
    """``created`` while the captured audio is still registered (``praktika transcribe`` can
    run again); ``discarded`` once the failure has purged it, as nothing can move it on."""
    try:
        live = any(m.deleted_at is None for m in rt.store.list_media(meeting_id))
    except Exception:  # pragma: no cover - the store failed as well; keep the recording
        live = True
    return MeetingState.created if live else MeetingState.discarded


@contextmanager
def honouring_abort(
    rt: ctx.Runtime, meeting: Meeting, *, first_ingest: bool = False
) -> Iterator[Stored]:
    """Track what a run stores; when it stops because its meeting was moved on
    (``RunStoppedError``, or any failure once the run's lock records an abort, a discard or a
    purge), remove it all: audio overwritten and deleted and its rows retired, the transcript
    erased, the vault put back as it was, the draft deleted. The stop is audited once as
    ``ingest.failed`` with ``stopped_by``, and raised as ``RunStoppedError`` naming what moved
    the meeting. With ``first_ingest``, any other failure (no speech, an empty VTT, the model
    down, Ctrl-C) is audited as ``ingest.failed`` with its stage and exception type, unless
    ``ingest_tracks`` has audited it already; what it stored is kept."""
    stored = Stored()
    try:
        yield stored
    except RunStoppedError as exc:
        raise _stop(rt, meeting, stored, exc) from exc
    except BaseException as exc:
        stop = _moved_on(rt, meeting.id)
        if stop is not None:
            raise _stop(rt, meeting, stored, stop) from exc
        if first_ingest and not stored.audited and isinstance(exc, Exception | KeyboardInterrupt):
            _audit_failed(rt, meeting, stored, exc, [])
        raise


def _stop(
    rt: ctx.Runtime, meeting: Meeting, stored: Stored, exc: RunStoppedError
) -> RunStoppedError:
    """Remove what the run stored, audit the stop once and return the error to raise."""
    removed, purged = _undo(rt, meeting, stored)
    if not stored.audited:
        _audit_failed(rt, meeting, stored, exc, purged)
    if not removed:
        return RunStoppedError(str(exc), stopped_by=exc.stopped_by)
    log.warning("run.stopped", meeting_id=meeting.id, stage=stored.stage, removed=removed)
    return RunStoppedError(
        f"{exc}; removed what it had stored: {', '.join(removed)}", stopped_by=exc.stopped_by
    )


def _audit_failed(
    rt: ctx.Runtime, meeting: Meeting, stored: Stored, exc: BaseException, purged: list[str]
) -> None:
    """Audit ``ingest.failed`` for this run (stage, exception type, purged files and, for a
    stop, what moved the meeting); never the exception's message."""
    stored.audited = True
    by = getattr(exc, "stopped_by", None)
    try:
        rt.audit.append(
            "ingest.failed",
            meeting.id,
            classification=meeting.classification.value,
            stage=stored.stage,
            error=type(exc).__name__,
            purged=purged,
            **({"stopped_by": by} if by else {}),
        )
    except Exception as err:  # noqa: BLE001 - the run's own failure is the one to report
        log.warning("ingest.failed_not_audited", meeting_id=meeting.id, error=type(err).__name__)


def _undo(rt: ctx.Runtime, meeting: Meeting, stored: Stored) -> tuple[list[str], list[str]]:
    """Remove what ``stored`` lists (only what this run stored); returns what was removed, for
    the message, and the audio files purged, for the audit event."""
    now = datetime.now(UTC)
    removed: list[str] = []
    purged: list[str] = []
    live = {m.id for m in rt.store.list_media(meeting.id) if m.deleted_at is None}
    for row_id, path in stored.media:
        if path.exists():
            purge_file(path)
            purged.append(path.name)
        if row_id in live:  # the abort may already have retired it
            rt.store.mark_deleted("media", row_id, now)
            if not purged or purged[-1] != path.name:
                removed.append(f"the registration of {path.name}")
    if purged:
        removed.append(f"{len(purged)} audio file(s) overwritten and deleted")
    for row_id in stored.transcripts:
        rt.store.erase_transcript(row_id, now)
    if stored.transcripts:
        removed.append("its transcript")
    if stored.vault_written:  # the vault it replaced goes back only while a transcript needs it
        if stored.vault_before is not None and rt.store.get_transcript(meeting.id) is not None:
            rt.store.save_vault(meeting.id, stored.vault_before)
        else:
            rt.store.delete_vault(meeting.id)
    for version in stored.minutes:
        rt.store.delete_minutes_version(meeting.id, version)
        removed.append(f"draft v{version}")
    return removed, purged


# --------------------------------------------------------------------------- transcription


def parse_vtt(rt: ctx.Runtime, meeting: Meeting, path: Path) -> Transcript:
    if not Path(path).is_file():
        raise PraktikaError(f"input file not found: {path}")
    text = Path(path).read_text(encoding="utf-8")
    transcript = parse_teams_vtt(text, meeting.id, meeting.room_identities)
    rt.audit.append(
        "ingest.vtt",
        meeting.id,
        classification=meeting.classification.value,
        object=Path(path).name,
        segments=len(transcript.segments),
    )
    return transcript


def parse_source(
    rt: ctx.Runtime,
    meeting: Meeting,
    path: Path,
    *,
    vtt: Path | None = None,
    stored: Stored | None = None,
) -> Transcript:
    """Parse ``path`` (``.vtt``, ``.docx`` or audio) into an unredacted transcript.

    Runs inside ``transcribing``. For audio, ``vtt`` optionally attaches a Teams transcript
    whose speaker names are inherited. Retained WAVs are registered as media rows with their
    ``delete_after`` inside ``ingest_tracks``' purge-on-error window, after checking that the
    meeting has not been aborted meanwhile. Raises ``PraktikaError`` for an unsupported suffix
    or a missing file.
    """
    path = Path(path)
    if not path.is_file():
        raise PraktikaError(f"input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(rt, meeting, path)
    if suffix == ".docx":
        transcript = parse_teams_docx(path, meeting.id, meeting.room_identities)
        rt.audit.append(
            "ingest.docx",
            meeting.id,
            classification=meeting.classification.value,
            object=path.name,
            segments=len(transcript.segments),
        )
        return transcript
    if suffix not in AUDIO_SUFFIXES:
        raise PraktikaError(
            f"unsupported input {path.name}: expected .vtt, .docx or one of "
            + ", ".join(sorted(AUDIO_SUFFIXES))
        )
    teams = parse_vtt(rt, meeting, vtt) if vtt is not None else None
    result = transcribe_audio(
        rt,
        meeting,
        stored if stored is not None else Stored(),
        {"file": path},
        vtt=teams,
        diarizer=build_diarizer(rt.settings),
    )
    for track in result.silent_tracks:
        ctx.err_console.print(
            f"WARNING: the {track} track is mostly silent; the transcript may be empty."
        )
    return result.transcript


def transcribe_audio(
    rt: ctx.Runtime,
    meeting: Meeting,
    stored: Stored,
    sources: dict[TrackName, Path],
    *,
    settings: Settings | None = None,
    diarizer: Any | None = None,
    vtt: Transcript | None = None,
    register: bool = True,
) -> IngestedAudio:
    """``ingest_tracks`` for this run (inside ``transcribing``) with ``audio_hooks``;
    ``register=False`` for a re-run over audio that already has its media rows."""
    return ingest_tracks(
        sources,
        meeting,
        settings or rt.settings,
        rt.audit,
        vtt=vtt,
        diarizer=diarizer,
        **audio_hooks(rt, meeting, stored, register=register),
    )


def audio_hooks(
    rt: ctx.Runtime, meeting: Meeting, stored: Stored, *, register: bool = True
) -> dict[str, Any]:
    """The ``ingest_tracks`` callbacks of a run: new WAVs are registered through
    ``registrar`` (unless ``register`` is off), purges are marked on the media rows, and a
    failure is first checked against the meeting, so audio that ``praktika abort`` removed
    mid-read ends the run as the ``RunStoppedError`` it is, not as a read error.
    ``ingest_tracks`` audits the failure itself, so the run does not audit it again."""

    def failed(exc: BaseException) -> BaseException | None:
        stored.audited = True
        if isinstance(exc, RunStoppedError) or not isinstance(exc, Exception):
            return None
        current = rt.require_meeting(meeting.id).state
        if current is MeetingState.transcribing:
            return _moved_on(rt, meeting.id)  # a DSAR purge under way, say
        return _stopped(rt, meeting.id, current)

    return {
        "on_purge": media_purged(rt, meeting.id),
        "on_retain": registrar(rt, meeting, stored) if register else None,
        "on_failure": failed,
    }


def media_purged(rt: ctx.Runtime, meeting_id: str) -> Callable[[Path], None]:
    """``ingest_tracks``'s ``on_purge``: mark the meeting's live media row for a purged file
    deleted, so the store never lists audio that is gone."""

    def mark(path: Path) -> None:
        for m in rt.store.list_media(meeting_id):
            if m.deleted_at is None and Path(m.path).resolve() == Path(path).resolve():
                rt.store.mark_deleted("media", m.id, datetime.now(UTC))

    return mark


def registrar(
    rt: ctx.Runtime, meeting: Meeting, stored: Stored
) -> Callable[[list[RetainedMedia]], None]:
    """``ingest_tracks``'s ``on_retain``: register every retained WAV with its ``delete_after``
    so the retention job can find and delete it (C-05). It runs inside ``ingest_tracks``'
    purge-on-error window, so a failed registration (or Ctrl-C) purges the files instead of
    leaving them unregistered, and it first checks that no abort has discarded the meeting."""

    def register(media: list[RetainedMedia]) -> None:
        stored.stage = "register"
        _still(rt, meeting.id, MeetingState.transcribing)
        for m in media:
            row = rt.store.save_media(
                meeting.id, m.path, m.sha256, kind=m.track, delete_after=m.delete_after
            )
            stored.media.append((row, Path(m.path)))

    return register


def redact_and_store(
    rt: ctx.Runtime, meeting: Meeting, transcript: Transcript, *, stored: Stored | None = None
) -> Transcript:
    """Tokenise identifiers, encrypt the vault, store the vault and the redacted transcript.

    Returns the redacted transcript. The vault and the transcript are written in one
    transaction (``save_transcript(..., vault=)``), so a failure or Ctrl-C can never leave a
    redacted transcript without the means to restore it, nor a vault that belongs to a
    transcript that was never stored. Runs inside ``transcribing``: with ``stored``, it checks
    before writing and confirms after writing (compare-and-set) that the meeting is still at
    ``transcribing``, and records what it wrote so a stop for an abort can remove it.
    """
    if stored is not None:
        stored.stage = "redact"
    if not transcript.segments:
        raise PraktikaError("the transcript has no segments; nothing to redact or draft")
    redacted, vault = Tokeniser(meeting.roster).apply(transcript)
    blob = encrypt_vault(vault, ctx.vault_key(rt.settings))
    if stored is not None:
        _still(rt, meeting.id, MeetingState.transcribing)
        stored.vault_before = rt.store.get_vault(meeting.id)
        stored.vault_written = True
    now = datetime.now(UTC)
    row = rt.store.save_transcript(
        redacted, delete_after=transcript_delete_after(rt, meeting, now), vault=blob
    )
    if stored is not None:
        stored.transcripts.append(row)
    rt.audit.append(
        "redact.applied",
        meeting.id,
        classification=meeting.classification.value,
        tokens=len(vault.entries),
        segments=len(redacted.segments),
        transcript_sha256=redacted.sha256(),
    )
    if stored is not None:
        advance(rt, meeting.id, MeetingState.transcribing, expected=(MeetingState.transcribing,))
    return redacted


# --------------------------------------------------------------------------- drafting


def draft_minutes(
    rt: ctx.Runtime,
    meeting: Meeting,
    transcript: Transcript,
    *,
    expected: MeetingState,
    restore: MeetingState | None = None,
    template: MeetingType | None = None,
    prompt_version: str | None = None,
    stored: Stored | None = None,
) -> Minutes:
    """Generate, verify and store a minutes version; the meeting becomes ``draft_ready``.

    The meeting moves ``expected`` -> ``drafting`` -> ``draft_ready``, each a compare-and-set,
    so an abort during the model call wins (``RunStoppedError``; the version stored is removed by
    ``honouring_abort``). A failure puts the meeting back to ``restore`` (default
    ``expected``) while it is still at ``drafting``.
    """
    settings = rt.settings
    meeting_type = template or meeting.meeting_type
    if meeting_type != meeting.meeting_type:
        meeting = meeting.model_copy(update={"meeting_type": meeting_type})
    version = prompt_version or settings.prompt_version
    prompts = pr.load(settings.prompts_dir, version, meeting_type)
    entries, glossary_sha = load_glossary(settings)
    opts = pipeline.GenerateOptions(
        template=meeting_type,
        prompt_version=version,
        full_context_max_tokens=settings.llm_full_context_max_tokens,
        long_transcript_tokens=settings.llm_long_transcript_tokens,
        fallback_model=settings.llm_fallback_model,
        known_terms=[t for e in entries for t in (e.canonical, *e.variants)],
    )
    stored = stored if stored is not None else Stored()
    stored.stage = "draft"
    advance(rt, meeting.id, MeetingState.drafting, expected=(expected,))
    try:
        minutes = pipeline.generate(
            transcript,
            meeting,
            ctx.llm_client(settings),
            prompts,
            glossary_sha,
            hashes_for_provenance(settings),
            opts,
            rt.audit,
        )
        stored_version = rt.store.save_minutes(minutes)
        stored.minutes.append(stored_version)
        advance(rt, meeting.id, MeetingState.draft_ready, expected=(MeetingState.drafting,))
    except BaseException:
        _put_back(rt, meeting.id, MeetingState.drafting, restore or expected)
        raise
    minutes = minutes.model_copy(update={"version": stored_version})
    rt.store.clear_index(meeting.id)  # a new draft supersedes any approved, indexed version
    log.info(
        "minutes.stored",
        meeting_id=meeting.id,
        version=stored_version,
        flags=len(minutes.flags),
        blocking=len(minutes.blocking_flags()),
    )
    return minutes


def run_pipeline(
    rt: ctx.Runtime,
    meeting: Meeting,
    produce: Callable[[Stored], Transcript],
    *,
    live_capture: bool = False,
) -> Minutes:
    """One ``ingest`` or ``start`` run: ``produce`` the transcript, redact and store it, draft.

    The caller holds the meeting's run lock (``run_lock``). A failure before the transcript is
    stored returns the meeting to ``created`` (or, for a live capture whose audio was purged,
    ``discarded``); a WAV already registered stays under its ``delete_after``, so no speech
    can be retried with ``praktika transcribe`` and ``praktika abort`` removes it. A failure
    while drafting returns the meeting to ``created`` with the transcript kept, so
    ``praktika generate`` can draft again. Every failure is audited as ``ingest.failed``
    (stage and exception type). An abort from another shell stops the run and removes what it
    stored (``honouring_abort``).
    """
    with honouring_abort(rt, meeting, first_ingest=True) as stored:
        with transcribing(rt, meeting, live_capture=live_capture):
            redacted = redact_and_store(rt, meeting, produce(stored), stored=stored)
        try:
            return draft_minutes(
                rt,
                meeting,
                redacted,
                expected=MeetingState.transcribing,
                restore=MeetingState.created,
                stored=stored,
            )
        except RunStoppedError:
            raise
        except BaseException as exc:
            # draft_minutes puts back what it moved; this covers a failure before its first
            # move (a prompt template that will not load, Ctrl-C), so no run ends at transcribing.
            _put_back(rt, meeting.id, MeetingState.transcribing, MeetingState.created)
            if isinstance(exc, Exception) and _moved_on(rt, meeting.id) is None:
                ctx.err_console.print(
                    f"The transcript of {meeting.id} is stored: once the error below is fixed, "
                    f"`praktika generate {meeting.id}` drafts the minutes from it."
                )
            raise


def announce_draft(rt: ctx.Runtime, meeting: Meeting, minutes: Minutes) -> None:
    """Print the outcome of a drafting run and the review URL (with a loud banner when the
    draft came from the fake provider)."""
    blocking = len(minutes.blocking_flags())
    if minutes.provenance.generator_model == "fake":
        ctx.err_console.print(
            "WARNING: these minutes were drafted by the FAKE provider (placeholder content). "
            "They are not minutes of the meeting. Unset PRAKTIKA_LLM_PROVIDER=fake."
        )
    ctx.console.print(
        f"Draft v{minutes.version} ready for {meeting.id}: {len(minutes.decisions)} decisions, "
        f"{len(minutes.actions)} actions, {len(minutes.flags)} flags "
        f"({blocking} blocking approval)."
    )
    ctx.console.print(f"Review: {ctx.review_url(rt.settings, meeting.id)}")


# --------------------------------------------------------------------------- approval


def refuse_stale_draft(rt: ctx.Runtime, meeting: Meeting, minutes: Minutes) -> None:
    """``approve``'s check: raise ``PraktikaError`` when ``stale_draft_reason`` has one."""
    reason = stale_draft_reason(rt.store, meeting, minutes)
    if reason is not None:
        raise PraktikaError(reason)
