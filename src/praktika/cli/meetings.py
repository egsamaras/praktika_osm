"""Meeting lifecycle commands: ``start`` and ``ingest`` (``meeting_ops`` holds the re-runs).

``start`` and ``ingest`` share the consent gate (``gate_prompts``) and the pipeline steps
(``steps.run_pipeline``). ``start`` checks that this host can record before the consent script
is printed, and writes a pid file next to the tracks so ``praktika abort`` from another shell
can signal it; on SIGTERM or after ``abort`` the in-flight audio is overwritten and unlinked
(C-04). A capture that cannot start, or that stops with no track, purges what it wrote and
ends at ``discarded``. Both hold the meeting's run lock from the consent gate to the end of
the run (``steps.run_lock``), so no other run and no review-page edit can work on the meeting
meanwhile; ``praktika abort`` still wins. ``--source sck`` is refused before anything else
unless the helper carries a Developer ID signature.
"""

from __future__ import annotations

import math
import os
import signal
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.live import Live
from rich.text import Text

from praktika.audio.capture import helper_is_signed, purge_file
from praktika.cli import context as ctx
from praktika.cli import gate_prompts as gp
from praktika.cli import steps
from praktika.config import default_data_dir
from praktika.errors import PraktikaError
from praktika.identity import normalise_upn
from praktika.ingest.audio_file import audio_dir, build_diarizer, ingest_tracks
from praktika.logging import get_logger
from praktika.models import (
    Classification,
    LanguageMode,
    Meeting,
    MeetingState,
    MeetingType,
    Minutes,
    Platform,
    Transcript,
)
from praktika.stt.router import require_language

log = get_logger(__name__)

PID_FILE = "capture.pid"
STATUS_INTERVAL_S = 0.5
#: The capture helper's default location: under this platform's default data directory.
DEFAULT_HELPER = default_data_dir() / "bin" / "praktika-capture"

TitleOpt = Annotated[str, typer.Option("--title", help="Meeting title.")]
TypeOpt = Annotated[MeetingType, typer.Option("--type", help="Minutes template.")]
ClassOpt = Annotated[Classification, typer.Option("--class", help="Information classification.")]
LangOpt = Annotated[
    LanguageMode,
    typer.Option(
        "--lang", help="Language routing mode (default en; auto samples language ID first)."
    ),
]
RosterOpt = Annotated[Path | None, typer.Option("--roster", help="Roster YAML.")]
PlatformOpt = Annotated[Platform, typer.Option("--platform", help="Where the meeting runs.")]

_abort_requested = False


def _on_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    global _abort_requested
    _abort_requested = True


SILENCE_LINE = "NO AUDIO DETECTED: the input is silent. Check the device or Teams audio routing."


def level_bar(amplitude: float, *, floor_db: float = -60.0, width: int = 10) -> str:
    """Render an RMS amplitude (0..1) as a ``#`` bar on a decibel scale.

    Speech into a laptop microphone sits near -40 dBFS (about 1% of full scale), so a linear
    bar never moves; on this scale silence is ".", quiet speech is a few blocks and full scale
    fills the bar. A -60 dBFS floor maps to zero blocks.
    """
    if amplitude <= 0.0:
        return "."
    db = 20.0 * math.log10(amplitude)
    blocks = int((db - floor_db) / (-floor_db) * width)
    return "#" * max(0, min(width, blocks)) or "."


def _status(
    meeting_id: str,
    mode: str,
    elapsed: float,
    levels: tuple[float, float],
    *,
    silent: bool = False,
) -> Text:
    bar = level_bar
    m, s = divmod(int(elapsed), 60)
    line = (
        f"REC {m:02d}:{s:02d}  sys [{bar(levels[0]):<10}]  mic [{bar(levels[1]):<10}]  "
        f"lang {mode}  {meeting_id}  (Ctrl-C to stop)"
    )
    return Text(line + "\n" + SILENCE_LINE if silent else line)


def _check_silence(capturer: Any) -> bool:
    """C-04: ask the capturer whether the track is mostly silent (False when it cannot say)."""
    check = getattr(capturer, "check_silence", None)
    return bool(check()) if callable(check) else False


def _require_capture(capturer: Any) -> None:
    """Refuse before the consent gate when this host cannot record (for example no PortAudio
    library), so no consent is recorded for a meeting that cannot be captured."""
    check = getattr(capturer, "preflight", None)
    if callable(check):
        check()


def _capture_failed(rt: Any, meeting: Meeting, out_dir: Path, exc: BaseException) -> None:
    """The capture could not start: purge what it wrote, discard the meeting, audit it."""
    (out_dir / PID_FILE).unlink(missing_ok=True)
    purged = []
    for wav in sorted(out_dir.glob("*.wav")):
        purge_file(wav)
        purged.append(wav.name)
    rt.store.transition(meeting.id, MeetingState.discarded, expected=(MeetingState.capturing,))
    rt.audit.append(
        "ingest.failed",
        meeting.id,
        classification=meeting.classification.value,
        stage="capture",
        error=type(exc).__name__,
        purged=purged,
    )


def _warn_silent_tracks(result: Any) -> None:
    for track in getattr(result, "silent_tracks", []) or []:
        ctx.err_console.print(
            f"WARNING: the {track} track is mostly silent; the transcript may be empty. "
            "Check the input device before the next meeting."
        )


@ctx.guarded
def start(
    title: TitleOpt,
    meeting_type: TypeOpt = MeetingType.general,
    classification: ClassOpt = Classification.internal,
    lang: LangOpt = LanguageMode.en,
    roster: RosterOpt = None,
    platform: PlatformOpt = Platform.in_room,
    source: Annotated[str, typer.Option("--source", help="mic | sck")] = "mic",
    helper: Annotated[Path, typer.Option("--helper", help="Signed capture helper.")] = (
        DEFAULT_HELPER
    ),
    device: Annotated[str | None, typer.Option("--device", help="Input device name.")] = None,
    duration: Annotated[
        float | None, typer.Option("--duration", help="Stop automatically after N seconds.")
    ] = None,
    notified: gp.NotifiedOpt = None,
    objections: gp.ObjectionsOpt = None,
    method: gp.MethodOpt = None,
    teams_transcription_started: gp.TeamsOpt = None,
    purpose: gp.PurposeOpt = None,
    scope_ack: gp.ScopeAckOpt = None,
    ack_all_scope: gp.AckAllOpt = False,
    tag: gp.TagOpt = None,
    foreign_hosted: gp.ForeignHostedOpt = False,
    external_participants: gp.ExternalOpt = False,
) -> None:
    """Read the script, run the consent gate, capture audio, then draft the minutes.

    A language mode this deployment cannot transcribe, and a host that cannot record, are
    refused before the consent script is printed or any gate answer is asked for.
    """
    if source not in ("mic", "sck"):
        raise PraktikaError("--source must be mic or sck")
    settings = ctx.load_settings()
    if source == "sck":
        if not settings.capture_team_id:
            raise ctx.GateIncompleteError(
                "PRAKTIKA_CAPTURE_TEAM_ID is not set; the capture helper's Team ID must be "
                "pinned before --source sck can be used"
            )
        if helper_is_signed(helper, team_id=settings.capture_team_id) is None:
            raise ctx.GateIncompleteError(
                f"capture helper {helper} is not Developer ID signed by team "
                f"{settings.capture_team_id}; refusing to start"
            )
    require_language(settings, lang)
    capturer = ctx.build_capturer(source, helper, device)
    if source == "sck":
        capturer.team_id = settings.capture_team_id
    _require_capture(capturer)
    rt = ctx.open_runtime(settings)
    gp.print_scripts()
    answers = gp.collect_answers(
        notified=notified,
        objections=objections,
        method=method,
        teams_transcription_started=teams_transcription_started,
        purpose=purpose,
        scope_ack=scope_ack,
        ack_all_scope=ack_all_scope,
    )
    meeting = gp.new_meeting(
        rt,
        title=title,
        meeting_type=meeting_type,
        classification=classification,
        language=lang,
        platform=platform,
        roster_path=roster,
        tags=gp.meeting_tags(tag, foreign_hosted=foreign_hosted, external=external_participants),
    )
    gp.run_gate(rt, meeting, answers)
    with steps.run_lock(rt, meeting, "start"):
        minutes = _capture_and_draft(rt, meeting, capturer, source, lang, duration)
    steps.announce_draft(rt, meeting, minutes)


def _capture_and_draft(
    rt: ctx.Runtime,
    meeting: Meeting,
    capturer: Any,
    source: str,
    lang: LanguageMode,
    duration: float | None,
) -> Minutes:
    """``start`` after the consent gate, under the meeting's run lock: capture until Ctrl-C,
    ``--duration`` or an abort, then transcribe, redact and draft (``steps.run_pipeline``)."""
    classification = meeting.classification
    out_dir = audio_dir(rt.settings, meeting.id)
    steps.advance(rt, meeting.id, MeetingState.capturing, expected=(MeetingState.created,))
    (out_dir / PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        capturer.start(out_dir)
    except BaseException as exc:
        _capture_failed(rt, meeting, out_dir, exc)
        if isinstance(exc, Exception) and not isinstance(exc, PraktikaError):
            raise PraktikaError(f"cannot start the capture: {exc}") from exc
        raise
    rt.audit.append(
        "capture.started", meeting.id, classification=classification.value, source=source
    )
    started = time.monotonic()
    silence_warned = False
    try:
        with Live(console=ctx.console, transient=True, refresh_per_second=4) as live:
            while not _abort_requested:
                elapsed = time.monotonic() - started
                silent = _check_silence(capturer)
                if silent and not silence_warned:
                    silence_warned = True
                    ctx.err_console.print(SILENCE_LINE)
                live.update(
                    _status(meeting.id, lang.value, elapsed, capturer.levels(), silent=silent)
                )
                if duration is not None and elapsed >= duration:
                    break
                time.sleep(STATUS_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        (out_dir / PID_FILE).unlink(missing_ok=True)
    if _abort_requested:
        capturer.abort()
        rt.audit.append("capture.aborted", meeting.id, classification=classification.value)
        rt.store.set_state(meeting.id, MeetingState.discarded)
        raise PraktikaError("capture aborted; audio overwritten and removed")
    tracks = capturer.stop()
    rt.audit.append(
        "capture.stopped",
        meeting.id,
        classification=classification.value,
        tracks=[t.name for t in tracks],
    )
    if not tracks:
        rt.store.transition(meeting.id, MeetingState.discarded, expected=(MeetingState.capturing,))
        raise PraktikaError(f"no audio track was captured; {meeting.id} is discarded")

    def transcribe_capture(stored: steps.Stored) -> Transcript:
        result = ingest_tracks(  # the captured tracks are not registered yet: this run owns them
            {t.name: t.path for t in tracks},
            meeting,
            rt.settings,
            rt.audit,
            diarizer=build_diarizer(rt.settings),
            own_sources=True,
            **steps.audio_hooks(rt, meeting, stored),
        )
        _warn_silent_tracks(result)
        return result.transcript

    return steps.run_pipeline(rt, meeting, transcribe_capture, live_capture=True)


@ctx.guarded
def ingest(
    path: Annotated[Path, typer.Argument(help="Audio file, Teams .vtt or Recap .docx.")],
    title: Annotated[str | None, typer.Option("--title", help="Meeting title.")] = None,
    meeting_type: TypeOpt = MeetingType.general,
    classification: ClassOpt = Classification.internal,
    lang: LangOpt = LanguageMode.en,
    roster: RosterOpt = None,
    platform: PlatformOpt = Platform.teams,
    vtt: Annotated[
        Path | None, typer.Option("--vtt", help="Teams transcript for an audio file.")
    ] = None,
    notified: gp.NotifiedOpt = None,
    objections: gp.ObjectionsOpt = None,
    method: gp.MethodOpt = None,
    teams_transcription_started: gp.TeamsOpt = None,
    purpose: gp.PurposeOpt = None,
    scope_ack: gp.ScopeAckOpt = None,
    ack_all_scope: gp.AckAllOpt = False,
    tag: gp.TagOpt = None,
    foreign_hosted: gp.ForeignHostedOpt = False,
    external_participants: gp.ExternalOpt = False,
    organiser: gp.OrganiserOpt = None,
) -> None:
    """File-first path: gate, then transcribe or parse, redact, draft and print the review URL.

    ``--organiser`` records a named organiser (UPN or e-mail, shape-checked before anything
    else runs) for a meeting an operator submits on someone's behalf; the operator stays the
    consent record's ``recorded_by`` and ``meeting.organiser_named`` audits both. A language
    mode this deployment cannot transcribe is refused before any gate answer is asked for.
    """
    named = normalise_upn(organiser) if organiser is not None else None
    settings = ctx.load_settings()
    require_language(settings, lang)
    rt = ctx.open_runtime(settings)
    answers = gp.collect_answers(
        notified=notified,
        objections=objections,
        method=method,
        teams_transcription_started=teams_transcription_started,
        purpose=purpose,
        scope_ack=scope_ack,
        ack_all_scope=ack_all_scope,
    )
    meeting = gp.new_meeting(
        rt,
        title=title or Path(path).stem,
        meeting_type=meeting_type,
        classification=classification,
        language=lang,
        platform=platform,
        roster_path=roster,
        tags=gp.meeting_tags(tag, foreign_hosted=foreign_hosted, external=external_participants),
        organiser=named,
    )
    gp.run_gate(rt, meeting, answers)
    with steps.run_lock(rt, meeting, "ingest"):
        minutes = steps.run_pipeline(
            rt,
            meeting,
            lambda stored: steps.parse_source(rt, meeting, path, vtt=vtt, stored=stored),
        )
    steps.announce_draft(rt, meeting, minutes)
