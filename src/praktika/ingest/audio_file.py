"""File-first audio ingest.

Contract: ``ingest_tracks`` converts every source to 16 kHz mono WAV inside the meeting's private
audio directory, runs VAD and language routing per track (``stt.router``), merges the tracks
into one segment list, optionally diarises and assigns speakers, normalises glossary terms and
returns an unredacted ``Transcript`` plus the retained WAV files. The converted WAV is deleted
immediately (zero-overwrite then unlink) when the classification's audio retention is 0 hours
(restricted), and that deletion is audited as ``retention.deleted``; otherwise its
``delete_after`` is the conversion time plus ``audio_hours`` and the caller registers it as
media through ``on_retain``, which runs inside the same guarded window. If anything fails
between conversion and the end of that registration (missing weights, an STT error, a locked
database, Ctrl-C) every WAV this call wrote is purged before the exception propagates, so no
unregistered audio can outlive the retention timers (C-05), and ``ingest.failed`` is audited
with the stage, the exception type and the purged file names. The caller's ``on_failure`` sees
the exception first and may name the real cause instead: when ``praktika abort`` removed the
audio mid-read, the read error becomes the caller's ``RunStoppedError``, which is what is
audited and raised.

A file that was already there is never purged here: a retained WAV re-transcribed in place
(``praktika transcribe``) belongs to its media row and survives a failed run. ``start`` hands
over its captured tracks, which nothing has registered yet, with ``own_sources``: the source
files are then this call's too, and a source under another name than its converted copy is
purged (and the purge audited on ``ingest.file``) as soon as the copy exists.
``ingest_file`` is the single-file convenience wrapper.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
from pydantic import BaseModel, ConfigDict

from praktika import glossary
from praktika.audio.capture import purge_file
from praktika.audio.convert import AudioInfo, file_sha256, is_mostly_silent, to_wav16k
from praktika.config import Settings
from praktika.diarize.assign import assign_speakers
from praktika.ingest.vtt import inherit_names
from praktika.logging import get_logger
from praktika.models import Meeting, RawSegment, Transcript
from praktika.stt import router

log = get_logger(__name__)

TrackName = Literal["mic", "system", "file"]
AUDIO_SUFFIXES: frozenset[str] = frozenset({".wav", ".m4a", ".mp3", ".mp4"})
AUDIO_SUBDIR = "audio"


class RetainedMedia(BaseModel):
    """A converted WAV kept on disk until ``delete_after`` (recorded by the caller as media)."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    sha256: str
    track: TrackName
    delete_after: datetime


class IngestedAudio(BaseModel):
    """Result of ``ingest_tracks``: the transcript, the WAVs that are still on disk and the
    tracks that were mostly silent (C-04: the caller warns the organiser on the console)."""

    model_config = ConfigDict(extra="forbid")

    transcript: Transcript
    media: list[RetainedMedia] = []
    silent_tracks: list[str] = []


def audio_dir(settings: Settings, meeting_id: str) -> Path:
    """The meeting's private audio directory (``data_dir/audio/<id>``, mode 0700)."""
    path = Path(settings.data_dir) / AUDIO_SUBDIR / meeting_id
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def load_glossary(settings: Settings) -> tuple[list[glossary.GlossaryEntry], str]:
    """Glossary entries and file hash; a missing file gives no entries and the hash ``missing``."""
    try:
        return glossary.load(settings.glossary_path)
    except FileNotFoundError:
        log.warning("glossary.missing", path=str(settings.glossary_path))
        return [], "missing"


def build_diarizer(settings: Settings) -> Any | None:
    """The configured diariser, or ``None`` when diarisation is off or the backend is ``fake``.

    ``fake`` returns ``None`` because the offline fake lives in the test suite and is injected
    through ``ingest_tracks(..., diarizer=...)``.
    """
    if not settings.diarize or settings.diarize_backend == "none":
        return None
    if settings.diarize_backend == "pyannote":
        from praktika.diarize.pyannote_backend import PyannoteDiarizer
        from praktika.models_registry.weights import ensure_verified

        ensure_verified(settings, None)
        return PyannoteDiarizer(Path(settings.models_dir) / "diarize")
    return None


def _prepare(src: Path, dst: Path) -> AudioInfo:
    """Convert ``src`` to ``dst``; a retained WAV re-transcribed in place is probed, not
    converted (converting a file onto itself would truncate it before ffmpeg reads it)."""
    if src.resolve() == dst.resolve():
        meta = sf.info(str(dst))
        return AudioInfo(
            path=dst,
            sample_rate=int(meta.samplerate),
            channels=int(meta.channels),
            duration_s=float(meta.frames) / float(meta.samplerate),
            sha256=file_sha256(dst),
        )
    return to_wav16k(src, dst)


def _engines(raw: list[RawSegment], settings: Settings) -> dict[str, str]:
    """Engine names per role: the configured backend, replaced by the engine that actually ran.

    With the Arabic path off (``stt_ar = "none"``) every segment came from the English engine,
    even one written in Arabic script, so the ``stt_ar`` role stays ``none``.
    """
    out = {"stt_en": settings.stt_en, "stt_ar": settings.stt_ar}
    arabic = settings.stt_ar != "none"
    for s in raw:
        role = "stt_ar" if arabic and s.language in ("ar", "mixed") else "stt_en"
        if s.engine:
            out[role] = s.engine
    return out


@dataclass
class _Run:
    """What one ``ingest_tracks`` call has done so far, for the clean-up when it fails."""

    own_sources: bool = False
    stage: str = "convert"
    converted: list[tuple[TrackName, Path, str]] = field(default_factory=list)
    #: Files this call may purge: the ones it wrote, plus the sources when they are handed over.
    owned: set[Path] = field(default_factory=set)


def _abandon(
    run: _Run,
    meeting: Meeting,
    audit: Any,
    exc: BaseException,
    on_purge: Callable[[Path], None] | None,
) -> None:
    """Purge every file ``run`` owns and audit ``ingest.failed`` (C-05); never transcript text."""
    purged: list[Path] = []
    for path in sorted(run.owned):
        if path.exists():
            purge_file(path)
            purged.append(path)
            log.warning("ingest.audio_purged_on_error", meeting_id=meeting.id, file=path.name)
            if on_purge is not None:
                on_purge(path)
    stopped_by = getattr(exc, "stopped_by", None)  # set on the caller's RunStoppedError
    audit.append(
        "ingest.failed",
        meeting.id,
        classification=meeting.classification.value,
        stage=run.stage,
        error=type(exc).__name__,
        purged=[p.name for p in purged],
        **({"stopped_by": stopped_by} if stopped_by else {}),
    )


def _cause(
    exc: BaseException, on_failure: Callable[[BaseException], BaseException | None] | None
) -> BaseException:
    """The exception to audit and raise for ``exc``: the one ``on_failure`` names instead, if
    any (a failing ``on_failure`` is logged and leaves ``exc`` as it is)."""
    if on_failure is None:
        return exc
    try:
        return on_failure(exc) or exc
    except Exception as check:  # noqa: BLE001 - the original failure is the one to report
        log.warning("ingest.failure_check_failed", error=type(check).__name__)
        return exc


def ingest_tracks(
    sources: dict[TrackName, Path],
    meeting: Meeting,
    settings: Settings,
    audit: Any,
    *,
    vtt: Transcript | None = None,
    diarizer: Any | None = None,
    engines: router.Engines | None = None,
    now: datetime | None = None,
    own_sources: bool = False,
    on_purge: Callable[[Path], None] | None = None,
    on_retain: Callable[[list[RetainedMedia]], None] | None = None,
    on_failure: Callable[[BaseException], BaseException | None] | None = None,
) -> IngestedAudio:
    """Convert, transcribe, merge, diarise and glossary-normalise one or more audio tracks.

    ``sources`` maps a track name to a file (``{"file": recording}`` for a downloaded recording,
    ``{"system": ..., "mic": ...}`` for a capture). ``vtt`` attaches a Teams transcript whose
    speaker names are inherited by time overlap. ``diarizer`` and ``engines`` override the
    configured backends (tests). ``own_sources`` says the source files are handed over to this
    call (``start``'s captured tracks, which nothing has registered yet): they and their
    converted copies are purged on failure and under zero retention like converted files.
    Without it a file already there (a retained WAV re-transcribed in place) is only read.
    ``on_retain`` is called once with the retained WAVs (only when there are any) before the
    guarded window closes, so the caller registers them there: if it raises, or Ctrl-C
    arrives before it returns, the files are purged like any other failure (stage
    ``register``). ``on_purge`` is called with each file purged on failure, so a caller holding
    media rows can mark them deleted. ``on_failure`` is called with the exception when the run
    fails, before the clean-up; it returns the exception to audit and raise instead (the
    caller's ``RunStoppedError`` when the meeting was aborted meanwhile, raised from the
    original), or ``None`` to keep it. Raises ``ValueError`` for an empty ``sources`` mapping and
    ``FfmpegError`` when conversion fails. Emits ``ingest.file`` per source,
    ``diarize.completed`` when diarisation ran (counts only, never who spoke how much),
    ``retention.deleted`` per WAV deleted under zero retention and ``ingest.failed`` when the
    run fails.
    """
    if not sources:
        raise ValueError("no audio sources given")
    entries, _ = load_glossary(settings)
    work = audio_dir(settings, meeting.id)
    started = now or datetime.now(UTC)
    run = _Run(own_sources=own_sources)
    try:
        segments, all_raw, silent = _convert_and_transcribe(
            sources, meeting, settings, audit, work, run, entries, engines, diarizer
        )
        run.stage = "assemble"
        transcript = Transcript(
            meeting_id=meeting.id,
            source="file" if set(sources) == {"file"} else "capture",
            engines=_engines(all_raw, settings),
            segments=glossary.apply(segments, entries),
        )
        if vtt is not None:
            transcript = inherit_names(transcript, vtt)
        run.stage = "register"
        media = _retain(run, meeting, settings, audit, started)
        if media and on_retain is not None:
            on_retain(media)
    except BaseException as exc:
        # Nothing has registered the files this call wrote (or the registration failed): purge
        # them so they cannot outlive C-05. A retained file it only read is not among them and
        # keeps its media row.
        failure = _cause(exc, on_failure)
        _abandon(run, meeting, audit, failure, on_purge)
        if failure is not exc:
            raise failure from exc
        raise
    log.info(
        "ingest.completed", meeting_id=meeting.id, tracks=list(sources), segments=len(segments)
    )
    return IngestedAudio(transcript=transcript, media=media, silent_tracks=silent)


def _retain(
    run: _Run, meeting: Meeting, settings: Settings, audit: Any, started: datetime
) -> list[RetainedMedia]:
    """The converted WAVs to keep, each with its ``delete_after``; under zero audio retention
    (restricted) every file this call owns is deleted now instead, and audited."""
    hours = int(settings.retention_audio_hours.get(meeting.classification.value, 0))
    media: list[RetainedMedia] = []
    for track, dst, sha in run.converted:
        if hours <= 0:
            if dst in run.owned:  # a retained file keeps its own row and timer
                purge_file(dst)
                audit.append(
                    "retention.deleted",
                    meeting.id,
                    classification=meeting.classification.value,
                    object=str(dst),
                    kind="audio",
                    reason=f"{meeting.classification.value}: audio deleted at transcription",
                    file_removed=True,
                )
                log.info("ingest.audio_purged", meeting_id=meeting.id, track=track)
            continue
        media.append(
            RetainedMedia(
                path=dst, sha256=sha, track=track, delete_after=started + timedelta(hours=hours)
            )
        )
    return media


def _convert_and_transcribe(
    sources: dict[TrackName, Path],
    meeting: Meeting,
    settings: Settings,
    audit: Any,
    work: Path,
    run: _Run,
    entries: list[glossary.GlossaryEntry],
    engines: router.Engines | None,
    diarizer: Any | None,
) -> tuple[list[Any], list[RawSegment], list[str]]:
    """Convert every source into ``work`` (recording each file in ``run`` before it is written,
    so a half-written conversion is purged too), transcribe, merge and diarise. Returns
    ``(segments, raw segments, silent tracks)``."""
    raw_by_track: dict[str, list[RawSegment]] = {}
    silent: list[str] = []
    converted = run.converted
    for track, src in sources.items():
        src, dst = Path(src), work / f"{track}.wav"
        run.stage = "convert"
        if run.own_sources:
            run.owned.update((src, dst))  # a handed-over source is this call's to purge
        elif not dst.exists():
            run.owned.add(dst)
        info = _prepare(src, dst)
        converted.append((track, dst, info.sha256))
        # A handed-over source under another name is superseded by its converted copy, which
        # is the one registered (or purged): delete it now, so it is never left unregistered.
        source_purged = run.own_sources and src.resolve() != dst.resolve() and src.exists()
        if source_purged:
            purge_file(src)
        audit.append(
            "ingest.file",
            meeting.id,
            classification=meeting.classification.value,
            object=src.name,
            track=track,
            sha256=info.sha256,
            duration_s=round(info.duration_s, 2),
            **({"source_purged": True} if source_purged else {}),
        )
        if is_mostly_silent(dst):
            silent.append(track)
            log.warning("ingest.mostly_silent", meeting_id=meeting.id, track=track)
        run.stage = "transcribe"
        raw_by_track[track] = router.transcribe_track(
            dst,
            track,
            meeting.language_mode,
            settings,
            audit,
            engines=engines,
            roster=meeting.roster,
            glossary_entries=entries,
            meeting_id=meeting.id,
            classification=meeting.classification.value,
        )
    segments = router.merge_tracks(raw_by_track)
    if diarizer is not None and segments:
        run.stage = "diarize"
        wav = (
            converted[0][1]
            if len(converted) == 1
            else next((p for t, p, _ in converted if t == "system"), converted[0][1])
        )
        turns = diarizer.diarize(wav, min_speakers=None, max_speakers=None)
        segments = assign_speakers(segments, turns)
        audit.append(
            "diarize.completed",
            meeting.id,
            classification=meeting.classification.value,
            turns=len(turns),
            labels=len({t.label for t in turns}),
            backend=getattr(diarizer, "name", type(diarizer).__name__),
        )
    all_raw = [s for segs in raw_by_track.values() for s in segs]
    return segments, all_raw, silent


def ingest_file(
    path: Path, meeting: Meeting, settings: Settings, audit: Any, **kw: Any
) -> Transcript:
    """Transcribe one recording file as the ``file`` track; see ``ingest_tracks``."""
    return ingest_tracks({"file": Path(path)}, meeting, settings, audit, **kw).transcript
