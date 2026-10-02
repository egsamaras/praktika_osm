"""Tests for ``praktika.stt.router``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from conftest import FakeDetector, FakeTranscriber

from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.models import LanguageMode, RawSegment, SpeechChunk
from praktika.stt import router
from praktika.stt.http_backend import HttpTranscriber
from praktika.stt.mlx_whisper_backend import MlxWhisperTranscriber

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_meeting.wav"
WAV = Path("/tmp/unused.wav")  # noqa: S108 — never opened; fakes ignore the path


def _chunks(n: int) -> list[SpeechChunk]:
    return [
        SpeechChunk(index=i, track="file", start=i * 10.0, end=i * 10.0 + 8.0) for i in range(n)
    ]


def _raw(start: float, text: str, lang: str = "en", engine: str = "fake") -> RawSegment:
    return RawSegment(
        start=start, end=start + 2.0, text=text, language=lang, confidence=0.8, engine=engine
    )


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, Any]]] = []

    def append(self, event: str, meeting_id: str | None, **detail: Any) -> None:
        self.events.append((event, meeting_id, detail))


# --------------------------------------------------------------------------- route


def test_en_mode_keeps_english_but_diverts_confident_arabic() -> None:
    """``en`` mode sends everything to the English engine except chunks LID is sure are
    Arabic (never turbo for Arabic); no detector means everything is English."""
    detector = FakeDetector("en", 0.99, per_chunk={2: ("ar", 0.95), 3: ("ar", 0.6)})
    table = router.route(_chunks(4), "en", detector, WAV, 0.9)
    assert [c.index for c in table["en"]] == [0, 1, 3]
    assert [c.index for c in table["ar"]] == [2]
    assert detector.calls == [0, 1, 2, 3]
    table = router.route(_chunks(4), "en", None, WAV, 0.9)
    assert [c.index for c in table["en"]] == [0, 1, 2, 3] and table["ar"] == []


def test_route_hints_follow_lid_confidence() -> None:
    """Whisper engines get the route's language only when LID agrees (p >= 0.5); a
    low-confidence chunk is decoded with ``None`` so Whisper detects rather than transliterates."""
    det = {0: ("en", 0.82), 1: ("ar", 0.97), 2: ("en", 0.95), 3: ("ar", 0.3), 4: ("en", 0.4)}
    routed = router.route_chunks(_chunks(5), "ar-mixed", det, 0.9)
    assert [(r.engine, r.hint) for r in routed] == [
        ("ar", None),  # English at 0.82 < threshold: Arabic engine, but no forced 'ar'
        ("ar", "ar"),
        ("en", "en"),
        ("ar", None),  # Arabic at 0.3: below CONFIDENT, no hint
        ("ar", None),
    ]
    routed = router.route_chunks(_chunks(5), "en", det, 0.9)
    assert [(r.engine, r.hint) for r in routed] == [
        ("en", "en"),
        ("ar", "ar"),
        ("en", "en"),
        ("en", None),
        ("en", None),
    ]
    assert [(r.engine, r.hint) for r in router.route_chunks(_chunks(2), "ar-mixed", None, 0.9)] == [
        ("ar", "ar"),
        ("ar", "ar"),
    ]


def test_ar_mixed_chunk_below_threshold_goes_to_ar() -> None:
    table = router.route(_chunks(1), "ar-mixed", FakeDetector("en", 0.89), WAV, 0.9)
    assert table["ar"] == _chunks(1) and table["en"] == []


def test_ar_mixed_chunk_at_exact_threshold_goes_to_en() -> None:
    table = router.route(_chunks(1), "ar-mixed", FakeDetector("en", 0.9), WAV, 0.9)
    assert table["en"] == _chunks(1) and table["ar"] == []


def test_ar_mixed_confident_arabic_goes_to_ar() -> None:
    table = router.route(_chunks(1), "ar-mixed", FakeDetector("ar", 0.99), WAV, 0.9)
    assert table["ar"] == _chunks(1)


def test_ar_mixed_without_detector_all_to_ar() -> None:
    table = router.route(_chunks(3), "ar-mixed", None, WAV, 0.9)
    assert len(table["ar"]) == 3 and table["en"] == []


def test_route_per_chunk_mixture_keeps_order() -> None:
    detector = FakeDetector("en", 0.95, per_chunk={1: ("ar", 0.9), 3: ("en", 0.5)})
    table = router.route(_chunks(5), "ar-mixed", detector, WAV, 0.9)
    assert [c.index for c in table["en"]] == [0, 2, 4]
    assert [c.index for c in table["ar"]] == [1, 3]


# --------------------------------------------------------------------------- resolve_mode


def test_auto_resolves_ar_share() -> None:
    chunks = _chunks(10)
    # Up to 24 chunks are sampled, so every chunk of a short meeting is examined.
    two_arabic = FakeDetector("en", 0.9, per_chunk={2: ("ar", 0.9), 9: ("ar", 0.9)})
    assert router.resolve_mode(LanguageMode.auto, two_arabic, WAV, chunks) == "ar-mixed"
    assert two_arabic.calls == list(range(10))
    # A single confidently Arabic chunk is enough: the Arabic engine also handles English.
    one_arabic = FakeDetector("en", 0.9, per_chunk={2: ("ar", 0.9)})
    assert router.resolve_mode(LanguageMode.auto, one_arabic, WAV, chunks) == "ar-mixed"
    # Weak Arabic guesses below the share threshold do not flip the mode.
    weak = FakeDetector("en", 0.9, per_chunk={2: ("ar", 0.55)})
    assert router.resolve_mode("auto", weak, WAV, chunks) == "en"
    # but three weak Arabic guesses out of ten reach the 0.3 share
    three_weak = FakeDetector("en", 0.9, per_chunk={i: ("ar", 0.55) for i in (1, 4, 8)})
    assert router.resolve_mode("auto", three_weak, WAV, chunks) == "ar-mixed"


def test_auto_catches_clustered_arabic_agenda_item() -> None:
    """100 chunks with one Arabic block (30-45, 16 %): six spread samples used to miss it and
    resolve to ``en``; now the block is sampled and the meeting is ``ar-mixed``."""
    chunks = _chunks(100)
    clustered = FakeDetector("en", 0.95, per_chunk={i: ("ar", 0.92) for i in range(30, 46)})
    assert router.resolve_mode(LanguageMode.auto, clustered, WAV, chunks) == "ar-mixed"
    assert len(clustered.calls) == 24 and any(30 <= i <= 45 for i in clustered.calls)
    # LID results already computed are reused rather than recomputed
    det = router.detect_all(chunks, clustered, WAV)
    before = len(clustered.calls)
    assert router.resolve_mode("auto", clustered, WAV, chunks, detections=det) == "ar-mixed"
    assert len(clustered.calls) == before


def test_explicit_modes_pass_through_without_detection() -> None:
    detector = FakeDetector("ar")
    assert router.resolve_mode(LanguageMode.en, detector, WAV, _chunks(3)) == "en"
    assert router.resolve_mode("ar-mixed", detector, WAV, _chunks(3)) == "ar-mixed"
    assert detector.calls == []


def test_auto_without_detector_or_chunks_is_en() -> None:
    """Without language ID (the HTTP speech path) there is no evidence of Arabic: ``auto`` is
    English, never ``ar-mixed`` (which sent every chunk to an Arabic engine an HTTP-only host
    does not serve)."""
    assert router.resolve_mode(LanguageMode.auto, None, WAV, _chunks(3)) == "en"
    assert router.resolve_mode(LanguageMode.auto, FakeDetector("en"), WAV, []) == "en"
    assert router.resolve_mode("auto", None, WAV, _chunks(3)) == "en"


def test_unknown_mode_rejected() -> None:
    with pytest.raises(PraktikaError):
        router.resolve_mode("fr", None, WAV, _chunks(1))


# --------------------------------------------------------------------------- merge_tracks


def test_merge_assigns_ids_and_self() -> None:
    merged = router.merge_tracks(
        {
            "system": [_raw(0.0, "Good morning."), _raw(6.0, "مرحبا", "ar")],
            "mic": [_raw(3.0, "Thanks, let us start.")],
            "file": [_raw(9.0, "Closing remarks.")],
        }
    )
    assert [s.id for s in merged] == ["S0001", "S0002", "S0003", "S0004"]
    assert [s.start for s in merged] == [0.0, 3.0, 6.0, 9.0]
    assert (merged[1].speaker, merged[1].speaker_kind, merged[1].track) == ("ME", "self", "mic")
    assert (merged[0].speaker, merged[0].speaker_kind, merged[0].track) == (
        "SPEAKER_00",
        "label",
        "system",
    )
    assert (merged[3].speaker, merged[3].speaker_kind, merged[3].track) == (
        "unknown",
        "unknown",
        "file",
    )
    assert merged[2].language == "ar" and merged[2].text == "مرحبا"
    assert all(s.engine == "fake" and s.confidence == 0.8 for s in merged)


def test_merge_rejects_unknown_track_and_handles_empty() -> None:
    assert router.merge_tracks({}) == []
    assert router.merge_tracks({"mic": []}) == []
    with pytest.raises(ValueError, match="unknown track"):
        router.merge_tracks({"vtt": [_raw(0.0, "x")]})


# --------------------------------------------------------------------------- transcribe_track


def test_engines_unloaded_in_order(tmp_settings: Settings) -> None:
    log: list[str] = []
    en = FakeTranscriber([_raw(10.0, "English part")], name="en", log=log)
    ar = FakeTranscriber([_raw(2.0, "الجزء العربي", "ar")], name="ar", log=log)
    detector = FakeDetector("en", 0.95, per_chunk={0: ("ar", 0.99)})
    audit = _Audit()

    segments = router.transcribe_track(
        FIXTURE,
        "file",
        LanguageMode.ar_mixed,
        tmp_settings,
        audit,
        engines=(en, ar, detector),
        meeting_id="M-20260916-a1b2",
    )

    assert log == ["transcribe:en", "unload:en", "transcribe:ar", "unload:ar"]
    assert [s.text for s in segments] == ["الجزء العربي", "English part"], "sorted by start"
    assert en.calls[0][2] == "en" and ar.calls[0][2] == "ar"
    routed = {c.index for c in en.calls[0][1]} | {c.index for c in ar.calls[0][1]}
    assert routed == set(detector.calls) and 0 in {c.index for c in ar.calls[0][1]}
    assert all(c.track == "file" for c in en.calls[0][1])

    event, meeting_id, detail = audit.events[0]
    assert (event, meeting_id) == ("stt.completed", "M-20260916-a1b2")
    assert detail["mode"] == "ar-mixed" and detail["segments"] == 2
    assert detail["engines"] == {"stt_en": "en", "stt_ar": "ar"}
    assert detail["chunks"] == sum(detail["routed"].values()) > 0


def test_en_mode_runs_only_english_engine(tmp_settings: Settings) -> None:
    log: list[str] = []
    en = FakeTranscriber([_raw(1.0, "hello")], name="en", log=log)
    ar = FakeTranscriber([_raw(1.0, "never")], name="ar", log=log)
    out = router.transcribe_track(FIXTURE, "file", "en", tmp_settings, None, engines=(en, ar, None))
    assert log == ["transcribe:en", "unload:en"]
    assert [s.text for s in out] == ["hello"]


class _RecordingWhisper(FakeTranscriber):
    """A Whisper-like fake: accepts ``language=None`` and records what it was given."""

    auto_language = True


def test_whisper_engines_get_no_forced_language_on_low_confidence(tmp_settings: Settings) -> None:
    """ar-mixed, LID says ('en', 0.8): the chunk goes to the Arabic engine, and a Whisper-class
    Arabic engine is not forced to 'ar' (it would transliterate the English); Cohere-class
    engines (``auto_language`` False) still get 'ar'."""
    log: list[str] = []
    en = _RecordingWhisper([], name="en", log=log)
    ar = _RecordingWhisper([_raw(1.0, "text")], name="ar", log=log)
    detector = FakeDetector("en", 0.8)
    audit = _Audit()
    router.transcribe_track(
        FIXTURE, "file", "ar-mixed", tmp_settings, audit, engines=(en, ar, detector)
    )
    assert en.calls == [] and len(ar.calls) == 1
    assert ar.calls[0][2] is None, "no language forced on a low-confidence route"
    assert audit.events[0][2]["unhinted"] == len(ar.calls[0][1]) > 0
    # the same route with a Cohere-class engine is decoded as Arabic
    cohere = FakeTranscriber([_raw(1.0, "نص")], name="ar", log=[])
    engines = (en, cohere, FakeDetector("en", 0.8))
    router.transcribe_track(FIXTURE, "file", "ar-mixed", tmp_settings, None, engines=engines)
    assert cohere.calls[0][2] == "ar"
    # explicit en mode with LID agreeing forces 'en'; LID saying Arabic diverts the chunk
    en2 = _RecordingWhisper([_raw(1.0, "hi")], name="en", log=[])
    ar2 = _RecordingWhisper([_raw(2.0, "مرحبا")], name="ar", log=[])
    router.transcribe_track(
        FIXTURE, "file", "en", tmp_settings, None, engines=(en2, ar2, FakeDetector("en", 0.97))
    )
    assert en2.calls and en2.calls[0][2] == "en" and ar2.calls == []
    en3 = _RecordingWhisper([], name="en", log=[])
    ar3 = _RecordingWhisper([_raw(2.0, "مرحبا")], name="ar", log=[])
    router.transcribe_track(
        FIXTURE, "file", "en", tmp_settings, None, engines=(en3, ar3, FakeDetector("ar", 0.97))
    )
    assert en3.calls == [] and ar3.calls[0][2] == "ar", "en mode never decodes Arabic as English"


def test_engine_unloaded_even_when_it_raises(tmp_settings: Settings) -> None:
    class Boom(FakeTranscriber):
        def transcribe(self, wav: Path, chunks: list[SpeechChunk], language: str) -> list[Any]:
            raise RuntimeError("engine crashed")

    log: list[str] = []
    en = Boom(name="en", log=log)
    with pytest.raises(RuntimeError, match="crashed"):
        router.transcribe_track(FIXTURE, "file", "en", tmp_settings, None, engines=(en, en, None))
    assert log == ["unload:en"]


def test_empty_track(tmp_settings: Settings, tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    log: list[str] = []
    en = FakeTranscriber([_raw(0.0, "x")], name="en", log=log)
    ar = FakeTranscriber([_raw(0.0, "y")], name="ar", log=log)
    audit = _Audit()

    out = router.transcribe_track(
        silent, "mic", "auto", tmp_settings, audit, engines=(en, ar, None)
    )

    assert out == []
    assert log == [], "no speech: no engine is run or unloaded"
    assert audit.events[0][2]["chunks"] == 0 and audit.events[0][2]["engines"] == {}


# --------------------------------------------------------------------------- build_transcribers


def test_build_transcribers_fake_settings(tmp_settings: Settings) -> None:
    en, ar, detector = router.build_transcribers(tmp_settings)
    assert isinstance(en, router.NullTranscriber) and isinstance(ar, router.NullTranscriber)
    assert detector is None
    assert en.transcribe(WAV, _chunks(1), "en") == [] and en.unload() is None


def test_build_transcribers_mlx_uses_vocab_prompt(
    tmp_settings: Settings, roster: list[Any]
) -> None:
    settings = tmp_settings.model_copy(update={"stt_en": "mlx_whisper", "stt_ar": "mlx_cohere"})
    en, ar, detector = router.build_transcribers(settings, roster=roster, glossary_entries=["ALCO"])
    assert isinstance(en, MlxWhisperTranscriber) and detector is en
    assert en.model_dir == settings.models_dir / "stt_en"
    assert en.initial_prompt is not None
    assert "F. Khalid" in en.initial_prompt and "ALCO" in en.initial_prompt
    assert ar.model_dir == settings.models_dir / "stt_ar"


def test_build_transcribers_http_requires_url(tmp_settings: Settings) -> None:
    with pytest.raises(PraktikaError, match="stt_http_url"):
        router.build_transcribers(tmp_settings.model_copy(update={"stt_en": "http"}))
    settings = Settings(
        _env_file=None,
        stt_en="http",
        stt_ar="http",
        stt_http_url="http://127.0.0.1:8000",
        models_dir=tmp_settings.models_dir,
        data_dir=tmp_settings.data_dir,
    )
    en, ar, detector = router.build_transcribers(settings)
    assert isinstance(en, HttpTranscriber) and isinstance(ar, HttpTranscriber)
    assert en.base_url == "http://127.0.0.1:8000" and detector is None


def test_http_path_auto_mode_routes_every_chunk_to_english(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Linux server profile: both engines ``http`` (vLLM Whisper), so there is no detector.
    ``auto`` must send every chunk to the English model, none to the Arabic one."""
    from praktika.config import AllowListTransport

    posted: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.content)
        return httpx.Response(
            200,
            json={
                "text": "Good morning.",
                "language": "english",
                "segments": [{"start": 0.0, "end": 1.0, "text": " Good morning."}],
            },
        )

    def mock_client(self: Settings, *, timeout: float | None = None) -> httpx.Client:
        inner = httpx.MockTransport(handler)
        return httpx.Client(transport=AllowListTransport(self.allowed_hosts, inner=inner))

    monkeypatch.setattr(Settings, "http_client", mock_client)
    settings = tmp_settings.model_copy(
        update={"stt_en": "http", "stt_ar": "http", "stt_http_url": "http://127.0.0.1:8000"}
    )
    audit = _Audit()

    segments = router.transcribe_track(
        FIXTURE, "file", LanguageMode.auto, settings, audit, meeting_id="M-20260925-a1b2"
    )

    detail = audit.events[-1][2]
    assert detail["mode"] == "en"
    assert detail["routed"] == {"en": detail["chunks"], "ar": 0} and detail["chunks"] > 0
    assert detail["engines"] == {"stt_en": "http/whisper-large-v3"}
    assert posted and all(b'name="model"\r\n\r\nwhisper-large-v3' in body for body in posted)
    assert not any(b"cohere-transcribe-arabic" in body for body in posted)
    assert segments and all(s.engine == "http/whisper-large-v3" for s in segments)


# --------------------------------------------------------------------------- weights integrity


def test_tampered_weights_refuse_transcription(
    tmp_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local backend never loads weights that differ from the register (C-13)."""
    from praktika.errors import ModelRegisterMismatch
    from praktika.models_registry import manage, weights
    from praktika.stt import engines as engines_mod

    weights.reset_cache()
    settings = tmp_settings.model_copy(update={"stt_en": "mlx_whisper", "stt_ar": "fake"})
    loaded: list[str] = []

    class _NeverLoads(FakeTranscriber):
        auto_language = True

        def transcribe(self, wav: Path, chunks: list[SpeechChunk], language: Any) -> list[Any]:
            loaded.append("transcribe")
            return []

    monkeypatch.setattr(
        engines_mod, "MlxWhisperTranscriber", lambda *a, **k: _NeverLoads(name="mlx")
    )
    # no register at all: refused before any backend is built
    with pytest.raises(ModelRegisterMismatch, match="no model register"):
        router.transcribe_track(FIXTURE, "file", "en", settings, _Audit())
    stt_en = settings.models_dir / "stt_en"
    stt_en.mkdir(parents=True)
    (stt_en / "weights.safetensors").write_bytes(b"\x00" * 64)
    manage.register("stt_en", stt_en, settings, repo="x/y", revision="r1", licence="MIT",
                    conversion=None)  # fmt: skip
    audit = _Audit()
    router.transcribe_track(FIXTURE, "file", "en", settings, audit, meeting_id="M-20260916-a1b2")
    assert [e[0] for e in audit.events][0] == "models.verified"
    assert audit.events[0][2]["roles"] == ["stt_en"]
    # verified once per process for an unchanged register
    router.transcribe_track(FIXTURE, "file", "en", settings, audit)
    assert [e[0] for e in audit.events].count("models.verified") == 1
    # tamper with the weights: the next run refuses and nothing is transcribed
    (stt_en / "weights.safetensors").write_bytes(b"\x01" * 64)
    weights.reset_cache()
    before = len(loaded)
    with pytest.raises(ModelRegisterMismatch, match="hash mismatch"):
        router.transcribe_track(FIXTURE, "file", "en", settings, _Audit())
    assert len(loaded) == before
    # remote/fake backends load nothing and need no register
    assert weights.loads_local_weights(tmp_settings) is False
    router.transcribe_track(FIXTURE, "file", "en", tmp_settings, None)


# --------------------------------------------------------------------------- Arabic switched off


def test_arabic_off_sends_every_chunk_to_english(tmp_settings: Settings) -> None:
    """With ``stt_ar = none`` a chunk LID would call Arabic still goes to the English engine as
    English: the regression where an English meeting reached a deleted Arabic model."""
    settings = tmp_settings.model_copy(update={"stt_ar": "none"})
    log: list[str] = []
    en = FakeTranscriber([_raw(1.0, "hello")], name="en", log=log)
    ar = FakeTranscriber([_raw(1.0, "never")], name="ar", log=log)
    detector = FakeDetector("ar", 0.99)
    audit = _Audit()

    segments = router.transcribe_track(
        FIXTURE, "file", LanguageMode.en, settings, audit, engines=(en, ar, detector)
    )

    assert log == ["transcribe:en", "unload:en"], "the Arabic engine is never touched"
    assert detector.calls == [], "no language ID when there is nothing to route to"
    assert {c[2] for c in en.calls} == {"en"}
    assert [s.text for s in segments] == ["hello"]
    detail = audit.events[0][2]
    assert detail["mode"] == "en" and detail["routed"]["ar"] == 0


def test_arabic_off_resolves_auto_to_english(tmp_settings: Settings) -> None:
    settings = tmp_settings.model_copy(update={"stt_ar": "none"})
    en = FakeTranscriber([_raw(1.0, "hello")], name="en")
    ar = FakeTranscriber([_raw(1.0, "never")], name="ar")
    router.transcribe_track(
        FIXTURE, "file", LanguageMode.auto, settings, None,
        engines=(en, ar, FakeDetector("ar", 0.99)),
    )  # fmt: skip
    assert en.calls and not ar.calls


def test_arabic_off_refuses_ar_mixed(tmp_settings: Settings) -> None:
    settings = tmp_settings.model_copy(update={"stt_ar": "none"})
    with pytest.raises(PraktikaError, match="Arabic transcription is switched off"):
        router.require_language(settings, LanguageMode.ar_mixed)
    with pytest.raises(PraktikaError, match="use --lang en"):
        router.transcribe_track(
            FIXTURE, "file", "ar-mixed", settings, None,
            engines=(FakeTranscriber([]), FakeTranscriber([]), None),
        )  # fmt: skip
    router.require_language(settings, LanguageMode.en)
    router.require_language(settings, LanguageMode.auto)
    router.require_language(tmp_settings.model_copy(update={"stt_ar": "fake"}), "ar-mixed")


def test_arabic_is_off_by_default() -> None:
    assert Settings.model_fields["stt_ar"].default == "none"
