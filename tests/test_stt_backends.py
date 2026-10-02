"""Tests for the STT backends and ``stt.base`` helpers.

Everything here runs without weights. The two ``@pytest.mark.models`` smoke tests run only when
``PRAKTIKA_MODELS_DIR`` points at mirrored weights (``stt_en`` / ``stt_ar`` sub-directories).
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import FIXTURES

from praktika.config import Settings
from praktika.errors import EgressError, PraktikaError
from praktika.models import Attendee, RawSegment, SpeechChunk, Word
from praktika.stt import base
from praktika.stt.faster_whisper_backend import FasterWhisperTranscriber
from praktika.stt.http_backend import HttpTranscriber
from praktika.stt.mlx_cohere_backend import MlxCohereTranscriber
from praktika.stt.mlx_whisper_backend import MlxWhisperTranscriber

MODELS_DIR = Path(os.environ.get("PRAKTIKA_MODELS_DIR", "")).expanduser()
FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_meeting.wav"


def _chunk(start: float, end: float, index: int = 0) -> SpeechChunk:
    return SpeechChunk(index=index, track="file", start=start, end=end)


def _raw(start: float, end: float, text: str, words: list[Word] | None = None) -> RawSegment:
    return RawSegment(
        start=start,
        end=end,
        text=text,
        language="en",
        confidence=0.7,
        engine="t",
        words=words or [],
    )


# --------------------------------------------------------------------------- HTTP backend

VERBOSE_JSON = {
    "text": "Good morning everyone. Let us start.",
    "language": "english",
    "duration": 1.5,
    "segments": [
        {"id": 0, "start": 0.0, "end": 0.8, "text": " Good morning everyone.", "avg_logprob": -0.2},
        {"id": 1, "start": 0.8, "end": 1.4, "text": " Let us start.", "avg_logprob": -0.5},
        {"id": 2, "start": 1.6, "end": 2.4, "text": " Beyond the chunk.", "avg_logprob": -0.1},
    ],
    "words": [
        {"word": "Good", "start": 0.0, "end": 0.3},
        {"word": "morning", "start": 0.3, "end": 0.6},
        {"word": "everyone.", "start": 0.6, "end": 0.8},
        {"word": "Let", "start": 0.8, "end": 1.0},
    ],
}


def test_http_transcriber_multipart_and_parse(tone_wav: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VERBOSE_JSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    stt = HttpTranscriber(
        "http://127.0.0.1:8000/", "whisper-large-v3", client, initial_prompt="Acme Bank"
    )

    out = stt.transcribe(tone_wav, [_chunk(1.0, 2.5)], "en")

    request = seen[0]
    assert request.url == "http://127.0.0.1:8000/v1/audio/transcriptions"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="file"; filename="chunk.wav"' in body and b"RIFF" in body
    assert b'name="model"\r\n\r\nwhisper-large-v3' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="response_format"\r\n\r\nverbose_json' in body
    assert b'name="prompt"\r\n\r\nAcme Bank' in body

    assert [s.text for s in out] == ["Good morning everyone.", "Let us start."]
    assert (out[0].start, out[0].end) == (1.0, 1.8), "times offset by the chunk start"
    assert out[1].end == 2.4 and out[1].start == 1.8
    assert out[0].confidence == pytest.approx(0.8) and out[1].confidence == pytest.approx(0.5)
    assert [w.text for w in out[0].words] == ["Good", "morning", "everyone."]
    assert out[0].words[1].start == pytest.approx(1.3)
    assert [w.text for w in out[1].words] == ["Let"]
    assert all(s.engine == "http/whisper-large-v3" and s.language == "en" for s in out)
    assert stt.unload() is None
    # no route hint: the language field is omitted so the server detects it (auto_language)
    assert HttpTranscriber.auto_language is True
    stt.transcribe(tone_wav, [_chunk(1.0, 2.5)], None)
    assert b'name="language"' not in seen[1].content


def test_http_transcriber_errors(tone_wav: Path) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    stt = HttpTranscriber(
        "http://127.0.0.1:8000", "m", httpx.Client(transport=httpx.MockTransport(failing))
    )
    with pytest.raises(PraktikaError, match="503"):
        stt.transcribe(tone_wav, [_chunk(0.0, 1.0)], "en")

    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>")

    stt = HttpTranscriber(
        "http://127.0.0.1:8000", "m", httpx.Client(transport=httpx.MockTransport(garbage))
    )
    with pytest.raises(PraktikaError, match="non-JSON"):
        stt.transcribe(tone_wav, [_chunk(0.0, 1.0)], "en")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    stt = HttpTranscriber(
        "http://127.0.0.1:8000", "m", httpx.Client(transport=httpx.MockTransport(down))
    )
    with pytest.raises(PraktikaError, match="request failed"):
        stt.transcribe(tone_wav, [_chunk(0.0, 1.0)], "en")


def test_http_transcriber_blocked_outside_allow_list(
    tone_wav: Path, tmp_settings: Settings
) -> None:
    client = tmp_settings.http_client(timeout=5)
    stt = HttpTranscriber("http://stt.example.com", "m", client)
    with pytest.raises(EgressError):
        stt.transcribe(tone_wav, [_chunk(0.0, 1.0)], "en")


def test_http_text_only_response_becomes_one_segment(tone_wav: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "خلاص نبدأ", "duration": 1.0})

    stt = HttpTranscriber(
        "http://127.0.0.1:8000", "m", httpx.Client(transport=httpx.MockTransport(handler))
    )
    out = stt.transcribe(tone_wav, [_chunk(0.5, 1.5)], "ar")
    assert len(out) == 1 and out[0].language == "ar" and out[0].confidence is None
    assert (out[0].start, out[0].end) == (0.5, 1.5)


# --------------------------------------------------------------------------- lazy backends


def test_faster_whisper_backend_constructs_without_model(
    monkeypatch: pytest.MonkeyPatch, tone_wav: Path
) -> None:
    stt = FasterWhisperTranscriber("/models/stt_en", device="cpu", compute_type="int8")
    assert stt.name == "faster_whisper" and stt.engine == "faster-whisper/stt_en"
    stt.unload()  # never loaded: must not raise

    real_import = importlib.import_module

    def no_faster_whisper(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "praktika.stt.faster_whisper_backend.importlib.import_module", no_faster_whisper
    )
    with pytest.raises(PraktikaError, match="cuda"):
        stt.transcribe(tone_wav, [_chunk(0.0, 1.0)], "en")
    with pytest.raises(PraktikaError, match="cuda"):
        stt.detect(tone_wav, _chunk(0.0, 1.0))


def test_mlx_backends_construct_without_weights() -> None:
    whisper = MlxWhisperTranscriber(Path("/nonexistent/stt_en"), initial_prompt="")
    assert whisper.name == "mlx_whisper" and whisper.engine == "mlx-whisper/stt_en"
    assert whisper.initial_prompt is None, "empty prompt is normalised to None"
    whisper.unload()
    cohere = MlxCohereTranscriber(Path("/nonexistent/stt_ar"))
    assert cohere.name == "mlx_cohere" and cohere.engine == "mlx-audio/stt_ar"
    cohere.unload()


# --------------------------------------------------------------------------- base helpers


class _Entry:
    def __init__(self, canonical: str, arabic: list[str] | None = None) -> None:
        self.canonical = canonical
        self.arabic_variants = arabic or []


def test_vocab_prompt_names_and_terms(roster: list[Attendee]) -> None:
    entries = [_Entry("ManCom", ["لجنة الإدارة"]), {"canonical": "ALCO"}, "KYC", "ALCO"]
    en = base.vocab_prompt(roster, entries, "en")
    assert en.startswith("Internal meeting. Attendees: ")
    assert "F. Khalid" in en and "Omar Nasser" in en and "T. Brennan" in en
    assert "AI Lab Meeting Room" not in en, "room devices are not spoken names"
    assert "Carlet" not in en and "Oma" not in en.split(), "misrendering aliases never primed"
    assert en.endswith("Terms: ManCom, ALCO, KYC.") and en.count("ALCO") == 1

    latin_only = Attendee(name="P. Okoro", role="Auditor", aliases=["Peter Okoro"])
    ar = base.vocab_prompt([*roster, latin_only], entries, "ar")
    assert "فيصل خالد" in ar and "F. Khalid" not in ar
    assert "توم برينان" in ar and "T. Brennan" not in ar
    assert "P. Okoro" in ar, "Latin name kept when no Arabic alias exists"
    assert "لجنة الإدارة" in ar and "ManCom" in ar


def test_vocab_prompt_empty_and_truncated(roster: list[Attendee]) -> None:
    assert base.vocab_prompt([], [], "en") == ""
    many = [f"Term{i:03d}" for i in range(400)]
    prompt = base.vocab_prompt(roster, many, "en")
    assert len(prompt) <= base.PROMPT_MAX_CHARS
    assert base.prompt_tokens(prompt) <= base.PROMPT_MAX_TOKENS
    assert "Term000" in prompt and "Term399" not in prompt and prompt.endswith(".")
    for a in roster:
        if a.status != "room":
            assert a.name in prompt, "truncation drops terms, never attendee names"


def test_vocab_prompt_arabic_budget_keeps_every_name(roster: list[Attendee]) -> None:
    """Whisper keeps the last 223 prompt tokens: the Arabic prompt must stay inside the token
    budget (not just 700 characters) with names first, framed in Arabic, or the names are the
    part Whisper drops."""
    import yaml

    from praktika import glossary

    entries, _ = glossary.load(Path(__file__).resolve().parent.parent / "glossary.yaml")
    # The example glossary is short, so pad it with Arabic terms until the budget really bites.
    padded = [*entries, *(f"مصطلح تجريبي رقم {i}" for i in range(60))]
    prompt = base.vocab_prompt(roster, padded, "ar")
    assert prompt.startswith("اجتماع داخلي. الحضور: ")
    assert base.prompt_tokens(prompt) <= base.PROMPT_MAX_TOKENS
    data = yaml.safe_load((FIXTURES / "roster_data_team.yaml").read_text(encoding="utf-8"))
    for a in data["attendees"]:
        if a["status"] == "room":
            continue
        arabic = next((al for al in a.get("aliases", []) if re.search(r"[؀-ۿ]", al)), None)
        assert (arabic or a["name"]) in prompt, a["name"]
    # the untruncated prompt would have been far over the budget, so terms were dropped
    assert "مصطلح تجريبي رقم 59" not in prompt
    full = base.vocab_prompt([], padded, "ar")
    assert base.prompt_tokens(full) <= base.PROMPT_MAX_TOKENS
    assert "مصطلح تجريبي رقم 59" not in full
    assert base.prompt_tokens("x" * 800) > base.PROMPT_MAX_TOKENS


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("Good morning everyone.", "en"),
        ("صباح الخير للجميع", "ar"),
        ("تحديث الـ Credit Policy يكون جاهز", "mixed"),
        ("250,000 ...", "unknown"),
        ("", "unknown"),
        ("The security review، نحتاجها قبل البدء", "mixed"),
    ],
)
def test_language_tag(text: str, tag: str) -> None:
    assert base.language_tag(text) == tag


def test_logprob_to_confidence() -> None:
    assert base.logprob_to_confidence(None) is None
    assert base.logprob_to_confidence(-0.25) == pytest.approx(0.75)
    assert base.logprob_to_confidence(-3.0) == 0.0
    assert base.logprob_to_confidence(0.5) == 1.0


def test_clean_segments_clips_and_dedupes() -> None:
    segs = [
        _raw(5.0, 7.0, "  second  "),
        _raw(0.0, 2.0, "First line"),
        _raw(2.0, 3.0, "   "),
        _raw(
            7.0,
            12.0,
            "second",
            words=[Word(start=7.0, end=8.0, text="second"), Word(start=11.0, end=11.5, text="x")],
        ),
        _raw(9.5, 12.0, "Tail clamped"),
        _raw(10.0, 13.0, "On the server"),
        _raw(10.5, 14.0, "on the server"),
    ]
    out = base.clean_segments(segs, duration_s=10.0)
    assert [s.text for s in out] == ["First line", "second", "Tail clamped"]
    assert out[1].end == 7.0, "consecutive duplicate dropped, first kept"
    assert out[2].end == 10.0, "end clamped to the audio duration"


def test_whisper_result_to_raw_clips_to_chunk() -> None:
    chunk = _chunk(10.0, 12.0)
    segs = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": " one ",
            "avg_logprob": -0.1,
            "words": [{"word": " one", "start": 0.0, "end": 1.0, "probability": 0.9}],
        },
        {
            "start": 1.5,
            "end": 3.5,
            "text": "two",
            "words": [
                {"word": "two", "start": 1.5, "end": 1.9},
                {"word": "late", "start": 2.5, "end": 3.0},
            ],
        },
        {"start": 2.0, "end": 3.0, "text": "beyond"},
        {"start": 0.5, "end": 0.6, "text": ""},
    ]
    out = base.whisper_result_to_raw(segs, chunk, engine="e")
    assert [(s.start, s.end, s.text) for s in out] == [(10.0, 11.0, "one"), (11.5, 12.0, "two")]
    assert out[0].words[0].text == "one" and out[0].words[0].prob == 0.9
    assert [w.text for w in out[1].words] == ["two"], "words starting past the chunk end dropped"
    assert out[1].confidence is None and out[0].confidence == pytest.approx(0.9)


def test_read_chunk_rejects_wrong_rate(tmp_path: Path, tone_wav: Path) -> None:
    import numpy as np
    import soundfile as sf

    audio = base.read_chunk(tone_wav, _chunk(1.0, 2.0))
    assert audio.shape == (16_000,) and audio.dtype == np.float32
    assert base.read_chunk(tone_wav, _chunk(2.9, 5.0)).shape == (1_600,), "clamped to the file"
    sf.write(tmp_path / "44k.wav", np.zeros(4410, dtype=np.float32), 44_100)
    with pytest.raises(ValueError, match="44100"):
        base.read_chunk(tmp_path / "44k.wav", _chunk(0.0, 0.1))


# --------------------------------------------------------------------------- real weights


@pytest.mark.models
@pytest.mark.skipif(not (MODELS_DIR / "stt_en").exists(), reason="no PRAKTIKA_MODELS_DIR/stt_en")
def test_mlx_whisper_real(roster: list[Attendee]) -> None:
    pytest.importorskip("mlx_whisper")
    from praktika.audio.vad import speech_chunks

    chunks = speech_chunks(FIXTURE)[:3]
    stt = MlxWhisperTranscriber(
        MODELS_DIR / "stt_en", initial_prompt=base.vocab_prompt(roster, [], "en")
    )
    try:
        lang, prob = stt.detect(FIXTURE, chunks[0])
        assert lang == "en" and prob > 0.5
        out = stt.transcribe(FIXTURE, chunks, "en")
    finally:
        stt.unload()
    text = " ".join(s.text for s in out).lower()
    assert out and "data team" in text.replace("-", " ")
    assert all(chunks[0].start <= s.start < s.end <= chunks[-1].end + 0.01 for s in out)
    assert out[0].words and out[0].confidence is not None
    assert json.dumps([s.model_dump() for s in out])  # serialisable


@pytest.mark.models
@pytest.mark.skipif(not (MODELS_DIR / "stt_ar").exists(), reason="no PRAKTIKA_MODELS_DIR/stt_ar")
def test_mlx_cohere_real() -> None:
    pytest.importorskip("mlx_audio")
    from praktika.audio.vad import speech_chunks

    chunks = [c for c in speech_chunks(FIXTURE) if 20.0 < c.start < 40.0][:2]  # the Arabic line
    stt = MlxCohereTranscriber(MODELS_DIR / "stt_ar")
    try:
        out = stt.transcribe(FIXTURE, chunks, "ar")
    finally:
        stt.unload()
    assert out and all(s.confidence is None for s in out)
    assert any(s.language in ("ar", "mixed") for s in out)
    assert [(s.start, s.end) for s in out] == [(c.start, c.end) for c in chunks[: len(out)]]
