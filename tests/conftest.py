"""Shared fixtures and offline fakes.

Every fake here satisfies the corresponding ``Protocol`` structurally, so the whole suite runs with
no network and no model weights. Fakes for models owned by other modules (``Turn``, ``Track``,
``Identity``) import the real class when it exists and otherwise fall back to a local twin with
the same fields.
"""

from __future__ import annotations

import base64
import copy
import importlib
import os
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import soundfile as sf
import yaml
from freezegun import freeze_time
from pydantic import BaseModel

from praktika.config import Settings
from praktika.errors import LLMError
from praktika.ids import segment_id
from praktika.models import Attendee, RawSegment, Segment, SpeechChunk, Transcript

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TONE_WAV = FIXTURES / "tone_16k.wav"
FROZEN_NOW = "2026-09-16T09:00:00+03:00"


def _model(module: str, name: str, fallback: type[BaseModel]) -> type[Any]:
    """Return ``module.name`` if importable, else ``fallback`` (same fields)."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return fallback


class _Turn(BaseModel):
    start: float
    end: float
    label: str


class _Track(BaseModel):
    name: Literal["system", "mic"]
    path: Path
    sample_rate: int


class _Identity(BaseModel):
    user: str
    display: str
    source: Literal["session", "local", "oidc", "fake"]
    groups: list[str] = []


# --------------------------------------------------------------------------- FakeLLM

_LINE_RE = re.compile(r"\[(S\d{4,5}) \d\d:\d\d:\d\d-\d\d:\d\d:\d\d [^\]]*\] ([^\n]*)")
_DEFAULT_QUOTE = "Good morning everyone, let us start with the notetaker pilot."

DEFAULT_RESPONSES: dict[str, dict[str, Any]] = {
    "ChunkFindings": {
        "decisions": [
            {
                "statement": "The notetaker pilot starts on 1 October, limited to the data team.",
                "kind": "approved",
                "decided_by": "Committee",
                "dissent_or_conditions": None,
                "refs": ["S0001"],
                "quote": _DEFAULT_QUOTE,
            }
        ],
        "actions": [
            {
                "description": "Draft the bilingual privacy notice and share it with Legal.",
                "owner": "Omar Nasser",
                "owner_confidence": "explicit",
                "due_text": "by Thursday",
                "refs": ["S0001"],
                "quote": _DEFAULT_QUOTE,
            }
        ],
        "questions": [],
        "risks": [],
        "key_points": [
            {"topic": "Notetaker pilot", "point": "Pilot scope agreed.", "refs": ["S0001"]}
        ],
        "figures": [],
    },
    "Narrative": {
        "summary": "The meeting agreed the notetaker pilot scope and assigned the privacy notice.",
        "topics": [
            {
                "title": "Notetaker pilot",
                "summary": "The pilot scope was agreed.",
                "key_points": ["Pilot scope agreed."],
                "refs": ["S0001"],
            }
        ],
    },
    "RetractionVerdict": {"retracted": False, "refs": [], "note": "No reversal found."},
    "ClassificationSuggestion": {
        "suggested": "internal",
        "reasons": [],
        "identifiers_seen": [],
        "mnpi_keywords": [],
    },
}
DEFAULT_RESPONSES["MergedFindings"] = {
    **copy.deepcopy(DEFAULT_RESPONSES["ChunkFindings"]),
    "retracted_decisions": [],
}


def _fill_refs(obj: Any, seg_id: str, quote: str) -> Any:
    """Point every ``refs`` at ``seg_id`` and every ``quote`` at ``quote`` (in place)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "refs" and isinstance(v, list) and v:
                obj[k] = [seg_id]
            elif k == "quote":
                obj[k] = quote[:240]
            else:
                _fill_refs(v, seg_id, quote)
    elif isinstance(obj, list):
        for item in obj:
            _fill_refs(item, seg_id, quote)
    return obj


@dataclass
class LLMCall:
    system: str
    user: str
    schema_title: str
    temperature: float
    seed: int
    max_tokens: int
    schema: dict[str, Any] = field(repr=False, default_factory=dict)


class FakeLLM:
    """Offline ``LLMClient``: canned schema-valid dicts keyed by the JSON schema's ``title``.

    ``playback={"ChunkFindings": [d1, d2]}`` returns queued dicts in order before falling back to
    the canned default. When the user prompt contains rendered transcript lines, canned refs and
    quotes are rewritten to the first line so they verify against that transcript. Every call is
    recorded in ``.calls``. ``fail_next_with_invalid_json()`` makes the next call return a dict
    that matches no schema, exercising the caller's retry path.
    """

    name = "fake"

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        playback: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.responses = {**copy.deepcopy(DEFAULT_RESPONSES), **(responses or {})}
        self.playback = {k: list(v) for k, v in (playback or {}).items()}
        self.calls: list[LLMCall] = []
        self._fail_next = False

    def fail_next_with_invalid_json(self) -> None:
        self._fail_next = True

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        seed: int = 7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        title = str(schema.get("title", ""))
        self.calls.append(LLMCall(system, user, title, temperature, seed, max_tokens, schema))
        if self._fail_next:
            self._fail_next = False
            return {"not_json": "«this is not valid output»"}
        queue = self.playback.get(title)
        if queue:
            return copy.deepcopy(queue.pop(0))
        if title not in self.responses:
            raise LLMError(f"FakeLLM has no canned response for schema {title!r}")
        out = copy.deepcopy(self.responses[title])
        lines = _LINE_RE.findall(user)
        if lines:
            _fill_refs(out, lines[0][0], lines[0][1])
        return out

    def model_digest(self) -> str:
        return "sha256:fake"


# --------------------------------------------------------------------------- STT / diarise


class FakeTranscriber:
    """``Transcriber`` returning ``RawSegment``s keyed by chunk index, or a flat list."""

    def __init__(
        self,
        segments: dict[int, list[RawSegment]] | list[RawSegment] | None = None,
        *,
        name: str = "fake",
        log: list[str] | None = None,
    ) -> None:
        self.name = name
        self.segments = segments if segments is not None else []
        self.log = log if log is not None else []
        self.calls: list[tuple[Path, list[SpeechChunk], str]] = []
        self.unloaded = 0

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: Literal["en", "ar"]
    ) -> list[RawSegment]:
        self.calls.append((wav, list(chunks), language))
        self.log.append(f"transcribe:{self.name}")
        if isinstance(self.segments, dict):
            return [seg for c in chunks for seg in self.segments.get(c.index, [])]
        return list(self.segments)

    def unload(self) -> None:
        self.unloaded += 1
        self.log.append(f"unload:{self.name}")


class FakeDetector:
    """``LanguageDetector`` with a fixed language, optionally overridden per chunk index."""

    def __init__(
        self,
        language: str = "en",
        prob: float = 0.99,
        per_chunk: dict[int, tuple[str, float]] | None = None,
    ) -> None:
        self.language, self.prob, self.per_chunk = language, prob, per_chunk or {}
        self.calls: list[int] = []

    def detect(self, wav: Path, chunk: SpeechChunk) -> tuple[str, float]:
        self.calls.append(chunk.index)
        return self.per_chunk.get(chunk.index, (self.language, self.prob))


class FakeDiarizer:
    """``Diarizer`` returning fixed turns; produces labels and times only, never embeddings."""

    def __init__(self, turns: list[tuple[float, float, str]] | None = None) -> None:
        self.turns = turns or []
        self.calls: list[Path] = []

    def diarize(
        self, wav: Path, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> list[Any]:
        turn_cls = _model("praktika.diarize.base", "Turn", _Turn)
        self.calls.append(wav)
        return [turn_cls(start=s, end=e, label=lbl) for s, e, lbl in self.turns]


# --------------------------------------------------------------------------- capture / identity


class FakeCapturer:
    """``Capturer`` writing a copy of the tone fixture per track (0600); ``abort`` purges it."""

    def __init__(self, tracks: tuple[str, ...] = ("system", "mic"), source: Path = TONE_WAV):
        self.track_names, self.source = tracks, source
        self.events: list[str] = []
        self._paths: list[Path] = []

    def start(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in self.track_names:
            dst = out_dir / f"{name}.wav"
            shutil.copyfile(self.source, dst)
            os.chmod(dst, 0o600)
            self._paths.append(dst)
        self.events.append("start")

    def stop(self) -> list[Any]:
        track_cls = _model("praktika.audio.capture", "Track", _Track)
        self.events.append("stop")
        return [
            track_cls(name=p.stem, path=p, sample_rate=16000) for p in self._paths if p.exists()
        ]

    def abort(self) -> None:
        for p in self._paths:
            if p.exists():
                size = p.stat().st_size
                p.write_bytes(b"\0" * size)
                p.unlink()
        self.events.append("abort")

    def levels(self) -> tuple[float, float]:
        return (0.2, 0.2)


class FakeIdentity:
    """``IdentityProvider`` returning a fixed synthetic identity."""

    def __init__(
        self,
        user: str = "f.khalid@acme.test",
        display: str = "F. Khalid",
        source: str = "fake",
        groups: tuple[str, ...] = ("Praktika-Users",),
    ) -> None:
        self.user, self.display, self.source, self.groups = user, display, source, list(groups)

    def current(self, request: Any = None) -> Any:
        identity_cls = _model("praktika.identity", "Identity", _Identity)
        return identity_cls(
            user=self.user, display=self.display, source=self.source, groups=self.groups
        )


# --------------------------------------------------------------------------- transcripts

_EN_LINES = [
    _DEFAULT_QUOTE,
    "The prototype transcribes English and Arabic locally; nothing leaves the bank's devices.",
    "I propose we start with volunteers from the data team only.",
    "Agreed. The pilot starts on the first of October, limited to the data team.",
    "I will draft the privacy notice and the consent script by Thursday and share it with Legal.",
    "One risk: transcripts may include customer names, so we redact before the model sees them.",
    "Open question: do we keep raw audio at all after the minutes are approved?",
    "The GPU business case goes to ManCom on the twenty-second of October.",
    "The budget line is BHD 250,000 for the first phase.",
    "Let us not decide the retention question today; we take it offline.",
    "To summarise: pilot approved from the first of October, notice by the eighteenth.",
    "Thank you all. Meeting closed.",
]
_AR_LINES = [
    "صباح الخير للجميع، نبدأ بمشروع مدوّن الملاحظات.",
    "النموذج يفرّغ العربية والإنجليزية محلياً ولا يخرج شيء من أجهزة البنك.",
    "أقترح أن نبدأ بالمتطوعين من فريق البيانات فقط.",
    "اتفقنا. يبدأ المشروع التجريبي في الأول من أكتوبر ويقتصر على فريق البيانات.",
    "سأعدّ إشعار الخصوصية ونص الموافقة قبل الخميس وأشاركه مع الإدارة القانونية.",
    "خطر واحد: قد تتضمن النصوص أسماء عملاء، لذلك نحذفها قبل أن يراها النموذج.",
    "سؤال مفتوح: هل نحتفظ بالتسجيل الصوتي بعد اعتماد المحضر؟",
    "تُعرض دراسة جدوى وحدات المعالجة الرسومية على لجنة الإدارة في الثاني والعشرين من أكتوبر.",
    "الميزانية المخصصة للمرحلة الأولى هي 250,000 دينار بحريني.",
    "لن نقرر مسألة الاحتفاظ اليوم؛ نؤجلها.",
    "للتلخيص: المشروع معتمد من الأول من أكتوبر، والإشعار قبل الثامن عشر.",
    "شكراً للجميع. انتهى الاجتماع.",
]
_MIXED_LINES = [
    "خلاص، we go with option two for the notetaker pilot.",
    "تحديث الـ Credit Policy يكون جاهز inshallah by Thursday.",
    "The InfoSec review، نحتاجها قبل البدء، by the twenty-fifth.",
    "طيب، the budget is BHD 250,000 للمرحلة الأولى.",
]
_SPEAKERS = ["F. Khalid", "R. Haddad", "L. Farouk", "Omar Nasser"]


def make_transcript(
    lang: Literal["en", "ar", "mixed"] = "en",
    n: int = 12,
    *,
    meeting_id: str = "M-20260916-a1b2",
    source: Literal["file", "vtt", "docx", "capture", "graph"] = "vtt",
    redacted: bool = True,
) -> Transcript:
    """Build a valid synthetic ``Transcript`` with ids ``S0001..`` and 10-second segments.

    ``lang="mixed"`` interleaves English, Arabic and code-switched lines with matching tags.
    """
    segments: list[Segment] = []
    for i in range(n):
        if lang == "en":
            text, tag = _EN_LINES[i % len(_EN_LINES)], "en"
        elif lang == "ar":
            text, tag = _AR_LINES[i % len(_AR_LINES)], "ar"
        else:
            pool = [(t, "en") for t in _EN_LINES[:4]] + [(t, "ar") for t in _AR_LINES[:4]]
            pool += [(t, "mixed") for t in _MIXED_LINES]
            text, tag = pool[i % len(pool)]
        segments.append(
            Segment(
                id=segment_id(i + 1),
                start=float(i * 10),
                end=float(i * 10 + 9),
                speaker=_SPEAKERS[i % len(_SPEAKERS)],
                speaker_kind="identity",
                language=tag,
                text=text,
                confidence=0.9 if tag == "en" else None,
                track="vtt" if source == "vtt" else "file",
                engine="fake",
            )
        )
    return Transcript(
        meeting_id=meeting_id,
        source=source,
        engines={"stt_en": "fake", "stt_ar": "fake"},
        segments=segments,
        redacted=redacted,
    )


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep developer ``PRAKTIKA_*`` / ``.env`` settings out of the suite."""
    for key in list(os.environ):
        if key.startswith("PRAKTIKA_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    models = tmp_path / "models"
    models.mkdir()
    return Settings(
        _env_file=None,
        data_dir=data,
        models_dir=models,
        prompts_dir=REPO / "prompts",
        glossary_path=REPO / "glossary.yaml",
        allowed_hosts=["localhost", "127.0.0.1"],
        llm_provider="fake",
        stt_en="fake",
        stt_ar="fake",
        diarize_backend="fake",
        identity_provider="fake",
        audit_sink="jsonl",
        pilot_smoke=True,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_identity() -> FakeIdentity:
    return FakeIdentity()


@pytest.fixture
def frozen_clock() -> Iterator[Any]:
    with freeze_time(FROZEN_NOW) as frozen:
        yield frozen


@pytest.fixture(scope="session")
def roster() -> list[Attendee]:
    data = yaml.safe_load((FIXTURES / "roster_data_team.yaml").read_text(encoding="utf-8"))
    return [Attendee(**a) for a in data["attendees"]]


@pytest.fixture(scope="session")
def room_identities() -> list[str]:
    data = yaml.safe_load((FIXTURES / "roster_data_team.yaml").read_text(encoding="utf-8"))
    return list(data["room_identities"])


@pytest.fixture
def transcript_factory() -> Callable[..., Transcript]:
    return make_transcript


@pytest.fixture(scope="session")
def fixed_key() -> bytes:
    """A deterministic Fernet key: urlsafe-base64 of 32 fixed bytes (44 bytes on the wire)."""
    return base64.urlsafe_b64encode(bytes(range(1, 33)))


@pytest.fixture(scope="session")
def tone_wav() -> Path:
    """Path to the 3-second 440 Hz 16 kHz mono tone; regenerated if missing."""
    if not TONE_WAV.exists():
        t = np.arange(int(3.0 * 16000)) / 16000
        sf.write(TONE_WAV, (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), 16000)
    return TONE_WAV
