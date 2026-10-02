"""Engine construction for the STT router: the configured backends, the
vocabulary prompt they are primed with, and the ``fake`` stand-in.

Split from ``stt/router`` to keep that module within the house line budget; ``router``
re-exports ``Engines``, ``NullTranscriber`` and ``build_transcribers``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.models import Attendee, RawSegment, SpeechChunk
from praktika.stt.base import LanguageDetector, SttLanguage, Transcriber, vocab_prompt
from praktika.stt.faster_whisper_backend import FasterWhisperTranscriber
from praktika.stt.http_backend import HttpTranscriber
from praktika.stt.mlx_cohere_backend import MlxCohereTranscriber
from praktika.stt.mlx_whisper_backend import MlxWhisperTranscriber

Engines = tuple[Transcriber, Transcriber, LanguageDetector | None]


class NullTranscriber:
    """The ``fake`` configuration value: transcribes nothing. Tests inject real fakes."""

    name = "fake"
    engine = "fake"
    auto_language = False

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]:
        return []

    def unload(self) -> None:
        return None


def build_transcribers(
    settings: Settings,
    *,
    roster: Sequence[Attendee] = (),
    glossary_entries: Sequence[Any] = (),
) -> Engines:
    """Instantiate ``(english, arabic, detector)`` from ``settings`` without loading weights.

    Whisper engines receive a vocabulary prompt built from ``roster`` and ``glossary_entries``.
    The detector is the English engine when it can identify language (Whisper backends), else
    ``None``. Raises ``PraktikaError`` when ``http`` is selected without ``stt_http_url``.
    """
    prompt_en = vocab_prompt(list(roster), list(glossary_entries), "en")
    prompt_ar = vocab_prompt(list(roster), list(glossary_entries), "ar")

    def http(model: str, prompt: str) -> HttpTranscriber:
        if settings.stt_http_url is None:
            raise PraktikaError("stt_http_url must be set when an STT backend is 'http'")
        client = settings.http_client()
        return HttpTranscriber(str(settings.stt_http_url), model, client, initial_prompt=prompt)

    en: Transcriber
    match settings.stt_en:
        case "mlx_whisper":
            en = MlxWhisperTranscriber(settings.models_dir / "stt_en", initial_prompt=prompt_en)
        case "faster_whisper":
            en = FasterWhisperTranscriber(
                str(settings.models_dir / "stt_en"), initial_prompt=prompt_en
            )
        case "http":
            en = http("whisper-large-v3", prompt_en)
        case _:
            en = NullTranscriber()

    ar: Transcriber
    match settings.stt_ar:
        case "mlx_cohere":
            ar = MlxCohereTranscriber(settings.models_dir / "stt_ar")
        case "mlx_whisper_full":
            ar = MlxWhisperTranscriber(
                settings.models_dir / "stt_ar_full", initial_prompt=prompt_ar
            )
        case "faster_whisper_full":
            ar = FasterWhisperTranscriber(
                str(settings.models_dir / "stt_ar_full"), initial_prompt=prompt_ar
            )
        case "http":
            ar = http("cohere-transcribe-arabic", prompt_ar)
        case _:  # "none" (Arabic off; the router never routes to it) and "fake"
            ar = NullTranscriber()

    # The English engine doubles as the language detector when it can identify language.
    detector = en if callable(getattr(en, "detect", None)) else None
    return en, ar, detector
