"""Cohere Transcribe Arabic via mlx-audio.

The primary Arabic / code-switched engine on Apple silicon. Cohere emits neither timestamps nor
language identification, so each segment spans exactly its VAD chunk (so citations are
chunk-precise, never wider), the language tag comes from the script ratio of the text, and
confidence is ``None``. All ``mlx_audio`` / ``mlx`` imports are lazy.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from praktika.audio.convert import duration_s
from praktika.logging import get_logger
from praktika.models import RawSegment, SpeechChunk
from praktika.stt.base import SAMPLE_RATE, SttLanguage, clean_segments, language_tag, read_chunk

log = get_logger(__name__)


class MlxCohereTranscriber:
    """``Transcriber`` over a local mlx-audio conversion of Cohere Transcribe Arabic."""

    name = "mlx_cohere"
    auto_language = False  # Cohere Transcribe Arabic always decodes as Arabic

    def __init__(self, model_dir: Path, *, max_tokens: int = 512) -> None:
        self.model_dir = Path(model_dir)
        self.max_tokens = max_tokens
        self.engine = f"mlx-audio/{self.model_dir.name}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            stt = importlib.import_module("mlx_audio.stt")
            self._model = stt.load(str(self.model_dir))
            log.info("stt.loaded", engine=self.engine)
        return self._model

    def unload(self) -> None:
        """Drop the model reference and release Metal buffers."""
        self._model = None
        try:
            importlib.import_module("mlx.core").clear_cache()
        except ImportError:
            return
        log.info("stt.unloaded", engine=self.engine)

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]:
        """One ``generate`` call per chunk; segment times are the chunk bounds. The engine
        handles Arabic and code-switched English alike, so ``language`` is always ``ar``."""
        language = "ar"
        model = self._load()
        total = duration_s(wav)
        raw: list[RawSegment] = []
        for chunk in chunks:
            audio = read_chunk(wav, chunk)
            if audio.size == 0:
                continue
            output = model.generate(
                audio, language=language, sample_rate=SAMPLE_RATE, max_tokens=self.max_tokens
            )
            text = str(getattr(output, "text", output)).strip()
            if not text:
                continue
            raw.append(
                RawSegment(
                    start=chunk.start,
                    end=chunk.end,
                    text=text,
                    language=language_tag(text),
                    confidence=None,
                    engine=self.engine,
                )
            )
        segments = clean_segments(raw, duration_s=total)
        log.info("stt.completed", engine=self.engine, chunks=len(chunks), segments=len(segments))
        return segments
