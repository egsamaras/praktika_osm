"""Whisper via faster-whisper / CTranslate2.

The non-Apple path (x86 CUDA hosts, Linux CI). ``faster_whisper`` is an optional extra and is
imported lazily, so the backend constructs anywhere; the first ``transcribe`` or ``detect`` on a
host without the package raises ``PraktikaError`` naming the missing extra. We run our own VAD,
so ``vad_filter=False``; decoding options mirror the mlx backend.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from praktika.audio.convert import duration_s
from praktika.errors import PraktikaError
from praktika.logging import get_logger
from praktika.models import RawSegment, SpeechChunk
from praktika.stt.base import SttLanguage, clean_segments, read_chunk, whisper_result_to_raw

log = get_logger(__name__)


def _segment_dict(seg: Any) -> dict[str, Any]:
    """Flatten a faster-whisper ``Segment`` (a NamedTuple) into the shared dict shape."""
    words = [
        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
        for w in (seg.words or [])
    ]
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "avg_logprob": seg.avg_logprob,
        "words": words,
    }


class FasterWhisperTranscriber:
    """``Transcriber`` + ``LanguageDetector`` over a CTranslate2 Whisper model."""

    name = "faster_whisper"
    auto_language = True  # ``language=None`` lets Whisper detect per chunk (router hints)

    def __init__(
        self,
        model_dir_or_name: str,
        device: str = "cuda",
        compute_type: str = "int8",
        *,
        word_timestamps: bool = True,
        initial_prompt: str | None = None,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self.model_dir_or_name = str(model_dir_or_name)
        self.device = device
        self.compute_type = compute_type
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt or None
        self.no_speech_threshold = no_speech_threshold
        self.engine = f"faster-whisper/{Path(self.model_dir_or_name).name}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                fw = importlib.import_module("faster_whisper")
            except ImportError as exc:
                raise PraktikaError(
                    "faster-whisper is not installed; install the 'cuda' extra"
                ) from exc
            self._model = fw.WhisperModel(
                self.model_dir_or_name,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )
            log.info("stt.loaded", engine=self.engine, device=self.device)
        return self._model

    def unload(self) -> None:
        """Drop the model reference; CTranslate2 frees device memory on collection."""
        self._model = None

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]:
        """Decode each chunk with ``language`` fixed (``None``: auto-detect per chunk)."""
        model = self._load()
        total = duration_s(wav)
        raw: list[RawSegment] = []
        for chunk in chunks:
            audio = read_chunk(wav, chunk)
            if audio.size == 0:
                continue
            segments, _info = model.transcribe(
                audio,
                language=language,
                task="transcribe",
                vad_filter=False,
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
                no_speech_threshold=self.no_speech_threshold,
            )
            dicts = [_segment_dict(s) for s in segments]
            raw.extend(whisper_result_to_raw(dicts, chunk, engine=self.engine))
        out = clean_segments(raw, duration_s=total)
        log.info("stt.completed", engine=self.engine, chunks=len(chunks), segments=len(out))
        return out

    def detect(self, wav: Path, chunk: SpeechChunk) -> tuple[str, float]:
        """``WhisperModel.detect_language`` on the chunk: ``(iso_code, probability)``."""
        model = self._load()
        audio = read_chunk(wav, chunk)
        if audio.size == 0:
            return ("unknown", 0.0)
        result = model.detect_language(audio)
        return (str(result[0]), float(result[1]))
