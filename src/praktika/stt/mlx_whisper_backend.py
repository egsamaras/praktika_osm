"""Whisper on Apple silicon via mlx-whisper.

Serves as the English engine (``whisper-large-v3-turbo``), the Arabic fallback engine
(``whisper-large-v3`` full, ``stt_ar="mlx_whisper_full"``) and the per-chunk language detector.
Every ``mlx_whisper`` / ``mlx`` import is inside a method so this module imports on any platform
and tests can construct the backend without weights.

Each chunk is decoded from its own in-memory slice with ``clip_timestamps=[0, chunk_len]``,
``condition_on_previous_text=False`` and the vocabulary ``initial_prompt``: without them Whisper
repeated itself past the end of the audio and misspelt attendee names.
Decoding a slice rather than the whole file with absolute clip bounds avoids re-decoding and
re-computing the mel spectrogram of the entire recording for every chunk.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from praktika.audio.convert import duration_s
from praktika.logging import get_logger
from praktika.models import RawSegment, SpeechChunk
from praktika.stt.base import (
    SttLanguage,
    clean_segments,
    read_chunk,
    whisper_result_to_raw,
)

log = get_logger(__name__)


class MlxWhisperTranscriber:
    """``Transcriber`` + ``LanguageDetector`` over a local mlx-whisper model directory.

    ``model_dir`` must hold ``config.json`` and ``weights.safetensors`` (a mirrored
    ``mlx-community/whisper-*`` snapshot). ``initial_prompt`` is the output of
    ``stt.base.vocab_prompt``. Nothing is loaded until the first ``transcribe`` or ``detect``.
    """

    name = "mlx_whisper"
    auto_language = True  # ``language=None`` lets Whisper detect per chunk (router hints)

    def __init__(
        self,
        model_dir: Path,
        *,
        word_timestamps: bool = True,
        initial_prompt: str | None = None,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt or None
        self.no_speech_threshold = no_speech_threshold
        self.engine = f"mlx-whisper/{self.model_dir.name}"
        self._model: Any = None

    # ----------------------------------------------------------------- lifecycle

    def _holder(self) -> Any:
        return importlib.import_module("mlx_whisper.transcribe").ModelHolder

    def _load(self) -> Any:
        """Load (or reuse) the model through mlx-whisper's own cache so ``transcribe`` shares it."""
        if self._model is None:
            mx = importlib.import_module("mlx.core")
            self._model = self._holder().get_model(str(self.model_dir), mx.float16)
            log.info("stt.loaded", engine=self.engine)
        return self._model

    def unload(self) -> None:
        """Drop every reference to the weights and release Metal buffers."""
        self._model = None
        try:
            holder = self._holder()
            holder.model = None
            holder.model_path = None
            importlib.import_module("mlx.core").clear_cache()
        except ImportError:  # never loaded on this host
            return
        log.info("stt.unloaded", engine=self.engine)

    # ----------------------------------------------------------------- transcription

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]:
        """Decode each chunk with ``language`` fixed (``None``: Whisper's own per-chunk
        detection); returns cleaned, time-ordered segments."""
        mlx_whisper = importlib.import_module("mlx_whisper")
        total = duration_s(wav)
        raw: list[RawSegment] = []
        for chunk in chunks:
            audio = read_chunk(wav, chunk)
            if audio.size == 0:
                continue
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=str(self.model_dir),
                language=language,
                task="transcribe",
                word_timestamps=self.word_timestamps,
                clip_timestamps=[0.0, chunk.end - chunk.start],
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
                no_speech_threshold=self.no_speech_threshold,
                verbose=None,
            )
            raw.extend(whisper_result_to_raw(result.get("segments", []), chunk, engine=self.engine))
        self._model = self._holder().model
        segments = clean_segments(raw, duration_s=total)
        log.info("stt.completed", engine=self.engine, chunks=len(chunks), segments=len(segments))
        return segments

    # ----------------------------------------------------------------- language id

    def detect(self, wav: Path, chunk: SpeechChunk) -> tuple[str, float]:
        """Whisper LID on the first 30 s of ``chunk``: ``(iso_code, probability)``."""
        model = self._load()
        audio_mod = importlib.import_module("mlx_whisper.audio")
        decoding = importlib.import_module("mlx_whisper.decoding")
        audio = read_chunk(wav, chunk)[: audio_mod.N_SAMPLES]
        if audio.size == 0:
            return ("unknown", 0.0)
        mel = audio_mod.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        mel = audio_mod.pad_or_trim(mel, audio_mod.N_FRAMES, axis=-2).astype(
            importlib.import_module("mlx.core").float16
        )
        _, probs = decoding.detect_language(model, mel)
        table = probs[0] if isinstance(probs, list) else probs
        code = max(table, key=table.get)
        return (str(code), float(table[code]))
