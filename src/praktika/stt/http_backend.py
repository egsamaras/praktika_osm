"""OpenAI-compatible HTTP transcription.

The server path: vLLM serving Whisper or Cohere behind your API gateway. Each chunk is posted as
an in-memory WAV to ``/v1/audio/transcriptions`` with ``response_format=verbose_json``. The
``httpx.Client`` is injected by the caller and must come from ``Settings.http_client()`` so the
egress allow-list is enforced per request (C-01); tests pass a ``MockTransport`` client.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from praktika.audio.convert import duration_s
from praktika.errors import PraktikaError
from praktika.logging import get_logger
from praktika.models import RawSegment, SpeechChunk
from praktika.stt.base import (
    SAMPLE_RATE,
    SttLanguage,
    clean_segments,
    read_chunk,
    whisper_result_to_raw,
)

log = get_logger(__name__)

TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"


def _attach_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the response segments, folding top-level ``words`` into them by time."""
    segments = [dict(s) for s in payload.get("segments") or []]
    words = payload.get("words") or []
    if not segments and payload.get("text"):
        segments = [
            {"start": 0.0, "end": float(payload.get("duration") or 0.0), "text": payload["text"]}
        ]
    if words and segments and not any(s.get("words") for s in segments):
        for s in segments:
            s["words"] = [
                w for w in words if float(s["start"]) <= float(w["start"]) < float(s["end"])
            ]
    return segments


class HttpTranscriber:
    """``Transcriber`` posting multipart chunks to an OpenAI-compatible transcription endpoint."""

    name = "http"
    auto_language = True  # no ``language`` field is sent when the router gives no hint

    def __init__(
        self, base_url: str, model: str, client: httpx.Client, *, initial_prompt: str | None = None
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.client = client
        self.initial_prompt = initial_prompt or None
        self.engine = f"http/{model}"

    def unload(self) -> None:
        """Nothing to free: the model lives on the server and the client is owned by the caller."""
        return None

    def _post(self, audio_bytes: bytes, language: SttLanguage | None) -> dict[str, Any]:
        data: dict[str, str] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language is not None:
            data["language"] = language
        if self.initial_prompt:
            data["prompt"] = self.initial_prompt
        try:
            resp = self.client.post(
                self.base_url + TRANSCRIPTIONS_PATH,
                data=data,
                files={"file": ("chunk.wav", audio_bytes, "audio/wav")},
            )
        except httpx.HTTPError as exc:
            raise PraktikaError(f"STT request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PraktikaError(f"STT server returned {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise PraktikaError("STT server returned non-JSON body") from exc
        if not isinstance(payload, dict):
            raise PraktikaError("STT server returned an unexpected JSON shape")
        return payload

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]:
        """Post each chunk; ``language=None`` lets the server detect the language."""
        total = duration_s(wav)
        raw: list[RawSegment] = []
        for chunk in chunks:
            audio = read_chunk(wav, chunk)
            if audio.size == 0:
                continue
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            payload = self._post(buf.getvalue(), language)
            raw.extend(whisper_result_to_raw(_attach_words(payload), chunk, engine=self.engine))
        out = clean_segments(raw, duration_s=total)
        log.info("stt.completed", engine=self.engine, chunks=len(chunks), segments=len(out))
        return out
