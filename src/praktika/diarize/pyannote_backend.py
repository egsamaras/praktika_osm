"""Diarisation with pyannote speaker-diarization-community-1.

Off by default (``PRAKTIKA_DIARIZE=1`` only after your data protection officer's written
position on transient speaker representations). The pipeline is loaded from a local mirror
(``config.yaml`` in ``model_dir``) with the Hub offline, and only the
``exclusive_speaker_diarization`` annotation is read: labels and times. The
pipeline's transient speaker representations are never accessed, stored or logged.
``pyannote.audio`` and ``torch`` are imported lazily (optional ``diarize`` extra).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from praktika.errors import PraktikaError
from praktika.logging import get_logger

from .base import Turn

log = get_logger(__name__)


def _annotation(output: Any) -> Any:
    """Pick the exclusive diarisation annotation from a pyannote 4 output (or an Annotation)."""
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        ann = getattr(output, attr, None)
        if ann is not None:
            return ann
    if hasattr(output, "itertracks"):
        return output
    raise PraktikaError("pyannote pipeline returned no diarisation annotation")


class PyannoteDiarizer:
    """``Diarizer`` over a locally mirrored pyannote pipeline."""

    name = "pyannote"

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                audio = importlib.import_module("pyannote.audio")
                torch = importlib.import_module("torch")
            except ImportError as exc:
                raise PraktikaError(
                    "pyannote.audio is not installed; install the 'diarize' extra"
                ) from exc
            config = self.model_dir / "config.yaml"
            if not config.exists():
                raise PraktikaError(f"diarisation pipeline config not found: {config}")
            pipeline = audio.Pipeline.from_pretrained(str(config))
            if pipeline is None:
                raise PraktikaError(f"could not load pyannote pipeline from {config}")
            self._pipeline = pipeline.to(torch.device(self.device))
            log.info("diarize.loaded", model_dir=str(self.model_dir), device=self.device)
        return self._pipeline

    def unload(self) -> None:
        self._pipeline = None

    def diarize(
        self, wav: Path, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> list[Turn]:
        """Run the pipeline and return time-ordered ``Turn``s (labels + times only)."""
        pipeline = self._load()
        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        output = pipeline(str(wav), **kwargs)
        turns = [
            Turn(start=float(seg.start), end=float(seg.end), label=str(label))
            for seg, _track, label in _annotation(output).itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        log.info("diarize.completed", turns=len(turns), speakers=len({t.label for t in turns}))
        return turns
