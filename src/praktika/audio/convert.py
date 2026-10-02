"""Audio conversion and inspection.

Every recording enters the pipeline as 16 kHz mono PCM-16 WAV so VAD, STT and diarisation
share one representation. Conversion is delegated to ffmpeg (a system prerequisite checked by
``praktika doctor``); the output file is created with mode 0600 before ffmpeg writes into it so
it is never world-readable, even transiently.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from praktika.errors import FfmpegError
from praktika.logging import get_logger

log = get_logger(__name__)

FFMPEG_BIN = "ffmpeg"
TARGET_RATE = 16_000
_FRAME_S = 0.02  # analysis window for the silence detector
_BLOCK_FRAMES = 16_000 * 30  # read audio in 30-second blocks to bound memory


class AudioInfo(BaseModel):
    """Facts about a converted WAV file. ``sha256`` is over the file bytes."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_s: float = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def create_private(path: Path) -> None:
    """Create (or truncate) ``path`` with mode 0600 so later writers inherit the mode.

    Parent directories are created as needed. Shared by conversion and capture so every audio
    file Praktika writes is private from the moment it exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of the file at ``path`` (streamed, 1 MiB blocks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_wav16k(src: Path, dst: Path, *, ffmpeg: str | None = None) -> AudioInfo:
    """Convert ``src`` (any ffmpeg-readable container) to 16 kHz mono PCM-16 WAV at ``dst``.

    ``dst`` is created with mode 0600 and overwritten if present. Raises ``FfmpegError`` when
    the ffmpeg binary cannot be executed or exits non-zero; the message carries ffmpeg's stderr.
    Returns an ``AudioInfo`` describing the written file.
    """
    src = Path(src)
    dst = Path(dst)
    binary = ffmpeg or FFMPEG_BIN
    if not src.exists():
        raise FfmpegError(f"input file does not exist: {src}")
    create_private(dst)
    cmd = [
        binary,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ar",
        str(TARGET_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dst),
    ]
    try:
        # Fixed argument list, no shell; ffmpeg is a declared system prerequisite.
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except (FileNotFoundError, PermissionError) as exc:
        dst.unlink(missing_ok=True)
        raise FfmpegError(f"ffmpeg not executable ({binary!r}): {exc}") from exc
    if proc.returncode != 0:
        dst.unlink(missing_ok=True)
        raise FfmpegError(
            f"ffmpeg exited with {proc.returncode} converting {src.name}: {proc.stderr.strip()}"
        )
    os.chmod(dst, 0o600)
    info = sf.info(str(dst))
    result = AudioInfo(
        path=dst,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        duration_s=float(info.frames) / float(info.samplerate),
        sha256=file_sha256(dst),
    )
    log.info("audio.converted", src=src.name, duration_s=round(result.duration_s, 2))
    return result


def duration_s(path: Path) -> float:
    """Return the duration of an audio file in seconds (0.0 for an empty file)."""
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _frame_db(block: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS level in dBFS for each complete ``frame_len``-sample frame of a mono block."""
    n = (len(block) // frame_len) * frame_len
    if n == 0:
        return np.empty(0, dtype=np.float64)
    frames = block[:n].reshape(-1, frame_len).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def is_mostly_silent(path: Path, threshold_db: float = -50.0, ratio: float = 0.95) -> bool:
    """Return True when at least ``ratio`` of 20 ms frames are below ``threshold_db`` dBFS.

    Multi-channel files are averaged to mono first. A file with no complete frame (including an
    empty file) counts as silent, because there is nothing to transcribe.
    """
    total = 0
    quiet = 0
    with sf.SoundFile(str(path)) as fh:
        frame_len = max(1, int(fh.samplerate * _FRAME_S))
        for block in fh.blocks(blocksize=_BLOCK_FRAMES, dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            levels = _frame_db(mono, frame_len)
            total += len(levels)
            quiet += int(np.count_nonzero(levels < threshold_db))
    if total == 0:
        return True
    return quiet / total >= ratio
