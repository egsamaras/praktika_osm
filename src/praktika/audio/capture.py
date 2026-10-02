"""Live capture (control C-04).

Two capturers behind one ``Capturer`` protocol:

* ``MicCapturer`` records an in-room microphone with ``sounddevice`` into a 16 kHz mono WAV
  (created 0600). The input stream is built by an injectable factory so tests drive it with a
  fake stream and no audio device. ``preflight`` checks the audio library and the input device
  so ``start`` can refuse a host that cannot record before the consent gate.
* ``SckCapturer`` launches the signed ``praktika-capture`` helper (not in this repository) and
  speaks its JSON status protocol: one object per stdout line with ``event`` in ``started``
  (``tracks``), ``level`` (``system``, ``mic`` in 0..1), ``error`` (``message``) and ``closed``
  (``tracks``).
  It refuses to start unless ``helper_is_signed`` finds a Developer ID identity.

Both implement ``abort``: stop, overwrite every in-flight file with zeros, unlink. There is no
silent mode: ``levels()`` feeds the visible status line and ``check_silence`` warns loudly.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from praktika.audio.convert import create_private
from praktika.audio.signing import CODESIGN, helper_is_signed
from praktika.errors import PraktikaError
from praktika.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CODESIGN",
    "MicCapturer",
    "SckCapturer",
    "Track",
    "helper_is_signed",
    "purge_file",
]

TrackName = Literal["system", "mic"]
BlockCallback = Callable[[np.ndarray, int, Any, Any], None]
StreamFactory = Callable[[BlockCallback], Any]  # returns an object with start()/stop()/close()


class Track(BaseModel):
    """One captured audio file."""

    model_config = ConfigDict(extra="forbid")

    name: TrackName
    path: Path
    sample_rate: int = Field(gt=0)


class Capturer(Protocol):
    def start(self, out_dir: Path) -> None: ...
    def stop(self) -> list[Track]: ...
    def abort(self) -> None: ...
    def levels(self) -> tuple[float, float]: ...


def purge_file(path: Path) -> None:
    """Overwrite ``path`` with zeros, flush, then unlink. Missing files are ignored."""
    if not path.exists():
        return
    size = path.stat().st_size
    with path.open("r+b") as fh:
        fh.write(b"\0" * size)
        fh.flush()
        os.fsync(fh.fileno())
    path.unlink()


def _default_stream_factory(device: str | None, sample_rate: int, blocksize: int) -> StreamFactory:
    def factory(callback: BlockCallback) -> Any:
        import sounddevice as sd  # lazy: needs PortAudio, which CI hosts may lack

        return sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=blocksize,
            callback=callback,
        )

    return factory


class MicCapturer:
    """Record one microphone track to ``<out_dir>/mic.wav`` (16 kHz mono PCM-16, mode 0600).

    ``stream_factory(callback)`` must return an object with ``start()``, ``stop()`` and
    ``close()`` that delivers float32 mono blocks to ``callback(indata, frames, time, status)``;
    the default builds a ``sounddevice.InputStream``. Blocks are queued and written by a
    background thread so the audio callback never blocks on disk.
    """

    def __init__(
        self,
        device: str | None = None,
        sample_rate: int = 16_000,
        *,
        stream_factory: StreamFactory | None = None,
        blocksize: int = 1_600,
        silence_threshold_db: float = -50.0,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._own_stream = stream_factory is None
        self._factory = stream_factory or _default_stream_factory(device, sample_rate, blocksize)
        self._quiet_amp = 10 ** (silence_threshold_db / 20.0)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: Any = None
        self._file: sf.SoundFile | None = None
        self._writer: threading.Thread | None = None
        self._path: Path | None = None
        self._level = 0.0
        self._samples = 0
        self._quiet_samples = 0
        self._silence_warned = False

    def preflight(self) -> None:
        """Check that this host can record from the input device, before anything is written
        or any consent is recorded; raises ``PraktikaError`` when it cannot (no PortAudio
        library, no such input device). A no-op with an injected ``stream_factory``."""
        if not self._own_stream:
            return
        try:
            import sounddevice as sd  # lazy: needs PortAudio, which a server may not have
        except (ImportError, OSError) as exc:
            raise PraktikaError(
                f"this host cannot record from a microphone ({exc}); "
                "use `praktika ingest` with a recording or a Teams transcript instead"
            ) from exc
        try:
            sd.query_devices(self.device, "input")
        except Exception as exc:  # sounddevice raises ValueError or PortAudioError
            name = self.device or "the default input device"
            raise PraktikaError(f"cannot record from {name}: {exc}") from exc

    def start(self, out_dir: Path) -> None:
        """Create the file, start the writer thread and the input stream. Not re-entrant."""
        if self._stream is not None:
            raise PraktikaError("capture already running")
        self._path = Path(out_dir) / "mic.wav"
        create_private(self._path)
        self._file = sf.SoundFile(
            str(self._path), mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16"
        )
        self._writer = threading.Thread(target=self._drain, name="mic-writer", daemon=True)
        self._writer.start()
        self._stream = self._factory(self._on_block)
        self._stream.start()
        log.info("capture.started", track="mic", device=self.device or "default")

    def _on_block(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        block = np.asarray(indata, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        rms = float(np.sqrt(np.mean(block * block)))
        # Peak-hold with decay: a block is ~50 ms, so a raw RMS meter flickers to zero
        # between words; decaying by ~10% per block keeps speech visible for ~1 s.
        self._level = min(1.0, max(rms, self._level * 0.9))
        self._samples += block.size
        if rms < self._quiet_amp:
            self._quiet_samples += block.size
        self._queue.put(block.copy())

    def _drain(self) -> None:
        while True:
            block = self._queue.get()
            if block is None:
                return
            if self._file is not None:
                self._file.write(block)

    def levels(self) -> tuple[float, float]:
        """``(system, mic)`` RMS levels in 0..1; the system level is always 0 for a mic track."""
        return (0.0, self._level)

    def elapsed_s(self) -> float:
        """Seconds of audio received so far (derived from samples, not wall-clock)."""
        return self._samples / float(self.sample_rate)

    def check_silence(self, *, min_elapsed_s: float = 60.0, ratio: float = 0.95) -> bool:
        """Return True (and log a warning once) when the track is > ``ratio`` silent after
        ``min_elapsed_s`` seconds of audio. Before that, always False."""
        if self.elapsed_s() < min_elapsed_s or self._samples == 0:
            return False
        silent = self._quiet_samples / self._samples > ratio
        if silent and not self._silence_warned:
            self._silence_warned = True
            log.warning("capture.silence", track="mic", elapsed_s=round(self.elapsed_s(), 1))
        return silent

    def stop(self) -> list[Track]:
        """Stop the stream, flush the writer and close the file. Idempotent."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._writer is not None:
            self._queue.put(None)
            self._writer.join(timeout=10)
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._path is None or not self._path.exists():
            return []  # never started, or already aborted and purged
        os.chmod(self._path, 0o600)
        log.info("capture.stopped", track="mic", elapsed_s=round(self.elapsed_s(), 1))
        return [Track(name="mic", path=self._path, sample_rate=self.sample_rate)]

    def abort(self) -> None:
        """Kill switch: stop, overwrite the file with zeros and unlink it."""
        tracks = self.stop()
        for t in tracks:
            purge_file(t.path)
        log.warning("capture.aborted", track="mic")


class SckCapturer:
    """Drive the signed ScreenCaptureKit helper: system + mic tracks written by the helper."""

    def __init__(
        self,
        helper: Path,
        *,
        sample_rate: int = 16_000,
        stop_timeout_s: float = 15.0,
        team_id: str | None = None,
    ):
        self.helper = Path(helper)
        self.team_id = team_id
        self.sample_rate = sample_rate
        self.stop_timeout_s = stop_timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self._tracks: list[Track] = []
        self._levels = (0.0, 0.0)
        self.errors: list[str] = []

    def start(self, out_dir: Path) -> None:
        """Verify the signature, launch the helper and start parsing its status lines.

        Raises ``PraktikaError`` when the helper is not Developer ID signed or cannot start.
        """
        identity = helper_is_signed(self.helper, team_id=self.team_id)
        if identity is None:
            raise PraktikaError(
                f"capture helper {self.helper} is not Developer ID signed"
                + (f" by team {self.team_id}" if self.team_id else "")
            )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cmd = [str(self.helper), "--out", str(out_dir), "--rate", str(self.sample_rate)]
        try:
            # Fixed argument list of a signature-verified binary; no shell.
            self._proc = subprocess.Popen(  # noqa: S603
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError as exc:
            raise PraktikaError(f"cannot launch capture helper: {exc}") from exc
        self._closed.clear()
        self._reader = threading.Thread(target=self._read, name="sck-reader", daemon=True)
        self._reader.start()
        log.info("capture.started", track="sck", identity=identity)

    def _handle(self, msg: dict[str, Any]) -> None:
        event = msg.get("event")
        if event in ("started", "closed") and "tracks" in msg:
            self._tracks = [
                Track(
                    name=t["name"],
                    path=Path(t["path"]),
                    sample_rate=int(t.get("sample_rate", self.sample_rate)),
                )
                for t in msg["tracks"]
            ]
        if event == "level":
            self._levels = (float(msg.get("system", 0.0)), float(msg.get("mic", 0.0)))
        elif event == "error":
            self.errors.append(str(msg.get("message", "")))
            log.error("capture.helper_error", message=msg.get("message"))
        elif event == "closed":
            self._closed.set()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None  # noqa: S101
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("capture.helper_noise", line=line[:120])
                continue
            if isinstance(msg, dict):
                self._handle(msg)
        self._closed.set()

    def levels(self) -> tuple[float, float]:
        return self._levels

    def stop(self) -> list[Track]:
        """SIGINT the helper, wait for ``closed``, chmod tracks 0600 and return them."""
        proc = self._proc
        if proc is None:
            return list(self._tracks)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        if not self._closed.wait(self.stop_timeout_s):
            proc.kill()
            log.error("capture.helper_timeout")
        proc.wait(timeout=self.stop_timeout_s)
        if self._reader is not None:
            self._reader.join(timeout=self.stop_timeout_s)
        self._proc = None
        for t in self._tracks:
            if t.path.exists():
                os.chmod(t.path, 0o600)
        log.info("capture.stopped", track="sck", tracks=[t.name for t in self._tracks])
        return list(self._tracks)

    def abort(self) -> None:
        """Kill switch: stop the helper, overwrite every track with zeros and unlink."""
        for t in self.stop():
            purge_file(t.path)
        log.warning("capture.aborted", track="sck")
