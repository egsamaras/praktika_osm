"""Model register: mirror, register, verify.

Contract: every model Praktika loads is listed in ``models.yaml`` (in ``settings.models_dir``)
with its Hugging Face repo and revision, licence, local path and the SHA-256 of every file.
``pull`` mirrors a repo with ``huggingface_hub.snapshot_download``; ``register`` records weights
that were mirrored by other means (air-gapped hosts, files downloaded on another machine);
``verify`` re-hashes everything and raises ``ModelRegisterMismatch`` on any difference;
``hashes_for_provenance`` gives one digest per role for the minutes' provenance record.

The Hub token is read from the process environment of the ``pull`` command only. It is passed
to the Hub client and never logged, stored in the register or written to disk by this module.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml

from praktika.config import Settings
from praktika.errors import ModelRegisterMismatch
from praktika.logging import get_logger
from praktika.models import ModelRecord, ModelRegister

REGISTER_FILE = "models.yaml"
ROLES: dict[str, tuple[str, str]] = {
    "stt_en": ("mlx-community/whisper-large-v3-turbo", "MIT"),
    "stt_ar_full": ("mlx-community/whisper-large-v3-mlx", "MIT"),
    "stt_ar": ("CohereLabs/cohere-transcribe-arabic-07-2026", "Apache-2.0"),
    "diarize": ("pyannote/speaker-diarization-community-1", "CC-BY-4.0"),
}
Downloader = Callable[[str, Path, "str | None"], tuple[Path, str]]
Converter = Callable[[Path], "str | None"]

log = get_logger(__name__)


def register_path(settings: Settings) -> Path:
    """Location of ``models.yaml``: inside ``models_dir`` so it travels with the weights."""
    return Path(settings.models_dir) / REGISTER_FILE


def load_register(settings: Settings) -> ModelRegister:
    """Parse ``models.yaml``; an absent file is an empty register."""
    path = register_path(settings)
    if not path.exists():
        return ModelRegister(models=[])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ModelRegister.model_validate(data)


def save_register(settings: Settings, register: ModelRegister) -> Path:
    path = register_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(register.model_dump(mode="json"), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under ``root`` keyed by POSIX relative path.

    Hidden entries (``.cache``, ``.gitattributes``, ``.lock`` files) and the register itself are
    skipped: they are Hub bookkeeping, not model content.
    """
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        if rel.name == REGISTER_FILE:
            continue
        out[rel.as_posix()] = sha256_file(p)
    return out


def _upsert(settings: Settings, record: ModelRecord) -> None:
    register = load_register(settings)
    models = [m for m in register.models if m.role != record.role] + [record]
    save_register(settings, ModelRegister(models=sorted(models, key=lambda m: m.role)))


def _hf_download(repo: str, local_dir: Path, token: str | None) -> tuple[Path, str]:
    """Resolve the repo's current revision and mirror it into ``local_dir``.

    ``config`` sets ``HF_HUB_OFFLINE=1`` for the whole process; this one call site lifts it for
    the duration of the download and restores it afterwards.
    """
    import huggingface_hub as hub
    from huggingface_hub import constants
    from huggingface_hub import utils as hub_utils

    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    hub_utils.close_session()
    try:
        sha = hub.HfApi(token=token).model_info(repo).sha or "unknown"
        path = hub.snapshot_download(repo, revision=sha, local_dir=str(local_dir), token=token)
    finally:
        constants.HF_HUB_OFFLINE = previous
        hub_utils.close_session()
    return Path(path), sha


def _convert_cohere(local_dir: Path) -> str | None:
    """Best-effort local 8-bit MLX conversion of the Cohere weights; ``None`` when unavailable.

    Community conversions are not shipped: the conversion is made locally from the official
    weights, and the parameters recorded here are what makes the local artefact reproducible.
    """
    try:
        from mlx_audio.convert import convert  # type: ignore[import-not-found]
    except ImportError as exc:  # not installed on this host, or its layout changed again
        log.warning("models.conversion_skipped", role="stt_ar", reason=str(exc))
        return None
    out = local_dir / "mlx-8bit"
    try:
        convert(hf_path=str(local_dir), mlx_path=str(out), quantize=True, q_bits=8, q_group_size=64)
    except (TypeError, AttributeError, OSError) as exc:  # API drift or missing files
        log.warning("models.conversion_failed", role="stt_ar", error=str(exc))
        return None
    return "mlx_audio.stt.convert quantize=True q_bits=8 q_group_size=64 -> mlx-8bit/"


def pull(
    role: str,
    settings: Settings,
    *,
    token_env: str = "HF_TOKEN",  # noqa: S107 - env var name
    downloader: Downloader | None = None,
    converter: Converter | None = None,
) -> ModelRecord:
    """Mirror the repo for ``role`` into ``models_dir/<role>`` and record it in the register.

    The token is ``os.environ[token_env]`` (absent is fine for ungated repos). ``downloader`` and
    ``converter`` exist for tests; the defaults use the Hub and ``mlx_audio``. Unknown roles raise
    ``ValueError``.
    """
    if role not in ROLES:
        raise ValueError(f"unknown model role {role!r}; known: {sorted(ROLES)}")
    repo, licence = ROLES[role]
    token = os.environ.get(token_env) or None
    local_dir = Path(settings.models_dir) / role
    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("models.pull", role=role, repo=repo, local_dir=str(local_dir), token=bool(token))
    path, revision = (downloader or _hf_download)(repo, local_dir, token)
    conversion = None
    if role == "stt_ar":
        conversion = (converter or _convert_cohere)(path)
    record = ModelRecord(
        role=role,
        repo=repo,
        revision=revision,
        licence=licence,
        files_sha256=hash_tree(path),
        local_path=path,
        conversion=conversion,
        pulled_at=datetime.now(UTC),
    )
    _upsert(settings, record)
    return record


def register(
    role: str,
    path: Path,
    settings: Settings,
    *,
    repo: str | None = None,
    revision: str = "local",
    licence: str | None = None,
    conversion: str | None = None,
) -> ModelRecord:
    """Register an already-mirrored directory for ``role`` (air-gapped or pre-downloaded weights).

    Hashes every file under ``path`` and writes the register. ``repo`` and ``licence`` default to
    the known values for the role; ``revision`` should be the Hub commit when known. Raises
    ``FileNotFoundError`` when ``path`` is not a directory containing at least one file.
    """
    path = Path(path).expanduser().resolve()
    files = hash_tree(path) if path.is_dir() else {}
    if not files:
        raise FileNotFoundError(f"no model files under {path}")
    known = ROLES.get(role, (path.name, "unknown"))
    record = ModelRecord(
        role=role,
        repo=repo or known[0],
        revision=revision,
        licence=licence or known[1],
        files_sha256=files,
        local_path=path,
        conversion=conversion,
        pulled_at=datetime.now(UTC),
    )
    _upsert(settings, record)
    log.info("models.registered", role=role, files=len(files))
    return record


def verify(settings: Settings) -> list[ModelRecord]:
    """Re-hash every registered model; raise ``ModelRegisterMismatch`` on the first difference.

    A missing register, a missing directory, a missing, changed or extra file all fail: the
    register is the complete description of what may be loaded.
    """
    reg = load_register(settings)
    if not reg.models:
        raise ModelRegisterMismatch(f"no models registered in {register_path(settings)}")
    for rec in reg.models:
        root = Path(rec.local_path)
        if not root.is_dir():
            raise ModelRegisterMismatch(f"{rec.role}: directory missing: {root}")
        actual = hash_tree(root)
        for name, digest in rec.files_sha256.items():
            if name not in actual:
                raise ModelRegisterMismatch(f"{rec.role}: file missing: {name}")
            if actual[name] != digest:
                raise ModelRegisterMismatch(f"{rec.role}: hash mismatch: {name}")
        extra = sorted(set(actual) - set(rec.files_sha256))
        if extra:
            raise ModelRegisterMismatch(f"{rec.role}: unregistered files present: {extra}")
    log.info("models.verified", roles=[m.role for m in reg.models])
    return reg.models


def hashes_for_provenance(settings: Settings) -> dict[str, str]:
    """One SHA-256 per role over the sorted ``name sha`` lines of its register entry.

    Reads the register only (no re-hashing); empty when nothing is registered.
    """
    out: dict[str, str] = {}
    for rec in load_register(settings).models:
        lines = "\n".join(f"{n} {s}" for n, s in sorted(rec.files_sha256.items()))
        out[rec.role] = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    return out
