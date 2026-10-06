"""Runtime configuration and the egress allow-list (control C-01).

``HF_HUB_OFFLINE=1`` is forced the moment this module is imported (a shell that exported
``HF_HUB_OFFLINE=0`` does not win), before any code path can import ``huggingface_hub``, so
model loaders never reach out to the Hub. ``get_settings`` repeats the assignment defensively.
``NUMBA_THREADING_LAYER`` defaults to ``workqueue`` for the same reason of ordering: it keeps
numba (Whisper word-timestamp alignment) from crashing after torch (silero VAD) is loaded.

Trust boundary: ``Settings()`` never reads a ``.env`` from the current working directory. The
only env file honoured is ``env_file_path()`` — ``PRAKTIKA_ENV_FILE`` or ``<data_dir>/.env``
under the owner-controlled default data directory (``default_data_dir``: the Application Support
folder on macOS, the XDG data directory on Linux) — and callers pass it explicitly
(``_env_file=env_file_path()``). A file named by ``PRAKTIKA_ENV_FILE`` must exist and be
readable (``ConfigError`` otherwise), so a mistyped or missing file stops every command instead
of letting it run on the defaults. Prompt and glossary paths default to the installed package's
repository root, not to the working directory, so a planted ``prompts/`` cannot rewrite the
system rules. An allow-list entry must contain a literal host label: ``"*"`` is refused.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
from collections.abc import Mapping
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field, HttpUrl, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from praktika.errors import ConfigError, EgressError

os.environ["HF_HUB_OFFLINE"] = "1"
# numba's default threading layer (OpenMP) segfaults once torch's own libomp is loaded in the
# same process, which is exactly the VAD (torch) -> Whisper word-timestamp DTW (numba) order of
# every audio ingest on macOS. The workqueue layer is single-runtime and safe; numba reads this
# only at first import, so it must be set here, before any backend is loaded.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

_URL_FIELDS = ("llm_base_url", "stt_http_url", "oidc_issuer", "oidc_jwks_url")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the checkout (src layout) or install root
MACOS_DATA_DIR = "~/Library/Application Support/Praktika"
XDG_DATA_HOME_VAR = "XDG_DATA_HOME"
XDG_DATA_FALLBACK = "~/.local/share"
ENV_FILE_VAR = "PRAKTIKA_ENV_FILE"


def default_data_dir(platform: str | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The owner-controlled default data directory for ``platform`` (default ``sys.platform``).

    macOS (``darwin``): ``~/Library/Application Support/Praktika``. Every other platform (Linux
    servers included): ``$XDG_DATA_HOME/praktika`` when ``XDG_DATA_HOME`` is an absolute
    path (the XDG spec says a relative value must be ignored), else
    ``~/.local/share/praktika``. ``PRAKTIKA_DATA_DIR`` overrides this through ``Settings``.
    The result is always expanded and absolute.
    """
    platform = sys.platform if platform is None else platform
    env = os.environ if environ is None else environ
    if platform == "darwin":
        return Path(MACOS_DATA_DIR).expanduser()
    xdg = env.get(XDG_DATA_HOME_VAR, "")
    base = Path(xdg).expanduser() if xdg and Path(xdg).expanduser().is_absolute() else None
    return (base or Path(XDG_DATA_FALLBACK).expanduser()) / "praktika"


#: The default data directory resolved at import (kept for callers that read a constant);
#: ``Settings`` and ``env_file_path`` resolve it afresh on every call.
DEFAULT_DATA_DIR = default_data_dir()


def env_file_path(platform: str | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The one ``.env`` location Praktika reads: ``$PRAKTIKA_ENV_FILE`` or ``.env`` inside
    ``default_data_dir(platform, environ)``. Never the current working directory.

    Raises ``ConfigError`` when ``PRAKTIKA_ENV_FILE`` names a file that does not exist, or when
    the file (named or default) exists but cannot be read. A default file that does not exist is
    fine: the defaults and the process environment apply. A default location this account
    cannot even look into is a ``ConfigError`` too, not "none loaded": that is what ``sudo -E``
    without ``PRAKTIKA_ENV_FILE`` gives, because it keeps the caller's ``HOME`` (and so a
    default data directory inside another account's home), and the operator has to export
    ``PRAKTIKA_ENV_FILE`` rather than run on the defaults.
    """
    env = os.environ if environ is None else environ
    override = env.get(ENV_FILE_VAR)
    path = Path(override).expanduser() if override else default_data_dir(platform, env) / ".env"
    if not override:
        try:
            present = path.exists()
        except OSError as exc:
            why = "is empty" if ENV_FILE_VAR in env else "is not set"
            raise ConfigError(
                f"{ENV_FILE_VAR} {why} and the default env file {path} cannot be checked "
                f"({exc.strerror or type(exc).__name__}): this account cannot look into that "
                "directory, which happens when HOME belongs to another account (sudo -E keeps "
                f"the caller's HOME). Export {ENV_FILE_VAR} naming this deployment's env file "
                f"(for example {ENV_FILE_VAR}=/etc/praktika/praktika.env) and run the command "
                "again"
            ) from exc
        if not present:
            return path
    named = f"{ENV_FILE_VAR}={path}" if override else f"the env file {path}"
    try:
        with path.open("rb"):
            pass
    except FileNotFoundError as exc:
        raise ConfigError(
            f"{named} does not exist; refusing to run on the defaults "
            f"(create the file, or unset {ENV_FILE_VAR})"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"{named} cannot be read ({exc.strerror or type(exc).__name__}); refusing to run on "
            "the defaults (give this account read access to it)"
        ) from exc
    return path


def pattern_has_literal_label(pattern: str) -> bool:
    """True when an allow-list glob names at least one literal host label (``*`` does not)."""
    stripped = pattern.strip()
    if not stripped or stripped in ("*", "**"):
        return False
    return any(label and not set(label) <= set("*?[]!") for label in stripped.split("."))


def host_allowed(host: str | None, allowed_hosts: list[str]) -> bool:
    """Return True when ``host`` matches any allow-list entry (case-insensitive fnmatch glob).

    ``None`` or an empty host (for example a relative URL) is never allowed. An IPv6 literal
    (``[fd00::10]`` in a URL) matches an entry naming that address with or without brackets, in
    short or long form, compared as an address (fnmatch would read the brackets as a character
    class); an entry with a glob character (``*fd00::*``) is still a glob.
    """
    if not host:
        return False
    h = _unbracket(host.lower())
    for pattern in allowed_hosts:
        p = _unbracket(pattern.lower())
        literal_ipv6 = ":" in p and not set(p) & set("*?[")
        if _same_address(h, p) if literal_ipv6 else fnmatchcase(h, p):
            return True
    return False


def _unbracket(value: str) -> str:
    """``[fd00::10]`` -> ``fd00::10``; anything else (a glob's ``[ab]`` class included) as is."""
    return value[1:-1] if value.startswith("[") and value.endswith("]") and ":" in value else value


def _same_address(host: str, pattern: str) -> bool:
    """IPv6 literals compared as addresses, so the long and short forms of one match."""
    try:
        return ipaddress.ip_address(host) == ipaddress.ip_address(pattern)
    except ValueError:
        return host == pattern


def _parse_host_list(v: Any) -> Any:
    """Accept a JSON list, a comma-separated string, or an already-parsed list."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            return json.loads(s)
        return [part.strip() for part in s.split(",") if part.strip()]
    return v


class Settings(BaseSettings):
    """All runtime configuration. Env prefix PRAKTIKA_. Never contains secrets (tokens come from
    the process environment of one-off commands only and are never persisted)."""

    model_config = SettingsConfigDict(
        env_prefix="PRAKTIKA_", env_file=None, extra="forbid", validate_default=True
    )

    mode: Literal["local", "service"] = "local"
    pilot: bool = True
    pilot_smoke: bool = False  # explicit opt-in for the fake LLM/identity providers
    data_dir: Path = Field(default_factory=default_data_dir)
    models_dir: Path = Path("~/praktika-models").expanduser()
    prompts_dir: Path = PACKAGE_ROOT / "prompts"
    prompt_version: str = "v1"
    glossary_path: Path = PACKAGE_ROOT / "glossary.yaml"

    # Declared before the URL fields so validators can read it from ``info.data``.
    allowed_hosts: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1"]

    llm_provider: Literal["ollama", "openai_compat", "fake"] = "ollama"
    llm_base_url: HttpUrl = "http://127.0.0.1:11434"  # type: ignore[assignment]
    llm_model: str = "qwen2.5:14b"
    llm_num_ctx: int = 32768
    llm_fallback_model: str | None = "llama3.1:8b"
    llm_long_transcript_tokens: int = 60000
    llm_full_context_max_tokens: int = 16000
    llm_timeout_s: int = 600

    stt_en: Literal["mlx_whisper", "faster_whisper", "http", "fake"] = "mlx_whisper"
    #: The Arabic speech path. ``none`` (the default) switches it off: meetings are English, every
    #: chunk is decoded by the English engine as English, language ID is skipped and
    #: ``--lang ar-mixed`` is refused before any audio is read.
    stt_ar: Literal[
        "none", "mlx_cohere", "mlx_whisper_full", "faster_whisper_full", "http", "fake"
    ] = "none"
    stt_http_url: HttpUrl | None = None
    stt_lid_threshold: float = 0.9

    diarize: bool = False
    diarize_backend: Literal["pyannote", "fake", "none"] = "none"

    #: Draft (unapproved) export over the API and CLI. Off everywhere by default: a query
    #: parameter alone must never produce an unapproved document, so the deployment has to opt in
    #: as well as the caller (C-07). Ignored while ``pilot`` is true, which refuses draft export
    #: outright.
    allow_draft_export: bool = False

    review_host: str = "127.0.0.1"
    review_port: int = 8793
    capture_team_id: str | None = None  # Apple Team ID the signed capture helper must carry

    retention_audio_hours: dict[str, int] = {"internal": 24, "confidential": 72, "restricted": 0}
    retention_transcript_days: dict[str, int] = {
        "internal": 14,
        "confidential": 14,
        "restricted": 7,
    }
    retention_draft_days: dict[str, int] = {"internal": 30, "confidential": 30, "restricted": 14}

    audit_sink: Literal["jsonl", "stdout"] = "jsonl"
    identity_provider: Literal["session", "oidc", "fake"] = "session"
    oidc_issuer: HttpUrl | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: HttpUrl | None = None  # the issuer's JWKS document (service mode only)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, v: Any) -> Any:
        return _parse_host_list(v)

    @field_validator("allowed_hosts", mode="after")
    @classmethod
    def _hosts_have_labels(cls, v: list[str]) -> list[str]:
        """Refuse wildcard-only patterns: ``"*"`` would make the allow-list a no-op."""
        bad = [h for h in v if not pattern_has_literal_label(h)]
        if bad:
            raise EgressError(f"allowed_hosts patterns must name a host label; refused: {bad}")
        return v

    @field_validator("data_dir", "models_dir", "prompts_dir", "glossary_path", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator(*_URL_FIELDS, mode="after")
    @classmethod
    def _host_allowed(cls, v: HttpUrl | None, info: ValidationInfo) -> HttpUrl | None:
        """Raise ``EgressError`` unless the URL host matches ``allowed_hosts`` (glob)."""
        if v is None:
            return v
        if "allowed_hosts" not in info.data:  # allowed_hosts itself is invalid: reported there
            return v
        allowed = info.data["allowed_hosts"]
        if not host_allowed(v.host, allowed):  # name the host only: a URL may carry a password
            raise EgressError(
                f"{info.field_name} targets host {v.host!r}, not in allowed_hosts {allowed}"
            )
        return v

    def assert_no_egress(self) -> None:
        """Re-check every configured URL against the allow-list; raise ``EgressError`` on the first
        violation. Called at startup and by ``praktika doctor``."""
        for name in _URL_FIELDS:
            url = getattr(self, name)
            if url is not None and not host_allowed(url.host, self.allowed_hosts):
                raise EgressError(f"{name} targets host {url.host!r} outside the allow-list")

    def http_client(self, *, timeout: float | None = None) -> httpx.Client:
        """Return an ``httpx.Client`` whose transport refuses any host outside ``allowed_hosts``.

        Every outbound HTTP call in Praktika (Ollama, vLLM, HTTP STT, OIDC) goes through a client
        built here, so the allow-list is enforced per request, not only at configuration time.
        """
        return httpx.Client(
            transport=AllowListTransport(self.allowed_hosts),
            timeout=timeout if timeout is not None else float(self.llm_timeout_s),
        )


class AllowListTransport(httpx.BaseTransport):
    """An httpx transport that raises ``EgressError`` for hosts outside the allow-list.

    Wraps an ``httpx.HTTPTransport`` (or any transport passed as ``inner``) and checks the request
    host before delegating.
    """

    def __init__(self, allowed_hosts: list[str], inner: httpx.BaseTransport | None = None) -> None:
        self.allowed_hosts = list(allowed_hosts)
        self._inner = inner if inner is not None else httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not host_allowed(host, self.allowed_hosts):
            raise EgressError(
                f"blocked request to {host!r}: not in allow-list {self.allowed_hosts}"
            )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` (cached), read from the environment and
    ``env_file_path()`` (``ConfigError`` for a named env file that is missing or unreadable).
    Sets ``HF_HUB_OFFLINE=1`` first."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    return Settings(_env_file=env_file_path())
