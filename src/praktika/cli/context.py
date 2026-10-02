"""Shared runtime for CLI commands: settings, store, identity, audit, console and exit codes.

Every command obtains its collaborators through this module so tests can substitute settings,
the LLM client, the vault key and the capturer by monkeypatching the module attributes
(``load_settings``, ``llm_client``, ``vault_key``, ``build_capturer``). Exit codes: 0 success,
1 domain error (``PraktikaError``), 2 refusal or incomplete gate answers.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from praktika.audit import AuditLog, AuditSink, JsonlAuditSink, StdoutAuditSink
from praktika.config import ENV_FILE_VAR, Settings, env_file_path
from praktika.errors import (
    ClassificationNotAllowed,
    ConfigError,
    ConsentRefused,
    EgressError,
    LanguageRefused,
    PraktikaError,
    ScopeError,
)
from praktika.identity import FakeIdentity, Identity, IdentityProvider, SessionIdentity
from praktika.logging import get_logger
from praktika.models import Attendee, Meeting
from praktika.store.repo import SqliteStore

log = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
DB_FILE = "praktika.db"
AUDIT_FILE = "audit.jsonl"
REVIEW_PATH = "/#/meetings/{meeting_id}"

console = Console(markup=False, highlight=False, emoji=False, soft_wrap=True)
err_console = Console(stderr=True, markup=False, highlight=False, emoji=False, soft_wrap=True)


class GateIncompleteError(PraktikaError):
    """A gate answer is missing and no interactive prompt is possible."""


#: The errors that are refusals (exit 2) rather than failures (exit 1).
REFUSALS: tuple[type[PraktikaError], ...] = (
    ConsentRefused,
    ScopeError,
    ClassificationNotAllowed,
    LanguageRefused,
    GateIncompleteError,
)


def fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    """Print ``error: <message>`` on stderr and exit with ``code``."""
    err_console.print(f"error: {message}")
    raise typer.Exit(code)


def guarded[F: Callable[..., Any]](fn: F) -> F:
    """Map domain errors raised by a command to exit codes (refusals exit 2, others 1)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except REFUSALS as exc:
            fail(str(exc), EXIT_REFUSED)
        except PraktikaError as exc:
            fail(str(exc), EXIT_ERROR)

    return wrapper  # type: ignore[return-value]


def checked_env_file() -> Path:
    """``config.env_file_path()``, exiting 1 (no traceback) when ``PRAKTIKA_ENV_FILE`` names a
    file that is missing or unreadable, or when it is unset and the default location cannot be
    looked into (``sudo -E`` keeping another account's ``HOME``). Every command calls it first
    (the application callback; skipped for ``--help``), so none of them runs on the defaults
    because the env file was mistyped or not created yet. A file that exists but sets no
    ``PRAKTIKA_`` variable is accepted here and reported by ``praktika doctor`` as a warning."""
    try:
        return env_file_path()
    except ConfigError as exc:
        fail(str(exc))


def load_settings() -> Settings:
    """``Settings`` from the environment and ``checked_env_file()`` (never a ``.env`` in the
    working directory); configuration errors exit 1, as does a prompts directory or glossary
    that does not exist (checked here so the gate never runs before them)."""
    env_file = checked_env_file()
    try:
        settings = Settings(_env_file=env_file)
    except EgressError as exc:
        fail(f"egress allow-list violation: {exc}")
    except ValidationError as exc:
        fail(f"invalid configuration: {exc}")
    missing = [
        f"{name}={path}"
        for name, path in (
            ("prompts_dir", settings.prompts_dir / settings.prompt_version),
            ("glossary_path", settings.glossary_path),
        )
        if not path.exists()
    ]
    if missing:
        fail(
            "configured path(s) not found: " + ", ".join(missing) + ". Praktika runs from a "
            "git checkout installed in editable mode (`uv sync` in the checkout does this) and "
            "reads its prompts and glossary from there; if it was installed another way, set "
            "PRAKTIKA_PROMPTS_DIR and PRAKTIKA_GLOSSARY_PATH to the checkout's prompts/ and "
            "glossary.yaml"
        )
    return settings


def smoke_allowed(settings: Settings) -> bool:
    """True when the fake LLM/identity providers may be used (``PRAKTIKA_PILOT_SMOKE=true``)."""
    return bool(settings.pilot_smoke)


def _refuse_fake(settings: Settings, what: str) -> None:
    if not smoke_allowed(settings):
        raise PraktikaError(
            f"{what}=fake is only allowed with PRAKTIKA_PILOT_SMOKE=true (placeholder output; "
            "never for a real meeting)"
        )


def ensure_data_dir(path: Path) -> Path:
    """Create ``path`` (mode 0700) if needed and tighten its mode; returns it.

    ``ConfigError`` (exit 1, no traceback) when this account cannot create or secure it, for
    example a default data directory under another account's home (``sudo -E`` keeps the
    caller's ``HOME``).
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise ConfigError(
            f"cannot use the data directory {path} ({exc.strerror or type(exc).__name__}); "
            "set PRAKTIKA_DATA_DIR in the env file to a directory this account owns, and check "
            f"that {ENV_FILE_VAR} is exported in this shell"
        ) from exc
    return path


def identity_provider(settings: Settings) -> IdentityProvider | None:
    """The CLI identity provider; ``oidc`` has no bearer token here and records ``system``.
    ``fake`` is refused unless ``pilot_smoke`` is set."""
    if settings.identity_provider == "fake":
        _refuse_fake(settings, "identity_provider")
        return FakeIdentity()
    if settings.identity_provider == "session":
        return SessionIdentity()
    log.warning("identity.oidc_unavailable_in_cli")
    return None


def server_identity(settings: Settings) -> IdentityProvider:
    """The identity provider for the review server: ``OidcIdentity`` over the allow-listed
    HTTP client in service mode, otherwise the same provider the CLI uses."""
    if settings.identity_provider != "oidc":
        return identity_provider(settings)  # type: ignore[return-value]
    if not (settings.oidc_issuer and settings.oidc_audience and settings.oidc_jwks_url):
        raise PraktikaError(
            "identity_provider=oidc needs PRAKTIKA_OIDC_ISSUER, PRAKTIKA_OIDC_AUDIENCE "
            "and PRAKTIKA_OIDC_JWKS_URL"
        )
    from praktika.identity import OidcIdentity

    return OidcIdentity(
        str(settings.oidc_issuer).rstrip("/"),
        settings.oidc_audience,
        str(settings.oidc_jwks_url),
        settings.http_client(timeout=10.0),
    )


def audit_sink(settings: Settings) -> AuditSink:
    if settings.audit_sink == "stdout":
        return StdoutAuditSink()
    return JsonlAuditSink(Path(settings.data_dir) / AUDIT_FILE)


@dataclass
class Runtime:
    """Open collaborators for one command invocation."""

    settings: Settings
    store: SqliteStore
    identity: IdentityProvider | None
    audit: AuditLog

    def current_identity(self) -> Identity:
        """The acting human; raises ``PraktikaError`` when only the system actor is available."""
        if self.identity is None:
            raise PraktikaError(
                "this command needs a user identity (identity_provider=oidc "
                "is only available behind the review server)"
            )
        return self.identity.current()

    def actor(self) -> str:
        return self.identity.current().user if self.identity is not None else "system"

    def require_meeting(self, meeting_id: str) -> Meeting:
        meeting = self.store.get_meeting(meeting_id)
        if meeting is None:
            raise PraktikaError(f"unknown meeting {meeting_id}")
        return meeting

    def close(self) -> None:
        self.store.close()


def open_runtime(settings: Settings | None = None, *, system_actor: bool = False) -> Runtime:
    """Open the store and the audit log under ``settings.data_dir``.

    ``system_actor`` records audit events as ``system`` (retention and other jobs). Every
    user command first runs the retention timers opportunistically (``sweep_retention``), so
    the C-05 promise does not depend on someone remembering ``praktika retention run``.
    """
    settings = settings or load_settings()
    ensure_data_dir(settings.data_dir)
    store = SqliteStore(Path(settings.data_dir) / DB_FILE)
    ident = None if system_actor else identity_provider(settings)
    rt = Runtime(settings, store, ident, AuditLog(audit_sink(settings), store, ident))
    if not system_actor:
        from praktika.cli.housekeeping import sweep_retention

        sweep_retention(rt)
    return rt


def review_url(settings: Settings, meeting_id: str) -> str:
    """The review page for ``meeting_id`` on the configured host and port; in local mode the
    link carries the persistent session token so it opens directly once ``serve`` runs."""
    host = settings.review_host if settings.review_host != "0.0.0.0" else "127.0.0.1"  # noqa: S104
    query = ""
    if settings.mode == "local":
        from praktika.server_auth import session_token_for

        query = f"?t={session_token_for(settings.data_dir)}"
    return f"http://{host}:{settings.review_port}/{query}" + REVIEW_PATH.format(
        meeting_id=meeting_id
    ).lstrip("/")


def llm_client(settings: Settings) -> Any:
    """The configured ``LLMClient``. ``fake`` is the offline stand-in (``llm/fake_client.py``)
    that drafts placeholder minutes from the transcript so the whole path runs without Ollama;
    it is refused unless ``pilot_smoke`` is set."""
    if settings.llm_provider == "ollama":
        from praktika.llm.ollama_client import OllamaClient

        return OllamaClient(
            str(settings.llm_base_url),
            settings.llm_model,
            settings.llm_num_ctx,
            settings.llm_timeout_s,
            settings.http_client(),
        )
    if settings.llm_provider == "openai_compat":
        from praktika.llm.openai_compat_client import OpenAICompatClient

        return OpenAICompatClient(
            str(settings.llm_base_url),
            settings.llm_model,
            lambda: os.environ.get("PRAKTIKA_LLM_TOKEN"),
            settings.http_client(),
            timeout_s=settings.llm_timeout_s,
        )
    _refuse_fake(settings, "llm_provider")
    from praktika.llm.fake_client import FakeLLM

    log.warning("llm.fake_provider", hint="PRAKTIKA_LLM_PROVIDER=fake drafts placeholder minutes")
    return FakeLLM()


def vault_key(settings: Settings | None = None) -> bytes:
    """The Fernet key for token vaults (the environment, or the platform key store where one
    exists; never persisted here).

    In service mode there is no key store, so ``PRAKTIKA_VAULT_KEY`` is required.
    """
    from praktika.redact.tokenise import VAULT_KEY_ENV, keychain_key

    if settings is not None and settings.mode == "service" and not os.environ.get(VAULT_KEY_ENV):
        raise PraktikaError(f"service mode needs {VAULT_KEY_ENV} in the process environment")
    return keychain_key()


def build_capturer(source: str, helper: Path, device: str | None = None) -> Any:
    """``MicCapturer`` for ``mic`` or ``SckCapturer`` for ``sck`` (signature checked on start)."""
    from praktika.audio.capture import MicCapturer, SckCapturer

    if source == "sck":
        return SckCapturer(helper)
    return MicCapturer(device)


def load_roster(path: Path | None) -> tuple[list[Attendee], list[str]]:
    """Attendees and room identities from a roster YAML (``attendees``, ``room_identities``)."""
    if path is None:
        return [], []
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        attendees = [Attendee.model_validate(a) for a in data.get("attendees", [])]
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise PraktikaError(f"cannot read roster {path}: {exc}") from exc
    return attendees, [str(r) for r in data.get("room_identities", [])]


def is_interactive() -> bool:
    """True when both stdin and stdout are terminals, so rich prompts can be used."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
