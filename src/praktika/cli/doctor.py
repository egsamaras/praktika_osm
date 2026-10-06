"""``praktika doctor``: readiness checks that degrade gracefully.

Each check reports ``ok``, ``warn`` or ``fail`` with a one-line detail; only ``fail`` makes the
command exit non-zero. Probes that touch the system (``which``, ``run_cmd``, ``http_client``,
``physical_memory_bytes``) are module functions so tests can substitute them, and every network
probe carries a short timeout so the command finishes in seconds when Ollama is down.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import httpx
import typer
from cryptography.fernet import Fernet
from dotenv import dotenv_values

from praktika.audit import chain_report
from praktika.cli import context as ctx
from praktika.config import ENV_FILE_VAR, Settings, env_file_path, hide_credentials
from praktika.errors import ConfigError, EgressError, ModelRegisterMismatch, PraktikaError
from praktika.llm import prompts as pr
from praktika.logging import get_logger
from praktika.models_registry import manage
from praktika.redact.tokenise import VAULT_KEY_ENV
from praktika.store.repo import SqliteStore

log = get_logger(__name__)

OLLAMA_TIMEOUT_S = 3.0
MIN_MEMORY_BYTES = 16 * 1024**3
FDESETUP = "/usr/bin/fdesetup"
FINDMNT = "/usr/bin/findmnt"
LSBLK = "/usr/bin/lsblk"
AGENT_MARKER = "endpoint-agent.ok"
#: What to set instead of an ``mlx_*`` speech backend on a host where it cannot run.
MLX_ELSEWHERE_FIX = {"en": "PRAKTIKA_STT_EN=http", "ar": "PRAKTIKA_STT_AR=http or none"}


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str


def which(name: str) -> str | None:
    return shutil.which(name)


def run_cmd(args: list[str], timeout: float = 5.0) -> str | None:
    """Run a fixed command; stdout on success, ``None`` on any failure (never raises)."""
    try:
        # Fixed absolute executables with constant arguments; no shell, no user input.
        proc = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def http_client(settings: Settings) -> httpx.Client:
    return settings.http_client(timeout=OLLAMA_TIMEOUT_S)


def physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def check_ffmpeg() -> Check:
    path = which("ffmpeg")
    if path:
        return Check("ffmpeg", "ok", path)
    return Check("ffmpeg", "warn", "ffmpeg not found: audio ingest unavailable (VTT/DOCX ok)")


def check_ollama(settings: Settings) -> Check:
    if settings.llm_provider == "fake":
        if ctx.smoke_allowed(settings):
            return Check("ollama", "warn", "llm_provider=fake: placeholder minutes (smoke test)")
        return Check(
            "ollama", "fail", "llm_provider=fake drafts placeholder minutes; refused unless "
            "PRAKTIKA_PILOT_SMOKE=true"
        )  # fmt: skip
    if settings.llm_provider != "ollama":
        return Check("ollama", "ok", f"llm_provider={settings.llm_provider}; Ollama not used")
    base = str(settings.llm_base_url).rstrip("/")
    try:
        with http_client(settings) as client:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
            names = {str(m.get("name", "")) for m in r.json().get("models", [])}
            if settings.llm_model not in names and f"{settings.llm_model}:latest" not in names:
                return Check(
                    "ollama",
                    "fail",
                    f"model {settings.llm_model} not pulled "
                    f"(have: {', '.join(sorted(names)) or 'none'})",
                )
            show = client.post(f"{base}/api/show", json={"model": settings.llm_model})
            params = str(show.json().get("parameters", "")) if show.is_success else ""
    except (httpx.HTTPError, EgressError, ValueError) as exc:
        return Check("ollama", "fail", f"Ollama unreachable at {base}: {exc}")
    for line in params.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            if int(parts[1]) < settings.llm_num_ctx:
                return Check(
                    "ollama",
                    "warn",
                    f"model num_ctx {parts[1]} < configured {settings.llm_num_ctx}",
                )
    return Check(
        "ollama",
        "ok",
        f"{settings.llm_model} present; num_ctx={settings.llm_num_ctx} sent per call",
    )


def praktika_keys(path: Path) -> list[str] | None:
    """The ``PRAKTIKA_`` variables an env file sets (parsed as the settings loader parses it),
    or ``None`` when the file cannot be read or decoded."""
    try:
        values = dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return sorted(k for k in values if k.upper().startswith("PRAKTIKA_"))


def check_env_file() -> Check:
    """The env file this run loaded: the one ``PRAKTIKA_ENV_FILE`` names, the default ``.env``
    in the per-account data directory, or none (the defaults and the process environment).

    A file that exists but sets no ``PRAKTIKA_`` variable (for example the empty placeholder a
    runbook creates before the settings are written) is a warning: nothing in it is applied,
    so the run uses the defaults exactly as if there were no file. A named file that is missing
    or unreadable never gets this far (every command refuses it first,
    ``context.checked_env_file``); the ``fail`` branch is for direct callers.
    """
    try:
        path = env_file_path()
    except ConfigError as exc:
        return Check("env_file", "fail", str(exc))
    named = bool(os.environ.get(ENV_FILE_VAR))
    if not named and not path.is_file():
        return Check(
            "env_file", "warn", f"none loaded: {ENV_FILE_VAR} is not set and {path} does not "
            "exist, so the defaults and the process environment apply"
        )  # fmt: skip
    where = f"named by {ENV_FILE_VAR}" if named else f"default location; {ENV_FILE_VAR} is not set"
    keys = praktika_keys(path)
    if keys is None:
        return Check("env_file", "warn", f"{path} ({where}) cannot be parsed as an env file")
    if not keys:
        return Check(
            "env_file", "warn", f"{path} ({where}) sets no PRAKTIKA_ variable, so the defaults "
            "and the process environment apply; write the settings into it (see "
            "docs/DEPLOYMENT.md)"
        )  # fmt: skip
    return Check("env_file", "ok", f"{path} ({where})")


def check_models(settings: Settings) -> Check:
    """Model register: absent is a setup step (warn), a mismatch is an integrity failure.

    A register that does not exist yet only warns. With no local weights to load (``http``,
    ``fake`` or ``none`` backends) ingest does not need one, and ``praktika models register``
    records the provenance of the weights the speech server loads; with local weights the fix
    is ``models pull`` on a connected machine or ``models register`` for weights copied in.
    A register whose hashes or files no longer match is a hard failure when an STT backend
    would load those weights. On a host other than macOS an ``mlx_*`` backend (the default
    ``stt_en=mlx_whisper``) cannot run at all, so the hint there is the HTTP speech server and
    docs/DEPLOYMENT.md, never ``models pull``, which would fetch weights that cannot run there;
    like all text read on a Linux server, it names no Apple product.
    """
    needs_weights = settings.stt_en not in ("fake", "http") or settings.stt_ar not in (
        "fake",
        "http",
        "none",
    )
    register = manage.register_path(settings)
    if not register.exists() and not needs_weights:
        over = "speech runs over HTTP" if settings.stt_en == "http" else "no weights are loaded"
        return Check(
            "models", "warn", f"no model register at {register}; stt_en={settings.stt_en}: "
            f"{over}, so ingest does not need one. To record the provenance of the weights the "
            "speech server loads, run `praktika models register stt_en <dir>` "
            "(see docs/DEPLOYMENT.md)"
        )  # fmt: skip
    mlx = {
        lang: backend
        for lang, backend in (("en", settings.stt_en), ("ar", settings.stt_ar))
        if backend.startswith("mlx_")
    }
    if not register.exists() and mlx and host_platform() != "darwin":
        names = ", ".join(f"stt_{lang}={backend}" for lang, backend in mlx.items())
        fix = " and ".join(MLX_ELSEWHERE_FIX[lang] for lang in mlx)
        return Check(
            "models", "warn", f"no model register at {register}; {names} cannot run on this "
            f"host: set {fix}, with PRAKTIKA_STT_HTTP_URL naming the speech server (see "
            "docs/DEPLOYMENT.md). Teams VTT/DOCX ingest needs no weights"
        )  # fmt: skip
    if not register.exists():
        return Check(
            "models", "warn", f"no model register at {register}; audio ingest loads local "
            "weights: run `praktika models pull stt_en` on a connected machine, or `praktika "
            "models register stt_en <dir>` for weights copied in (Teams VTT/DOCX ingest needs "
            "no weights)"
        )  # fmt: skip
    try:
        records = manage.verify(settings)
    except ModelRegisterMismatch as exc:
        return Check("models", "fail" if needs_weights else "warn", str(exc))
    return Check("models", "ok", "verified: " + ", ".join(r.role for r in records))


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing directory at or above ``path`` (the data directory may not exist
    yet on a first run, but the volume it will live on does)."""
    path = Path(path).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_disk_encryption(settings: Settings) -> Check:
    """Is the data directory on an encrypted volume?

    Linux (a server such as an NVIDIA DGX Spark): the volume holding the data directory must be
    a dm-crypt mapping (``findmnt`` for its source, ``lsblk`` for its type). Anything else is a
    warning, not a failure, because self-encrypting drives and storage-array encryption are
    invisible to the operating system; the record of truth is IT's confirmation. Developer
    workstations with the platform's full-disk-encryption tool are checked with it, and a disk
    reported as unencrypted there is a hard failure.
    """
    if Path(FINDMNT).exists() and Path(LSBLK).exists():
        target = _existing_ancestor(Path(settings.data_dir))
        source = run_cmd([FINDMNT, "-no", "SOURCE", "--target", str(target)])
        if not source:
            return Check(
                "disk_encryption", "warn", "cannot resolve the data volume; confirm with IT"
            )
        dev = source.strip().split("[")[0]
        kind = run_cmd([LSBLK, "-no", "TYPE", dev]) or ""
        if "crypt" in kind.split():
            return Check("disk_encryption", "ok", f"data directory on encrypted volume {dev}")
        return Check(
            "disk_encryption",
            "warn",
            f"{dev} is not a dm-crypt volume; confirm disk encryption at rest with IT",
        )
    if Path(FDESETUP).exists():
        out = run_cmd([FDESETUP, "status"])
        if out is None:
            return Check("disk_encryption", "warn", "encryption status unavailable; verify")
        if "is On" in out:
            return Check("disk_encryption", "ok", "full-disk encryption is on")
        return Check("disk_encryption", "fail", out.strip() or "full-disk encryption is off")
    return Check("disk_encryption", "warn", "cannot verify on this platform; confirm with IT")


def check_data_dir(settings: Settings) -> Check:
    path = Path(settings.data_dir)
    if not path.is_dir():
        return Check("data_dir", "warn", f"{path} does not exist yet (created on first use)")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o700:
        return Check("data_dir", "fail", f"{path} mode is {mode:o}, expected 700")
    return Check("data_dir", "ok", f"{path} mode 700")


def check_egress(settings: Settings) -> Check:
    """Every configured URL is inside the allow-list; the OK line lists the allow-list as plain
    comma-separated host patterns (``allowed_hosts=localhost, 127.0.0.1``), not a Python list."""
    try:
        settings.assert_no_egress()
    except EgressError as exc:
        return Check("no_egress", "fail", str(exc))
    hosts = ", ".join(settings.allowed_hosts) or "(none)"
    return Check("no_egress", "ok", f"allowed_hosts={hosts}")


def check_memory() -> Check:
    total = physical_memory_bytes()
    if total is None:
        return Check("memory", "warn", "cannot determine physical memory")
    gib = total / 1024**3
    if total < MIN_MEMORY_BYTES:
        return Check("memory", "warn", f"{gib:.0f} GiB physical memory; 16 GiB recommended")
    return Check("memory", "ok", f"{gib:.0f} GiB physical memory")


def check_identity(settings: Settings) -> Check:
    if settings.mode == "service" and settings.identity_provider != "oidc":
        return Check(
            "identity", "fail", f"service mode with identity_provider={settings.identity_provider}"
            " authenticates nobody (every caller becomes the console user); set "
            "PRAKTIKA_IDENTITY_PROVIDER=oidc"
        )  # fmt: skip
    if settings.identity_provider == "fake" and not ctx.smoke_allowed(settings):
        return Check(
            "identity", "fail", "identity_provider=fake attributes records to a fictional user; "
            "refused unless PRAKTIKA_PILOT_SMOKE=true"
        )  # fmt: skip
    provider = ctx.identity_provider(settings)
    if provider is None:
        return Check("identity", "warn", "oidc identity is only available behind the server")
    try:
        who = provider.current()
    except PraktikaError as exc:
        return Check("identity", "warn", f"identity unavailable: {exc}")
    status = "ok" if who.source in ("session", "oidc") else "warn"
    return Check("identity", status, f"{who.user} (source={who.source})")


def check_audit(settings: Settings) -> Check:
    """JSONL chain cross-checked against ``audit.jsonl.head`` and the store's audit table."""
    path = Path(settings.data_dir) / ctx.AUDIT_FILE
    db = Path(settings.data_dir) / ctx.DB_FILE
    store = SqliteStore(db) if db.exists() else None
    try:
        report = chain_report(path, store)
    finally:
        if store is not None:
            store.close()
    return Check("audit_chain", report.status, report.detail)


def check_prompts(settings: Settings) -> Check:
    """Prompt files present and matching the pinned hash (a planted prompt set is a warning)."""
    try:
        digest = pr.version_sha256(settings.prompts_dir, settings.prompt_version)
    except FileNotFoundError:
        return Check(
            "prompts", "fail", f"{settings.prompts_dir}/{settings.prompt_version} not found"
        )
    if not Path(settings.glossary_path).is_file():
        return Check("prompts", "fail", f"glossary not found at {settings.glossary_path}")
    pinned = pr.PINNED_SHA256.get(settings.prompt_version)
    if pinned is None:
        return Check("prompts", "warn", f"no pinned hash for prompts {settings.prompt_version}")
    if digest != pinned:
        return Check(
            "prompts", "warn", f"prompts {settings.prompt_version} differ from the pinned set "
            f"({digest[:12]} != {pinned[:12]}); review before drafting"
        )  # fmt: skip
    return Check("prompts", "ok", f"{settings.prompt_version} matches pin {pinned[:12]}")


def host_platform() -> str:
    """``sys.platform``, behind a function so tests can exercise either platform's checks."""
    return sys.platform


def _vault_key_valid(value: str) -> bool:
    try:
        Fernet(value.encode("utf-8"))
    except (ValueError, TypeError):
        return False
    return True


def check_vault_key(settings: Settings) -> Check:
    """Where the token-vault key will come from, and whether it is usable.

    ``PRAKTIKA_VAULT_KEY`` wins when set: ``ok`` if it is a valid Fernet key, ``fail``
    otherwise (every ingest would fail on it). Unset: service mode fails (a service has no key
    store), local mode on a developer workstation whose platform key store holds or creates
    the key (``darwin``) is ``ok``, and local mode on Linux fails, because there is no key
    store there and the vault cannot be opened without the variable. The messages name no
    platform product.
    """
    value = os.environ.get(VAULT_KEY_ENV)
    if value:
        if not _vault_key_valid(value):
            return Check("vault_key", "fail", f"{VAULT_KEY_ENV} is set but is not a Fernet key")
        return Check("vault_key", "ok", f"{VAULT_KEY_ENV} present (never persisted)")
    if settings.mode == "service":
        return Check(
            "vault_key", "fail", f"service mode needs {VAULT_KEY_ENV} in the process "
            "environment (a service has no key store)"
        )  # fmt: skip
    if host_platform() == "darwin":
        return Check("vault_key", "ok", "local mode: vault key held in the platform key store")
    return Check(
        "vault_key", "fail", f"{VAULT_KEY_ENV} is not set and there is no key store on this "
        "host; set it in the process environment (never in the .env file)"
    )  # fmt: skip


def check_agent(settings: Settings) -> Check:
    marker = Path(settings.data_dir) / AGENT_MARKER
    if marker.exists():
        return Check("endpoint_agent", "ok", f"marker present: {marker}")
    return Check(
        "endpoint_agent", "warn", f"no log-shipping marker at {marker}; confirm that the "
        "log-shipping agent forwards the audit log"
    )  # fmt: skip


def _checked(name: str, probe: Callable[[], Check]) -> Check:
    """Run one check; a file-system error it did not expect (for example a data directory
    under a home this account cannot search) becomes that check's ``fail`` line, so doctor
    always prints every check instead of stopping at a traceback. Any URL in the result loses
    its user name and password (``config.hide_credentials``), whatever the check printed."""
    check = _probe(name, probe)
    return Check(check.name, check.status, hide_credentials(check.detail))


def _probe(name: str, probe: Callable[[], Check]) -> Check:
    try:
        return probe()
    except OSError as exc:
        where = f" ({exc.filename})" if exc.filename else ""
        return Check(name, "fail", f"cannot check: {exc.strerror or type(exc).__name__}{where}")


def run_checks(settings: Settings) -> list[Check]:
    """The thirteen checks, in the order doctor prints them."""
    return [
        _checked("env_file", check_env_file),
        _checked("ffmpeg", check_ffmpeg),
        _checked("ollama", lambda: check_ollama(settings)),
        _checked("models", lambda: check_models(settings)),
        _checked("disk_encryption", lambda: check_disk_encryption(settings)),
        _checked("data_dir", lambda: check_data_dir(settings)),
        _checked("no_egress", lambda: check_egress(settings)),
        _checked("memory", check_memory),
        _checked("identity", lambda: check_identity(settings)),
        _checked("audit_chain", lambda: check_audit(settings)),
        _checked("prompts", lambda: check_prompts(settings)),
        _checked("vault_key", lambda: check_vault_key(settings)),
        _checked("endpoint_agent", lambda: check_agent(settings)),
    ]


def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check the env file, ffmpeg, Ollama, models, disk encryption, data dir, egress, memory,
    identity, audit chain, prompt pin, vault key and the log-shipping marker."""
    settings = ctx.load_settings()
    results = run_checks(settings)
    if as_json:
        ctx.console.print(json.dumps([asdict(c) for c in results], indent=2))
    else:
        for c in results:
            ctx.console.print(f"[{c.status.upper():<4}] {c.name:<15} {c.detail}")
    failures = [c for c in results if c.status == "fail"]
    if failures:
        ctx.err_console.print(f"{len(failures)} hard failure(s)")
        raise typer.Exit(ctx.EXIT_ERROR)
