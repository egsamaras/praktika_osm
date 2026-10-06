"""Egress allow-list and offline guarantees (control C-01)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from praktika.config import AllowListTransport, Settings, get_settings, host_allowed
from praktika.errors import EgressError, PraktikaError

REPO = Path(__file__).resolve().parent.parent
PUBLIC = ["https://api.openai.com", "https://x.azure.com", "http://10.0.0.5"]


def settings(**over: object) -> Settings:
    """Settings built from keyword arguments only: no ``.env`` file, no developer environment."""
    return Settings(_env_file=None, **over)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- plan tests


@pytest.mark.parametrize("url", PUBLIC)
@pytest.mark.parametrize("field", ["llm_base_url", "stt_http_url", "oidc_issuer", "oidc_jwks_url"])
def test_public_hosts_rejected(field: str, url: str) -> None:
    with pytest.raises(EgressError) as exc:
        settings(**{field: url})
    host = url.split("://", 1)[1]
    assert field in str(exc.value) and repr(host) in str(exc.value)
    assert url not in str(exc.value), "the message names the host, never the URL (credentials)"
    assert isinstance(exc.value, PraktikaError)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://LOCALHOST:8000/v1",
        "https://localhost",
    ],
)
def test_local_hosts_accepted(url: str) -> None:
    s = settings(llm_base_url=url, stt_http_url=url, oidc_issuer=url, oidc_jwks_url=url)
    assert s.llm_base_url.host in {"127.0.0.1", "localhost"}
    s.assert_no_egress()
    assert settings().llm_base_url.host == "127.0.0.1", "default URL is validated and local"


def test_service_mode_gateway_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = ["localhost", "127.0.0.1", "*.acme.internal"]
    s = settings(
        mode="service", allowed_hosts=hosts, llm_base_url="http://llm-gw.acme.internal:8000"
    )
    assert s.llm_base_url.host == "llm-gw.acme.internal"
    s.assert_no_egress()
    for bad in (
        "http://llm-gw.acme.internal.evil.example",
        "http://acme.internal",
        "https://api.openai.com",
    ):
        with pytest.raises(EgressError):
            settings(mode="service", allowed_hosts=hosts, llm_base_url=bad)
    # The allow-list arrives from the environment as JSON or as a comma-separated string.
    monkeypatch.setenv("PRAKTIKA_ALLOWED_HOSTS", '["127.0.0.1", "*.acme.internal"]')
    monkeypatch.setenv("PRAKTIKA_LLM_BASE_URL", "http://vllm.acme.internal")
    assert settings().llm_base_url.host == "vllm.acme.internal"
    monkeypatch.setenv("PRAKTIKA_ALLOWED_HOSTS", "127.0.0.1, *.acme.internal")
    assert settings().allowed_hosts == ["127.0.0.1", "*.acme.internal"]
    monkeypatch.setenv("PRAKTIKA_ALLOWED_HOSTS", "127.0.0.1")
    with pytest.raises(EgressError):
        settings()


def test_hf_offline_set_before_import() -> None:
    """Importing ``praktika.config`` in a fresh interpreter sets HF_HUB_OFFLINE before
    ``huggingface_hub`` reads its constants, so the Hub is unreachable from any loader."""
    code = (
        "import os, sys; assert 'HF_HUB_OFFLINE' not in os.environ; "
        "assert 'huggingface_hub' not in sys.modules; "
        "import praktika.config; "
        "from huggingface_hub import constants; "
        "print(os.environ['HF_HUB_OFFLINE'], constants.HF_HUB_OFFLINE)"
    )
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    env["PYTHONPATH"] = str(REPO / "src")
    # Deliberate subprocess: the test needs a fresh interpreter with a clean module table.
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["1", "True"]
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    get_settings.cache_clear()


def test_transport_guard_blocks_other_hosts(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Settings-driven client refuses non-allow-listed hosts before any socket is opened."""
    reached: list[str] = []

    def fake_handle(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        reached.append(request.url.host)
        if not host_allowed(request.url.host, tmp_settings.allowed_hosts):
            raise AssertionError(f"guard let {request.url.host} through")
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
    client = tmp_settings.http_client(timeout=1.0)
    for url in ("https://api.openai.com/v1/models", "http://10.0.0.5:11434/api/tags"):
        with pytest.raises(EgressError):
            client.get(url)
    assert reached == [], "blocked requests must never reach the network transport"
    assert client.get("http://127.0.0.1:11434/api/tags").json() == {"models": []}
    assert reached == ["127.0.0.1"]
    client.close()


# --------------------------------------------------------------------------- extras


def test_assert_no_egress_recheck_catches_later_narrowing() -> None:
    s = settings(
        allowed_hosts=["127.0.0.1", "*.acme.internal"], llm_base_url="http://llm.acme.internal"
    )
    narrowed = s.model_copy(update={"allowed_hosts": ["127.0.0.1"]})
    with pytest.raises(EgressError):
        narrowed.assert_no_egress()


def test_host_allowed_edge_cases() -> None:
    assert host_allowed("LLM.ACME.INTERNAL", ["*.acme.internal"])
    assert not host_allowed(None, ["*"])
    assert not host_allowed("", ["*"])
    assert not host_allowed("localhost", [])
    assert not host_allowed("localhost.evil.example", ["localhost"])


def test_allow_list_transport_wraps_inner_transport() -> None:
    seen: list[str] = []

    def inner(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(204)

    t = AllowListTransport(["127.0.0.1"], inner=httpx.MockTransport(inner))
    with httpx.Client(transport=t) as client:
        assert client.get("http://127.0.0.1:8793/api/health").status_code == 204
        with pytest.raises(EgressError):
            client.get("http://169.254.169.254/latest/meta-data")
    assert seen == ["http://127.0.0.1:8793/api/health"]


def test_settings_forbid_unknown_keys_and_no_secret_fields() -> None:
    with pytest.raises(Exception, match="skip_consent"):
        settings(skip_consent=True)
    names = set(Settings.model_fields)
    secret_like = {
        n
        for n in names
        if n.endswith(("_token", "_key", "_secret", "_password"))
        or any(w in n for w in ("secret", "password", "api_key", "hf_token"))
    }
    assert not secret_like, "secrets come from the process environment, never from Settings"
    assert "pilot" in names and settings().pilot is True and settings().mode == "local"


# --------------------------------------------------------------------------- config trust


def test_env_file_never_read_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A planted ``.env`` in the working directory cannot widen the allow-list or switch mode."""
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / ".env").write_text(
        "PRAKTIKA_ALLOWED_HOSTS=evil.example\nPRAKTIKA_LLM_BASE_URL=http://evil.example:11434\n"
        "PRAKTIKA_PILOT=false\nPRAKTIKA_MODE=service\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(planted)
    s = Settings()
    assert s.allowed_hosts == ["localhost", "127.0.0.1"] and s.pilot is True
    assert s.mode == "local" and s.llm_base_url.host == "127.0.0.1"
    # the honoured location is explicit: PRAKTIKA_ENV_FILE or data_dir/.env
    from praktika.config import default_data_dir, env_file_path

    assert env_file_path() == default_data_dir() / ".env"
    own = tmp_path / "own.env"
    own.write_text("PRAKTIKA_PILOT=false\n", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(own))
    assert env_file_path() == own
    assert Settings(_env_file=env_file_path()).pilot is False


def test_default_data_dir_is_platform_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS keeps Application Support; Linux uses the XDG data directory, and the
    fallback env file follows the data directory."""
    from praktika.config import default_data_dir, env_file_path

    home = Path("~").expanduser()
    mac = default_data_dir("darwin", {"XDG_DATA_HOME": str(tmp_path)})
    assert mac == home / "Library" / "Application Support" / "Praktika"
    assert env_file_path("darwin", {}) == mac / ".env"

    xdg = tmp_path / "xdg"
    assert default_data_dir("linux", {"XDG_DATA_HOME": str(xdg)}) == xdg / "praktika"
    assert env_file_path("linux", {"XDG_DATA_HOME": str(xdg)}) == xdg / "praktika" / ".env"
    fallback = home / ".local" / "share" / "praktika"
    assert default_data_dir("linux", {}) == fallback
    assert default_data_dir("linux", {"XDG_DATA_HOME": ""}) == fallback
    assert default_data_dir("linux", {"XDG_DATA_HOME": "relative/dir"}) == fallback, (
        "the XDG spec says a relative XDG_DATA_HOME is invalid and must be ignored"
    )
    assert "Library" not in str(default_data_dir("linux", {}))


def test_linux_settings_default_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Settings().data_dir`` resolves the platform default at construction time;
    ``PRAKTIKA_DATA_DIR`` and ``PRAKTIKA_ENV_FILE`` still override."""
    from praktika import config

    xdg = tmp_path / "xdg"
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert settings().data_dir == xdg / "praktika"
    assert config.env_file_path() == xdg / "praktika" / ".env"
    monkeypatch.setenv("PRAKTIKA_DATA_DIR", str(tmp_path / "own-data"))
    assert settings().data_dir == tmp_path / "own-data"
    (tmp_path / "own.env").write_text("", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(tmp_path / "own.env"))
    assert config.env_file_path() == tmp_path / "own.env"


def test_named_env_file_must_exist_and_be_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped or not-yet-written PRAKTIKA_ENV_FILE used to be skipped silently, so every
    command ran on the defaults (the development speech backends, the default data directory).
    Now it is a configuration error; only the unnamed default location may be absent."""
    from praktika import config
    from praktika.errors import ConfigError

    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert config.env_file_path() == tmp_path / "xdg" / "praktika" / ".env", "absent is fine"

    missing = tmp_path / "praktika.env"
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(missing))
    with pytest.raises(ConfigError, match=r"PRAKTIKA_ENV_FILE=.*praktika\.env does not exist"):
        config.env_file_path()
    with pytest.raises(ConfigError, match="does not exist"):
        config.get_settings.__wrapped__()
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(tmp_path))
    with pytest.raises(ConfigError, match="cannot be read"):
        config.env_file_path()

    missing.write_text("PRAKTIKA_REVIEW_PORT=8800\n", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(missing))
    assert config.env_file_path() == missing
    assert Settings(_env_file=config.env_file_path()).review_port == 8800
    if os.geteuid() != 0:  # root reads any file
        missing.chmod(0o000)
        try:
            with pytest.raises(ConfigError, match=r"cannot be read \(Permission denied\)"):
                config.env_file_path()
        finally:
            missing.chmod(0o600)


def test_wildcard_allow_list_refused() -> None:
    for bad in ("*", "**", "*.*", "?"):
        with pytest.raises(EgressError, match="host label"):
            settings(allowed_hosts=[bad])
    ok = settings(allowed_hosts=["*.gateway.acme.internal", "127.0.0.1"])
    assert ok.allowed_hosts == ["*.gateway.acme.internal", "127.0.0.1"]


def test_prompts_and_glossary_default_to_the_package_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prompts").mkdir()
    monkeypatch.chdir(tmp_path)
    s = settings()
    assert s.prompts_dir == REPO / "prompts" and s.prompts_dir.is_absolute()
    assert s.glossary_path == REPO / "glossary.yaml"
    assert (s.prompts_dir / "v1" / "system_common.md").is_file()


def test_env_example_loads_cleanly() -> None:
    """The shipped template must be a valid env file as-is (comments must not become values),
    and it is the Linux server profile: speech over HTTP, Arabic off."""
    s = Settings(_env_file=REPO / ".env.example")
    assert s.stt_en == "http" and s.stt_ar == "none"
    assert str(s.stt_http_url) == "http://127.0.0.1:8801/"
    assert s.oidc_issuer is None and s.oidc_jwks_url is None
    assert s.mode == "local" and s.llm_provider == "ollama" and s.review_port == 8793
    assert s.allowed_hosts == ["localhost", "127.0.0.1"]


def test_numba_threading_layer_set_on_import() -> None:
    """VAD (torch) then word-timestamp DTW (numba) segfaults on the OpenMP layer; the config
    module must pin the single-runtime workqueue layer before any backend import."""
    import os

    import praktika.config  # noqa: F401  (import side effect under test)

    assert os.environ.get("NUMBA_THREADING_LAYER") == "workqueue"


def test_host_matching_keeps_glob_classes_and_compares_ipv6_addresses() -> None:
    assert host_allowed("llm1.example.test", ["llm[12].example.test"])
    assert host_allowed("a.example.test", ["[ab].example.test"])
    assert host_allowed("[fd00::10]", ["fd00:0:0:0:0:0:0:10"])
    assert host_allowed("[fd00::10]", ["[fd00::10]"])
    assert not host_allowed("[fd00::11]", ["fd00::10"])
    assert host_allowed("[fd00::10]", ["*fd00::*"]) and host_allowed("fd00::10", ["fd00::*"])


def test_a_url_with_a_password_is_never_printed(tmp_path: Path) -> None:
    """Configuration errors name the setting and the host, never a value."""
    from pydantic import ValidationError

    from praktika.cli import context as ctx

    with pytest.raises(EgressError) as info:
        settings(llm_base_url="https://svc:s3cret@api.example.com")
    assert "s3cret" not in str(info.value) and "'api.example.com'" in str(info.value)
    with pytest.raises(ValidationError) as bad:
        settings(allowed_hosts='["unclosed', llm_base_url="http://127.0.0.1:11434")
    shown = ctx.describe_invalid(bad.value)
    assert shown.startswith("allowed_hosts:") and "unclosed" not in shown


def test_hide_credentials_removes_only_the_user_and_password() -> None:
    from praktika.config import hide_credentials

    assert hide_credentials("at http://svc:s3cret@127.0.0.1:1/api: refused") == (
        "at http://***@127.0.0.1:1/api: refused"
    )
    assert hide_credentials("a https://u@x.test and HTTP://a:b@y.test/p?q=1") == (
        "a https://***@x.test and HTTP://***@y.test/p?q=1"
    )
    assert hide_credentials("http://u:p@ss@host/x") == "http://***@host/x", "a raw @ in a password"
    plain = "mail f.khalid@example.test; url http://x.test/a@b; http://x.test:8080/"
    assert hide_credentials(plain) == plain


def test_every_log_line_loses_url_passwords(capsys: pytest.CaptureFixture[str]) -> None:
    """The HTTP library logs each request's full URL at INFO, and our own lines may carry one."""
    import logging

    from praktika.logging import configure_logging, get_logger

    try:
        configure_logging(level="INFO")
        logging.getLogger("httpx").info(
            "HTTP Request: %s %s", "GET", "http://svc:s3cret@127.0.0.1:1/api/tags"
        )
        get_logger("praktika.test").info("probe", url="https://u:pw@llm.example.test/v1")
        err = capsys.readouterr().err
    finally:
        configure_logging(level="WARNING")
    assert "s3cret" not in err and "u:pw" not in err
    assert "http://***@127.0.0.1:1/api/tags" in err and "https://***@llm.example.test" in err
