"""Identity providers (control C-08)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from helpers_core import Request, Signer

from praktika import identity as ident
from praktika.identity import (
    FakeIdentity,
    Identity,
    IdentityError,
    OidcIdentity,
    SessionIdentity,
    parse_dscl,
)

ISSUER = "https://login.acme.test/tenant/v2.0"
AUDIENCE = "api://praktika-test"
JWKS_URL = "http://127.0.0.1/keys"


@pytest.fixture(scope="module")
def signer() -> Signer:
    return Signer()


def _client(signer: Signer, calls: list[str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(200, json=signer.jwks())

    return httpx.Client(transport=httpx.MockTransport(handler))


def _claims(**over: Any) -> dict[str, Any]:
    now = int(time.time())
    base = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + 600,
        "nbf": now - 10,
        "sub": "abc123",
        "preferred_username": "R.Haddad@acme.test",
        "name": "R. Haddad",
        "groups": ["Praktika-DPO", "Praktika-Users", "Finance-Readers"],
    }
    base.update(over)
    return base


def _oidc(signer: Signer, calls: list[str] | None = None) -> OidcIdentity:
    return OidcIdentity(ISSUER, AUDIENCE, JWKS_URL, _client(signer, calls))


# --------------------------------------------------------------------------- session


def test_session_identity_marks_source_local_when_not_ad_bound() -> None:
    def runner(args: list[str]) -> str:
        assert args[0] == ident.DSCL and args[3] == "/Users/layla"
        return "OriginalNodeName:\n /Local/Default\n"

    who = SessionIdentity(user="layla", runner=runner, platform="darwin").current()
    assert who.source == "local" and who.user == "layla" and who.audit_source() == "local"


def test_session_identity_resolves_upn_when_ad_bound() -> None:
    output = (
        "AltSecurityIdentities: Kerberos:L.Farouk@ACME.TEST\n"
        "OriginalNodeName:\n /Active Directory/ACME/acme.test\n"
    )
    who = SessionIdentity(user="lfarouk", runner=lambda _: output, platform="darwin").current()
    assert who.source == "session" and who.user == "l.farouk@acme.test"


def test_session_identity_ad_bound_without_kerberos_uses_domain() -> None:
    output = "OriginalNodeName: /Active Directory/ACME/acme.test\n"
    who = SessionIdentity(user="omar", runner=lambda _: output, platform="darwin").current()
    assert who.source == "session" and who.user == "omar@acme.test"


def test_session_identity_never_hangs_or_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_run(*a: Any, **kw: Any) -> Any:
        assert kw["timeout"] <= 5, "the directory lookup must be bounded by a short timeout"
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=kw["timeout"])

    monkeypatch.setattr(ident.subprocess, "run", slow_run)
    who = SessionIdentity(user="omar", platform="darwin").current()
    assert who.source == "local" and who.user == "omar"

    monkeypatch.setattr(ident.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert SessionIdentity(user="omar", platform="darwin").current().source == "local"


def test_session_identity_on_linux_is_the_invoking_user_not_console_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a Linux server ``/dev/console`` belongs to root; the invoking user must be recorded."""
    monkeypatch.setattr(ident, "console_user", lambda: "root")

    def no_dscl(args: list[str]) -> str:
        raise AssertionError("no directory lookup on Linux")

    who = SessionIdentity(runner=no_dscl, platform="linux", invoker=lambda: "layla").current()
    assert (who.user, who.source, who.audit_source()) == ("layla", "local", "local")


def test_invoking_user_prefers_the_login_uid_over_the_real_uid() -> None:
    """``sudo -u praktika`` keeps the kernel login uid, so the engineer is still recorded."""
    names = {1001: "layla", 1002: "praktika"}
    via_sudo = ident.invoking_user(uid=1002, loginuid=lambda: 1001, account=names.__getitem__)
    assert via_sudo == "layla"
    service = ident.invoking_user(uid=1002, loginuid=lambda: None, account=names.__getitem__)
    assert service == "praktika"


def test_invoking_user_ignores_forged_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: SUDO_USER / USER / LOGNAME are caller-controlled and must not name the actor."""
    for var in ("SUDO_USER", "USER", "LOGNAME", "LNAME", "USERNAME"):
        monkeypatch.setenv(var, "ceo")
    monkeypatch.setattr(ident.getpass, "getuser", lambda: "ceo")
    real = ident.account_name(ident.os.getuid())
    assert real != "ceo"
    assert ident.invoking_user(loginuid=lambda: None) == real
    linux = SessionIdentity(platform="linux").current()
    assert linux.user != "ceo"


def test_login_uid_reads_the_kernel_value(tmp_path: Path) -> None:
    f = tmp_path / "loginuid"
    f.write_text("1001", encoding="ascii")
    assert ident.login_uid(f) == 1001
    f.write_text(str(ident.UNSET_LOGINUID), encoding="ascii")
    assert ident.login_uid(f) is None
    f.write_text("garbage", encoding="ascii")
    assert ident.login_uid(f) is None
    assert ident.login_uid(tmp_path / "missing") is None
    assert ident.account_name(2**31 - 7).startswith("uid:")


def test_session_identity_on_macos_still_reads_the_console_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ident, "console_user", lambda: "lfarouk")
    seen: list[str] = []

    def runner(args: list[str]) -> str:
        seen.append(args[3])
        return "OriginalNodeName:\n /Local/Default\n"

    who = SessionIdentity(runner=runner, platform="darwin", invoker=lambda: "x").current()
    assert who.user == "lfarouk" and seen == ["/Users/lfarouk"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("L.Farouk@Acme.test", "l.farouk@acme.test"),
        ("  r.haddad@acme.test ", "r.haddad@acme.test"),
        ("o'neil+minutes@mail.acme.test", "o'neil+minutes@mail.acme.test"),
    ],
)
def test_normalise_upn_accepts_upn_shapes(value: str, expected: str) -> None:
    assert ident.normalise_upn(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "layla",
        "layla@localhost",
        "@acme.test",
        "a@b@acme.test",
        "has space@acme.test",
        ".lead@acme.test",
        "a..b@acme.test",
        "x@-bad.example",
        "x@acme.test\nforged",
        "a" * 250 + "@ex.test",
    ],
)
def test_normalise_upn_rejects_other_shapes(value: str) -> None:
    with pytest.raises(IdentityError, match="UPN"):
        ident.normalise_upn(value)


def test_parse_dscl_handles_inline_and_indented_values() -> None:
    parsed = parse_dscl("A: one\nB:\n two words\nC:\nD: x:y\n")
    assert parsed == {"A": "one", "B": "two words", "C": "", "D": "x:y"}


# --------------------------------------------------------------------------- OIDC


def test_oidc_accepts_valid_token(signer: Signer) -> None:
    who = _oidc(signer).current(Request(authorization=f"Bearer {signer.token(_claims())}"))
    assert who == Identity(
        user="r.haddad@acme.test",
        display="R. Haddad",
        source="oidc",
        groups=["Praktika-DPO", "Praktika-Users"],
    )


def test_oidc_rejects_bad_signature(signer: Signer) -> None:
    other = Signer(kid=signer.kid)  # same kid, different private key
    token = other.token(_claims())
    with pytest.raises(IdentityError, match="signature"):
        _oidc(signer).validate(token)
    # A tampered payload with the genuine signature fails too.
    h, p, s = signer.token(_claims()).split(".")
    tampered = signer.token(_claims(preferred_username="admin@acme.test")).split(".")[1]
    with pytest.raises(IdentityError, match="signature"):
        _oidc(signer).validate(f"{h}.{tampered}.{s}")


def test_oidc_rejects_wrong_audience(signer: Signer) -> None:
    with pytest.raises(IdentityError, match="audience"):
        _oidc(signer).validate(signer.token(_claims(aud="api://someone-else")))
    with pytest.raises(IdentityError, match="issuer"):
        _oidc(signer).validate(signer.token(_claims(iss="https://evil.example.test")))
    # Audience may be a list that contains ours.
    claims = _oidc(signer).validate(signer.token(_claims(aud=["x", AUDIENCE])))
    assert claims["sub"] == "abc123"


def test_oidc_rejects_expired_and_not_yet_valid(signer: Signer) -> None:
    past = int(time.time()) - 3600
    with pytest.raises(IdentityError, match="expired"):
        _oidc(signer).validate(signer.token(_claims(exp=past)))
    with pytest.raises(IdentityError, match="expired"):
        _oidc(signer).validate(signer.token({**_claims(), "exp": None}))
    with pytest.raises(IdentityError, match="not yet valid"):
        _oidc(signer).validate(signer.token(_claims(nbf=int(time.time()) + 3600)))


def test_oidc_rejects_non_rs256_and_unknown_kid(signer: Signer) -> None:
    with pytest.raises(IdentityError, match="RS256"):
        _oidc(signer).validate(signer.token(_claims(), alg="none"))
    with pytest.raises(IdentityError, match="RS256"):
        _oidc(signer).validate(signer.token(_claims(), alg="HS256"))
    with pytest.raises(IdentityError, match="key id"):
        _oidc(signer).validate(signer.token(_claims(), kid="rotated-away"))
    with pytest.raises(IdentityError):
        _oidc(signer).validate("not.a.jwt.at.all")
    with pytest.raises(IdentityError):
        _oidc(signer).validate("garbage")


def test_oidc_maps_groups_to_roles(signer: Signer) -> None:
    who = _oidc(signer).current(Request(authorization=f"Bearer {signer.token(_claims())}"))
    assert who.roles() == ["dpo", "user"]
    assert "Finance-Readers" not in who.groups, "unrelated AD groups must not be recorded"
    nobody = _oidc(signer).current(
        Request(authorization=f"Bearer {signer.token(_claims(groups=['Something']))}")
    )
    assert nobody.groups == [] and nobody.roles() == []


def test_oidc_maps_app_roles_claim_to_roles(signer: Signer) -> None:
    """The gateway sign-in assigns access through app roles (the ``roles`` claim); the group
    names and the bare role names are both accepted, anything else is dropped."""
    token = signer.token(
        {
            **{k: v for k, v in _claims().items() if k != "groups"},
            "roles": ["Praktika-Secretaries", "admin", "Finance.Reader", 7],
        }
    )
    who = _oidc(signer).current(Request(authorization=f"Bearer {token}"))
    assert who.roles() == ["admin", "secretary"]
    assert who.groups == ["Praktika-Admins", "Praktika-Secretaries"]
    assert "Finance.Reader" not in who.groups


def test_oidc_merges_groups_and_roles_claims_without_duplicates(signer: Signer) -> None:
    claims = _claims(groups=["Praktika-DPO", "Praktika-Users"], roles=["user", "Praktika-DPO"])
    who = _oidc(signer).current(Request(authorization=f"Bearer {signer.token(claims)}"))
    assert who.groups == ["Praktika-DPO", "Praktika-Users"] and who.roles() == ["dpo", "user"]


def test_oidc_token_with_neither_groups_nor_roles_has_no_role(signer: Signer) -> None:
    bare = {k: v for k, v in _claims().items() if k != "groups"}
    who = _oidc(signer).current(Request(authorization=f"Bearer {signer.token(bare)}"))
    assert who.groups == [] and who.roles() == []
    # malformed claims (not lists) are treated as absent, never as a role
    odd = _claims(groups="Praktika-Admins", roles={"admin": True})
    who = _oidc(signer).current(Request(authorization=f"Bearer {signer.token(odd)}"))
    assert who.groups == [] and who.roles() == []


def test_bare_role_names_only_count_in_the_roles_claim() -> None:
    """A tenant-wide group called ``admin`` is not Praktika's admin role."""
    assert ident.praktika_groups({"groups": ["admin", "dpo"]}) == []
    assert ident.praktika_groups({"roles": ["dpo"]}) == ["Praktika-DPO"]


def test_oidc_jwks_fetched_once_and_refetched_on_rotation(signer: Signer) -> None:
    calls: list[str] = []
    provider = _oidc(signer, calls)
    provider.validate(signer.token(_claims()))
    provider.validate(signer.token(_claims()))
    assert calls == [JWKS_URL]
    with pytest.raises(IdentityError, match="key id"):
        provider.validate(signer.token(_claims(), kid="new-kid"))
    assert len(calls) == 2, "an unknown kid triggers exactly one refetch"


def test_oidc_jwks_fetch_failure_is_identity_error(signer: Signer) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(IdentityError, match="JWKS"):
        OidcIdentity(ISSUER, AUDIENCE, JWKS_URL, client).validate(signer.token(_claims()))


def test_headers_are_never_trusted(signer: Signer) -> None:
    provider = _oidc(signer)
    # Proxy-style identity headers alone: refused.
    with pytest.raises(IdentityError, match="bearer"):
        provider.current(Request(x_auth_user="ceo@acme.test", x_auth_groups="Admins"))
    with pytest.raises(IdentityError, match="bearer"):
        provider.current(Request(x_forwarded_user="ceo@acme.test"))
    with pytest.raises(IdentityError, match="bearer"):
        provider.current(None)
    with pytest.raises(IdentityError, match="bearer"):
        provider.current(Request(authorization="Basic abc"))
    # With a valid token, the headers must not override anything in the token.
    who = provider.current(
        Request(
            authorization=f"Bearer {signer.token(_claims())}",
            x_auth_user="ceo@acme.test",
            x_auth_groups="Praktika-Admins",
        )
    )
    assert who.user == "r.haddad@acme.test" and "admin" not in who.roles()


# --------------------------------------------------------------------------- Identity model


def test_fake_identity_is_recorded_as_local_in_audit() -> None:
    who = FakeIdentity().current()
    assert who.source == "fake" and who.audit_source() == "local"
    assert FakeIdentity(source="session").current().audit_source() == "session"
    with pytest.raises(ValueError):
        Identity(user="x", display="x", source="oidc", extra_field=1)  # type: ignore[call-arg]
