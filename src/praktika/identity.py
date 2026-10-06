"""Who is acting: session identity on the laptop, OIDC identity behind the gateway (C-08).

Contract: every ``IdentityProvider`` returns an ``Identity`` whose ``source`` says how much it can
be trusted. ``SessionIdentity`` on macOS reads the console user and, when the Mac is bound to
Active Directory, resolves the UPN; otherwise it marks the identity ``source="local"``. On Linux
(a headless server) the owner of ``/dev/console`` is root, so the invoking user is used instead
(the kernel login uid, which survives ``sudo``, else the real uid; never an environment
variable) and the identity is always ``source="local"``.
``OidcIdentity`` accepts only a bearer JWT whose RS256 signature, issuer, audience and expiry
verify against a JWKS document fetched through an injected ``httpx.Client`` (so the egress
allow-list applies); roles come from the ``groups`` claim (AD security groups) and the ``roles``
claim (app roles). ``X-Auth-*`` and similar headers are never read.
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import pwd
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel, ConfigDict

from praktika.errors import PraktikaError
from praktika.logging import get_logger

log = get_logger(__name__)

Source = Literal["session", "local", "oidc", "fake"]
AuditSource = Literal["session", "local", "oidc"]

#: AD security groups that carry a Praktika role. Groups not listed here are dropped, so an
#: identity never records a person's unrelated memberships.
ROLE_BY_GROUP: dict[str, str] = {
    "Praktika-Users": "user",
    "Praktika-Secretaries": "secretary",
    "Praktika-DPO": "dpo",
    "Praktika-Admins": "admin",
}

#: App-role values (the ``roles`` claim) accepted as well as the group names above: an app
#: role is scoped to Praktika's own app registration, so the bare role name is unambiguous there.
GROUP_BY_APP_ROLE: dict[str, str] = {role: group for group, role in ROLE_BY_GROUP.items()}

DSCL = "/usr/bin/dscl"
SUBPROCESS_TIMEOUT_S = 3.0
LOGINUID_FILE = Path("/proc/self/loginuid")
UNSET_LOGINUID = 4294967295  # (uint32)-1: no login session (services, early boot)

#: Shape of a UPN or e-mail address named with ``--organiser``: one ``@``, a dotted domain.
_DNS_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_UPN_RE = re.compile(rf"[A-Za-z0-9._%+'-]+@{_DNS_LABEL}(?:\.{_DNS_LABEL})+")
UPN_MAX_LEN = 254


class IdentityError(PraktikaError):
    """A bearer token is missing, malformed or fails validation."""


def normalise_upn(value: str) -> str:
    """Return ``value`` stripped and lower-cased when it has the shape of a UPN or e-mail
    address (``local@domain.tld``, at most 254 characters, no whitespace, no leading or
    trailing dot in the local part); raise ``IdentityError`` otherwise. Shape only: the
    directory is not consulted."""
    candidate = value.strip()
    local = candidate.partition("@")[0]
    if (
        not candidate
        or len(candidate) > UPN_MAX_LEN
        or not _UPN_RE.fullmatch(candidate)
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
    ):
        raise IdentityError(f"not a UPN or e-mail address: {value!r}")
    return candidate.lower()


class Identity(BaseModel):
    """The acting user. ``groups`` holds only the Praktika AD groups from ``ROLE_BY_GROUP``."""

    model_config = ConfigDict(extra="forbid")

    user: str
    display: str
    source: Source
    groups: list[str] = []

    def roles(self) -> list[str]:
        """Role names derived from ``groups`` via ``ROLE_BY_GROUP``, sorted, without duplicates."""
        return sorted({ROLE_BY_GROUP[g] for g in self.groups if g in ROLE_BY_GROUP})

    def audit_source(self) -> AuditSource:
        """The ``actor_source`` to record in audit and consent records.

        A ``fake`` identity (tests, ``identity_provider="fake"``) is recorded as ``local``, the
        least trusted real source, so no record ever claims more assurance than it has.
        """
        return "local" if self.source == "fake" else self.source


class IdentityProvider(Protocol):
    def current(self, request: Any = None) -> Identity: ...


class FakeIdentity:
    """``IdentityProvider`` returning a fixed synthetic identity (``source="fake"``)."""

    def __init__(
        self,
        user: str = "f.khalid@acme.test",
        display: str = "F. Khalid",
        source: Source = "fake",
        groups: tuple[str, ...] = ("Praktika-Users",),
    ) -> None:
        self.user, self.display, self.source, self.groups = user, display, source, list(groups)

    def current(self, request: Any = None) -> Identity:
        return Identity(
            user=self.user, display=self.display, source=self.source, groups=self.groups
        )


# --------------------------------------------------------------------------- session identity


def _run(args: list[str]) -> str:
    """Run a fixed command with a timeout; return stdout, or "" on any failure (never raises)."""
    try:
        # Fixed absolute executable, argument list built from the console user name only.
        proc = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("identity.subprocess_failed", command=args[0], error=str(exc))
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def console_user() -> str:
    """The macOS console user (owner of ``/dev/console``), falling back to the process user."""
    try:
        return pwd.getpwuid(os.stat("/dev/console").st_uid).pw_name
    except (OSError, KeyError):
        return getpass.getuser()


def account_name(uid: int) -> str:
    """The account name of ``uid`` from the password database, or ``uid:<n>`` without one."""
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid:{uid}"


def login_uid(path: Path = LOGINUID_FILE) -> int | None:
    """The kernel audit login uid of this process (``/proc/self/loginuid``), or ``None``.

    ``pam_loginuid`` sets it once when a person logs in (SSH or console) and it survives
    ``sudo`` and ``su``; an unprivileged process cannot change it. ``None`` when the file is
    missing or unreadable (not Linux, some containers) or the value is unset (4294967295, as
    for a systemd service).
    """
    try:
        raw = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not raw.isdigit() or int(raw) == UNSET_LOGINUID:
        return None
    return int(raw)


def invoking_user(
    *,
    uid: int | None = None,
    loginuid: Callable[[], int | None] = login_uid,
    account: Callable[[int], str] = account_name,
) -> str:
    """The person who ran the command on Linux, from kernel facts only.

    The login uid (``login_uid``) when it is set, so an engineer who logged in as ``layla`` and
    runs ``sudo -u praktika praktika ingest`` is recorded as ``layla``; otherwise the account of
    the real uid (``uid``, default ``os.getuid()``). Environment variables (``SUDO_USER``,
    ``LOGNAME``, ``USER``) are never read: the caller controls them, so honouring them would let
    anyone be recorded as anyone. Never the owner of ``/dev/console``, which is root on a
    headless server.
    """
    who = loginuid()
    return account(who if who is not None else (os.getuid() if uid is None else uid))


def parse_dscl(output: str) -> dict[str, str]:
    """Parse ``dscl -read`` output (``Key: value`` or ``Key:`` followed by an indented line)."""
    result: dict[str, str] = {}
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(" ") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if not value and i + 1 < len(lines) and lines[i + 1].startswith(" "):
            value = lines[i + 1].strip()
        result[key.strip()] = value
    return result


class SessionIdentity:
    """Identity of the local user; ``source="session"`` only on an AD-bound Mac.

    macOS: the console user, resolved to a UPN through ``dscl`` when the Mac is AD-bound.
    Linux and every other platform: the invoking user (``invoking_user``), ``source="local"``,
    with no directory lookup. ``runner``, ``user``, ``platform`` (default ``sys.platform`` at
    call time) and ``invoker`` (default ``invoking_user``) are injectable for tests. The
    directory lookup is bounded by a timeout and every failure degrades to ``source="local"``;
    it never hangs or raises.
    """

    def __init__(
        self,
        *,
        user: str | None = None,
        runner: Callable[[list[str]], str] | None = None,
        platform: str | None = None,
        invoker: Callable[[], str] | None = None,
    ) -> None:
        self._user = user
        self._runner = runner or _run
        self._platform = platform
        self._invoker = invoker or invoking_user

    def current(self, request: Any = None) -> Identity:
        platform = self._platform or sys.platform
        if platform != "darwin":
            name = self._user or self._invoker()
            return Identity(user=name, display=_display_name(name), source="local")
        name = self._user or console_user()
        record = parse_dscl(
            self._runner(
                [DSCL, ".", "-read", f"/Users/{name}", "AltSecurityIdentities", "OriginalNodeName"]
            )
        )
        node = record.get("OriginalNodeName", "")
        if "active directory" in node.lower():
            alt = record.get("AltSecurityIdentities", "")
            upn = alt.split(":", 1)[1] if alt.lower().startswith("kerberos:") else ""
            user = upn.lower() or f"{name}@{node.rstrip('/').rsplit('/', 1)[-1].lower()}"
            return Identity(user=user, display=_display_name(name), source="session")
        return Identity(user=name, display=_display_name(name), source="local")


def _display_name(name: str) -> str:
    try:
        gecos = pwd.getpwnam(name).pw_gecos.split(",", 1)[0].strip()
    except KeyError:
        gecos = ""
    return gecos or name


# --------------------------------------------------------------------------- OIDC identity


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise IdentityError("token is not base64url") from exc


def _json_segment(segment: str) -> dict[str, Any]:
    try:
        value = json.loads(_b64url_decode(segment))
    except (ValueError, RecursionError):  # deeply nested JSON from an untrusted token
        raise IdentityError("token segment is not JSON") from None
    if not isinstance(value, dict):
        raise IdentityError("token segment is not an object")
    return value


def rsa_key_from_jwk(jwk: dict[str, Any]) -> rsa.RSAPublicKey:
    """Build an RSA public key from a JWK with ``n`` and ``e``; ``IdentityError`` otherwise."""
    if jwk.get("kty") != "RSA" or "n" not in jwk or "e" not in jwk:
        raise IdentityError("JWK is not an RSA key")
    n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
    e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
    return rsa.RSAPublicNumbers(e=e, n=n).public_key()


#: Seconds between two JWKS fetches, so a stream of tokens naming unknown key ids cannot make the
#: server hammer the identity provider; a rotated key is picked up within this time.
JWKS_REFETCH_S = 60.0


class OidcIdentity:
    """Validate an RS256 bearer JWT against a JWKS document and map groups and app roles to
    Praktika roles (``praktika_groups``).

    ``client`` must be an ``httpx.Client`` built by ``Settings.http_client`` so the JWKS fetch is
    subject to the allow-list. Keys are cached; an unknown ``kid`` triggers a refetch at most
    once per ``JWKS_REFETCH_S`` (rotation). ``current(request)`` reads only
    ``request.headers["authorization"]``; any other header is ignored. Raises ``IdentityError``
    on any validation failure.
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str,
        client: httpx.Client,
        *,
        leeway_s: int = 60,
        clock: Callable[[], float] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.issuer, self.audience, self.jwks_url = issuer, audience, jwks_url
        self._client, self._leeway, self._clock = client, leeway_s, clock or time.time
        #: Times the refetch interval; never steps backwards, unlike the clock for exp and nbf.
        self._monotonic = monotonic or time.monotonic
        self._keys: dict[str, rsa.RSAPublicKey] = {}
        self._fetched_at: float | None = None
        self._last_failure: str | None = None
        #: One fetch at a time; a request that needs the keys meanwhile waits for its result.
        self._fetch_lock = threading.Lock()

    def _load_keys(self) -> None:
        """Fetch the signing keys and keep the RSA signing keys among them (an EC key or an
        encryption key the provider also publishes is skipped, as is a malformed entry).

        A failure reaches the caller only as ``JWKS fetch failed: <reason>`` (the HTTP status,
        the error type, or a document with no key list), never the JWKS URL: that message is the
        401 body sent to an unauthenticated caller and the server's access-denied log line, and
        the HTTP library's own error text names the URL (an internal host, possibly with a user
        name and password). The full cause goes to the operator log, where any password is
        masked. Called with ``_fetch_lock`` held."""
        self._fetched_at = self._monotonic()
        try:
            resp = self._client.get(self.jwks_url)
            resp.raise_for_status()
            document = resp.json()
        except httpx.HTTPStatusError as exc:
            self._fail(f"HTTP {exc.response.status_code}", exc)
        except (
            httpx.HTTPError,
            PraktikaError,
            ValueError,
        ) as exc:  # EgressError is a PraktikaError
            self._fail(type(exc).__name__, exc)
        keys = document.get("keys") if isinstance(document, dict) else None
        if not isinstance(keys, list):
            self._fail("no key list in the document", None)
        found: dict[str, rsa.RSAPublicKey] = {}
        for jwk in keys:
            if not isinstance(jwk, dict) or not isinstance(jwk.get("kid"), str):
                continue
            if jwk.get("kty") != "RSA" or jwk.get("use", "sig") != "sig":
                continue
            try:
                found[jwk["kid"]] = rsa_key_from_jwk(jwk)
            except (IdentityError, TypeError, ValueError):
                log.warning("identity.jwks_key_skipped", kid=jwk["kid"])
        self._keys, self._last_failure = found, None

    def _fail(self, reason: str, exc: Exception | None) -> NoReturn:
        self._last_failure = reason
        log.warning("identity.jwks_fetch_failed", reason=reason, error=str(exc) if exc else None)
        raise IdentityError(f"JWKS fetch failed: {reason}") from None

    def _key(self, kid: str) -> rsa.RSAPublicKey:
        if kid not in self._keys:
            with self._fetch_lock:  # a fetch already running answers this request too
                if kid not in self._keys:
                    if self._fetch_due():
                        self._load_keys()
                    elif self._last_failure is not None:
                        raise IdentityError(f"JWKS fetch failed: {self._last_failure}")
        if kid not in self._keys:
            raise IdentityError("token key id not in JWKS")
        return self._keys[kid]

    def _fetch_due(self) -> bool:
        return self._fetched_at is None or self._monotonic() - self._fetched_at >= JWKS_REFETCH_S

    def validate(self, token: str) -> dict[str, Any]:
        """Verify signature, ``iss``, ``aud``, ``exp`` (and ``nbf`` if present); return claims."""
        parts = token.split(".")
        if len(parts) != 3:
            raise IdentityError("token is not a JWS compact serialisation")
        header = _json_segment(parts[0])
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise IdentityError("token must be RS256 with a kid")
        try:
            signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        except UnicodeEncodeError:
            raise IdentityError("token is not base64url") from None
        try:
            self._key(header["kid"]).verify(
                _b64url_decode(parts[2]), signing_input, padding.PKCS1v15(), hashes.SHA256()
            )
        except InvalidSignature as exc:
            raise IdentityError("token signature invalid") from exc
        claims = _json_segment(parts[1])
        if claims.get("iss") != self.issuer:
            raise IdentityError("token issuer mismatch")
        aud = claims.get("aud")
        if not (aud == self.audience or (isinstance(aud, list) and self.audience in aud)):
            raise IdentityError("token audience mismatch")
        now = self._clock()
        exp, nbf = claims.get("exp"), claims.get("nbf", now)
        if not isinstance(exp, int | float) or now > exp + self._leeway:
            raise IdentityError("token expired or has no exp")
        if not isinstance(nbf, int | float) or now + self._leeway < nbf:
            raise IdentityError("token not yet valid")
        return claims

    def current(self, request: Any = None) -> Identity:
        headers = getattr(request, "headers", None)
        auth = headers.get("authorization", "") if headers is not None else ""
        scheme, _, token = str(auth).partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise IdentityError("missing bearer token")
        claims = self.validate(token.strip())
        user = claims.get("preferred_username") or claims.get("upn") or claims.get("sub")
        if not isinstance(user, str) or not user:
            raise IdentityError("token carries no user claim")
        groups = praktika_groups(claims)
        display = claims.get("name") if isinstance(claims.get("name"), str) else user
        log.info("identity.oidc_validated", user=user, roles=len(groups))
        return Identity(user=user.lower(), display=display, source="oidc", groups=groups)


def _claim_list(claims: dict[str, Any], name: str) -> list[str]:
    """String members of a list claim; a missing or non-list claim yields ``[]``."""
    value = claims.get(name, [])
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def praktika_groups(claims: dict[str, Any]) -> list[str]:
    """The Praktika groups a validated token carries, sorted and without duplicates.

    ``groups`` (AD security groups): only names in ``ROLE_BY_GROUP``. ``roles`` (app roles, how
    the gateway sign-in assigns access): names in ``ROLE_BY_GROUP`` or the bare role names
    (``user``, ``secretary``, ``dpo``, ``admin``), mapped to their group. Anything else is
    dropped, so unrelated memberships are never recorded; a token with neither claim has no
    Praktika role.
    """
    found = {g for g in _claim_list(claims, "groups") if g in ROLE_BY_GROUP}
    for role in _claim_list(claims, "roles"):
        if role in ROLE_BY_GROUP:
            found.add(role)
        elif role in GROUP_BY_APP_ROLE:
            found.add(GROUP_BY_APP_ROLE[role])
    return sorted(found)
