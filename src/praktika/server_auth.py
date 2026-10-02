"""Authorisation and request-origin guards of the review server (controls C-01, C-07, C-08).

Authorisation (``require_access``, ``visible``): the organiser of a meeting may read and change
it; a ``secretary`` may do the same for meetings that are not private; ``dpo`` and ``admin``
may read every meeting but never act on one; every other identity is refused (403, raised as
``AccessDeniedError`` so the application can audit it as ``auth.denied``). Identities that carry
no Praktika role at all are refused outright (401) — except the laptop's own console user
(``source`` ``session``/``local``), who is the sole operator of a local data directory and is
treated as a ``user``.

Origin guard (``RequestGuard``): the ``Host`` header must name the bound address (loopback, or
the configured review host and egress allow-list in service mode) or the request is refused
with 400, which defeats DNS rebinding; every ``POST`` must carry ``X-Praktika-Review: 1`` (a
custom header no cross-site form can set) and, when ``Origin``/``Referer`` are present, they
must name a trusted host, or the request is refused with 403 (CSRF). In local mode a per-run
session token printed in the review URL must accompany every ``/api`` request as
``X-Praktika-Token`` (401 otherwise), so a page loaded from elsewhere cannot read minutes.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from praktika.config import Settings, host_allowed
from praktika.identity import Identity, IdentityError
from praktika.logging import get_logger
from praktika.models import Meeting

log = get_logger(__name__)

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
REVIEW_HEADER = "x-praktika-review"
TOKEN_HEADER = "x-praktika-token"  # noqa: S105 - header name, not a secret
UNBOUND_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})  # noqa: S104 - wildcard binds
READ_ONLY_ROLES = frozenset({"dpo", "admin"})
_ANY = frozenset({"user", "secretary", "dpo", "admin"})


class AccessDeniedError(Exception):
    """An authorisation refusal (403) carrying who was refused what, for the audit trail."""

    def __init__(self, user: Identity, meeting: Meeting, reason: str, message: str) -> None:
        super().__init__(message)
        self.user, self.meeting_id, self.reason = user, meeting.id, reason


def new_session_token() -> str:
    """A fresh random token for one ``praktika serve`` run (local mode)."""
    return secrets.token_urlsafe(24)


TOKEN_FILE = "review.token"  # noqa: S105 - a file name, not a secret


def session_token_for(data_dir: Path) -> str:
    """The local-mode session token for ``data_dir``, created on first use (file mode 0600).

    Persisting the token per data directory (rather than per ``serve`` run) keeps the review
    links printed by ``start``, ``ingest`` and ``generate`` valid, and lets a browser tab
    survive a server restart. The protection is unchanged: the token only stops other local
    processes and pages from calling the API, and it never leaves the machine.
    """
    path = Path(data_dir) / TOKEN_FILE
    try:
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 24:
            return token
    except FileNotFoundError:
        pass
    token = new_session_token()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    os.chmod(path, 0o600)
    return token


def effective_roles(user: Identity) -> list[str]:
    """Roles derived from AD groups; the laptop console user is a ``user`` by construction."""
    roles = user.roles()
    if not roles and user.source in ("session", "local"):
        return ["user"]
    return roles


def require_roles(user: Identity) -> Identity:
    """Raise ``IdentityError`` (401) for an identity that carries no Praktika role."""
    if not effective_roles(user):
        raise IdentityError("identity carries no Praktika role; access refused")
    return user


def visible(user: Identity, meeting: Meeting) -> bool:
    """True when ``user`` may read ``meeting``."""
    roles = set(effective_roles(user))
    if meeting.organiser.lower() == user.user.lower():
        return True
    if "secretary" in roles and not meeting.private:
        return True
    return bool(roles & READ_ONLY_ROLES)


def require_access(user: Identity, meeting: Meeting, *, write: bool) -> None:
    """Raise ``AccessDeniedError`` (mapped to 403 and audited) unless ``user`` may read (or, with
    ``write``, change) ``meeting``: organiser always; secretary on non-private meetings;
    dpo/admin read-only."""
    if not visible(user, meeting):
        raise AccessDeniedError(
            user, meeting, "not_visible", "you are not allowed to access this meeting"
        )
    if not write:
        return
    roles = set(effective_roles(user))
    organiser = meeting.organiser.lower() == user.user.lower()
    if organiser or ("secretary" in roles and not meeting.private):
        return
    raise AccessDeniedError(
        user, meeting, "read_only_role", "your role is read-only for this meeting"
    )


def trusted_hosts(settings: Settings) -> list[str]:
    """Host patterns a request may name: loopback, plus the review host and the egress
    allow-list in service mode (the gateway's FQDN is what browsers send there)."""
    hosts = set(LOOPBACK)
    if settings.mode == "service":
        if settings.review_host not in UNBOUND_HOSTS:
            hosts.add(settings.review_host)
        hosts.update(settings.allowed_hosts)
    return sorted(hosts)


def _hostname(value: str) -> str:
    """The host part of a ``Host`` header value or a URL, without port, lower-cased."""
    if "://" in value:
        value = urlsplit(value).netloc
    host = value.strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class RequestGuard(BaseHTTPMiddleware):
    """Host, CSRF and (local mode) session-token checks for every request."""

    def __init__(self, app: ASGIApp, settings: Settings, token: str | None) -> None:
        super().__init__(app)
        self.trusted = trusted_hosts(settings)
        self.token = token

    def _trusted(self, value: str) -> bool:
        return host_allowed(_hostname(value), self.trusted)

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        host = request.headers.get("host", "")
        if not self._trusted(host):
            log.warning("server.host_refused", host=host[:80])
            return JSONResponse({"detail": "host not trusted"}, status_code=400)
        path = request.url.path
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            for header in ("origin", "referer"):
                value = request.headers.get(header)
                if value and not self._trusted(value):
                    log.warning("server.origin_refused", header=header, value=value[:80])
                    return JSONResponse({"detail": "cross-site request refused"}, 403)
            if request.headers.get(REVIEW_HEADER) != "1":
                return JSONResponse({"detail": "missing X-Praktika-Review header"}, 403)
        if self.token is not None and path.startswith("/api/") and path != "/api/health":
            given = request.headers.get(TOKEN_HEADER, "")
            if not secrets.compare_digest(given, self.token):
                log.warning("server.session_token_refused", path=path[:80])
                return JSONResponse({"detail": "missing or wrong session token"}, 401)
        return await call_next(request)
