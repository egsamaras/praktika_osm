"""Review API and static UI (controls C-07, C-08).

``create_app`` wires a FastAPI application over an injected ``Store``, ``IdentityProvider`` and
``AuditLog``. Every mutating route resolves the acting identity from the request, records a
``ReviewItem`` or review status on the latest minutes, moves the meeting state machine and
appends one audit event on behalf of that identity. Approval is refused (403) while
``Minutes.blocking_flags()`` is non-empty; export follows ``policy.export_allowed``; the audio
slice route serves retained WAV only and answers 404 for restricted meetings or deleted media.
In local mode the app refuses to bind anything but loopback; service mode is refused unless
the identity provider is ``oidc``, whatever the bind address (fail closed: behind the gateway
a loopback bind is still reachable by every network caller). Authorisation and the
host/CSRF/session token guards live in ``server_auth`` and apply to every route; denials and
read-only accesses are audited (``auth.denied``, ``review.opened`` with ``read_only``,
``audio.read``).

The routes live in ``server_review`` (review actions) and ``server_export`` (audio, export);
the shared helpers and request models in ``server_support`` are re-exported here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from praktika.audit import AuditLog
from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.identity import IdentityError, IdentityProvider
from praktika.llm import pipeline
from praktika.llm.base import LLMClient
from praktika.logging import get_logger
from praktika.models import MeetingState
from praktika.server_auth import (
    AccessDeniedError,
    RequestGuard,
    new_session_token,
    session_token_for,
    visible,
)
from praktika.server_export import register_media_routes
from praktika.server_review import register_review_routes
from praktika.server_support import (
    LOOPBACK,
    MAX_SLICE_S,
    REASON_CODES,
    STATIC_DIR,
    ReasonRequest,
    RegenerateRequest,
    ReviewContext,
    ReviewItemRequest,
    item_kind,
    locate_item,
    rename_ref_speakers,
    sorted_flags,
    wav_slice,
)
from praktika.store.records import Store

__all__ = [
    "LOOPBACK",
    "MAX_SLICE_S",
    "REASON_CODES",
    "ReasonRequest",
    "RegenerateRequest",
    "ReviewContext",
    "ReviewItemRequest",
    "bind_address",
    "create_app",
    "require_service_identity",
    "item_kind",
    "locate_item",
    "new_session_token",
    "session_token_for",
    "rename_ref_speakers",
    "serve",
    "sorted_flags",
    "wav_slice",
]

log = get_logger(__name__)


def require_service_identity(settings: Settings) -> None:
    """Raise ``PraktikaError`` when service mode runs without ``identity_provider=oidc``.

    The check is keyed on the mode, not the bind address: in the container a loopback bind
    sits behind a gateway or sidecar that forwards every network caller, and with ``session``
    or ``fake`` each of them would be authenticated as the console user with the organiser's
    rights on every meeting.
    """
    if settings.mode == "service" and settings.identity_provider != "oidc":
        raise PraktikaError(
            f"service mode with identity_provider={settings.identity_provider!r} authenticates "
            "nobody: every caller would act as the console user; set "
            "PRAKTIKA_IDENTITY_PROVIDER=oidc (with issuer, audience and JWKS URL)"
        )


def bind_address(settings: Settings) -> tuple[str, int]:
    """The (host, port) the review server may bind.

    Local mode permits loopback only. Service mode requires ``identity_provider=oidc``
    whatever the host (``require_service_identity``).
    """
    host = settings.review_host
    if settings.mode == "local" and host not in LOOPBACK:
        raise PraktikaError(
            f"review_host={host!r} is not loopback; local mode serves 127.0.0.1 only "
            "(PRAKTIKA_MODE=service behind the gateway may bind elsewhere)"
        )
    require_service_identity(settings)
    return host, settings.review_port


def serve(app: FastAPI, settings: Settings) -> None:
    """Run ``app`` with uvicorn on ``bind_address(settings)``."""
    import uvicorn  # local import: the test suite never starts a server

    host, port = bind_address(settings)
    log.info("server.start", host=host, port=port, mode=settings.mode)
    uvicorn.run(app, host=host, port=port, log_config=None)


def review_url(settings: Settings, token: str | None, meeting_id: str | None = None) -> str:
    """The URL to open the review UI (local mode carries the session token as ``?t=``)."""
    host = settings.review_host if settings.review_host not in ("0.0.0.0", "::") else "127.0.0.1"  # noqa: S104
    base = f"http://{host}:{settings.review_port}/"
    query = f"?t={token}" if token else ""
    fragment = f"#/meetings/{meeting_id}" if meeting_id else ""
    return base + query + fragment


def create_app(
    settings: Settings,
    store: Store,
    identity: IdentityProvider,
    audit: AuditLog,
    *,
    llm_client: LLMClient | None = None,
    clock: Callable[[], datetime] | None = None,
    session_token: str | None = None,
) -> FastAPI:
    """Build the review application. Raises ``PraktikaError`` when local mode would bind a
    non-loopback ``review_host``. ``llm_client`` enables ``/regenerate`` (503 without it);
    ``clock`` is the timestamp source for review records (tests pass a fixed instant);
    ``session_token``, when given, must accompany every ``/api`` request (local mode).

    Handlers are plain functions, so FastAPI runs them in its worker threads: ``SqliteStore``
    serialises access under its own lock, and a slow regenerate call never blocks the event
    loop. Every audit event is recorded with the identity resolved for that request
    (``actor=``), which is what makes service mode with bearer tokens work.
    """
    bind_address(settings)
    ctx = ReviewContext(
        settings=settings,
        store=store,
        identity=identity,
        audit=audit,
        llm_client=llm_client,
        now=clock or (lambda: datetime.now(UTC)),
    )
    app = FastAPI(title="Praktika review", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestGuard, settings=settings, token=session_token)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(IdentityError)
    async def _unauthorised(request: Request, exc: IdentityError) -> JSONResponse:
        ctx.deny(request, None, None, reason="unauthenticated", detail=str(exc), status=401)
        return JSONResponse({"detail": str(exc)}, status_code=401)

    @app.exception_handler(AccessDeniedError)
    async def _forbidden(request: Request, exc: AccessDeniedError) -> JSONResponse:
        ctx.deny(request, exc.user, exc.meeting_id, reason=exc.reason, detail=str(exc))
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(PraktikaError)
    async def _domain_error(_: Request, exc: PraktikaError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Liveness for the container HEALTHCHECK: no identity, no data."""
        return {"status": "ok", "mode": settings.mode}

    @app.get("/api/me")
    def me(request: Request) -> dict[str, Any]:
        return ctx.who(request).model_dump(mode="json")

    @app.get("/api/meetings")
    def list_meetings(request: Request) -> list[dict[str, Any]]:
        """Meetings the caller may read (organiser's own; secretary: non-private; dpo/admin:
        all)."""
        user = ctx.who(request)
        rows = []
        for m in store.list_meetings():
            if not visible(user, m):
                continue
            latest = store.latest_minutes(m.id)
            rows.append(
                {
                    **m.model_dump(
                        mode="json",
                        include={
                            "id",
                            "title",
                            "meeting_type",
                            "classification",
                            "state",
                            "started_at",
                            "organiser",
                            "private",
                        },
                    ),
                    "version": latest.version if latest else None,
                    "review_status": latest.review.status if latest else None,
                    "blocking": len(latest.blocking_flags()) if latest else 0,
                }
            )
        return rows

    @app.get("/api/meetings/{meeting_id}")
    def meeting_detail(meeting_id: str, request: Request) -> dict[str, Any]:
        """The review payload; a reviewer opening a draft-ready meeting moves it to
        ``in_review`` (a read-only role does not)."""
        user = ctx.who(request)
        meeting = ctx.meeting_or_404(meeting_id)
        ctx.authorise(user, meeting, write=False)
        transcript = store.get_transcript(meeting_id)
        minutes = store.latest_minutes(meeting_id)
        writer = ctx.may_write(user, meeting)
        if meeting.state == MeetingState.draft_ready and minutes is not None and writer:
            ctx.move(meeting, MeetingState.in_review)
            audit.append(
                "review.opened",
                meeting_id,
                actor=user,
                classification=meeting.classification.value,
                version=minutes.version,
            )
        elif not writer:  # a DPO/admin reading minutes leaves a trace
            audit.append(
                "review.opened",
                meeting_id,
                actor=user,
                classification=meeting.classification.value,
                version=minutes.version if minutes else None,
                read_only=True,
            )
        segs = transcript.segments if transcript else []
        segments = [s.model_dump(mode="json", exclude={"words"}) for s in segs]
        labels = sorted({s.speaker for s in segs if s.speaker_kind in ("label", "unknown")})
        media = [
            r for r in store.list_media(meeting_id) if r.deleted_at is None and r.path.exists()
        ]
        return {
            "meeting": meeting.model_dump(mode="json"),
            "segments": segments,
            "minutes": minutes.model_dump(mode="json") if minutes else None,
            "flags": sorted_flags(minutes) if minutes else [],
            "unmapped_speakers": labels,
            "reason_codes": list(REASON_CODES),
            "sections": list(pipeline.SECTIONS),
            "regenerate_available": llm_client is not None,
            "audio_available": bool(media) and meeting.classification.value != "restricted",
            "viewer": user.model_dump(mode="json"),
        }

    register_review_routes(app, ctx)
    register_media_routes(app, ctx)
    return app
