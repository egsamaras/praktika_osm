"""Audio-slice and export routes of the review server (controls C-05, C-06, C-07).

The audio route serves a slice of retained WAV only: 404 for restricted meetings (audio is
never kept) and for deleted or missing media. Export follows ``policy.export_allowed`` (draft
export is 403 under the pilot), never de-tokenises (that needs the Keychain key and is a CLI
route), writes DOCX files 0600 under ``data_dir/exports`` and audits ``export.written``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from praktika import policy
from praktika.render.docx import render_docx
from praktika.render.markdown import render_markdown
from praktika.server_support import MAX_SLICE_S, ReviewContext, wav_slice


def register_media_routes(app: FastAPI, ctx: ReviewContext) -> None:
    """Attach the audio and export routes to ``app``."""
    store, audit, settings = ctx.store, ctx.audit, ctx.settings

    @app.get("/api/meetings/{meeting_id}/audio")
    def audio(
        meeting_id: str,
        request: Request,
        start: float = Query(ge=0),
        end: float = Query(gt=0),
        track: str | None = Query(default=None, max_length=16),
    ) -> Response:
        """A slice of the retained WAV; ``track`` (``mic``/``system``/``file``) picks the
        media row of the cited segment's track, else the last retained row is served."""
        user = ctx.who(request)
        meeting = ctx.meeting_or_404(meeting_id)
        ctx.authorise(user, meeting, write=False)
        if meeting.classification.value == "restricted":
            raise HTTPException(404, "audio is never retained for restricted meetings")
        if end <= start or end - start > MAX_SLICE_S:
            raise HTTPException(422, f"slice must be 0 < end - start <= {MAX_SLICE_S:g} s")
        wavs = [
            r
            for r in store.list_media(meeting_id)
            if r.deleted_at is None and r.path.suffix.lower() == ".wav" and r.path.exists()
        ]
        if not wavs:
            raise HTTPException(404, "audio has been deleted or was never retained")
        chosen = next((r for r in wavs if track and r.kind == track), wavs[-1])
        payload = wav_slice(chosen.path, start, end)
        audit.append(
            "audio.read",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            object=chosen.path.name,
            start=start,
            end=end,
            read_only=not ctx.may_write(user, meeting),
        )
        return Response(
            payload,
            status_code=206,
            media_type="audio/wav",
            headers={"Content-Range": f"seconds {start:g}-{end:g}", "Cache-Control": "no-store"},
        )

    def export(meeting_id: str, fmt: str, request: Request, allow_draft: bool) -> Response:
        user = ctx.who(request)
        meeting = ctx.meeting_or_404(meeting_id)
        ctx.authorise(user, meeting, write=False)
        minutes = ctx.minutes_or_404(meeting_id)
        decision = policy.export_allowed(minutes, settings, allow_draft=allow_draft)
        if not decision.allowed:
            raise HTTPException(403, decision.reason)
        transcript = store.get_transcript(meeting_id)
        out_dir = settings.data_dir / "exports"
        out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fmt == "md":
            text = render_markdown(minutes, meeting, transcript, settings=settings, now=ctx.now())
            resp: Response = Response(text, media_type="text/markdown; charset=utf-8")
            target = f"{meeting_id}.v{minutes.version}.md"
        else:
            path = render_docx(
                minutes,
                meeting,
                transcript,
                out_dir / f"{meeting_id}.v{minutes.version}.docx",
                settings=settings,
                now=ctx.now(),
            )
            resp, target = FileResponse(path, filename=path.name), path.name
        resp.headers["Content-Disposition"] = f'attachment; filename="{target}"'
        audit.append(
            "export.written",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            object=target,
            format=fmt,
            version=minutes.version,
            draft=minutes.review.status != "approved",
            obligations=decision.obligations,
        )
        return resp

    @app.get("/api/export/{meeting_id}.md")
    def export_md(meeting_id: str, request: Request, allow_draft: bool = False) -> Response:
        return export(meeting_id, "md", request, allow_draft)

    @app.get("/api/export/{meeting_id}.docx")
    def export_docx(meeting_id: str, request: Request, allow_draft: bool = False) -> Response:
        return export(meeting_id, "docx", request, allow_draft)
