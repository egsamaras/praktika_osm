"""Review-side commands: ``approve``, ``export``, ``serve``, ``search``, ``actions``.

``export`` applies ``policy.export_allowed`` (drafts refused unless ``--allow-draft`` outside
the pilot, then watermarked and audited) and ``policy.detokenise_allowed`` before touching the
vault (C-06, C-07); it writes under ``data_dir/exports`` by default and refuses (without
``--force``) any target inside iCloud Drive, so minutes never land in a cloud-synced folder by
default (C-01). ``approve`` applies the same blocking-flag and closed-state rules as the review
UI. ``serve`` binds loopback only in local mode.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from praktika import policy
from praktika.cli import context as ctx
from praktika.cli import steps
from praktika.cli.export_paths import resolve_export_target
from praktika.errors import NotApproved, PraktikaError
from praktika.llm.prompts import TEMPLATES
from praktika.logging import get_logger
from praktika.models import MeetingState, Minutes
from praktika.redact.tokenise import TokenVault, decrypt_vault
from praktika.render.docx import render_docx
from praktika.render.markdown import render_markdown
from praktika.server_support import CLOSED_STATES
from praktika.store import search as search_mod

log = get_logger(__name__)


def _latest(rt: ctx.Runtime, meeting_id: str) -> Minutes:
    minutes = rt.store.latest_minutes(meeting_id)
    if minutes is None:
        raise PraktikaError(f"no minutes drafted for {meeting_id}")
    return minutes


@ctx.guarded
def approve(
    meeting_id: Annotated[str, typer.Argument(help="Meeting id.")],
    reason: Annotated[str, typer.Option("--reason", help="Approval reason code.")] = "reviewed",
) -> None:
    """Approve the latest minutes version (refused while a priority-1 flag is open, and once
    the meeting is approved, discarded or purged)."""
    rt = ctx.open_runtime()
    meeting = rt.require_meeting(meeting_id)
    if meeting.state in CLOSED_STATES:
        raise PraktikaError(f"{meeting_id} is {meeting.state.value}; its minutes are closed")
    minutes = _latest(rt, meeting_id)
    if minutes.review.status == "discarded":
        raise PraktikaError(f"{meeting_id} v{minutes.version} was discarded; nothing to approve")
    steps.refuse_stale_draft(rt, meeting, minutes)  # never a draft of an older transcript
    blocking = minutes.blocking_flags()
    if blocking:
        raise NotApproved(
            f"{len(blocking)} blocking flag(s) open on v{minutes.version}; "
            "resolve them in the review UI first"
        )
    who = rt.current_identity()
    review = minutes.review.model_copy(
        update={
            "status": "approved",
            "reviewer": who.user,
            "reviewer_source": who.audit_source(),
            "reviewed_at": datetime.now(UTC),
        }
    )
    minutes = minutes.model_copy(update={"review": review})
    # Compare-and-set, and only while no run holds the meeting: a run that started after the
    # checks above, or an abort, wins, and nothing is approved.
    if not rt.store.transition(
        meeting_id, MeetingState.approved, expected=(meeting.state,), unlocked=True
    ):
        raise PraktikaError(
            f"{meeting_id} changed while it was being approved (a run started, or it was "
            "aborted); nothing was approved"
        )
    try:
        rt.store.set_review_status(minutes)
    except BaseException:  # never an approved meeting whose minutes are not marked approved
        rt.store.transition(meeting_id, meeting.state, expected=(MeetingState.approved,))
        raise
    search_mod.index_minutes(rt.store, minutes, TEMPLATES[minutes.meeting_type])
    rt.audit.append(
        "review.approved",
        meeting_id,
        classification=minutes.classification.value,
        version=minutes.version,
        reason_code=reason,
    )
    ctx.console.print(f"Approved {meeting_id} v{minutes.version} ({reason}).")


def _vault(rt: ctx.Runtime, minutes: Minutes) -> TokenVault:
    decision = policy.detokenise_allowed(minutes)
    if not decision.allowed:
        raise NotApproved(f"de-tokenisation refused: {decision.reason}")
    blob = rt.store.get_vault(minutes.meeting_id)
    if blob is None:
        raise PraktikaError(f"no vault stored for {minutes.meeting_id}")
    return decrypt_vault(blob, ctx.vault_key(rt.settings))


@ctx.guarded
def export(
    meeting_id: Annotated[str, typer.Argument(help="Meeting id.")],
    fmt: Annotated[str, typer.Option("--format", help="md | docx")] = "md",
    out: Annotated[
        Path | None, typer.Option("--out", help="Output path (default: data_dir/exports/).")
    ] = None,
    detokenise: Annotated[
        bool,
        typer.Option(
            "--detokenise",
            help="Restore identifiers from the token vault (approved, non-restricted only).",
        ),
    ] = False,
    allow_draft: Annotated[
        bool, typer.Option("--allow-draft", help="Export an unapproved draft (not in pilot mode).")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Write into a cloud-synced folder anyway.")
    ] = False,
) -> None:
    """Write the latest minutes as Markdown or DOCX (approved minutes only under the pilot)."""
    if fmt not in ("md", "docx"):
        raise PraktikaError("--format must be md or docx")
    rt = ctx.open_runtime()
    meeting = rt.require_meeting(meeting_id)
    minutes = _latest(rt, meeting_id)
    decision = policy.export_allowed(minutes, rt.settings, allow_draft=allow_draft)
    if not decision.allowed:
        raise NotApproved(f"export refused: {decision.reason}")
    vault = _vault(rt, minutes) if detokenise else None
    transcript = rt.store.get_transcript(meeting_id)
    target = resolve_export_target(
        rt.settings, out, f"{meeting_id}-v{minutes.version}.{fmt}", force=force
    )
    if fmt == "docx":
        render_docx(minutes, meeting, transcript, target, vault=vault, settings=rt.settings)
    else:
        text = render_markdown(minutes, meeting, transcript, vault=vault, settings=rt.settings)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(target, 0o600)
    rt.audit.append(
        "export.written",
        meeting_id,
        classification=minutes.classification.value,
        object=str(target),
        format=fmt,
        version=minutes.version,
        watermarked="watermark_draft" in decision.obligations,
        detokenised=vault is not None,
    )
    ctx.console.print(f"Exported {meeting_id} v{minutes.version} to {target}")


@ctx.guarded
def serve(
    port: Annotated[int | None, typer.Option("--port", help="Review UI port.")] = None,
) -> None:
    """Run the review UI; local mode binds 127.0.0.1 only and prints the URL with the data
    directory's persistent session token, which every API request must carry.

    The app is built by ``server.create_app`` over the CLI's store and audit log, the identity
    provider from ``context.server_identity`` and the configured LLM client (so the reviewer's
    "regenerate section" works), then served by ``server.serve``. The retention timers run
    once before the server starts.
    """
    from praktika import server
    from praktika.server_auth import TOKEN_FILE

    settings = ctx.load_settings()
    if port is not None:
        settings = settings.model_copy(update={"review_port": port})
    server.bind_address(settings)
    rt = ctx.open_runtime(settings)
    token = server.session_token_for(settings.data_dir) if settings.mode == "local" else None
    app = server.create_app(
        settings,
        rt.store,
        ctx.server_identity(settings),
        rt.audit,
        llm_client=ctx.llm_client(settings),
        session_token=token,
    )
    ctx.console.print(f"Review UI on {server.review_url(settings, token)}")
    if token:
        ctx.console.print(
            "Open exactly that link: the ?t= token is this data directory's persistent session "
            "key, the same in every review link and after a restart (to issue a new one, stop "
            f"serve and delete {Path(settings.data_dir) / TOKEN_FILE})."
        )
    server.serve(app, settings)


@ctx.guarded
def search(
    query: Annotated[str, typer.Argument(help="Free-text query (EN or AR).")],
    include_private: Annotated[bool, typer.Option("--include-private")] = False,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Full-text search over indexed minutes."""
    rt = ctx.open_runtime()
    hits = search_mod.search(rt.store, query, include_private=include_private, limit=limit)
    if not hits:
        ctx.console.print("No matches.")
        return
    table = Table("Meeting", "Version", "Title", "Snippet")
    for h in hits:
        table.add_row(h.meeting_id, str(h.version), h.title, h.snippet)
    ctx.console.print(table)


@ctx.guarded
def actions(
    owner: Annotated[str | None, typer.Option("--owner", help="Filter by owner.")] = None,
    overdue: Annotated[bool, typer.Option("--overdue", help="Only past-due actions.")] = False,
) -> None:
    """Cross-meeting register of open actions from the latest minutes of every meeting."""
    rt = ctx.open_runtime()
    today = date.today()  # noqa: DTZ011 - local calendar day is what an overdue list means
    rows: list[Any] = []
    for item in rt.store.open_actions(owner):
        due = item.action.due_date
        if overdue and (due is None or due >= today):
            continue
        rows.append(item)
    if not rows:
        ctx.console.print("No open actions.")
        return
    table = Table("Meeting", "Id", "Owner", "Due", "Description")
    for item in rows:
        a = item.action
        table.add_row(
            item.meeting_id,
            a.id,
            a.owner or "?",
            due_display(a.due_date, a.due_text),
            a.description,
        )
    ctx.console.print(table)


def due_display(due_date: date | None, due_text: str | None) -> str:
    """A due date as the reviewer should see it: the resolved date with the phrase actually
    spoken beside it (``2026-10-02 ("by November the 2nd")``), so a wrongly resolved date is
    caught at review; the phrase alone when nothing was resolved. Exports keep their format."""
    spoken = (due_text or "").strip()
    if due_date is None:
        return spoken
    resolved = due_date.isoformat()
    return f'{resolved} ("{spoken}")' if spoken and spoken != resolved else resolved
