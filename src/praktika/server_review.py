"""Review-action routes: speaker mapping, item verdicts, flag clearing, regenerate, approve,
discard (controls C-07, C-08).

Every route resolves the request identity, checks it may change the meeting (``server_auth``,
403 otherwise), refuses closed meetings (409), refuses every write while a run holds the
meeting's lock (409 "a <command> run is working on this meeting"; a discard is the exception,
and the run then stops), refuses reviewer text that carries a raw identifier (422; C-06: the
text is stored and later fed to the model), claims the state move as a compare-and-set before
writing anything (409 when the meeting has moved on since the request read it), records the
change on the latest minutes version and appends one audit event with ``actor=`` set to that
identity. Approval is refused (403) while ``Minutes.blocking_flags()`` is non-empty.
Regeneration takes the meeting's run lock for the model call, goes through ``AuditedClient``
so every model call is recorded as ``llm.call`` against the reviewer, and keeps the new
version only if the meeting has not moved on meanwhile. Discarding or regenerating clears the
search-index row: only approved minutes are searchable. ``GET /api/meetings/{id}/run`` tells
the page whether a run is working on the meeting, so it can pause its edits.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Request

from praktika.diarize.assign import apply_names
from praktika.drafts import RUNNING_STATES, stale_draft_reason
from praktika.identity import Identity
from praktika.llm import pipeline
from praktika.llm import prompts as pr
from praktika.llm.base import AuditedClient
from praktika.models import MeetingState, Minutes, Review, ReviewItem
from praktika.retention import TRANSCRIPT_MAX_DAYS
from praktika.server_support import (
    ITEM_LISTS,
    ITEM_MODELS,
    OPEN_STATES,
    ReasonRequest,
    RegenerateRequest,
    ReviewContext,
    ReviewItemRequest,
    item_kind,
    locate_item,
    refuse_identifiers,
    rename_ref_speakers,
    sorted_flags,
)
from praktika.store import search


def restore_item(
    ctx: ReviewContext, minutes: Minutes, item_id: str
) -> tuple[None, str, dict[str, Any]]:
    """Re-insert the removed item whose JSON id is ``item_id`` and clear its flag."""
    for n, flag in enumerate(minutes.flags):
        if flag.kind != "uncited_item_removed" or flag.cleared or not flag.item_json:
            continue
        data = json.loads(flag.item_json)
        if data.get("id") != item_id:
            continue
        name = item_kind(data)
        restored = ITEM_MODELS[name].model_validate(data)
        cleared = flag.model_copy(update={"cleared_by": "restore", "cleared_at": ctx.now()})
        flags = [*minutes.flags[:n], cleared, *minutes.flags[n + 1 :]]
        return (
            None,
            getattr(restored, ITEM_LISTS[name]),
            {name: [*getattr(minutes, name), restored], "flags": flags},
        )
    raise HTTPException(404, f"no removed item {item_id!r} to restore")


class _ActorAudit:
    """``AuditLike`` that records every event on behalf of one request identity."""

    def __init__(self, ctx: ReviewContext, actor: Identity) -> None:
        self._ctx, self._actor = ctx, actor

    def append(self, event: str, meeting_id: str | None = None, **detail: Any) -> Any:
        return self._ctx.audit.append(event, meeting_id, actor=self._actor, **detail)


def register_review_routes(app: FastAPI, ctx: ReviewContext) -> None:
    """Attach the review-action routes to ``app``."""
    store, audit, settings = ctx.store, ctx.audit, ctx.settings

    def open_for_write(
        request: Request, meeting_id: str, *, during_runs: bool = False
    ) -> tuple[Identity, Any]:
        user = ctx.who(request)
        meeting = ctx.meeting_or_404(meeting_id)
        ctx.authorise(user, meeting, write=True)
        ctx.require_open(meeting)
        if not during_runs:
            ctx.refuse_running(meeting_id)
        return user, meeting

    @app.get("/api/meetings/{meeting_id}/run")
    def run_status(meeting_id: str, request: Request) -> dict[str, Any]:
        """Whether a run is working on the meeting; the page pauses its edits meanwhile."""
        user = ctx.who(request)
        meeting = ctx.meeting_or_404(meeting_id)
        ctx.authorise(user, meeting, write=False)
        lock = store.live_run_lock(meeting_id)
        return {
            "running": lock is not None,
            "command": lock.command if lock else None,
            "since": lock.started_at.isoformat() if lock else None,
            "message": lock.busy if lock else None,
            "state": meeting.state.value,
        }

    @app.post("/api/meetings/{meeting_id}/speakers")
    def map_speakers(
        meeting_id: str, request: Request, mapping: Annotated[dict[str, str], Body()]
    ) -> dict[str, Any]:
        user, meeting = open_for_write(request, meeting_id)
        clean = {k: v.strip() for k, v in mapping.items() if v.strip()}
        for label, name in clean.items():
            # A speaker name is reviewer-typed text that is stored in the transcript and rendered
            # into every later model call, so it is held to the same rule as an edited item (C-06).
            refuse_identifiers(name, meeting.roster, f"speaker name for {label}")
        transcript = ctx.transcript_or_404(meeting_id)
        minutes = store.latest_minutes(meeting_id)
        # Claimed before anything is written: 409 while a run holds the meeting or once it
        # has moved on since this request read it.
        ctx.claim(meeting, MeetingState.in_review if minutes is not None else meeting.state)
        renamed = transcript.model_copy(
            update={"segments": apply_names(transcript.segments, clean)}
        )
        days = TRANSCRIPT_MAX_DAYS[meeting.classification.value]
        deadline = store.transcript_delete_after(meeting_id) or ctx.now() + timedelta(days=days)
        store.save_transcript(renamed, delete_after=deadline)
        if minutes is not None:
            minutes = ctx.commit(meeting, rename_ref_speakers(minutes, clean), claimed=True)
        audit.append(
            "review.speaker_mapped",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            labels=sorted(clean),
        )
        return {"mapped": len(clean), "state": meeting.state.value}

    @app.post("/api/minutes/{meeting_id}/items/{item_id}")
    def review_item(
        meeting_id: str, item_id: str, body: ReviewItemRequest, request: Request
    ) -> dict[str, Any]:
        user, meeting = open_for_write(request, meeting_id)
        minutes = ctx.minutes_or_404(meeting_id)
        update: dict[str, Any] = {}
        before, after = body.before, body.after
        if body.action == "restore":
            before, after, update = restore_item(ctx, minutes, item_id)
        else:
            found = locate_item(minutes, item_id)
            if found is None:
                raise HTTPException(404, f"no item {item_id!r} in the minutes body")
            name, i = found
            items = list(getattr(minutes, name))
            before = getattr(items[i], ITEM_LISTS[name])
            if body.action == "reject":
                del items[i]
            elif body.action == "modify":
                if not after:
                    raise HTTPException(422, "modify needs 'after' text")
                refuse_identifiers(after, meeting.roster, "'after' text")
                items[i] = items[i].model_copy(update={ITEM_LISTS[name]: after})
            update[name] = items
        item = ReviewItem(
            item_id=item_id,
            action=body.action,
            reason_code=body.reason_code,
            before=before,
            after=after,
            by=user.user,
            at=ctx.now(),
        )
        review = minutes.review.model_copy(update={"items": [*minutes.review.items, item]})
        minutes = ctx.commit(meeting, minutes.model_copy(update={**update, "review": review}))
        store.save_review_item(meeting_id, minutes.version, item)
        audit.append(
            "review.item",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            object=item_id,
            action=body.action,
            reason=body.reason_code,
            version=minutes.version,
        )
        return {"minutes": minutes.model_dump(mode="json"), "flags": sorted_flags(minutes)}

    @app.post("/api/minutes/{meeting_id}/flags/{n}/clear")
    def clear_flag(meeting_id: str, n: int, request: Request) -> dict[str, Any]:
        user, meeting = open_for_write(request, meeting_id)
        minutes = ctx.minutes_or_404(meeting_id)
        if not 0 <= n < len(minutes.flags):
            raise HTTPException(404, "no such flag")
        flags = list(minutes.flags)
        flags[n] = flags[n].model_copy(update={"cleared_by": user.user, "cleared_at": ctx.now()})
        minutes = ctx.commit(meeting, minutes.model_copy(update={"flags": flags}))
        audit.append(
            "review.item",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            object=f"flag:{n}",
            action="clear_flag",
            kind=flags[n].kind,
        )
        return {"flags": sorted_flags(minutes), "blocking": len(minutes.blocking_flags())}

    def refuse_running_state(meeting: Any) -> None:
        if meeting.state in RUNNING_STATES:
            raise HTTPException(
                409,
                f"meeting is {meeting.state.value}; draft it with `praktika generate "
                f"{meeting.id}` before regenerating a section",
            )

    @app.post("/api/minutes/{meeting_id}/regenerate")
    def regenerate(meeting_id: str, body: RegenerateRequest, request: Request) -> dict[str, Any]:
        user, meeting = open_for_write(request, meeting_id)
        if ctx.llm_client is None:
            raise HTTPException(503, "no LLM client configured for regeneration")
        refuse_running_state(meeting)
        ctx.minutes_or_404(meeting_id)  # 404s come before the lock is taken
        ctx.transcript_or_404(meeting_id)
        refuse_identifiers(body.instruction, meeting.roster, "instruction")
        with ctx.running(meeting, "regenerate", user) as lock:
            meeting = ctx.meeting_or_404(meeting_id)  # as it is now that no run can move it
            ctx.require_open(meeting)
            refuse_running_state(meeting)
            minutes = ctx.minutes_or_404(meeting_id)
            transcript = ctx.transcript_or_404(meeting_id)
            prompts = pr.load(settings.prompts_dir, settings.prompt_version, minutes.meeting_type)
            client = AuditedClient(
                ctx.llm_client,
                _ActorAudit(ctx, user),
                meeting_id,
                prompts.sha256,
                classification=meeting.classification.value,
            )
            fresh = pipeline.regenerate_section(
                minutes, transcript, body.section, body.instruction, client, prompts
            )
            version = store.save_minutes(fresh)
            try:  # kept only if nothing moved the meeting during the model call (an abort)
                ctx.claim(meeting, MeetingState.draft_ready, own_lock=lock.token)
            except HTTPException:
                store.delete_minutes_version(meeting_id, version)
                raise
            fresh = fresh.model_copy(update={"version": version})
            store.clear_index(meeting_id)  # a new draft supersedes any indexed version
            audit.append(
                "minutes.drafted",
                meeting_id,
                actor=user,
                classification=meeting.classification.value,
                object=body.section,
                version=version,
                instruction_chars=len(body.instruction),
            )
        return {"minutes": fresh.model_dump(mode="json"), "flags": sorted_flags(fresh)}

    @app.post("/api/minutes/{meeting_id}/approve")
    def approve(meeting_id: str, body: ReasonRequest, request: Request) -> dict[str, Any]:
        user, meeting = open_for_write(request, meeting_id)
        minutes = ctx.minutes_or_404(meeting_id)
        stale = stale_draft_reason(store, meeting, minutes)
        if stale is not None:
            raise HTTPException(409, stale)
        blocking = minutes.blocking_flags()
        if blocking:
            raise HTTPException(403, f"{len(blocking)} priority-1 flag(s) must be resolved first")
        review = Review(
            status="approved",
            reviewer=user.user,
            reviewer_source=user.audit_source(),
            reviewed_at=ctx.now(),
            items=minutes.review.items,
        )
        minutes = minutes.model_copy(update={"review": review})
        before = meeting.state
        ctx.claim(meeting, MeetingState.approved)  # 409, nothing written, if it moved on
        try:
            store.set_review_status(minutes)
        except BaseException:  # never an approved meeting whose minutes are not marked approved
            store.transition(meeting_id, before, expected=(MeetingState.approved,))
            raise
        search.index_minutes(store, minutes, pr.TEMPLATES[minutes.meeting_type])
        audit.append(
            "review.approved",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            version=minutes.version,
            reason=body.reason_code,
            items=len(review.items),
        )
        return {"state": meeting.state.value, "review": review.model_dump(mode="json")}

    @app.post("/api/minutes/{meeting_id}/discard")
    def discard(meeting_id: str, body: ReasonRequest, request: Request) -> dict[str, Any]:
        # Like `praktika abort`, a discard is allowed while a run works on the meeting: the
        # run stops and removes what it stored. Not while a `generate --reopen` run re-drafts
        # approved minutes, though: the approved record is never discarded from here.
        user, meeting = open_for_write(request, meeting_id, during_runs=True)
        minutes = ctx.minutes_or_404(meeting_id)
        lock = store.live_run_lock(meeting_id)
        if minutes.review.status == "approved" or (
            lock is not None and lock.reopened_from is MeetingState.approved
        ):
            raise HTTPException(
                409, "these minutes were approved and are being re-drafted; they stay approved"
            )
        ctx.claim(
            meeting,
            MeetingState.discarded,
            expected=OPEN_STATES,
            during_runs=True,
            stopped_by="discard",
        )
        review = minutes.review.model_copy(
            update={
                "status": "discarded",
                "reviewer": user.user,
                "reviewer_source": user.audit_source(),
                "reviewed_at": ctx.now(),
            }
        )
        store.set_review_status(minutes.model_copy(update={"review": review}))
        store.clear_index(meeting_id)  # discarded content is never searchable
        audit.append(
            "review.discarded",
            meeting_id,
            actor=user,
            classification=meeting.classification.value,
            version=minutes.version,
            reason=body.reason_code,
        )
        return {"state": meeting.state.value}
