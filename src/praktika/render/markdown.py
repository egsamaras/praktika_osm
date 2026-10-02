"""Markdown export.

Contract: ``render_markdown`` renders one Jinja2 template per meeting type from
``templates/render/``. Drafts carry the ``DRAFT — NOT APPROVED`` banner; every export carries
the classification in header and footer and a provenance footer; citations re-attach transcript
times; Arabic quotes are verbatim followed by an English gloss line; items the verifier removed
are listed under "Reviewer flags" with their full text. Identifier tokens (``«IBAN_1»``) are
replaced from the vault only when ``praktika.policy.detokenise_allowed`` says so; when the policy
module is absent the answer is "not allowed" (fail closed, C-06).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from praktika.config import Settings
from praktika.logging import get_logger
from praktika.models import (
    ActionItem,
    Flag,
    MancomMinutes,
    Meeting,
    Minutes,
    OneToOneMinutes,
    Ref,
    Transcript,
)

ARABIC_RE = re.compile(r"[؀-ۿ]")
BANNER_DRAFT = "DRAFT — NOT APPROVED"
REMOVED_KIND = "uncited_item_removed"
PACKAGE_TEMPLATES = Path(__file__).resolve().parents[3] / "templates" / "render"
_BLANK_RUN = re.compile(r"\n{3,}")

log = get_logger(__name__)


def templates_dir(settings: Settings) -> Path:
    """Locate ``templates/render``: the repository checkout first, then beside ``prompts_dir``."""
    for candidate in (
        PACKAGE_TEMPLATES,
        Path(settings.prompts_dir).resolve().parent / "templates" / "render",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("templates/render directory not found")


def template_name(minutes: Minutes) -> str:
    """The template family: decided by the minutes class first, then by ``meeting_type``."""
    if isinstance(minutes, MancomMinutes):
        return "mancom"
    if isinstance(minutes, OneToOneMinutes):
        return "one_to_one"
    return minutes.meeting_type.value


def hms(seconds: float) -> str:
    """``HH:MM:SS`` for a non-negative number of seconds (fractions truncated)."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detokeniser(minutes: Minutes, vault: Any | None, settings: Settings) -> Callable[[str], str]:
    """Return a text transform that restores identifiers from ``vault.entries`` (token ->
    plaintext) when policy allows, otherwise the identity function.

    ``praktika.policy`` is imported lazily via ``importlib`` so tests can substitute a stub; if it
    is missing, or ``detokenise_allowed`` is missing or false, tokens are left in place.
    """
    entries: dict[str, str] = dict(getattr(vault, "entries", {}) or {})
    if not entries or not _detokenise_allowed(minutes, settings):
        return lambda text: text
    ordered = sorted(entries.items(), key=lambda kv: len(kv[0]), reverse=True)

    def apply(text: str) -> str:
        for token, plain in ordered:
            text = text.replace(token, plain)
        return text

    return apply


def _detokenise_allowed(minutes: Minutes, settings: Settings) -> bool:
    try:
        policy = importlib.import_module("praktika.policy")
    except ImportError:
        log.warning("render.detokenise_denied", reason="policy module unavailable")
        return False
    fn = getattr(policy, "detokenise_allowed", None)
    if fn is None:
        return False
    try:
        verdict = fn(minutes, settings)
    except TypeError:
        verdict = fn(minutes)
    return bool(getattr(verdict, "allowed", verdict))


def ref_view(ref: Ref, gloss: str, by_id: dict[str, Any]) -> dict[str, Any]:
    """A citation for templates: id, re-attached time span, speaker, verbatim quote and, for an
    Arabic quote, the English gloss (the item's own wording)."""
    seg = by_id.get(ref.segment_id)
    start, end = (seg.start, seg.end) if seg is not None else (ref.start_s, ref.end_s)
    arabic = bool(ARABIC_RE.search(ref.quote))
    return {
        "id": ref.segment_id,
        "time": f"{hms(start)}–{hms(end)}",
        "speaker": ref.speaker,
        "quote": ref.quote,
        "arabic": arabic,
        "gloss": gloss if arabic else None,
    }


def _action_view(a: ActionItem, by_id: dict[str, Any]) -> dict[str, Any]:
    due = a.due_date.isoformat() if a.due_date else (a.due_text or "—")
    return {
        "id": a.id,
        "description": a.description,
        "owner": a.owner or "Unassigned",
        "owner_confidence": a.owner_confidence,
        "due": due,
        "source_language": a.source_language,
        "refs": [ref_view(r, a.description, by_id) for r in a.refs],
    }


def _flag_view(f: Flag, by_id: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": f.kind,
        "detail": f.detail,
        "priority": f.priority,
        "cleared": f.cleared,
        "refs": [ref_view(r, f.detail, by_id) for r in f.refs],
    }


def build_view(
    minutes: Minutes,
    meeting: Meeting,
    transcript: Transcript | None,
    *,
    settings: Settings,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The template context shared by the Markdown and DOCX renderers (tokens untouched)."""
    by_id = transcript.by_id() if transcript is not None else {}
    approved = minutes.review.status == "approved"
    open_flags = [f for f in minutes.flags if f.kind != REMOVED_KIND]
    removed = [f for f in minutes.flags if f.kind == REMOVED_KIND]
    view: dict[str, Any] = {
        "title": minutes.title,
        "banner": None if approved else BANNER_DRAFT,
        "approved": approved,
        "classification": minutes.classification.value.upper(),
        "meeting_id": minutes.meeting_id,
        "version": minutes.version,
        "date": minutes.date.isoformat(),
        "meeting_type": minutes.meeting_type.value,
        "platform": meeting.platform.value,
        "chair": meeting.chair or "—",
        "organiser": meeting.organiser,
        "language_profile": ", ".join(f"{k} {v:.0%}" for k, v in minutes.language_profile.items()),
        "attendees": [
            {
                "name": a.name,
                "role": a.role or "",
                "organisation": a.organisation,
                "status": a.status,
            }
            for a in minutes.attendees
        ],
        "summary": minutes.summary,
        "topics": [
            {
                "title": t.title,
                "summary": t.summary,
                "key_points": t.key_points,
                "refs": [ref_view(r, t.title, by_id) for r in t.refs],
            }
            for t in minutes.topics
        ],
        "decisions": [
            {
                "id": d.id,
                "statement": d.statement,
                "kind": d.kind,
                "decided_by": d.decided_by,
                "dissent": d.dissent_or_conditions,
                "refs": [ref_view(r, d.statement, by_id) for r in d.refs],
            }
            for d in minutes.decisions
        ],
        "actions": [_action_view(a, by_id) for a in minutes.actions],
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "raised_by": q.raised_by,
                "owner": q.owner,
                "refs": [ref_view(r, q.question, by_id) for r in q.refs],
            }
            for q in minutes.open_questions
        ],
        "risks": [
            {
                "id": r.id,
                "description": r.description,
                "severity": r.severity,
                "owner": r.owner,
                "mitigation": r.mitigation,
                "refs": [ref_view(x, r.description, by_id) for x in r.refs],
            }
            for r in minutes.risks
        ],
        "follow_ups": list(minutes.follow_ups),
        "flags": [_flag_view(f, by_id) for f in open_flags],
        "removed": [_flag_view(f, by_id) for f in removed],
        "review": {
            "status": minutes.review.status,
            "reviewer": minutes.review.reviewer or "—",
            "reviewer_source": minutes.review.reviewer_source or "—",
            "reviewed_at": minutes.review.reviewed_at.isoformat()
            if minutes.review.reviewed_at
            else "—",
        },
        "provenance": minutes.provenance.model_dump(mode="json"),
        "rendered_at": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        "pilot": settings.pilot,
    }
    if isinstance(minutes, MancomMinutes):
        view.update(_mancom_view(minutes, by_id))
    if isinstance(minutes, OneToOneMinutes):
        view.update(_one_to_one_view(minutes, by_id))
    return view


def _mancom_view(m: MancomMinutes, by_id: dict[str, Any]) -> dict[str, Any]:
    return {
        "agenda": [a.model_dump() for a in m.agenda],
        "matters_arising": [
            {
                "previous_action_id": x.previous_action_id,
                "status": x.status,
                "note": x.note,
                "refs": [ref_view(r, x.note, by_id) for r in x.refs],
            }
            for x in m.matters_arising
        ],
        "figures": [f.model_dump() for f in m.figures_mentioned],
        "escalations": list(m.escalations_to_board),
    }


def _one_to_one_view(m: OneToOneMinutes, by_id: dict[str, Any]) -> dict[str, Any]:
    listed = {a.id for a in m.my_commitments} | {a.id for a in m.their_commitments}
    return {
        "private": m.private,
        "my_commitments": [_action_view(a, by_id) for a in m.my_commitments],
        "their_commitments": [_action_view(a, by_id) for a in m.their_commitments],
        "other_actions": [_action_view(a, by_id) for a in m.actions if a.id not in listed],
        "next_one_to_one": m.next_one_to_one.isoformat() if m.next_one_to_one else "—",
    }


def render_markdown(
    minutes: Minutes,
    meeting: Meeting,
    transcript: Transcript | None,
    *,
    vault: Any | None = None,
    settings: Settings,
    now: datetime | None = None,
) -> str:
    """Render the Markdown export for ``minutes`` (see module docstring for the guarantees)."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir(settings))),
        autoescape=False,  # noqa: S701 - Markdown output, not HTML; autoescape would corrupt it
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    template = env.get_template(f"{template_name(minutes)}.md.j2")
    text = template.render(**build_view(minutes, meeting, transcript, settings=settings, now=now))
    text = _BLANK_RUN.sub("\n\n", text)
    return detokeniser(minutes, vault, settings)(text)
