"""Consent-gate answers for ``start`` and ``ingest`` (C-03).

Every answer can be given as a flag (non-interactive, scriptable) or, when the command runs in a
terminal and a flag is missing, through a rich prompt. When a flag is missing and there is no
terminal the command refuses with exit 2 (``GateIncompleteError``). Scope items are acknowledged one
key at a time (``--scope-ack KEY``) or all at once (``--ack-all-scope``); either way every key is
recorded explicitly in the consent record. Nothing here can skip ``consent.gate``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import typer
from rich.prompt import Confirm, Prompt

from praktika import consent, scope
from praktika.cli import context as ctx
from praktika.errors import PraktikaError
from praktika.identity import normalise_upn
from praktika.ids import new_meeting_id
from praktika.llm.prompts import TEMPLATES
from praktika.models import (
    Classification,
    ConsentRecord,
    LanguageMode,
    Meeting,
    MeetingType,
    Platform,
)
from praktika.store.search import normalise_ar
from praktika.stt.router import require_language

METHODS = ("spoken", "chat", "teams_transcription", "placard")

NotifiedOpt = Annotated[
    bool | None,
    typer.Option("--notified/--no-notified", help="Attendees were told the notetaker is on."),
]
ObjectionsOpt = Annotated[
    bool | None,
    typer.Option("--objections/--no-objections", help="Whether anyone objected."),
]
MethodOpt = Annotated[
    str | None,
    typer.Option("--method", help="How notice was given: spoken|chat|teams_transcription|placard."),
]
TeamsOpt = Annotated[
    bool | None,
    typer.Option(
        "--teams-transcription-started/--no-teams-transcription-started",
        help="Teams native transcription was started as well.",
    ),
]
PurposeOpt = Annotated[
    str | None, typer.Option("--purpose", help="Purpose of the recording (10-500 characters).")
]
ScopeAckOpt = Annotated[
    list[str] | None,
    typer.Option("--scope-ack", help="Confirm one scope checklist key (repeatable)."),
]
AckAllOpt = Annotated[
    bool, typer.Option("--ack-all-scope", help="Confirm every scope checklist key explicitly.")
]
TagOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--tag",
        help="Meeting tag (repeatable); a pilot exclusion tag (board, hr, customer_call, "
        "regulator, legal_privileged, foreign_hosted) is refused in code.",
    ),
]
ForeignHostedOpt = Annotated[
    bool,
    typer.Option(
        "--foreign-hosted",
        help="The meeting is hosted by an entity in another jurisdiction (excluded).",
    ),
]
OrganiserOpt = Annotated[
    str | None,
    typer.Option(
        "--organiser",
        help="UPN or e-mail of the organiser when an operator submits the meeting on their "
        "behalf (default: the acting user). The audit log records both.",
    ),
]
ExternalOpt = Annotated[
    bool,
    typer.Option(
        "--external-participants",
        help="Non-organisation participants take part (tagged 'external').",
    ),
]

#: Title words that suggest a pilot exclusion; the organiser is warned and must confirm. Arabic
#: words are written as ``normalise_ar`` leaves them (bare alef, haa for taa marbuta).
_TITLE_HINTS: dict[str, tuple[str, ...]] = {
    "board": ("board", "مجلس الادار", "مجلس ادار"),
    "hr": ("grievance", "disciplinary", "appraisal", "hr", "تظلم", "تاديب", "الموارد البشريه"),
    "customer_call": ("customer", "client call", "عميل", "عملاء"),
    "regulator": (
        "central bank", "supervisory", "regulator", "auditor", "المركزي", "مدقق",
    ),
    "legal_privileged": ("privileged", "litigation", "امتياز"),
}  # fmt: skip
_LATIN = re.compile(r"[a-z]")
#: Word breaks inside camelCase and acronym-led titles ("BoardMeeting", "HRReview"), but not
#: before an acronym's plural ("KPIs").
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]{2})")
#: Hints that begin common words when run on ("auditorium", "hrs"), so they must also end at a
#: word break.
_ENDS_AT_A_BREAK = frozenset({"auditor", "hr"})


def _hint_pattern(word: str) -> re.Pattern[str]:
    """A Latin-script hint must start a word: no Latin letter before it, so 'board' never matches
    'dashboard', 'keyboard' or 'onboarding'. Run-on file stems and digits still match
    ('boardmeeting', 'Q4Board', 'Board_Meeting', 'الـBoard'). The few hints in
    ``_ENDS_AT_A_BREAK`` must also end at one (plural allowed, except for 'hr', so '2 hrs' is not
    HR), and 'hr' may not follow a digit ('3hr'). An Arabic hint matches anywhere, because Arabic
    attaches prefixes such as و and ب to the word itself."""
    if not _LATIN.search(word):
        return re.compile(re.escape(word))
    body = re.escape(word).replace(r"\ ", r"[\s_-]+")
    before = r"(?<![a-z0-9])" if word == "hr" else r"(?<![a-z])"
    if word not in _ENDS_AT_A_BREAK:
        return re.compile(before + body)
    plural = "s?" if len(word) > 3 else ""
    return re.compile(rf"{before}{body}{plural}(?![a-z])")


_HINT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    key: tuple(_hint_pattern(normalise_ar(w)) for w in words) for key, words in _TITLE_HINTS.items()
}


@dataclass
class GateAnswers:
    notified: bool
    objections: bool
    method: str
    teams_transcription_started: bool
    purpose: str
    scope_checks: dict[str, bool]


def collect_answers(
    *,
    notified: bool | None,
    objections: bool | None,
    method: str | None,
    teams_transcription_started: bool | None,
    purpose: str | None,
    scope_ack: list[str] | None,
    ack_all_scope: bool,
    interactive: bool | None = None,
) -> GateAnswers:
    """Resolve every gate answer from flags, prompting for the missing ones in a terminal.

    Raises ``GateIncompleteError`` listing the missing flags when a prompt is not possible, and
    ``PraktikaError`` for an unknown scope key or method. Declined scope items are kept as
    ``False`` so the gate refuses them explicitly.
    """
    interactive = ctx.is_interactive() if interactive is None else interactive
    missing: list[str] = []

    def ask_bool(value: bool | None, flag: str, prompt: str) -> bool:
        if value is not None:
            return value
        if interactive:
            return Confirm.ask(prompt, console=ctx.console)
        missing.append(flag)
        return False

    def ask_text(value: str | None, flag: str, prompt: str, choices: list[str] | None) -> str:
        if value is not None:
            return value
        if interactive:
            return Prompt.ask(prompt, console=ctx.console, choices=choices)
        missing.append(flag)
        return ""

    notified_v = ask_bool(notified, "--notified/--no-notified", "Were all attendees notified?")
    objections_v = ask_bool(objections, "--objections/--no-objections", "Did anyone object?")
    method_v = ask_text(method, "--method", "How was notice given?", list(METHODS))
    teams_v = ask_bool(
        teams_transcription_started,
        "--teams-transcription-started/--no-teams-transcription-started",
        "Was Teams native transcription started as well?",
    )
    purpose_v = ask_text(purpose, "--purpose", "Purpose of the recording", None)
    keys = scope.scope_checklist()
    acked = set(scope_ack or [])
    unknown = sorted(acked - {k for k, _, _ in keys})
    if unknown:
        raise PraktikaError(f"unknown scope checklist key(s): {', '.join(unknown)}")
    checks: dict[str, bool] = {}
    for key, en, ar in keys:
        if ack_all_scope or key in acked:
            checks[key] = True
        elif interactive:
            checks[key] = Confirm.ask(f"{en}\n{ar}", console=ctx.console)
        else:
            missing.append(f"--scope-ack {key}")
    if missing:
        raise ctx.GateIncompleteError(
            "consent gate answers missing and no terminal to ask: " + ", ".join(missing)
        )
    if method_v not in METHODS:
        raise PraktikaError(f"--method must be one of {', '.join(METHODS)}")
    return GateAnswers(notified_v, objections_v, method_v, teams_v, purpose_v.strip(), checks)


def meeting_tags(
    tags: list[str] | None, *, foreign_hosted: bool = False, external: bool = False
) -> set[str]:
    """Normalised tag set from ``--tag`` plus the ``--foreign-hosted`` /
    ``--external-participants`` flags; ``scope.check`` refuses any pilot exclusion among them."""
    out = {t.strip().lower() for t in (tags or []) if t.strip()}
    if foreign_hosted:
        out.add("foreign_hosted")
    if external:
        out.add("external")
    return out


def title_hints(title: str) -> list[str]:
    """Exclusion keys a meeting title hints at (``grievance`` -> ``hr``); a warning, not a
    refusal, because the organiser's attestation and tags decide."""
    text = normalise_ar(_CAMEL.sub(" ", title))
    return sorted(key for key, pats in _HINT_PATTERNS.items() if any(p.search(text) for p in pats))


def warn_title(title: str) -> None:
    hints = title_hints(title)
    if hints:
        ctx.err_console.print(
            f"WARNING: the title suggests a pilot exclusion ({', '.join(hints)}); if so, stop: "
            "such meetings are out of scope (--tag <exclusion> refuses in code)."
        )


def print_scripts(copy_banner: bool = True) -> None:
    """Print the bilingual consent script and banner; optionally copy both banners (EN, then
    AR) to the clipboard so an ar-mixed meeting gets a bilingual chat notice."""
    ctx.console.print(f"Consent script (version {consent.SCRIPT_VERSION})")
    ctx.console.print("")
    for lang in ("en", "ar"):
        consent.print_script(ctx.console, lang)  # type: ignore[arg-type]
        ctx.console.print("")
    if copy_banner:
        copied = consent.copy_banner_to_clipboard("both")
        ctx.console.print(
            "Chat notice copied to the clipboard; paste it into the meeting chat."
            if copied
            else "Could not copy the chat notice; paste the banner above into the meeting chat."
        )


def new_meeting(
    rt: ctx.Runtime,
    *,
    title: str,
    meeting_type: MeetingType,
    classification: Classification,
    language: LanguageMode,
    platform: Platform,
    roster_path: Any,
    now: datetime | None = None,
    tags: set[str] | None = None,
    organiser: str | None = None,
) -> Meeting:
    """Build a ``Meeting`` (not yet stored); ``tags`` reach ``consent.gate`` -> ``scope.check``
    unchanged.

    The organiser is the acting identity unless ``organiser`` names someone else (an operator
    submitting on their behalf); a named organiser must have the shape of a UPN or e-mail
    address (``IdentityError`` otherwise, before the roster is read) and is stored lower-case.
    """
    require_language(rt.settings, language)
    named = normalise_upn(organiser) if organiser is not None else None
    attendees, rooms = ctx.load_roster(roster_path)
    started = now or datetime.now(UTC)
    warn_title(title)
    return Meeting(
        id=new_meeting_id(started),
        title=title,
        meeting_type=meeting_type,
        classification=classification,
        language_mode=language,
        platform=platform,
        started_at=started,
        organiser=named or rt.current_identity().user,
        roster=attendees,
        room_identities=rooms,
        private=TEMPLATES[meeting_type].default_private,
        tags=set(tags or ()),
    )


def run_gate(rt: ctx.Runtime, meeting: Meeting, answers: GateAnswers) -> ConsentRecord:
    """Run ``consent.gate`` and, on success, persist the meeting and its consent record.

    Refusals propagate as ``ConsentRefused`` / ``ScopeError`` / ``ClassificationNotAllowed``
    (the gate has already audited them); nothing is stored in that case. When the meeting names
    an organiser other than the acting identity, ``meeting.organiser_named`` is audited after the
    gate passes and *before* the meeting is stored, so a stored meeting whose organiser someone
    else named always has that audit event (if the audit append fails, nothing is stored).
    """
    record = consent.gate(
        meeting,
        notified=answers.notified,
        objections=answers.objections,
        method=answers.method,
        teams_transcription_started=answers.teams_transcription_started,
        purpose=answers.purpose,
        scope_checks=answers.scope_checks,
        identity=rt.current_identity(),
        settings=rt.settings,
        audit=rt.audit,
    )
    audit_named_organiser(rt, meeting)
    rt.store.save_meeting(meeting)
    rt.store.save_consent(record)
    return record


def audit_named_organiser(rt: ctx.Runtime, meeting: Meeting) -> bool:
    """Audit ``meeting.organiser_named`` when the stored organiser is not the acting identity.

    Called by ``run_gate`` once the gate has passed and before the meeting is stored. The event
    is recorded on behalf of the operator and carries both the operator and the named organiser
    (and the operator's identity source), so a meeting submitted for someone else is always
    attributable to the person who submitted it. Returns ``True`` when an event was written.
    """
    operator = rt.current_identity()
    if meeting.organiser.lower() == operator.user.lower():
        return False
    rt.audit.append(
        "meeting.organiser_named",
        meeting.id,
        actor=operator,
        classification=meeting.classification.value,
        operator=operator.user,
        operator_source=operator.audit_source(),
        organiser=meeting.organiser,
    )
    return True


def transcript_delete_after(rt: ctx.Runtime, meeting: Meeting, now: datetime) -> datetime:
    """Hard maximum life of a transcript row (``retention.TRANSCRIPT_MAX_DAYS``)."""
    from praktika.retention import TRANSCRIPT_MAX_DAYS

    return now + timedelta(days=TRANSCRIPT_MAX_DAYS[meeting.classification.value])
