"""The consent and scope gate (C-03; controls are listed in docs/CONTROLS.md).

The spoken script and the chat banner below are the notice attendees hear and read. Any
change to their wording is a new ``SCRIPT_VERSION``, which every consent record carries.

Contract: ``gate`` is the only way to obtain a ``ConsentRecord``. It refuses unless the
organiser attests that attendees were notified and nobody objected, the meeting is inside pilot
scope, its classification is permitted and every scope checklist item is confirmed. There is no
parameter, flag or environment variable that skips any of these checks. A refusal is audited as
``scope.refused`` with the reason "recording not used"; it never records who objected.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import ValidationError

from praktika import policy, scope
from praktika.audit import AuditLog
from praktika.config import Settings
from praktika.errors import ClassificationNotAllowed, ConsentRefused, ScopeError
from praktika.identity import Identity
from praktika.logging import get_logger
from praktika.models import ConsentRecord, Meeting

log = get_logger(__name__)

SCRIPT_VERSION = "2026-10-02"

# Every statement below holds with the default settings (pilot, so Internal meetings only, in
# local mode) once the hourly retention scheduler runs (retention.py): the first run after
# approval or discard deletes the audio, and the first run 24 hours after conversion does so
# otherwise; the transcript goes 14 days after approval and at most 60 days after it is first
# saved; a legal hold stops every timer; nothing is exported before approval. docs/CONTROLS.md
# quotes these texts verbatim (a test checks it) and says when a deployment must change them.
SCRIPT_EN = (
    "Before we start: this meeting is being recorded and transcribed by our internal AI "
    "notetaker, which runs only on our own systems, to produce the minutes. The recording is "
    "deleted after the minutes are approved or discarded, and within about a day in any case. "
    "The transcript is deleted within two weeks of the minutes being approved, and within 60 "
    "days in any case. A legal hold can require us to keep them longer. I check and approve "
    "the minutes before they are shared. Nothing is used to evaluate individuals. If anyone "
    "prefers not to be recorded, say so now or message me privately and we will take notes "
    "manually instead. Does everyone agree to proceed?"
)

SCRIPT_AR = (
    "قبل أن نبدأ: يتم تسجيل هذا الاجتماع وتفريغه نصياً بواسطة أداة تدوين المحاضر الداخلية "
    "لدينا بالذكاء الاصطناعي، والتي تعمل فقط على أنظمتنا الخاصة، وذلك لإعداد محضر "
    "الاجتماع. يُحذف التسجيل الصوتي بعد اعتماد المحضر أو إلغائه، وفي جميع الأحوال خلال يوم "
    "تقريباً. ويُحذف النص المفرَّغ خلال أسبوعين من اعتماد المحضر، وفي جميع الأحوال خلال ستين "
    "يوماً. وقد يقتضي حفظ قانوني الاحتفاظ بهما مدة أطول. وأقوم بمراجعة المحضر واعتماده قبل "
    "مشاركته. لا تُستخدم هذه المعلومات لتقييم أداء الأفراد. إذا كان أي منكم يفضّل عدم "
    "تسجيله، فليُبلغني الآن أو عبر رسالة خاصة وسنقوم بتدوين الملاحظات يدوياً. هل يوافق "
    "الجميع على المتابعة؟"
)

BANNER_EN = (
    "Our internal AI notetaker is recording and transcribing for minutes, on our systems only. "
    "Audio deleted within about a day, transcript within 60 days, unless on legal hold. Object "
    "via the organiser."
)

BANNER_AR = (
    "أداة تدوين المحاضر الداخلية لدينا تسجّل الاجتماع وتفرّغه لإعداد المحضر، على أنظمتنا فقط. "
    "يُحذف الصوت خلال يوم تقريباً والنص خلال ستين يوماً ما لم يُفرض حفظ قانوني. يمكنك "
    "الاعتراض عبر المنظّم."
)

BANNER_MAX_CHARS = 200
SCRIPTS: dict[str, str] = {"en": SCRIPT_EN, "ar": SCRIPT_AR}
BANNERS: dict[str, str] = {"en": BANNER_EN, "ar": BANNER_AR}
PBCOPY = "/usr/bin/pbcopy"
REFUSAL_REASON = "recording not used"

Lang = Literal["en", "ar"]


def _text(table: dict[str, str], lang: str) -> str:
    try:
        return table[lang]
    except KeyError:
        raise ValueError(f"unsupported language {lang!r}; use 'en' or 'ar'") from None


def print_script(console: Any, lang: Lang) -> None:
    """Print the spoken script and the chat banner for ``lang`` on a ``rich`` console."""
    console.print(_text(SCRIPTS, lang))
    console.print("")
    console.print(_text(BANNERS, lang))


def copy_banner_to_clipboard(lang: Lang | Literal["both"]) -> bool:
    """Copy the chat banner (``"both"``: English then Arabic, one per line) to the clipboard
    with ``pbcopy``; False on any failure.

    A host with no clipboard tool (a headless Linux server) returns False at once and
    logs nothing: that is the normal state there, not a failure worth a warning, and the
    caller already tells the organiser to paste the printed banner instead.
    """
    if not Path(PBCOPY).is_file():
        return False
    try:
        text = BANNER_EN + "\n" + BANNER_AR if lang == "both" else _text(BANNERS, lang)
        # Fixed absolute executable, no arguments; the banner is passed on stdin.
        subprocess.run(  # noqa: S603
            [PBCOPY], input=text.encode("utf-8"), check=True, timeout=3.0, capture_output=True
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        log.warning("consent.clipboard_failed", error=str(exc))
        return False
    return True


def _refuse(audit: AuditLog, meeting: Meeting, exc: Exception) -> NoReturn:
    """Audit and log a refusal with the one neutral reason; which check failed is told only to
    the organiser through ``exc`` (an audit line naming ``objections`` on a 1:1 would identify
    the objector; C-03)."""
    audit.append(
        "scope.refused",
        meeting.id,
        classification=meeting.classification.value,
        reason=REFUSAL_REASON,
    )
    log.warning("consent.refused", meeting_id=meeting.id, reason=REFUSAL_REASON)
    raise exc


def gate(
    meeting: Meeting,
    *,
    notified: bool,
    objections: bool,
    method: str,
    teams_transcription_started: bool,
    purpose: str,
    scope_checks: dict[str, bool],
    identity: Identity,
    settings: Settings,
    audit: AuditLog,
) -> ConsentRecord:
    """Run every pre-capture check and return the organiser's ``ConsentRecord``.

    Raises ``ConsentRefused`` unless ``notified and not objections``, when a checklist item is
    missing or false, or when ``method``/``purpose`` are invalid; ``ScopeError`` for a pilot
    exclusion; ``ClassificationNotAllowed`` when policy refuses the classification. Emits
    ``consent.recorded`` on success and ``scope.refused`` on every refusal.
    """
    if not notified:
        _refuse(audit, meeting, ConsentRefused("attendees were not notified"))
    if objections:
        _refuse(audit, meeting, ConsentRefused("an objection was received"))
    try:
        scope.check(meeting.meeting_type.value, meeting.tags, settings)
    except ScopeError as exc:
        _refuse(audit, meeting, exc)
    decision = policy.classification_allowed(meeting.classification, settings)
    if not decision.allowed:
        _refuse(audit, meeting, ClassificationNotAllowed(decision.reason))
    missing = sorted(scope.scope_check_keys() - set(scope_checks))
    if missing:
        _refuse(
            audit, meeting, ConsentRefused(f"scope checklist not confirmed: {', '.join(missing)}")
        )
    unchecked = sorted(k for k, ok in scope_checks.items() if not ok)
    if unchecked:
        declined = ", ".join(unchecked)
        _refuse(audit, meeting, ConsentRefused(f"scope checklist items declined: {declined}"))
    try:
        record = ConsentRecord(
            meeting_id=meeting.id,
            notified=notified,
            objections=objections,
            method=method,  # type: ignore[arg-type]
            teams_transcription_started=teams_transcription_started,
            script_version=SCRIPT_VERSION,
            purpose=purpose,
            scope_checks=dict(scope_checks),
            recorded_by=identity.user,
            recorded_by_source=identity.audit_source(),
            recorded_at=datetime.now(UTC),
        )
    except ValidationError as exc:
        _refuse(audit, meeting, ConsentRefused(f"consent record invalid: {exc}"))
    audit.append(
        "consent.recorded",
        meeting.id,
        classification=meeting.classification.value,
        method=record.method,
        teams_transcription_started=record.teams_transcription_started,
        script_version=record.script_version,
        purpose=record.purpose,
        scope_checks=record.scope_checks,
    )
    log.info("consent.recorded", meeting_id=meeting.id, method=record.method)
    return record
