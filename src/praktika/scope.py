"""Pilot scope exclusions (C-10): refused in code under ``PRAKTIKA_PILOT=1``.

Contract: ``check`` raises ``ScopeError`` with a bilingual reason when a meeting type or tag
falls under an exclusion while ``settings.pilot`` is true; outside the pilot it is a no-op.
``scope_checklist`` lists the attestations the consent gate requires the organiser to confirm.
"""

from __future__ import annotations

from praktika.config import Settings
from praktika.errors import ScopeError

PILOT_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "board",
        "board_subcommittee",
        "hr",
        "customer_call",
        "regulator",
        "legal_privileged",
        "foreign_hosted",
    }
)

#: (checklist key, English prompt, Arabic prompt). Every key must be confirmed at the gate.
_CHECKLIST: tuple[tuple[str, str, str], ...] = (
    (
        "not_board",
        "This is not a Board or Board sub-committee meeting",
        "هذا ليس اجتماعاً لمجلس الإدارة أو إحدى لجانه",
    ),
    (
        "not_foreign_hosted",
        "This meeting is not hosted by an entity in another jurisdiction",
        "هذا الاجتماع ليس مستضافاً من جهة تابعة لولاية قضائية أخرى",
    ),
    (
        "not_customer_call",
        "This is not a customer call or customer meeting",
        "هذا ليس اتصالاً أو اجتماعاً مع عميل",
    ),
    (
        "not_hr",
        "This is not an HR, disciplinary or grievance meeting",
        "هذا ليس اجتماعاً للموارد البشرية أو اجتماعاً تأديبياً أو اجتماع تظلم",
    ),
    (
        "not_regulator",
        "No regulator or auditor is taking part",
        "لا يشارك في الاجتماع أي جهة رقابية أو مدقق",
    ),
    (
        "not_legal_privileged",
        "The meeting is not legally privileged",
        "الاجتماع لا يخضع للامتياز القانوني",
    ),
)

_REASON_AR: dict[str, str] = {
    "board": "اجتماعات مجلس الإدارة خارج نطاق المشروع التجريبي",
    "board_subcommittee": "اجتماعات لجان مجلس الإدارة خارج نطاق المشروع التجريبي",
    "hr": "اجتماعات الموارد البشرية خارج نطاق المشروع التجريبي",
    "customer_call": "اجتماعات العملاء خارج نطاق المشروع التجريبي",
    "regulator": "اجتماعات الجهات الرقابية والمدققين خارج نطاق المشروع التجريبي",
    "legal_privileged": "الاجتماعات ذات الامتياز القانوني خارج نطاق المشروع التجريبي",
    "foreign_hosted": "الاجتماعات المستضافة من جهة في ولاية قضائية أخرى خارج نطاق المشروع التجريبي",
}


def scope_checklist() -> list[tuple[str, str, str]]:
    """Return ``(key, EN prompt, AR prompt)`` triples the organiser must confirm at the gate."""
    return list(_CHECKLIST)


def scope_check_keys() -> frozenset[str]:
    """The set of checklist keys a complete ``scope_checks`` mapping must contain."""
    return frozenset(key for key, _, _ in _CHECKLIST)


def excluded_reasons(meeting_type: str, tags: set[str]) -> list[str]:
    """Exclusion keys triggered by ``meeting_type`` (exact or ``board*``) or any tag, sorted."""
    hits: set[str] = set()
    mt = meeting_type.strip().lower()
    if mt in PILOT_EXCLUSIONS or mt.startswith("board"):
        hits.add("board" if mt.startswith("board") and mt not in PILOT_EXCLUSIONS else mt)
    for tag in tags:
        t = tag.strip().lower()
        if t in PILOT_EXCLUSIONS:
            hits.add(t)
    return sorted(hits)


def check(meeting_type: str, tags: set[str], settings: Settings) -> None:
    """Raise ``ScopeError`` (bilingual message) if the meeting is excluded from the pilot.

    Exclusions apply only while ``settings.pilot`` is true; the message names every triggered
    exclusion in English and Arabic and never mentions participants.
    """
    if not settings.pilot:
        return
    hits = excluded_reasons(meeting_type, tags)
    if not hits:
        return
    en = "Out of pilot scope: " + ", ".join(h.replace("_", " ") for h in hits)
    ar = " / ".join(_REASON_AR[h] for h in hits)
    raise ScopeError(f"{en} | {ar}")
