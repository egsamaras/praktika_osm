"""Turning LLM findings into ``Minutes`` items with citations resolved from the transcript.

Split out of ``pipeline.py`` to keep that module readable. Nothing here calls a model: these
are pure functions over Pydantic objects. The provenance helpers (git SHA, model hashes) live
in ``llm.provenance`` and are re-exported here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from praktika.llm.provenance import git_sha as git_sha
from praktika.llm.provenance import model_hashes as model_hashes
from praktika.llm.quotes import cited_runs, quote_matches, supports
from praktika.models import (
    ActionItem,
    Decision,
    Flag,
    MergedFindings,
    Minutes,
    Narrative,
    OpenQuestion,
    Ref,
    Risk,
    Segment,
    TopicSummary,
)

_SEG_ID = re.compile(r"^S\d{4,5}$")
_ISO_DATE = re.compile(r"\b(?P<year>\d{4})-(?P<mm>\d{2})-(?P<day>\d{2})\b")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
#: Weekday spellings accepted next to a day ("Mon the 21st"); on their own the abbreviations are
#: ordinary words (sat, sun, wed, mon) and are never read as a weekday.
_WEEKDAY_FORMS = {name: i for i, name in enumerate(_WEEKDAYS)} | {
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}  # fmt: skip

Items = dict[str, list[Any]]


def _quotes_for(ids: list[str], quote: str | None, by_id: dict[str, Segment]) -> dict[str, str]:
    """The quote each cited (present) segment should carry.

    The model returns one quote per item. It is attached to the segment(s) that contain it —
    on their own or as a run of consecutive cited segments — and every other cited segment
    carries its own text, so a supporting citation is not later dropped for carrying a quote
    that belongs to its neighbour. When no cited segment contains the quote, every ref keeps
    the model's quote: the verifier then removes the item and flags it for the reviewer.
    """
    present = [sid for sid in ids if sid in by_id]
    if not quote:
        return {sid: by_id[sid].text for sid in present}
    supported = {sid for sid in present if quote_matches(quote, by_id[sid].text)}
    if not supported:  # perhaps the quote straddles a boundary of consecutive cited segments
        # ``by_id`` preserves transcript order (``Transcript.by_id``), which the run check needs
        runs = cited_runs(present, list(by_id.values()))
        supported = {sid for sid in present if supports(quote, by_id[sid], runs.get(sid))}
    if not supported:
        return dict.fromkeys(present, quote)
    return {sid: quote if sid in supported else by_id[sid].text for sid in present}


def resolve_refs(ids: list[str], quote: str | None, by_id: dict[str, Segment]) -> list[Ref]:
    """Turn cited ids into ``Ref``s with real times.

    Malformed ids are dropped. Well-formed ids missing from the transcript become placeholder
    refs (speaker ``unknown``, empty quote) that the verifier removes, so a fabricated citation
    is visible rather than silently accepted. Without a quote the segment text is used; with
    one, only the segment(s) that contain it carry it (see ``_quotes_for``).
    """
    wanted = [sid for sid in dict.fromkeys(ids) if _SEG_ID.match(sid)]
    quotes = _quotes_for(wanted, quote, by_id)
    refs: list[Ref] = []
    for sid in wanted:
        seg = by_id.get(sid)
        if seg is None:
            refs.append(Ref(segment_id=sid, start_s=0, end_s=0, speaker="unknown", quote=""))
            continue
        refs.append(
            Ref(
                segment_id=sid,
                start_s=seg.start,
                end_s=seg.end,
                speaker=seg.speaker,
                quote=quotes[sid][:240],
            )
        )
    return refs


_NOT_A_PERSON = re.compile(r"^(unknown|none|null|n/a|tbd|tbc|room|speaker[_ ]?\d+)$", re.I)


def _opt(value: str | None) -> str | None:
    """Draft models carry optional text as ``""`` (constrained-decoding friendly); the minutes
    models keep ``None`` for "not stated". Whitespace-only counts as absent."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _person(value: str | None) -> str | None:
    """An owner/raiser name, or ``None`` when the model echoed a speaker label or placeholder
    ("unknown", "SPEAKER_01", "Room") instead of a person (seen with qwen2.5:14b when the
    transcript has no speaker names)."""
    name = _opt(value)
    if name is None or _NOT_A_PERSON.match(name):
        return None
    return name


_MONTH_NAMES = (
    ("january", "jan", "يناير"),
    ("february", "feb", "فبراير"),
    ("march", "mar", "مارس"),
    ("april", "apr", "أبريل", "ابريل"),
    ("may", "مايو"),
    ("june", "jun", "يونيو"),
    ("july", "jul", "يوليو"),
    ("august", "aug", "أغسطس", "اغسطس"),
    ("september", "sep", "sept", "سبتمبر"),
    ("october", "oct", "أكتوبر", "اكتوبر"),
    ("november", "nov", "نوفمبر"),
    ("december", "dec", "ديسمبر"),
)
_MONTHS = {name: i + 1 for i, names in enumerate(_MONTH_NAMES) for name in names}


def _alternation(names: Iterable[str]) -> str:
    """A regex alternation of ``names``, longest first so "sept" wins over "sep"."""
    return "|".join(sorted(names, key=len, reverse=True))


#: "may" is the month only where a date puts it: after "of", before a day number, or at the end
#: of the phrase, before punctuation or before a year ("the 21st may slip" is not a date).
_MAY_AS_MONTH = r"may(?=\s*$|\s*[^\w\s]|\s+\d{4}\b)"
_MONTH_RE = _alternation(n for n in _MONTHS if n != "may")
_ORDINAL = r"(?:st|nd|rd|th)"
#: Four digits said after the date: the year ("21st May 2027", "March 3, 2027") or a 24-hour
#: time ("30 September 1700"); ``_in_year`` decides which.
_WITH_YEAR = r"(?:,?\s+(?P<year>\d{4})\b)?"
#: A year is four digits within a few years of the meeting's: one back, three ahead.
_YEARS_NEAR = range(-1, 4)
#: Four digits up to this many years from the meeting's could be a year or a time ("2 October
#: 2033"): never guessed. Further off, a 19xx or 20xx that reads as a time is one ("30 Sep 2000").
_YEARS_OR_TIME = 10
_CLOCK = re.compile(r"(?:[01]\d|2[0-3])[0-5]\d")
#: A 24-hour time said as four digits with "hrs" or after "at": "1600 hrs", "at 1700". Blanked
#: out first, so it is never read as a year.
_CLOCK_TIME = re.compile(
    r"\b(?:at\s+)?(?:[01]\d|2[0-3])[0-5]\d\s*(?:hrs?|hours|h)\b|\bat\s+(?:[01]\d|2[0-3])[0-5]\d\b",
    re.I,
)
#: "18th of September", "30 Sept", "the 10th in November", "٢٥ سبتمبر".
_DAY_MONTH = re.compile(
    rf"\b(?P<day>\d{{1,2}}){_ORDINAL}?"
    rf"(?:\s+(?:of|in)\s+(?P<of>may|{_MONTH_RE})|\s+(?P<month>{_MAY_AS_MONTH}|{_MONTH_RE}))\b\.?"
    rf"{_WITH_YEAR}",
    re.I,
)
#: "September 29", "Nov. 10th", "November the 2nd".
_MONTH_DAY = re.compile(
    rf"\b(?P<month>may|{_MONTH_RE})\b\.?\s+(?:the\s+)?(?P<day>\d{{1,2}}){_ORDINAL}?\b{_WITH_YEAR}",
    re.I,
)
#: A month named anywhere: full, abbreviated ("Nov.", "Dec") or Arabic, and "may" where a date
#: puts it. Bare "jan" and "mar" are a given name and a verb as often as a month, so they count
#: only with a period.
_ANY_MONTH = re.compile(
    rf"\b(?:{_alternation(n for n in _MONTHS if n not in ('may', 'jan', 'mar'))}"
    rf"|jan\.|mar\.|{_MAY_AS_MONTH})(?!\w)",
    re.I,
)
_ORDINAL_DAY = re.compile(rf"\b(?P<day>\d{{1,2}}){_ORDINAL}\b", re.I)
_WEEKDAY_RE = _alternation(_WEEKDAY_FORMS)
_FULL_WEEKDAY_RE = "|".join(_WEEKDAYS)
#: Only full weekday names count on their own (sat, sun, wed and mon are ordinary words).
_FULL_WEEKDAY = re.compile(rf"\b(?:{_FULL_WEEKDAY_RE})\b", re.I)
#: Arabic weekday names, used only to check a date said in full ("الخميس ١ أكتوبر").
_ARABIC_WEEKDAYS = {
    "الاثنين": 0, "الإثنين": 0, "الثلاثاء": 1, "الأربعاء": 2, "الاربعاء": 2, "الخميس": 3,
    "الجمعة": 4, "السبت": 5, "الأحد": 6, "الاحد": 6,
}  # fmt: skip
_ARABIC_WEEKDAY = re.compile(rf"(?<!\w)(?:{_alternation(_ARABIC_WEEKDAYS)})(?!\w)")
#: A weekday said with the day it names: "Monday the 21st", "Mon, 21st", "Friday (the 2nd)",
#: "Friday, i.e. the 2nd", "the 21st, a Monday", "21st (Monday)", "the 3rd, which is a Friday".
#: A weekday anywhere else in the phrase says nothing about that day.
_SAID_BEFORE = r"\.?\s*(?:[,:;(–—-]\s*)?(?:i\.?e\.?,?\s*)?(?:the\s+)?"
_SAID_AFTER = r"\s*[,:;(–—-]\s*(?:i\.?e\.?,?\s*)?(?:a\s+|on\s+a\s+|which\s+is\s+(?:a\s+)?)?"
_WEEKDAY_BEFORE_DAY = re.compile(
    rf"\b(?P<weekday>{_WEEKDAY_RE})\b{_SAID_BEFORE}\d{{1,2}}{_ORDINAL}\b", re.I
)
_WEEKDAY_AFTER_DAY = re.compile(
    rf"\b\d{{1,2}}{_ORDINAL}{_SAID_AFTER}(?P<weekday>{_WEEKDAY_RE})\b", re.I
)
#: A day number said without its suffix straight after a full weekday name: "Sunday 27", "by
#: Friday, 2". Not a time ("Friday 2pm", "Friday 5.30", "Thursday 1700"), and not a numeric
#: date or a range ("Thursday 1/10", "Friday 2-3"), which ``_NUMBER_BY_WEEKDAY`` refuses.
_TIME_AFTER = r"\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock|noon|midday|midnight|h|hrs?|hours)(?!\w)"
_PLAIN_DAY = re.compile(
    rf"\b(?P<weekday>{_FULL_WEEKDAY_RE})\b{_SAID_BEFORE}(?P<day>\d{{1,2}})\b"
    rf"(?![:.]\d|\s*[/.-]\s*\d|{_TIME_AFTER})",
    re.I,
)
#: Any number said beside a weekday that is not a time of day: a day ("Sunday 27", "27,
#: Sunday"), a numeric date ("Thursday 1/10", "Friday, 2/10", "Friday 9.10") or a range
#: ("Friday 2-3"). "Friday 5.30", "Friday 9.10am" and "Friday 17:00" are times.
_TIME_NUMBER = rf":\d|\.(?:00|1[3-9]|[2-5]\d)\b|\.\d{{1,2}}{_TIME_AFTER}|{_TIME_AFTER}"
_NUMBER_BY_WEEKDAY = re.compile(
    rf"\b(?:{_FULL_WEEKDAY_RE})\b{_SAID_BEFORE}\d{{1,2}}(?!\d|{_ORDINAL}\b|{_TIME_NUMBER})"
    rf"|(?<![:.])\b\d{{1,2}}(?:[/.-]\d{{1,2}}(?:[/.-]\d{{2,4}})?)?\s*[,:;(–—-]?\s*(?:the\s+)?"
    rf"(?:{_FULL_WEEKDAY_RE})\b",
    re.I,
)
#: The same, next to a date said in full ("Friday 2nd October", "5 October (Monday)").
_WEEKDAY_BEFORE_DATE = re.compile(rf"\b(?P<weekday>{_WEEKDAY_RE})\b{_SAID_BEFORE}$", re.I)
_WEEKDAY_AFTER_DATE = re.compile(rf"{_SAID_AFTER}(?P<weekday>{_WEEKDAY_RE})\b\)?", re.I)
#: A weekday that describes a date said elsewhere in the phrase: "the 2nd of next month, a
#: Monday". On its own ("on a Friday") it is any Friday.
_A_WEEKDAY = re.compile(
    rf"\b(?P<intro>a|which\s+is|i\.?e\.?)\s+(?P<weekday>{_FULL_WEEKDAY_RE})\b", re.I
)
#: A counted weekday: "the first Monday", "last Friday", "every other Friday".
_COUNTED_WEEKDAY = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|every|each|other|alternate)"
    rf"\s+(?:{_WEEKDAY_RE}|may)\b",
    re.I,
)
#: "a 3rd", "every 2nd": an ordinal that counts.
_BEFORE_A_COUNT = re.compile(r"\b(?:a|an|every|each|per)\s+$", re.I)
_RELATIVE_MONTH = r"(?:(?P<this>this|current)|next|following|coming)\s+month\b"
#: Words that can follow a day of the month ("by the 30th at the latest", "the 30th inshallah").
#: An ordinal followed by any other word counts something ("the 2nd week", "the 4th quarter",
#: "the 2nd MANCOM", "the 2nd reading", "21st Monday", "the 21st may slip") and is not a day.
_DAY_FOLLOWERS = (
    "and", "or", "but", "not", "nor", "so", "then", "if", "unless", "when", "which", "that", "as",
    "since", "because", "though", "while", "at", "by", "on", "before", "ahead", "after", "for",
    "to", "with", "without", "via", "including", "inclusive", "latest", "earliest", "sharp", "max",
    "maximum", "please", "pls", "ideally", "hopefully", "inshallah", "insha", "god", "the", "i",
    "we", "you", "he", "she", "they", "it", "everyone", "is", "are", "will", "would", "should",
    "could", "must", "shall", "can", "works", "suits", "deadline", "morning", "afternoon",
    "evening", "night", "noon", "midday", "midnight", "lunchtime", "eod", "cob", "cop", "close",
    "end", "am", "pm", "tops",
)  # fmt: skip
_AFTER_A_DAY = re.compile(
    r"\s*(?:$|[^\w\s-]|-(?![a-z])"  # end of phrase, or punctuation: "the 21st, a Monday"
    r"|\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\d{1,2}[:.]\d{2}\b"  # a time: "the 30th 5pm"
    r"|in\s+the\s+(?:morning|afternoon|evening)\b"
    r"|(?:of\s+|in\s+)?(?:the\s+)?(?:this|current|next|following|coming)\s+month\b"
    rf"|(?:{_alternation(_DAY_FOLLOWERS)})\b)",
    re.I,
)
#: A month said with a bare day: "the 15th of next month", "next month on the 10th".
_MONTH_AFTER_DAY = re.compile(rf"\s*,?\s*(?:of\s+|in\s+)?(?:the\s+)?{_RELATIVE_MONTH}", re.I)
_MONTH_BEFORE_DAY = re.compile(rf"\b{_RELATIVE_MONTH},?\s+(?:on\s+|by\s+)?(?:the\s+)?$", re.I)
_MONTH_WORD = re.compile(r"\bmonths?\b", re.I)
_WEEK_WORD = re.compile(r"\b(?:weeks?|fortnights?)\b|\bw/c\b", re.I)
_YEAR_WORD = re.compile(r"\byears?\b|\b(?:19|20)\d{2}\b", re.I)
_YEAR_HINT = re.compile(r"\b(?:(?P<next>next|following)|this|current)\s+year\b", re.I)
#: "Friday next week", "Wednesday of next week", "next week on Tuesday", "Friday this week".
_WEEKDAY_THEN_WEEK = re.compile(
    rf"\b(?:{_FULL_WEEKDAY_RE})\b(?:'s)?,?\s+(?:of\s+)?(?:the\s+)?"
    r"(?P<which>next|following|this)\s+week\b",
    re.I,
)
_WEEK_THEN_WEEKDAY = re.compile(
    r"\b(?P<which>next|following|this)\s+week\b,?\s+(?:on\s+)?(?:the\s+)?"
    rf"(?:{_FULL_WEEKDAY_RE})\b",
    re.I,
)
_NEXT_WEEKDAY = re.compile(
    rf"\bnext\s+(?:{_FULL_WEEKDAY_RE})\b|\b(?:{_FULL_WEEKDAY_RE})\s+next\b", re.I
)
#: Words that fix a weekday, a weekend or an eve by something said after them.
_ANCHOR = (
    r"(?:(?:just|right|immediately|directly)\s+)?"
    r"(?:before(?:hand)?|prior(?:\s+to)?|previous(?:\s+to)?|preceding|ahead\s+of|"
    r"in\s+advance\s+of|leading\s+up\s+to|after(?:wards?)?|following)"
)
#: A weekday fixed by something else, said or not: "the Friday before the board", "the Monday
#: after next", "the Thursday prior", "the previous Thursday", "that Friday".
_ANCHORED_WEEKDAY = re.compile(
    rf"\b(?:{_FULL_WEEKDAY_RE})\b(?:'s)?\s+(?:{_ANCHOR}|of)\b|\bafter\s+next\b"
    rf"|\b(?:previous|preceding|following|prior|that|same)\s+(?:{_FULL_WEEKDAY_RE})\b",
    re.I,
)
#: A weekday, weekend, eve or part of a day fixed by a day said with it: "the Monday before the
#: 2nd", "the Friday prior to the 30th", "the weekend before 2 October", "the eve of the 2nd",
#: "the night before the 2nd". The day is the anchor, not the deadline.
_ANCHORED_TO_A_DAY = re.compile(
    rf"\b(?:{_FULL_WEEKDAY_RE}|weekends?|eve)\b(?:'s)?(?:,?\s+{_ANCHOR}|\s+of)\b"
    rf"|\b(?:morning|afternoon|evening|night)\s+{_ANCHOR}\b",
    re.I,
)
#: A second time offered with the date: "the 30th this month or next", "the 2nd or later",
#: "by the 30th or the week after" (not "the 30th and next steps").
_ALTERNATIVE = re.compile(
    r"\b(?:or|and(?:/or)?)\s+(?:the\s+|a\s+)?(?:"
    rf"(?:next|following)(?=\s*(?:$|[^\w\s]|(?:day|week|month|year|one|{_FULL_WEEKDAY_RE})\b))"
    r"|(?:later|after(?:wards)?|thereafter|beyond)\b"
    r"|(?:day|week|month|one)\s+(?:after|before|later|earlier)\b)",
    re.I,
)
_TODAY = re.compile(r"\b(?:(?P<today>today|tonight)|tomorrow)\b", re.I)
#: A day counted from another: "the day before Friday", "10 days from the 1st".
_DAY_OFFSET = re.compile(
    r"\bdays\b|\bday\s+(?:before|after|following|prior|preceding|ahead)\b"
    r"|\b\d+\s*(?:h|hrs?|hours?)\s+(?:before|after|following|prior|preceding|ahead)\b",
    re.I,
)
#: With two dates in a phrase, one of these words makes it a correction or an alternative.
_CHANGE_WORDS = re.compile(
    r"\b(?:or|but|either|instead|rather|alternatively|otherwise|move[ds]?|moving|push(?:ed)?|"
    r"postponed?|brought|change[ds]?|rescheduled?|delayed|extended|slipped|shifted|originally|"
    r"previously|formerly|now|was|were|sorry|correction|actually|mean)\b",
    re.I,
)
#: A date said as one that is not the deadline: one that no longer holds ("not the 30th",
#: "moved from 2 October") or a start ("after Friday", "from the 1st", "not before the 30th").
_NOT_DUE = re.compile(
    r"\b(?:not(?:\s+before)?|no\s+earlier\s+than|from|since|after|starting|beginning|commencing|"
    r"effective|was|were|originally|previously|formerly|instead\s+of|rather\s+than)\s+"
    r"(?:(?:on|by|the|until|till)\s+)*$",
    re.I,
)
#: A day that is one end of a range or one of two: "5-6 October", "October 5 or 6".
_RANGE_LINK = r"\s*(?:[-–—/&]|\b(?:to|till|until|through|thru|or|and)\b)\s*"
_RANGE_BEFORE = re.compile(rf"\d{_ORDINAL}?{_RANGE_LINK}(?:the\s+)?$", re.I)
_RANGE_AFTER = re.compile(rf"{_RANGE_LINK}(?:the\s+)?\d{{1,2}}(?!\d|\s*(?:am|pm)\b|[:.]\d)", re.I)
#: A number straight after a date said in full: "5 October 27" (a year?), "30 September, 2".
_NUMBER_AFTER = re.compile(r",?\s+'?\d{1,2}\b(?!\s*(?:am|pm)\b|[:.]\d)", re.I)
#: A quarter or half of the year: the date must fall in it ("the 2nd, early Q4").
_QUARTER = re.compile(r"\b(?P<kind>q|h)(?P<n>[1-4])\b", re.I)
#: "Jan, the 15th", "by the 15th, Jan": a given name or the month, so the month is not known.
_NAME_OR_MONTH = re.compile(
    rf"\b(?:jan|mar)\b\.?,?\s+(?:the\s+)?\d{{1,2}}{_ORDINAL}?\b"
    rf"|\b\d{{1,2}}{_ORDINAL}?\s*,\s*(?:jan|mar)\b",
    re.I,
)
_FIRST_THING = re.compile(r"\b(?:1st|first)\s+thing\b", re.I)
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


_TIME_WORDS = re.compile(
    r"\d|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|"
    r"week|weeks|month|months|quarter|year|end of|eod|cob|asap|by|before|after|until|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|q[1-4]|h[12]|"
    r"mancom|committee|council|board|meeting|session|next|following)\b|"
    r"[\u0660-\u0669]|(?:الأسبوع|الاسبوع|الشهر|غدا|غداً|اليوم|قبل|بعد|بحلول|نهاية|الخميس|الأحد|"
    r"الاثنين|الثلاثاء|الأربعاء|الجمعة|السبت|يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|"
    r"سبتمبر|أكتوبر|نوفمبر|ديسمبر|الأول|الثاني|الثالث|الرابع|الخامس|العاشر|العشرين)",
    re.IGNORECASE,
)


def due_phrase(value: str | None) -> str | None:
    """A due phrase only if it refers to a time; otherwise ``None``.

    A model output can put a speaker's request ("please send me your view on which supplier
    we should shortlist") into ``due_text`` when no deadline was spoken. A phrase with no
    digit, weekday, month, relative-time word or milestone reference is not a due date and
    must not reach the minutes.
    """
    text = _opt(value)
    if text is None or not _TIME_WORDS.search(text):
        return None
    return text


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


class _Stated(NamedTuple):
    """A date said in full, and where it sits in the phrase (with any weekday said with it)."""

    start: int
    end: int
    day: int
    month: int
    year: int | None
    weekday: int | None
    #: Four digits said after a day and month: a year or a 24-hour time (``_in_year``).
    four: str | None = None


def due_date_from_text(due_text: str | None, meeting_date: date) -> date | None:
    """Resolve a spoken due phrase to a calendar date, or ``None`` unless the phrase states it.

    The reviewer sees the phrase next to the date, so ``None`` is always safe and a wrong date is
    not. Resolved:

    - an ISO date, or a day with its month in English or Arabic ("18th of September", "Nov.
      10th", "November the 2nd", "the 10th in November", "٢٥ سبتمبر"), in the year said with
      it (four digits within a few years of the meeting; "30 September 1700" is a time), else
      "next year" or "this year", else the meeting's year, rolled to the next year when it
      falls more than a month before the meeting; a weekday said in the phrase must fall on a
      date whose year is said or rolled;
    - a bare ordinal day ("by the 21st"), or a day number said straight after a weekday
      ("Sunday 27"), in the month said with it ("of next month", "this month"), else the next
      such day after the meeting, and only if a weekday said with it ("Monday the 21st", "the
      21st, a Monday") falls on it; "Friday 2" may be Friday at 2, so a day up to 12 is kept
      only when it is also the coming such weekday;
    - one weekday ("by Thursday") as the next such day after the meeting; with "next week" or
      "this week", or as "next Thursday", only when a Sunday-start week (the Gulf working week)
      and a Monday-start week (ISO) give the same day.

    Anything else stays ``None`` (system rule 4), in particular: a month named anywhere but not
    with the day ("in November, on the 10th"); an ordinal that counts ("the 2nd week of
    October", "the 4th quarter", "the 2nd MANCOM", "the first Monday", "the 21st may slip");
    a week or day offset ("Friday week", "a week on Friday", "Friday after next", "the day
    before Friday"); a weekday, weekend or eve fixed by the day said ("the Monday before the
    2nd", "the eve of the 2nd"); two different dates, a range, an alternative, a correction or
    a start ("moved from 2 October to 9 October", "the 30th this month or next", "not the
    30th", "after Friday"); a numeric date beside a weekday ("Thursday 1/10"); "today" or
    "tomorrow" said with another date; and the meeting's own weekday or day of the month
    (today, or a week or a month on).
    """
    if not due_text:
        return None
    text = _FIRST_THING.sub(_blank, due_text.translate(_ARABIC_INDIC))  # "1st thing Monday"
    text = _CLOCK_TIME.sub(_blank, text)  # "1600 hrs", "at 1700": never a year
    stated, rest = _stated_dates(text)
    days = sorted([*_ORDINAL_DAY.finditer(rest), *_PLAIN_DAY.finditer(rest)], key=_start)
    if (
        _DAY_OFFSET.search(rest)
        or _ALTERNATIVE.search(rest)
        or ((stated or days) and _ANCHORED_TO_A_DAY.search(rest))
        or _counts_something(rest, stated, days)
        or _corrected(rest, stated, days)
    ):
        return None
    if stated:
        due = _stated_date(stated, rest, days, meeting_date)
    elif days:
        due = _bare_day(rest, days, meeting_date)
    else:
        due = _weekday_only(rest, meeting_date)
    return due if due and _agrees(rest, due, meeting_date) else None


def _blank(m: re.Match[str]) -> str:
    return "#" * len(m[0])


def _start(m: re.Match[str]) -> int:
    return m.start()


def _agrees(rest: str, due: date, meeting_date: date) -> bool:
    """Whether ``due`` fits a "today", "tomorrow", quarter or half of the year said with it: in
    "tomorrow, the 17th" said on the 15th, or "the 2nd, early Q4" said in December, one of the
    two is wrong."""
    for m in _TODAY.finditer(rest):
        if due != meeting_date + timedelta(days=0 if m["today"] else 1):
            return False
    for m in _QUARTER.finditer(rest):
        months = 3 if m["kind"].lower() == "q" else 6
        if (due.month - 1) // months + 1 != int(m["n"]):
            return False
    return True


def _stated_dates(text: str) -> tuple[list[_Stated], str]:
    """Each date said in full (ISO, or a day with its month), and ``text`` with those dates and
    any weekday said with them blanked out, so that what is left can be checked against them.
    A month followed by a count ("October 2nd half", "November the 2nd reading") is no date."""
    found: list[_Stated] = []
    for pattern in (_ISO_DATE, _DAY_MONTH, _MONTH_DAY):
        for m in pattern.finditer(text):
            if pattern is _MONTH_DAY and not _AFTER_A_DAY.match(text, m.end()):
                continue
            (start, end), weekday = m.span(), None
            if w := _WEEKDAY_BEFORE_DATE.search(text, 0, start):
                start, weekday = w.start(), w["weekday"]
            elif w := _WEEKDAY_AFTER_DATE.match(text, end):
                end, weekday = w.end(), w["weekday"]
            g = m.groupdict()
            name = g.get("of") or g.get("month")
            iso = pattern is _ISO_DATE
            found.append(
                _Stated(
                    start,
                    end,
                    day=int(g["day"]),
                    month=_MONTHS[name.lower()] if name else int(g["mm"]),
                    year=int(g["year"]) if iso else None,
                    weekday=_WEEKDAY_FORMS[weekday.lower()] if weekday else None,
                    four=None if iso else g["year"],
                )
            )
            text = text[:start] + "#" * (end - start) + text[end:]
    return sorted(found), text


def _counts_something(rest: str, stated: list[_Stated], days: list[re.Match[str]]) -> bool:
    """An ordinal that counts rather than dates: "the 2nd week", "every 3rd", "first Monday"."""
    if _COUNTED_WEEKDAY.search(rest):
        return True
    starts = [s.start for s in stated] + [m.start() for m in days]
    return any(_BEFORE_A_COUNT.search(rest, 0, start) for start in starts) or any(
        not _AFTER_A_DAY.match(rest, m.end()) for m in days
    )


def _loose_weekdays(rest: str, days: list[re.Match[str]]) -> list[int]:
    """Where each full weekday name that is not said with a day starts."""
    with_a_day = {
        m.start("weekday")
        for pattern in (_WEEKDAY_BEFORE_DAY, _WEEKDAY_AFTER_DAY)
        for m in pattern.finditer(rest)
    }
    with_a_day |= {m.start("weekday") for m in days if m.re is _PLAIN_DAY}
    return [m.start() for m in _FULL_WEEKDAY.finditer(rest) if m.start() not in with_a_day]


def _corrected(rest: str, stated: list[_Stated], days: list[re.Match[str]]) -> bool:
    """The date is corrected, offered as one of two, a start, or one end of a range.

    Two dates with a change word ("moved from Monday to the 30th", "Friday or the 2nd"); the
    date itself said as one that is not the deadline ("Friday, not the 2nd", "moved from 2
    October", "after Friday"); or a day joined to another number ("5-6 October", "October 5
    or 6", "5 October 27"). A weekday beside a day that does not change it stays harmless ("by
    the 30th, not Friday").
    """
    loose = _loose_weekdays(rest, days)
    said = [s.start for s in stated] + [m.start() for m in days] + loose
    said += [m.start() for m in _TODAY.finditer(rest)]
    if len(said) > 1 and _CHANGE_WORDS.search(rest):
        return True
    spans = [(s.start, s.end) for s in stated] or [m.span() for m in days]
    chosen = [start for start, _ in spans] or loose
    return any(_NOT_DUE.search(rest, 0, start) for start in chosen) or any(
        _RANGE_BEFORE.search(rest, 0, start)
        or _RANGE_AFTER.match(rest, end)
        or (bool(stated) and _NUMBER_AFTER.match(rest, end))
        for start, end in spans
    )


def _stated_date(
    stated: list[_Stated], rest: str, days: list[re.Match[str]], meeting_date: date
) -> date | None:
    """The one date said in full, unless the rest of the phrase names a different day or month,
    or talks of weeks, months or another year."""
    hints = list(_YEAR_HINT.finditer(rest))
    if (
        _WEEK_WORD.search(rest)
        or _MONTH_WORD.search(rest)
        or len(hints) > 1
        or len(_YEAR_WORD.findall(rest)) != len(hints)
    ):
        return None
    years_ahead = int(bool(hints[0]["next"])) if hints else None
    weekdays = {_WEEKDAYS.index(m[0].lower()) for m in _FULL_WEEKDAY.finditer(rest)}
    weekdays |= {_ARABIC_WEEKDAYS[m[0]] for m in _ARABIC_WEEKDAY.finditer(rest)}
    resolved = {_in_year(s, years_ahead, weekdays, meeting_date) for s in stated}
    due = resolved.pop() if len(resolved) == 1 else None
    if due is None:
        return None
    other_days = {int(m["day"]) for m in days} - {due.day}
    other_months = {_MONTHS[m[0].lower().rstrip(".")] for m in _ANY_MONTH.finditer(rest)}
    return None if other_days or other_months - {due.month} else due


def _in_year(
    s: _Stated, years_ahead: int | None, weekdays: set[int], meeting_date: date
) -> date | None:
    """``s`` in the year said with it, else the one "next/this year" gives, else the meeting's
    year, rolled to the next year when that falls more than a month before the meeting. The
    roll is an inference, so a weekday said in the phrase that does not fall on it undoes it;
    so does one that does not fall on a date said with its year ("Sunday 27 September 2027").

    Four digits after the day and month are the year only within a few years of the meeting's
    (``_YEARS_NEAR``); well away from it, digits that read as a 24-hour time are one ("by 30
    September 1700", "2 October 2359") and the year is inferred; anything else ("2 October
    2033", "30 September 1999") is not resolved."""
    said = weekdays | ({s.weekday} if s.weekday is not None else set())
    year = s.year
    if s.four is not None:
        offset = int(s.four) - meeting_date.year
        if offset in _YEARS_NEAR:
            year = int(s.four)
        elif abs(offset) <= _YEARS_OR_TIME or not _CLOCK.fullmatch(s.four):
            return None
    if year is not None:
        due = _safe_date(year, s.month, s.day) if years_ahead is None else None
        return due if due and not said - {due.weekday()} else None
    if years_ahead is not None:
        return _safe_date(meeting_date.year + years_ahead, s.month, s.day)
    resolved = _safe_date(meeting_date.year, s.month, s.day)
    if resolved is None or (meeting_date - resolved).days <= 31:
        return resolved
    rolled = _safe_date(meeting_date.year + 1, s.month, s.day)
    return rolled if rolled and not said - {rolled.weekday()} else None


def _bare_day(rest: str, days: list[re.Match[str]], meeting_date: date) -> date | None:
    """One day of the month, in the month ``_month_of`` gives, checked against a weekday said
    with it or describing it ("the 2nd of next month, a Monday"): the month is inferred, so
    "Monday the 21st" that lands on a Wednesday is a guess and the reviewer gets the phrase
    instead. Two different days ("Wednesday the 30th or Tuesday the 29th"), or a month, week or
    year named anywhere ("in November, on the 10th"), are never resolved.

    A day up to 12 said without its suffix after a weekday ("Friday 2") may be the hour, so it
    is kept only when it is also the coming such weekday: both readings give the same date."""
    if len({int(m["day"]) for m in days}) != 1 or any(
        pattern.search(rest) for pattern in (_ANY_MONTH, _NAME_OR_MONTH, _WEEK_WORD, _YEAR_WORD)
    ):
        return None
    day = int(days[0]["day"])
    year_month = _month_of(day, rest, days, meeting_date)
    resolved = _safe_date(*year_month, day) if year_month else None
    stated = {
        _WEEKDAY_FORMS[m["weekday"].lower()]
        for pattern in (_WEEKDAY_BEFORE_DAY, _WEEKDAY_AFTER_DAY, _A_WEEKDAY, _PLAIN_DAY)
        for m in pattern.finditer(rest)
    }
    if resolved is None or len(stated) > 1 or (stated and resolved.weekday() not in stated):
        return None
    if day <= 12 and any(m.re is _PLAIN_DAY for m in days):
        ahead = (resolved.weekday() - meeting_date.weekday()) % 7
        if not ahead or resolved != meeting_date + timedelta(days=ahead):
            return None
    return resolved


def _month_of(
    day: int, rest: str, days: list[re.Match[str]], meeting_date: date
) -> tuple[int, int] | None:
    """``(year, month)`` for a bare day: "this month" or "next month" said with it; else the
    meeting's month, or the next one when the day has passed. ``None`` for any other talk of
    months ("the 10th, and the rest next month") and for the meeting's own day of the month,
    which may be today or a month on."""
    after = [_MONTH_AFTER_DAY.match(rest, d.end()) for d in days]
    before = [_MONTH_BEFORE_DAY.search(rest, 0, d.start()) for d in days]
    said = [m for m in after + before if m]
    if len(said) > 1 or len(_MONTH_WORD.findall(rest)) != len(said):
        return None
    if said:
        months_ahead = 0 if said[0]["this"] else 1
    elif day == meeting_date.day:
        return None
    else:
        months_ahead = int(day < meeting_date.day)
    index = meeting_date.month - 1 + months_ahead
    return meeting_date.year + index // 12, index % 12 + 1


def _weekday_only(rest: str, meeting_date: date) -> date | None:
    """One full weekday name, as the next such day after the meeting ("by Thursday").

    With "next week" or "this week" ("Friday next week", "next week on Tuesday"), or as "next
    Thursday", only when Sunday-start and Monday-start weeks agree (``_in_week``). ``None`` for
    the meeting's own weekday (today, or a week on), any other week offset ("Friday week", "a
    week on Friday", "Friday after next"), a weekday fixed by something else ("the Friday
    before the board") or any one of them ("on a Friday"), two weekdays, with a month or a
    year ("first Monday of October"), or with a number beside it that is not a time ("Thursday
    1/10", "27, Sunday"): that is a day, and never the next such weekday.
    """
    named = {_WEEKDAYS.index(m[0].lower()) for m in _FULL_WEEKDAY.finditer(rest)}
    if len(named) != 1 or any(
        pattern.search(rest)
        for pattern in (_ANY_MONTH, _MONTH_WORD, _YEAR_WORD, _NUMBER_BY_WEEKDAY)
    ):
        return None
    (weekday,) = named
    if weeks := _WEEK_WORD.findall(rest):
        m = _WEEKDAY_THEN_WEEK.search(rest) or _WEEK_THEN_WEEKDAY.search(rest)
        if m is None or len(weeks) > 1:
            return None
        return _in_week(weekday, meeting_date, next_week=m["which"].lower() != "this")
    if _ANCHORED_WEEKDAY.search(rest) or any(
        m["intro"].lower() == "a" for m in _A_WEEKDAY.finditer(rest)
    ):
        return None
    ahead = (weekday - meeting_date.weekday()) % 7
    if _NEXT_WEEKDAY.search(rest):  # the next one, or the one in next week?
        coming = meeting_date + timedelta(days=ahead or 7)
        return coming if coming == _in_week(weekday, meeting_date, next_week=True) else None
    return meeting_date + timedelta(days=ahead) if ahead else None


def _in_week(weekday: int, meeting_date: date, next_week: bool) -> date | None:
    """``weekday`` in the meeting's week or the next, when a Sunday-start week (the Gulf working
    week) and a Monday-start week (ISO) give the same day and it falls after the meeting."""
    found = set()
    for first in (6, 0):  # Sunday, Monday
        start = meeting_date - timedelta(days=(meeting_date.weekday() - first) % 7)
        found.add(start + timedelta(days=(7 if next_week else 0) + (weekday - first) % 7))
    day = found.pop() if len(found) == 1 else None
    return day if day and day > meeting_date else None


def source_language(refs: list[Ref], by_id: dict[str, Segment]) -> Literal["en", "ar", "mixed"]:
    """The language of the cited segments: ``mixed`` when they disagree or are code-switched."""
    langs = {by_id[r.segment_id].language for r in refs if r.segment_id in by_id}
    if "mixed" in langs or {"en", "ar"} <= langs:
        return "mixed"
    return "ar" if langs == {"ar"} else "en"


PLACEHOLDER_SEGMENT = "S0000"


def placeholder_ref(quote: str) -> Ref:
    """A citation to no segment, carrying the model's quote, for an item whose citations were
    all malformed: it lets the item be stored as a real ``Decision``/``ActionItem`` (which
    require at least one ref) so a reviewer can restore it from its flag."""
    return Ref(
        segment_id=PLACEHOLDER_SEGMENT, start_s=0, end_s=0, speaker="unknown", quote=quote[:240]
    )


def uncited_flag(item: BaseModel, label: str, priority: int) -> Flag:
    """Flag for a body item whose citations were all malformed (it never reached the body).

    ``item`` is the assembled ``Decision``/``ActionItem`` (with its id and a placeholder ref),
    so ``item_json`` is exactly what the review server's restore route validates.
    """
    return Flag(
        kind="uncited_item_removed",
        detail=f"{label} — removed: no well-formed citation",
        item_json=json.dumps(item.model_dump(mode="json"), ensure_ascii=False),
        priority=priority,
    )


def assemble_items(
    merged: MergedFindings, by_id: dict[str, Segment], meeting_date: date
) -> tuple[Items, list[Flag]]:
    """Body lists (decisions, actions, open_questions, risks) from merged findings.

    Decisions and actions whose ids are all malformed are returned as priority-1 flags instead;
    their JSON is a complete, restorable item (id ``D<n>``/``A<n>``, placeholder ref).
    """
    out: Items = {"decisions": [], "actions": [], "open_questions": [], "risks": []}
    flags: list[Flag] = []
    for i, d in enumerate(merged.decisions, 1):
        refs = resolve_refs(d.refs, d.quote, by_id)
        item = Decision(
            id=f"D{i}",
            statement=d.statement,
            kind=d.kind,
            decided_by=d.decided_by,
            dissent_or_conditions=_opt(d.dissent_or_conditions),
            refs=refs or [placeholder_ref(d.quote)],
        )
        if not refs:
            flags.append(uncited_flag(item, d.statement, 1))
            continue
        out["decisions"].append(item)
    for i, a in enumerate(merged.actions, 1):
        refs = resolve_refs(a.refs, a.quote, by_id)
        item = ActionItem(
            id=f"A{i}",
            description=a.description,
            owner=_person(a.owner),
            owner_confidence=a.owner_confidence if _person(a.owner) else "unknown",
            due_date=due_date_from_text(due_phrase(a.due_text), meeting_date),
            due_text=due_phrase(a.due_text),
            source_language=source_language(refs, by_id),
            refs=refs or [placeholder_ref(a.quote)],
        )
        if not refs:
            flags.append(uncited_flag(item, a.description, 1))
            continue
        out["actions"].append(item)
    for i, q in enumerate(merged.questions, 1):
        out["open_questions"].append(
            OpenQuestion(
                id=f"Q{i}",
                question=q.question,
                raised_by=_person(q.raised_by),
                owner=_person(q.owner),
                refs=resolve_refs(q.refs, None, by_id),
            )
        )
    for i, r in enumerate(merged.risks, 1):
        out["risks"].append(
            Risk(
                id=f"R{i}",
                description=r.description,
                severity=r.severity,
                owner=_person(r.owner),
                mitigation=_opt(r.mitigation),
                refs=resolve_refs(r.refs, None, by_id),
            )
        )
    return out, flags


def split_commitments(
    actions: list[ActionItem], by_id: dict[str, Segment]
) -> tuple[list[ActionItem], list[ActionItem]]:
    """``(mine, theirs)`` for one-to-one minutes: an action is the organiser's when any of its
    citations comes from a segment spoken by ``self`` (the mic track). Computed from the
    *verified* actions, so an action the verifier removed never lingers as a commitment."""
    mine = [
        a
        for a in actions
        if any(by_id[r.segment_id].speaker_kind == "self" for r in a.refs if r.segment_id in by_id)
    ]
    theirs = [a for a in actions if a not in mine]
    return mine, theirs


def topics_from(narrative: Narrative, by_id: dict[str, Segment]) -> list[TopicSummary]:
    """Topic summaries with their ids resolved against the transcript."""
    return [
        TopicSummary(
            title=t.title,
            summary=t.summary,
            key_points=t.key_points,
            refs=resolve_refs(t.refs, None, by_id),
        )
        for t in narrative.topics
    ]


def findings_from_minutes(m: Minutes) -> MergedFindings:
    """Rebuild reduce-stage findings from a minutes body (for narrative regeneration)."""

    def ids(refs: list[Ref]) -> list[str]:
        return [r.segment_id for r in refs]

    return MergedFindings.model_validate(
        {
            "decisions": [
                {
                    "statement": d.statement,
                    "kind": d.kind,
                    "decided_by": d.decided_by,
                    "dissent_or_conditions": d.dissent_or_conditions,
                    "refs": ids(d.refs),
                    "quote": d.refs[0].quote,
                }
                for d in m.decisions
            ],
            "actions": [
                {
                    "description": a.description,
                    "owner": a.owner,
                    "owner_confidence": a.owner_confidence,
                    "due_text": a.due_text,
                    "refs": ids(a.refs),
                    "quote": a.refs[0].quote,
                }
                for a in m.actions
            ],
            "questions": [
                {
                    "question": q.question,
                    "raised_by": q.raised_by,
                    "owner": q.owner,
                    "refs": ids(q.refs),
                }
                for q in m.open_questions
            ],
            "risks": [
                {
                    "description": r.description,
                    "severity": r.severity,
                    "owner": r.owner,
                    "mitigation": r.mitigation,
                    "refs": ids(r.refs),
                }
                for r in m.risks
            ],
            "key_points": [
                {"topic": t.title, "point": p, "refs": ids(t.refs)}
                for t in m.topics
                for p in t.key_points
            ],
        }
    )
