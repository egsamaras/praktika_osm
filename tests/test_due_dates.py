"""Spoken due phrases → calendar dates (``llm.assemble.due_date_from_text``).

The first real Ollama run on the synthetic speech recording produced ``by Thursday the 18th
of September`` (resolved to the wrong Thursday) and ``before the 25th of September`` (not
resolved at all), so explicit day-and-month phrases, English and Arabic, are pinned here.
"""

from __future__ import annotations

from datetime import date

import pytest

from praktika.llm.assemble import due_date_from_text

MEETING = date(2026, 9, 15)  # a Tuesday


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("2026-10-01", date(2026, 10, 1)),
        ("by Thursday the 18th of September", date(2026, 9, 18)),  # explicit day beats weekday
        ("before the 25th of September", date(2026, 9, 25)),
        ("September 29", date(2026, 9, 29)),
        ("on Monday the 21st", date(2026, 9, 21)),
        ("by the 3rd", date(2026, 10, 3)),  # bare ordinal: next such day after the meeting
        ("15 January", date(2027, 1, 15)),  # more than a month back rolls to next year
        ("قبل ٢٥ سبتمبر", date(2026, 9, 25)),  # Arabic-Indic digits and Arabic month
        ("by Thursday", date(2026, 9, 17)),
        ("inshallah by Friday", date(2026, 9, 18)),
        ("before October ManCom", None),  # no day stated: never guessed
        ("next week", None),
        ("", None),
        (None, None),
        ("31st of September", None),  # impossible date stays unresolved
    ],
)
def test_due_date_from_text(phrase: str | None, expected: date | None) -> None:
    assert due_date_from_text(phrase, MEETING) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # Meeting on Friday 25 September: the next 21st is Wednesday 21 October.
        ("I will circulate the final numbers on Monday the 21st", None),
        ("by Wednesday the 21st", date(2026, 10, 21)),
        ("by the 21st", date(2026, 10, 21)),
        ("by Monday the 28th", date(2026, 9, 28)),
    ],
)
def test_bare_day_must_agree_with_a_stated_weekday(phrase: str, expected: date | None) -> None:
    """An inferred month is checked against the weekday said with it; a mismatch is a guess.

    A regression on the synthetic recording: "Monday
    the 21st" was dated Wednesday 21 October.
    """
    assert due_date_from_text(phrase, date(2026, 9, 25)) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # Meeting on Friday 25 September: the next 21st is Wednesday 21 October.
        ("Monday the 21st", None),
        ("Monday 21st", None),
        ("Monday, the 21st", None),
        ("the 21st, a Monday", None),
        ("21st (Monday)", None),
        ("Mon the 21st", None),  # abbreviations count next to the day
        ("Weds the 21st", date(2026, 10, 21)),
        ("the 28th (Monday)", date(2026, 9, 28)),
        ("Wed the 30th", date(2026, 9, 30)),
        # A weekday that is not said with the day says nothing about it.
        ("circulate Monday's board pack by the 30th", date(2026, 9, 30)),
        ("send the draft Monday and the final version by the 30th", date(2026, 9, 30)),
        ("by the 30th, not Friday", date(2026, 9, 30)),
        ("by Friday, i.e. the 2nd", date(2026, 10, 2)),
        # Two dates, or an ordinal that is not a day, are never resolved.
        ("by Wednesday the 30th or Tuesday the 29th", None),
        ("the 2nd Friday of October", None),
        ("first Monday of October", None),
        ("21st Monday", None),
    ],
)
def test_only_an_adjacent_weekday_constrains_a_bare_day(phrase: str, expected: date | None) -> None:
    """Found in review of the weekday check: a weekday anywhere in the phrase vetoed the day
    ("Monday's board pack by the 30th" lost its date), and the ordinal in "the 2nd Friday of
    October" was read as the 2nd."""
    assert due_date_from_text(phrase, date(2026, 9, 25)) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # Meeting on Tuesday 15 September: the 21st is a Monday.
        ("Monday the 21st", date(2026, 9, 21)),
        ("Mon the 21st", date(2026, 9, 21)),
        ("the 21st, a Monday", date(2026, 9, 21)),
        ("21st (Monday)", date(2026, 9, 21)),
        ("Tuesday the 21st", None),
    ],
)
def test_adjacent_weekday_that_agrees_keeps_the_date(phrase: str, expected: date | None) -> None:
    assert due_date_from_text(phrase, MEETING) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("21st of May", date(2027, 5, 21)),
        ("21 May", date(2027, 5, 21)),
        ("21st May 2027", date(2027, 5, 21)),
        ("May 21", date(2027, 5, 21)),
        ("by 21 May.", date(2027, 5, 21)),
        ("Monday the 21st may be too late", None),
        ("the 21st may slip", None),
    ],
)
def test_may_is_a_month_only_in_a_date_position(phrase: str, expected: date | None) -> None:
    """Neither the 21st of May nor a firm 21st: "the 21st may slip"."""
    assert due_date_from_text(phrase, date(2026, 9, 25)) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("by Monday", date(2026, 9, 28)),
        ("Monday may be too late", date(2026, 9, 28)),
        ("by Monday in October", None),  # a month but no day: which Monday is a guess
        ("by Monday or Friday", None),
        ("by mon", None),  # abbreviations are ordinary words on their own
        ("sat with the team by sun", None),
        ("wed", None),
    ],
)
def test_weekday_alone(phrase: str, expected: date | None) -> None:
    assert due_date_from_text(phrase, date(2026, 9, 25)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Omar", "Omar"),
        ("unknown", None),
        ("SPEAKER_01", None),
        ("Room", None),
        (" ", None),
        ("", None),
        (None, None),
        ("TBD", None),
    ],
)
def test_person_drops_speaker_labels_and_placeholders(
    value: str | None, expected: str | None
) -> None:
    """A model that echoes the transcript's speaker label as the owner must not produce an
    action owned by "unknown"; such owners become None with owner_confidence "unknown"."""
    from praktika.llm.assemble import _person

    assert _person(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("by Thursday the 18th of September", "by Thursday the 18th of September"),
        ("before October ManCom", "before October ManCom"),
        ("قبل الخامس والعشرين من سبتمبر", "قبل الخامس والعشرين من سبتمبر"),
        ("next week", "next week"),
        ("please send me your view on which supplier we should shortlist", None),
        ("as discussed", None),
        ("", None),
        (None, None),
    ],
)
def test_due_phrase_requires_a_time_reference(value: str | None, expected: str | None) -> None:
    """A model output may put a request sentence into due_text when no deadline was spoken;
    only phrases with a time reference survive into the minutes."""
    from praktika.llm.assemble import due_phrase

    assert due_phrase(value) == expected


TUE_15_SEP, FRI_25_SEP, MON_28_DEC = date(2026, 9, 15), date(2026, 9, 25), date(2026, 12, 28)


@pytest.mark.parametrize(
    ("phrase", "on_15_sep", "on_25_sep"),
    [
        # A month named but not next to the day lost to the inferred month.
        ("November the 2nd", date(2026, 11, 2), date(2026, 11, 2)),
        ("March the 3rd", date(2027, 3, 3), date(2027, 3, 3)),
        ("December the 1st", date(2026, 12, 1), date(2026, 12, 1)),
        ("Nov. 10th", date(2026, 11, 10), date(2026, 11, 10)),
        ("Dec. 1st", date(2026, 12, 1), date(2026, 12, 1)),
        ("the 10th in November", date(2026, 11, 10), date(2026, 11, 10)),
        ("in November, on the 10th", None, None),
        # "next month" was ignored.
        ("the 20th of next month", date(2026, 10, 20), date(2026, 10, 20)),
        ("the 15th of next month", date(2026, 10, 15), date(2026, 10, 15)),
        ("the 28th next month", date(2026, 10, 28), date(2026, 10, 28)),
        # An ordinal that counts something was read as a day.
        ("the 2nd week of October", None, None),
        ("in the 3rd week of October", None, None),
        ("by the 4th quarter", None, None),
        ("in the 1st half of next year", None, None),
        ("at the 2nd MANCOM", None, None),
        ("before the 3rd board meeting", None, None),
        ("after the 2nd reading", None, None),
        ("1st thing Monday", date(2026, 9, 21), date(2026, 9, 28)),
        # A weekday with a week offset was read as the coming weekday.
        ("Friday next week", date(2026, 9, 25), date(2026, 10, 2)),
        ("Wednesday of next week", date(2026, 9, 23), date(2026, 9, 30)),
        ("Monday week", None, None),
        ("Friday week", None, None),
        ("a week on Friday", None, None),
        ("two weeks from Monday", None, None),
        ("the Monday after next", None, None),
        ("Friday after next", None, None),
        # A correction or a second date: the first date won.
        ("moved from 2 October to 9 October", None, None),
        ("not the 30th of September but the 7th of October", None, None),
        ("originally 2026-09-30, now 2026-10-07", None, None),
        ("draft by 25 September, final by 30 September", None, None),
    ],
)
def test_fuzzed_phrases_that_gave_a_date_never_stated(
    phrase: str, on_15_sep: date | None, on_25_sep: date | None
) -> None:
    """Found by fuzzing spoken due phrases after review: each gave a confident date that the
    phrase does not state, and the minutes show the date instead of the phrase. A date is
    returned only when the phrase states it; otherwise the reviewer sees the phrase."""
    assert due_date_from_text(phrase, TUE_15_SEP) == on_15_sep
    assert due_date_from_text(phrase, FRI_25_SEP) == on_25_sep


def _at(phrase: str, expected: tuple[date | None, date | None, date | None]) -> None:
    got = tuple(due_date_from_text(phrase, m) for m in (TUE_15_SEP, FRI_25_SEP, MON_28_DEC))
    assert got == expected, phrase


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("next Monday", (date(2026, 9, 21), date(2026, 9, 28), date(2027, 1, 4))),
        # Said early in the week, "next Friday" may be this Friday or next week's.
        ("next Friday", (None, date(2026, 10, 2), None)),
        ("next Wednesday", (None, date(2026, 9, 30), None)),
        ("next Tuesday", (date(2026, 9, 22), date(2026, 9, 29), None)),
        ("Friday next week", (date(2026, 9, 25), date(2026, 10, 2), date(2027, 1, 8))),
        ("next week Tuesday", (date(2026, 9, 22), date(2026, 9, 29), date(2027, 1, 5))),
        (
            "the Friday of the following week",
            (date(2026, 9, 25), date(2026, 10, 2), date(2027, 1, 8)),
        ),
        # Sunday-start (Gulf) and Monday-start (ISO) weeks put this Sunday in different weeks.
        ("next week on Sunday", (None, None, None)),
        ("Thursday this week", (date(2026, 9, 17), None, date(2026, 12, 31))),
        ("this Thursday", (date(2026, 9, 17), date(2026, 10, 1), date(2026, 12, 31))),
        # The meeting's own weekday: today, or a week on.
        ("Tuesday", (None, date(2026, 9, 29), date(2026, 12, 29))),
        ("by Monday", (date(2026, 9, 21), date(2026, 9, 28), None)),
        ("on a Friday", (None, None, None)),
        ("last Friday", (None, None, None)),
        ("every other Friday", (None, None, None)),
        ("the Friday before the board meeting", (None, None, None)),
        ("Friday fortnight", (None, None, None)),
        ("Friday next month", (None, None, None)),
        ("tomorrow or Friday", (None, None, None)),
    ],
)
def test_weekdays_resolve_only_when_every_reading_agrees(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    _at(phrase, expected)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("by the 2nd", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 1, 2))),
        # The meeting's own day of the month: today, or a month on.
        ("the 15th", (None, date(2026, 10, 15), date(2027, 1, 15))),
        ("the 28th", (date(2026, 9, 28), date(2026, 9, 28), None)),
        ("the 15th of this month", (date(2026, 9, 15), date(2026, 9, 15), date(2026, 12, 15))),
        ("the 5th of next month", (date(2026, 10, 5), date(2026, 10, 5), date(2027, 1, 5))),
        ("next month, the 12th", (date(2026, 10, 12), date(2026, 10, 12), date(2027, 1, 12))),
        ("Monday the 5th of next month", (date(2026, 10, 5), date(2026, 10, 5), None)),
        ("the 10th, and the rest next month", (None, None, None)),
        ("the 2nd of next month, a Monday", (None, None, None)),  # 2 October is a Friday
        ("the 3rd, which is a Saturday", (date(2026, 10, 3), date(2026, 10, 3), None)),
        ("Friday (the 3rd)", (None, None, None)),
        ("by the 30th at the latest", (date(2026, 9, 30), date(2026, 9, 30), date(2026, 12, 30))),
        ("the 30th-ish", (None, None, None)),
        ("the 1st of each month", (None, None, None)),
        ("every 2nd Tuesday", (None, None, None)),
        ("the 1st business day of October", (None, None, None)),
        ("the 2nd draft by the 30th", (None, None, None)),
        ("Jan, the 15th", (None, None, None)),  # a given name or the month
        ("by the 21st may be tight", (None, None, None)),
        ("the 1st next year", (None, None, None)),
    ],
)
def test_bare_days_take_only_the_month_said_with_them(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    _at(phrase, expected)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("by 5 January", (date(2027, 1, 5),) * 3),
        ("by 30 December", (date(2026, 12, 30),) * 3),
        ("Oct. 1st", (date(2026, 10, 1), date(2026, 10, 1), date(2027, 10, 1))),
        # Rolled to next year, 1 October is a Friday: the weekday said undoes the roll.
        ("Thursday, 1 October", (date(2026, 10, 1), date(2026, 10, 1), None)),
        ("الخميس ١ أكتوبر", (date(2026, 10, 1), date(2026, 10, 1), None)),
        ("Friday the 1st of January", (date(2027, 1, 1),) * 3),
        ("30 September 2027", (date(2027, 9, 30),) * 3),
        ("by the 30th of September next year", (date(2027, 9, 30),) * 3),
        ("Sept 30 deadline", (date(2026, 9, 30), date(2026, 9, 30), date(2027, 9, 30))),
        ("October 2nd half", (None, None, None)),
        ("5-6 October", (None, None, None)),
        ("October 5 or 6", (None, None, None)),
        ("between 5 and 9 October", (None, None, None)),
        ("5 October 27", (None, None, None)),
        ("by 30 September for the October board", (None, None, None)),
        ("2027-01-04 or 2027-01-05", (None, None, None)),
    ],
)
def test_dates_said_in_full(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    _at(phrase, expected)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("moved to 12 October", (date(2026, 10, 12), date(2026, 10, 12), date(2027, 10, 12))),
        ("moved from 12 October", (None, None, None)),
        ("pushed back from the 30th", (None, None, None)),
        ("brought forward to the 30th", (date(2026, 9, 30), date(2026, 9, 30), date(2026, 12, 30))),
        ("the 2nd, not Friday", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 1, 2))),
        ("Friday, not the 2nd", (None, None, None)),
        ("originally Friday, now the 2nd", (None, None, None)),
        ("after the 30th", (None, None, None)),
        ("not before the 30th", (None, None, None)),
        ("the day before Friday", (None, None, None)),
        ("tomorrow, the 17th", (None, None, None)),
        ("tomorrow, the 16th", (date(2026, 9, 16), None, None)),
        ("the 2nd, early Q4", (date(2026, 10, 2), date(2026, 10, 2), None)),
    ],
)
def test_corrections_starts_and_contradictions_are_never_resolved(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    _at(phrase, expected)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # A four-digit clock time after a date was read as the year ("1700-09-30").
        ("by 30 September 1700", (date(2026, 9, 30), date(2026, 9, 30), date(2027, 9, 30))),
        ("COB 1 October 1600 hrs", (date(2026, 10, 1), date(2026, 10, 1), date(2027, 10, 1))),
        ("Oct 2 1200", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 10, 2))),
        ("15th October 0900", (date(2026, 10, 15), date(2026, 10, 15), date(2027, 10, 15))),
        ("by 2 October 2359", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 10, 2))),
        ("30 Sep, 1700", (date(2026, 9, 30), date(2026, 9, 30), date(2027, 9, 30))),
        # Rolled to 2027, 27 September is a Monday: the weekday said undoes the roll.
        ("Sunday 27 September 1400", (date(2026, 9, 27), date(2026, 9, 27), None)),
        ("by 30 September at 2000", (date(2026, 9, 30), date(2026, 9, 30), date(2027, 9, 30))),
        ("Thursday 2000 hrs", (date(2026, 9, 17), date(2026, 10, 1), date(2026, 12, 31))),
        # A year is four digits near the meeting's; nearby but further off is never guessed.
        ("1 Oct 2027", (date(2027, 10, 1),) * 3),
        ("by 2 October 2033", (None, None, None)),
        ("30 September 1999", (None, None, None)),
        # A weekday said with a year must fall on the date (27 September 2027 is a Monday).
        ("Sunday 27 September 2027", (None, None, None)),
        ("Monday 27 September 2027", (date(2027, 9, 27),) * 3),
        ("on 2026-10-05 (Monday)", (date(2026, 10, 5),) * 3),
        ("2026-10-05 (Tuesday)", (None, None, None)),
    ],
)
def test_four_digits_after_a_date_are_a_year_only_near_the_meeting(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    """Fuzzing found: "by 30 September 1700" gave 1700-09-30 and "Sunday 27 September 1400"
    a Saturday in 1400: any four digits were the year, and a weekday was never checked once a
    year was present."""
    _at(phrase, expected)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # A day number said without its suffix was ignored and the next weekday returned.
        ("Sunday 27", (date(2026, 9, 27), date(2026, 9, 27), None)),
        ("by Wednesday 30", (date(2026, 9, 30), date(2026, 9, 30), date(2026, 12, 30))),
        ("Tuesday 29", (date(2026, 9, 29), date(2026, 9, 29), date(2026, 12, 29))),
        ("COB Thursday 24", (date(2026, 9, 24), None, None)),
        # "Friday 2" may be Friday at 2: kept only when the day is also the coming Friday.
        ("Friday 2", (None, None, None)),
        ("Friday 1", (None, None, date(2027, 1, 1))),
        ("Saturday 3", (None, None, None)),
        ("by Friday 9", (None, None, None)),
        ("Wednesday, 3 at the latest", (None, None, None)),
        ("Friday 2pm", (date(2026, 9, 18), None, date(2027, 1, 1))),
        ("Wednesday at 3", (date(2026, 9, 16), date(2026, 9, 30), date(2026, 12, 30))),
        # A numeric date or a number before the weekday is never the next such weekday.
        ("Thursday 1/10", (None, None, None)),
        ("Friday, 2/10", (None, None, None)),
        ("27, Sunday", (None, None, None)),
        ("Friday 2-3", (None, None, None)),
        ("Friday 9.10", (None, None, None)),  # 9 October (a Friday) or 9.10 in the morning
        ("Friday 5.30", (date(2026, 9, 18), None, date(2027, 1, 1))),
    ],
)
def test_a_day_number_beside_a_weekday_is_that_day_or_nothing(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    """Fuzzing found: "Sunday 27" said on 15 September gave Sunday 20 September."""
    _at(phrase, expected)


@pytest.mark.parametrize(
    "phrase",
    [
        # The day said is the anchor, not the deadline.
        "the Monday before the 2nd",
        "the Friday prior to the 30th",
        "the Thursday before 2 October",
        "the Monday following the 2nd",
        "the Sunday ahead of the 1st",
        "the Wednesday preceding 7 October",
        "the weekend before the 2nd",
        "the eve of the 2nd",
        "the Friday before the 16 October ALCO",
        "the night before the 2nd",
        "the working day preceding the 30th",
        "the Thursday prior to the board",
        # A weekday fixed by something left unsaid.
        "the Thursday prior",
        "the previous Thursday",
        "the following Thursday",
        "that Friday",
        "the Friday beforehand",
        "the 2nd, the Friday prior",
        # An hour offset from the day said (confirmation round).
        "48 hours before 2 October",
        "48h before 2 Oct",
        "48 hours prior to 2 October",
        "48 hours after the board on 2 October",
        "24 hours before the 2nd",
        "72 hours ahead of 5 October",
        "36 hrs before Friday",
        # Alternatives.
        "the 30th this month or next",
        "the 2nd or later",
        "by the 30th or the week after",
    ],
)
def test_anchored_days_and_alternatives_are_never_resolved(phrase: str) -> None:
    """Fuzzing found: "the Monday before the 2nd" gave Friday 2 October, and "the 30th this
    month or next" gave the 30th of this month."""
    _at(phrase, (None, None, None))


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("on or before the 2nd", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 1, 2))),
        (
            "by Thursday, before the board",
            (date(2026, 9, 17), date(2026, 10, 1), date(2026, 12, 31)),
        ),
        ("the morning of the 2nd", (date(2026, 10, 2), date(2026, 10, 2), date(2027, 1, 2))),
        ("by the 30th or thereabouts", (date(2026, 9, 30), date(2026, 9, 30), date(2026, 12, 30))),
        (
            "report by the 30th and next steps by Friday",
            (date(2026, 9, 30), date(2026, 9, 30), date(2026, 12, 30)),
        ),
        ("the coming Saturday", (date(2026, 9, 19), date(2026, 9, 26), date(2027, 1, 2))),
    ],
)
def test_what_the_anchor_and_alternative_checks_leave_alone(
    phrase: str, expected: tuple[date | None, date | None, date | None]
) -> None:
    _at(phrase, expected)
