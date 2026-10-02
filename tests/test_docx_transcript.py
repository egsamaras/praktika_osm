"""Tests for ``praktika.ingest.docx_transcript``."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from praktika.errors import PraktikaError
from praktika.ingest.docx_transcript import parse_teams_docx
from praktika.ingest.vtt import parse_teams_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MEETING = "M-20260916-a1b2"
ROOMS = ["AI Lab Meeting Room"]


def _write_docx(path: Path, paragraphs: list[str]) -> Path:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))
    return path


def test_parse_recap_docx() -> None:
    t = parse_teams_docx(FIXTURES / "synthetic_recap.docx", MEETING, ROOMS)
    assert t.source == "docx"
    assert t.engines == {"docx": "teams-recap"}
    assert t.redacted is False
    assert t.meeting_id == MEETING
    assert 55 <= len(t.segments) <= 75

    first = t.segments[0]
    assert first.id == "S0001"
    assert (first.speaker, first.speaker_kind) == ("F. Khalid", "identity")
    assert first.text.startswith("Good morning everyone")
    assert (first.track, first.engine, first.confidence) == ("docx", "teams-docx", None)

    # ids are dense and times are monotonic; every block ends where the next one starts
    assert [s.id for s in t.segments] == [f"S{i:04d}" for i in range(1, len(t.segments) + 1)]
    starts = [s.start for s in t.segments]
    assert starts == sorted(starts)
    assert all(s.end > s.start for s in t.segments)
    for a, b in zip(t.segments, t.segments[1:], strict=False):
        assert a.end <= b.start + 1e-9

    rooms = [s for s in t.segments if s.speaker_kind == "room"]
    assert len(rooms) == 1 and rooms[0].speaker == "Room"
    unknown = [s for s in t.segments if s.speaker_kind == "unknown"]
    assert len(unknown) == 1 and unknown[0].text.startswith("Sorry, the room dropped")

    # the Recap DOCX carries the same words as the VTT export of the same meeting
    vtt = parse_teams_vtt((FIXTURES / "synthetic_en.vtt").read_text("utf-8"), MEETING, ROOMS)
    assert " ".join(s.text for s in t.segments) == " ".join(s.text for s in vtt.segments)
    assert t.segments[-1].start > 20 * 60


def test_malformed_paragraphs_skipped(tmp_path: Path) -> None:
    path = _write_docx(
        tmp_path / "recap.docx",
        [
            "Transcript",  # title, no timestamp
            "Data team weekly - 16 September 2026",  # date line
            "",  # empty
            "R. Haddad   0:05",
            "First block, first paragraph.",
            "   ",  # whitespace-only paragraph inside a block is ignored
            "First block, second paragraph.",
            "L. Farouk   0:20",  # header with no text: dropped
            "Omar Nasser   0:31",
            "Second block.",
            "12:34",  # a bare timestamp is not a header; it is text of the current block
            "Unknown Speaker   1:02:03",  # h:mm:ss header, unattributed
            "Third block.",
        ],
    )
    t = parse_teams_docx(path, MEETING, [])
    assert [(s.speaker, s.speaker_kind, s.text) for s in t.segments] == [
        ("R. Haddad", "identity", "First block, first paragraph. First block, second paragraph."),
        ("Omar Nasser", "identity", "Second block. 12:34"),
        ("unknown", "unknown", "Third block."),
    ]
    assert [s.start for s in t.segments] == [5.0, 31.0, 3723.0]
    assert [s.end for s in t.segments][:2] == [31.0, 3723.0]
    assert t.segments[2].end > t.segments[2].start  # last block gets an estimated duration


def test_room_identity_and_empty_document(tmp_path: Path) -> None:
    path = _write_docx(
        tmp_path / "room.docx", ["AI Lab Meeting Room   0:02", "Hello from the lab."]
    )
    t = parse_teams_docx(path, MEETING, ROOMS)
    assert [(s.speaker, s.speaker_kind) for s in t.segments] == [("Room", "room")]

    empty = parse_teams_docx(_write_docx(tmp_path / "empty.docx", []), MEETING, ROOMS)
    assert empty.segments == []
    assert empty.language_profile() == {}


def test_missing_or_non_docx_raises(tmp_path: Path) -> None:
    with pytest.raises(PraktikaError, match="cannot read DOCX"):
        parse_teams_docx(tmp_path / "nope.docx", MEETING, ROOMS)
    not_docx = tmp_path / "text.docx"
    not_docx.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\n", encoding="utf-8")
    with pytest.raises(PraktikaError, match="cannot read DOCX"):
        parse_teams_docx(not_docx, MEETING, ROOMS)


def test_utterance_ending_in_a_clock_time_is_not_a_header(tmp_path: Path) -> None:
    """ "We meet again at 10:30" stays text of F. Khalid's block: the name part is speech
    (lower-case function words), so no bogus speaker at 630 s jumps the block out of order."""
    from praktika.ingest.docx_transcript import _blocks, looks_like_name

    assert _blocks(
        [
            "F. Khalid 0:05",
            "Good morning everyone.",
            "We meet again at 10:30",
            "and then finalise the budget.",
            "R. Haddad 0:20",
            "Agreed.",
        ]
    ) == [
        ("F. Khalid", 5.0, "Good morning everyone. We meet again at 10:30 and then finalise the budget."),  # noqa: E501
        ("R. Haddad", 20.0, "Agreed."),
    ]  # fmt: skip
    # a plausible name whose time runs backwards or jumps hours is speech too
    assert _blocks(
        [
            "Omar Nasser 12:00",
            "Lunch is at 11:30",
            "Budget 3:15:00",
            "L. Farouk 12:10",
            "Fine.",
        ]
    ) == [
        ("Omar Nasser", 720.0, "Lunch is at 11:30 Budget 3:15:00"),
        ("L. Farouk", 730.0, "Fine."),
    ]
    for name in (
        "F. Khalid",
        "S. Al-Mahmood",
        "Omar Nasser",
        "AI Lab Meeting Room",
        "Tom Brennan (Guest)",
        "Unknown Speaker",
        "عمر ناصر",
        "Rania bint Haddad",
    ):
        assert looks_like_name(name), name
    for name in ("We meet again at", "the call is at", "Meeting At Noon Or Later Still",
                 "F. Khalid,", "so Omar"):  # fmt: skip
        assert not looks_like_name(name), name
    path = _write_docx(
        tmp_path / "time.docx",
        ["F. Khalid 0:05", "We meet again at 10:30", "R. Haddad 0:20", "Agreed."],
    )
    t = parse_teams_docx(path, MEETING, [])
    assert [(s.speaker, s.start, s.text) for s in t.segments] == [
        ("F. Khalid", 5.0, "We meet again at 10:30"),
        ("R. Haddad", 20.0, "Agreed."),
    ]
