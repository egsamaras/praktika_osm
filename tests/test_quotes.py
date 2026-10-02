"""Quote verification guards (``praktika.llm.quotes``) and the citation handling that depends
on them: polarity/figure flips, quotes spanning consecutive segments, per-ref quotes."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_transcript

from praktika.llm import assemble, verify
from praktika.llm.quotes import cited_runs, quote_matches
from praktika.models import Attendee, Decision, DecisionDraft, MergedFindings, Ref, Segment

T = make_transcript("en", n=12)


def _segment(sid: str, start: float, text: str, speaker: str = "F. Khalid") -> Segment:
    return Segment(
        id=sid,
        start=start,
        end=start + 4.0,
        speaker=speaker,
        speaker_kind="label" if speaker.startswith("SPEAKER_") else "identity",
        language="en",
        text=text,
        confidence=0.9,
        track="file",
        engine="fake",
    )


def _split_transcript() -> tuple[object, dict[str, Segment]]:
    """Whisper-style short segments: a decision straddles S0004/S0005."""
    segs = [
        _segment("S0001", 0.0, "Good morning everyone, let us start with the pilot."),
        _segment("S0002", 4.0, "The prototype transcribes English and Arabic locally."),
        _segment("S0003", 8.0, "We will not proceed with the vendor until legal signs off."),
        _segment("S0004", 12.0, "Okay so we are agreed then, the pilot", "SPEAKER_01"),
        _segment("S0005", 16.0, "starts on the first of October for the data team only."),
        _segment("S0006", 20.0, "Thank you all, meeting closed."),
    ]
    t = T.model_copy(update={"segments": segs})
    return t, t.by_id()


def _decision(refs: list[Ref], statement: str = "Pilot starts in October") -> Decision:
    return Decision(id="D1", statement=statement, kind="approved", decided_by="Chair", refs=refs)


def _ref(seg: Segment, quote: str) -> Ref:
    return Ref(
        segment_id=seg.id, start_s=seg.start, end_s=seg.end, speaker=seg.speaker, quote=quote
    )


# --------------------------------------------------------------------------- polarity / figures


@pytest.mark.parametrize(
    ("quote", "text"),
    [
        ("we will proceed with the vendor", "we will not proceed with the vendor until legal"),
        ("Omar will own the migration", "Omar will not own the migration, Layla will"),
        ("we will not proceed with the vendor", "we will proceed with the vendor once legal signs"),
        ("we don't keep the audio after approval", "we do keep the audio after approval, a week"),
        ("إننا نوافق على الميزانية الجديدة", "قال إننا لن نوافق على الميزانية الجديدة هذا العام"),
        ("the pilot starts on 30 October", "Agreed. The pilot starts on 20 October, limited."),
        ("budget of 5 million for phase one", "the budget of 3 million for phase one was approved"),
    ],
)  # fmt: skip
def test_negation_or_number_flip_never_verifies(quote: str, text: str) -> None:
    assert not quote_matches(quote, text)


@pytest.mark.parametrize(
    ("quote", "text"),
    [
        ("we will not proceed with the vendor", "we will not proceed with the vendor until legal"),
        ("the pilot starts on 20 October", "Agreed. The pilot starts on 20 October, limited."),
        ("لن نوافق على الميزانية الجديدة", "قال إننا لن نوافق على الميزانية الجديدة هذا العام"),
        ("budget line is BHD 250,000", "The budget line is BHD 250,000 for the first phase."),
    ],
)  # fmt: skip
def test_faithful_quotes_with_negations_and_numbers_still_verify(quote: str, text: str) -> None:
    assert quote_matches(quote, text)


def test_flipped_polarity_decision_is_removed_by_verifier(roster: list[Attendee]) -> None:
    t, by_id = _split_transcript()
    seg = by_id["S0003"]
    from helpers_foundation import minutes as base_minutes

    m = base_minutes(
        decisions=[_decision([_ref(seg, "we will proceed with the vendor")], "Vendor approved")],
        actions=[], open_questions=[], risks=[], topics=[], follow_ups=[],
    )  # fmt: skip
    out = verify.apply(m, t, roster)
    assert out.decisions == []
    removed = [f for f in out.flags if f.kind == "uncited_item_removed"]
    assert len(removed) == 1 and removed[0].priority == 1
    assert "Vendor approved" in removed[0].detail


# --------------------------------------------------------------------------- consecutive runs


def test_cited_runs_joins_consecutive_segments_only() -> None:
    _, by_id = _split_transcript()
    segs = list(by_id.values())
    runs = cited_runs(["S0005", "S0004", "S0001", "S9999"], segs)
    joined = by_id["S0004"].text + " " + by_id["S0005"].text
    assert runs == {"S0004": joined, "S0005": joined}
    assert cited_runs(["S0001", "S0003"], segs) == {}
    assert cited_runs(["S0002"], segs) == {}


def test_quote_spanning_two_segments_keeps_both_refs(roster: list[Attendee]) -> None:
    """A genuine quote that straddles a segment boundary verifies against the joined text
    instead of being removed with a blocking flag."""
    t, by_id = _split_transcript()
    quote = "we are agreed then, the pilot starts on the first of October"
    assert not quote_matches(quote, by_id["S0004"].text)
    assert not quote_matches(quote, by_id["S0005"].text)
    from helpers_foundation import minutes as base_minutes

    refs = [_ref(by_id["S0004"], quote), _ref(by_id["S0005"], quote)]
    m = base_minutes(
        decisions=[_decision(refs)], actions=[], open_questions=[], risks=[], topics=[],
        follow_ups=[],
    )  # fmt: skip
    out = verify.apply(m, t, roster)
    assert [r.segment_id for r in out.decisions[0].refs] == ["S0004", "S0005"]
    assert not any(f.kind == "uncited_item_removed" for f in out.flags)
    # the unmapped speaker of S0004 is still surfaced to the reviewer
    assert any(f.kind == "unresolved_speaker" and "SPEAKER_01" in f.detail for f in out.flags)


def test_quote_spanning_non_consecutive_segments_is_not_evidence(roster: list[Attendee]) -> None:
    """A citation only counts when the quote is verbatim in the cited segment, its run of
    cited neighbours, or the cited segment joined with one immediate neighbour. Here the
    quote straddles S0004/S0005: the S0002 citation is dropped (two segments away) while the
    S0005 citation survives through its S0004 neighbour."""
    t, by_id = _split_transcript()
    quote = "we are agreed then, the pilot starts on the first of October"
    from helpers_foundation import minutes as base_minutes

    refs = [_ref(by_id["S0002"], quote), _ref(by_id["S0005"], quote)]
    m = base_minutes(
        decisions=[_decision(refs)], actions=[], open_questions=[], risks=[], topics=[],
        follow_ups=[],
    )  # fmt: skip
    out = verify.apply(m, t, roster)
    assert [r.segment_id for r in out.decisions[0].refs] == ["S0005"]
    # A quote whose halves are two segments apart is not evidence for either citation.
    far = f"{by_id['S0002'].text} {by_id['S0005'].text}"
    m2 = base_minutes(
        decisions=[_decision([_ref(by_id["S0005"], far)])], actions=[], open_questions=[],
        risks=[], topics=[], follow_ups=[],
    )  # fmt: skip
    out2 = verify.apply(m2, t, roster)
    assert out2.decisions == []
    assert any(f.kind == "uncited_item_removed" and f.priority == 1 for f in out2.flags)


# --------------------------------------------------------------------------- per-ref quotes


def test_resolve_refs_attaches_quote_only_to_its_segment() -> None:
    """A supporting citation carries its own text, not the neighbour's quote, so the verifier
    keeps it and its advisories (unresolved speaker, low confidence) still fire."""
    _, by_id = _split_transcript()
    quote = by_id["S0005"].text
    refs = assemble.resolve_refs(["S0004", "S0005"], quote, by_id)
    assert [(r.segment_id, r.quote) for r in refs] == [
        ("S0004", by_id["S0004"].text),
        ("S0005", quote),
    ]
    spanning = "we are agreed then, the pilot starts on the first of October"
    refs = assemble.resolve_refs(["S0004", "S0005"], spanning, by_id)
    assert [r.quote for r in refs] == [spanning, spanning], "a run carries the whole quote"
    # a quote nothing contains stays on every ref so the verifier removes the item
    bogus = "words the transcript never contained at any point whatsoever"
    refs = assemble.resolve_refs(["S0004", "S0005"], bogus, by_id)
    assert [r.quote for r in refs] == [bogus, bogus]
    assert assemble.resolve_refs(["S0004"], None, by_id)[0].quote == by_id["S0004"].text


def test_multi_ref_item_keeps_supporting_citations_after_verification(
    roster: list[Attendee],
) -> None:
    t, by_id = _split_transcript()
    merged = MergedFindings(
        decisions=[
            DecisionDraft(
                statement="Pilot starts on the first of October",
                kind="approved",
                decided_by="Chair",
                refs=["S0004", "S0005"],
                quote=by_id["S0005"].text,
            )
        ]
    )
    items, flags = assemble.assemble_items(merged, by_id, date(2026, 9, 16))
    assert flags == []
    from helpers_foundation import minutes as base_minutes

    m = base_minutes(
        decisions=items["decisions"], actions=[], open_questions=[], risks=[], topics=[],
        follow_ups=[],
    )  # fmt: skip
    out = verify.apply(m, t, roster)
    assert [r.segment_id for r in out.decisions[0].refs] == ["S0004", "S0005"]
    assert any(f.kind == "unresolved_speaker" and "SPEAKER_01" in f.detail for f in out.flags)
