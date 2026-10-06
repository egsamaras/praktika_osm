"""SQLite store, search and index rules."""

from __future__ import annotations

import os
import sqlite3
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import FROZEN_NOW, make_transcript
from helpers_foundation import audit_event, consent_kwargs
from helpers_foundation import minutes as base_minutes

from praktika.llm.prompts import TEMPLATES
from praktika.models import (
    ActionItem,
    Attendee,
    Classification,
    ConsentRecord,
    LanguageMode,
    MancomMinutes,
    Meeting,
    MeetingState,
    MeetingType,
    OneToOneMinutes,
    Platform,
    Ref,
    Review,
    ReviewItem,
)
from praktika.store import db, search
from praktika.store.repo import GENESIS_HASH, SqliteStore

NOW = datetime.fromisoformat(FROZEN_NOW)


def meeting(mid: str = "M-20260916-a1b2", **over: Any) -> Meeting:
    kw: dict[str, Any] = {
        "id": mid,
        "title": "Data team weekly",
        "meeting_type": MeetingType.general,
        "classification": Classification.internal,
        "language_mode": LanguageMode.en,
        "platform": Platform.teams,
        "started_at": NOW,
        "organiser": "f.khalid@acme.test",
        "roster": [
            Attendee(name="F. Khalid", aliases=["Faisal", "فيصل"], upn="f.khalid@acme.test"),
            Attendee(name="Omar Nasser", aliases=["Omar"]),
        ],
    }
    kw.update(over)
    return Meeting(**kw)


def action(aid: str, owner: str | None, description: str = "Do the thing") -> ActionItem:
    return ActionItem(
        id=aid,
        description=description,
        owner=owner,
        owner_confidence="explicit",
        due_date=None,
        due_text="soon",
        source_language="en",
        refs=[Ref(segment_id="S0001", start_s=0, end_s=9, speaker="F. Khalid", quote="q")],
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "praktika.db")
    yield s
    s.close()


def test_round_trips(store: SqliteStore, tmp_path: Path) -> None:
    m = meeting(tags={"weekly"}, chair="F. Khalid")
    store.save_meeting(m)
    assert store.get_meeting(m.id) == m
    assert store.get_meeting("M-20260916-ffff") is None
    store.set_state(m.id, MeetingState.drafting)
    assert store.get_meeting(m.id).state is MeetingState.drafting
    with pytest.raises(KeyError):
        store.set_state("M-20260916-ffff", MeetingState.drafting)

    consent = ConsentRecord(**consent_kwargs())
    store.save_consent(consent)
    assert store.get_consent(m.id) == consent

    media_id = store.save_media(
        m.id, tmp_path / "x.wav", "ab" * 32, kind="audio", delete_after=NOW + timedelta(hours=24)
    )
    rec = store.list_media(m.id)[0]
    assert rec.id == media_id and rec.path == tmp_path / "x.wav" and rec.deleted_at is None
    assert rec.delete_after == NOW + timedelta(hours=24)

    t = make_transcript("mixed", n=6)
    store.save_transcript(t, delete_after=None)
    assert store.get_transcript(m.id) == t and store.get_transcript("M-20260916-ffff") is None

    store.save_vault(m.id, b"\x00blob")
    assert store.get_vault(m.id) == b"\x00blob"
    store.save_vault(m.id, b"new")
    assert store.get_vault(m.id) == b"new"

    mins = base_minutes()
    assert store.save_minutes(mins) == 1
    assert store.latest_minutes(m.id) == mins
    item = ReviewItem(
        item_id="D1",
        action="accept",
        reason_code="ok",
        before=None,
        after=None,
        by="f.khalid",
        at=NOW,
    )
    store.save_review_item(m.id, 1, item)
    assert store.list_review_items(m.id, 1) == [item]

    store.set_hold(m.id, True, "litigation", "dpo")
    assert store.get_meeting(m.id).legal_hold is True
    store.set_hold(m.id, False, "released", "dpo")
    assert store.get_meeting(m.id).legal_hold is False
    # Updating a meeting must never cascade-delete its children (REPLACE would).
    store.set_state(m.id, MeetingState.approved)
    assert store.get_consent(m.id) == consent and store.list_media(m.id)[0].id == media_id
    assert store.get_transcript(m.id) == t and store.get_vault(m.id) == b"new"
    assert store.latest_minutes(m.id) == mins and store.list_review_items(m.id, 1) == [item]


def test_minutes_subclasses_round_trip(store: SqliteStore) -> None:
    store.save_meeting(meeting())
    mancom = MancomMinutes(**base_minutes().model_dump(), escalations_to_board=["GPU paper"])
    mancom = mancom.model_copy(update={"meeting_type": MeetingType.mancom})
    store.save_minutes(mancom)
    got = store.latest_minutes(mancom.meeting_id)
    assert isinstance(got, MancomMinutes) and got.escalations_to_board == ["GPU paper"]
    one = OneToOneMinutes(**base_minutes().model_dump(), my_commitments=[action("A9", "F. Khalid")])
    one = one.model_copy(update={"meeting_type": MeetingType.one_to_one, "version": 2})
    store.save_minutes(one)
    got = store.get_minutes(one.meeting_id, 2)
    assert isinstance(got, OneToOneMinutes) and got.private and got.my_commitments[0].id == "A9"


APPROVED = Review(
    status="approved", reviewer="F. Khalid", reviewer_source="session", reviewed_at=NOW
)
GENERAL = TEMPLATES[MeetingType.general]


def approved(**over: Any) -> Any:
    """Approved minutes: only these are ever indexed for search."""
    return base_minutes(review=APPROVED, **over)


def test_fts_excludes_restricted_and_private(store: SqliteStore) -> None:
    ids = ["M-20260916-0001", "M-20260916-0002", "M-20260916-0003", "M-20260916-0004"]
    store.save_meeting(meeting(ids[0]))
    store.save_meeting(meeting(ids[1], classification=Classification.restricted))
    store.save_meeting(meeting(ids[2], meeting_type=MeetingType.one_to_one, private=True))
    store.save_meeting(meeting(ids[3], private=True))
    common = "The vendor shortlist for the Arabic bake-off"
    draft = base_minutes(meeting_id=ids[0], summary=common)
    assert search.index_minutes(store, draft, GENERAL) is False
    assert search.search(store, "vendor shortlist") == [], "unreviewed drafts are not searchable"
    assert search.index_minutes(store, approved(meeting_id=ids[0], summary=common), GENERAL) is True
    assert (
        search.index_minutes(
            store,
            approved(meeting_id=ids[1], summary=common, classification=Classification.restricted),
            GENERAL,
        )
        is False
    )
    assert (
        search.index_minutes(
            store,
            approved(meeting_id=ids[2], summary=common, meeting_type=MeetingType.one_to_one),
            TEMPLATES[MeetingType.one_to_one],
        )
        is False
    )
    assert search.index_minutes(store, approved(meeting_id=ids[3], summary=common), GENERAL) is True
    hits = search.search(store, "vendor shortlist")
    assert [h.meeting_id for h in hits] == [ids[0]]
    hits = search.search(store, "vendor shortlist", include_private=True)
    assert sorted(h.meeting_id for h in hits) == [ids[0], ids[3]]
    assert search.search(store, "   ") == []
    assert search.search(store, "nothing-matches-here") == []
    # Re-indexing a now-restricted meeting removes the stale row.
    search.index_minutes(
        store,
        approved(meeting_id=ids[0], summary=common, classification=Classification.restricted),
        GENERAL,
    )
    assert search.search(store, "vendor shortlist") == []


def test_arabic_search_normalised(store: SqliteStore) -> None:
    mid = "M-20260916-0009"
    store.save_meeting(meeting(mid))
    summary = "تمّت الموافقة على الإشعار قبل الخميس، والميزانية ٢٥٠ ألف دينار."
    search.index_minutes(store, approved(meeting_id=mid, summary=summary), GENERAL)
    for q in ("الاشعار", "الإشعار", "الموافقه", "250", "٢٥٠", "تمت"):
        assert [h.meeting_id for h in search.search(store, q)] == [mid], q
    hit = search.search(store, "الاشعار")[0]
    assert hit.title and hit.rank <= 0 and "[" in hit.snippet
    assert search.normalise_ar("أَحْمَد إلى الْبَنْك") == "احمد الي البنك"
    assert search.normalise_ar("Notice ٢٥٠") == "notice 250"
    assert search.fts_query('a "b" c') == '"a" "b" "c"'


def test_open_actions_across_meetings(store: SqliteStore) -> None:
    a, b = "M-20260916-00aa", "M-20260916-00bb"
    store.save_meeting(meeting(a, title="First", started_at=NOW - timedelta(days=1)))
    store.save_meeting(meeting(b, title="Second"))
    store.save_minutes(
        approved(meeting_id=a, actions=[action("A1", "Omar Nasser"), action("A2", "R. Haddad")])
    )
    store.save_minutes(base_minutes(meeting_id=b, actions=[action("A1", "omar nasser")]))
    assert {o.meeting_id for o in store.open_actions()} == {a}, "drafts are not on the register"
    store.set_review_status(approved(meeting_id=b, actions=[action("A1", "omar nasser")]))
    store.save_review_item(
        a,
        1,
        ReviewItem(
            item_id="A2",
            action="reject",
            reason_code="dup",
            before="x",
            after=None,
            by="me",
            at=NOW,
        ),
    )
    got = store.open_actions()
    assert [(o.meeting_id, o.action.id) for o in got] == [(b, "A1"), (a, "A1")]
    assert got[0].meeting_title == "Second"
    assert [(o.meeting_id, o.action.id) for o in store.open_actions(owner="Omar Nasser")] == [
        (b, "A1"),
        (a, "A1"),
    ]
    assert store.open_actions(owner="R. Haddad") == []
    # A newer version supersedes; a discarded version is ignored.
    store.save_minutes(approved(meeting_id=b, version=2, actions=[action("A7", "L. Farouk")]))
    assert [o.action.id for o in store.open_actions() if o.meeting_id == b] == ["A7"]
    discarded = base_minutes(
        meeting_id=b, version=3, review=Review(status="discarded"), actions=[action("A8", "X")]
    )
    store.save_minutes(discarded)
    assert [o.action.id for o in store.open_actions() if o.meeting_id == b] == []
    # A private (one-to-one) meeting never feeds the cross-meeting register.
    c = "M-20260916-00cc"
    store.save_meeting(meeting(c, title="Private", private=True))
    store.save_minutes(approved(meeting_id=c, actions=[action("A1", "Omar Nasser")]))
    assert c not in {o.meeting_id for o in store.open_actions()}


def test_minutes_versioning(store: SqliteStore) -> None:
    store.save_meeting(meeting())
    m1 = base_minutes(summary="first")
    assert store.save_minutes(m1) == 1
    assert store.save_minutes(m1.model_copy(update={"summary": "second"})) == 2
    assert store.save_minutes(base_minutes(version=5, summary="fifth")) == 5
    assert store.save_minutes(base_minutes(version=2, summary="clash")) == 6
    assert store.latest_minutes(m1.meeting_id).summary == "clash"
    assert store.get_minutes(m1.meeting_id, 1).summary == "first"
    assert store.get_minutes(m1.meeting_id, 2).summary == "second"
    assert store.get_minutes(m1.meeting_id, 6).summary == "clash"
    assert store.latest_minutes(m1.meeting_id).version == 6
    assert store.get_minutes(m1.meeting_id, 3) is None
    approved = store.get_minutes(m1.meeting_id, 2).model_copy(
        update={"review": Review(status="approved", reviewer="me", reviewed_at=NOW)}
    )
    store.set_review_status(approved)
    assert store.get_minutes(m1.meeting_id, 2).review.status == "approved"
    assert store.conn.execute("SELECT status FROM minutes WHERE version = 2").fetchone()[0] == (
        "approved"
    )
    with pytest.raises(KeyError):
        store.set_review_status(base_minutes(version=42))


def test_dsar_find_by_participant(store: SqliteStore) -> None:
    a, b, c = "M-20260916-d001", "M-20260916-d002", "M-20260916-d003"
    store.save_meeting(meeting(a))
    store.save_meeting(meeting(b, roster=[Attendee(name="Layla Farouk", aliases=["ليلى"])]))
    store.save_meeting(meeting(c, roster=[Attendee(name="T. Brennan")]))
    t = make_transcript("en", n=4, meeting_id=c)
    t = t.model_copy(
        update={"segments": [s.model_copy(update={"speaker": "Rania Haddad"}) for s in t.segments]}
    )
    store.save_transcript(t, delete_after=None)
    assert store.dsar_find("khalid") == [a]
    assert store.dsar_find("فيصل") == [a]
    # F. Khalid organised all three meetings: as organiser they are a data subject in each
    assert sorted(store.dsar_find("f.khalid@acme.test")) == [a, b, c]
    assert store.dsar_matches("f.khalid@acme.test")[0][1] == "organiser"
    store.save_meeting(meeting("M-20260916-d004", organiser="l.farouk@acme.test"))
    assert store.dsar_find("L.Farouk@acme.test") == ["M-20260916-d004"]
    # a full name in a title is found; a single word never is ('May', 'Ali', 'Bond')
    store.save_meeting(meeting("M-20260916-d005", title="Credit review - Fatima Al Zayani"))
    store.save_meeting(meeting("M-20260916-d006", title="ALCO May 2026"))
    assert ("M-20260916-d005", "title") in store.dsar_matches("Fatima al-Zayani")
    assert "M-20260916-d006" not in store.dsar_find("May")
    assert store.dsar_find("Layla") == [b]
    assert store.dsar_find("ليلى") == [b]
    assert store.dsar_find("rania haddad") == [c]
    assert store.dsar_find("nobody") == []
    assert store.dsar_find("") == []


def test_db_file_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "praktika.db"
    conn = db.connect(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.current_version(conn) == db.SCHEMA_VERSION
    assert db.migrate(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO consent (meeting_id, record_json, recorded_at) VALUES "
            "('M-20260916-none', '{}', 'now')"
        )
    conn.close()
    os.chmod(path, 0o644)
    db.connect(path).close()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_audit_chain_and_broken_prev_hash(store: SqliteStore) -> None:
    assert store.last_audit_hash() == GENESIS_HASH
    first = audit_event(prev_hash=GENESIS_HASH)
    h1 = store.append_audit(first)
    assert h1 == first.sealed().hash and store.last_audit_hash() == h1
    h2 = store.append_audit(audit_event(prev_hash=h1, event="review.approved"))
    assert h2 != h1 and store.last_audit_hash() == h2
    with pytest.raises(ValueError, match="chain broken"):
        store.append_audit(audit_event(prev_hash=h1))
    assert store.last_audit_hash() == h2


def test_list_meetings_filters(store: SqliteStore) -> None:
    store.save_meeting(meeting("M-20260916-1001", started_at=NOW - timedelta(days=2)))
    store.save_meeting(meeting("M-20260916-1002", state=MeetingState.approved))
    assert [m.id for m in store.list_meetings()] == ["M-20260916-1002", "M-20260916-1001"]
    assert [m.id for m in store.list_meetings({"state": MeetingState.approved})] == [
        "M-20260916-1002"
    ]
    assert [m.id for m in store.list_meetings({"since": NOW - timedelta(days=1)})] == [
        "M-20260916-1002"
    ]
    assert store.list_meetings({"organiser": "nobody"}) == []
