"""Tests for the Microsoft Graph seam (not wired to a live poller yet).

No network: the fake source below reads the committed VTT fixture and satisfies
``GraphTranscriptSource`` structurally, which is exactly how a future poller would be tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from praktika.errors import PraktikaError
from praktika.ingest.graph_stub import (
    GRAPH_ERROR_HINTS,
    GraphError,
    GraphErrorCode,
    GraphTranscriptSource,
)
from praktika.ingest.vtt import parse_teams_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORGANISER = "f.khalid@acme.test"


class FixtureGraphSource:
    """Fixture-backed fake: one organiser, transcripts served from ``tests/fixtures``."""

    def __init__(self, enabled_organisers: set[str]) -> None:
        self._enabled = enabled_organisers
        self._catalogue: dict[str, dict[str, Any]] = {
            "delta-0": {
                "meeting_id": "M-20260916-a1b2",
                "transcript_id": "tr-en",
                "path": FIXTURES / "synthetic_en.vtt",
            },
            "delta-1": {
                "meeting_id": "M-20260916-c3d4",
                "transcript_id": "tr-ar",
                "path": FIXTURES / "synthetic_ar_mixed.vtt",
            },
        }
        self._order = list(self._catalogue)

    def _check(self, organiser_upn: str) -> None:
        if organiser_upn not in self._enabled:
            raise GraphError(GraphErrorCode.ApplicationAccessPolicyMissing, organiser_upn)

    def list_new(
        self, organiser_upn: str, delta_link: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        self._check(organiser_upn)
        start = 0 if delta_link is None else self._order.index(delta_link) + 1
        new = [
            {k: v for k, v in self._catalogue[key].items() if k != "path"}
            for key in self._order[start:]
        ]
        return new, self._order[-1]

    def fetch_vtt(self, organiser_upn: str, meeting_id: str, transcript_id: str) -> str:
        self._check(organiser_upn)
        for entry in self._catalogue.values():
            if (entry["meeting_id"], entry["transcript_id"]) == (meeting_id, transcript_id):
                return Path(entry["path"]).read_text(encoding="utf-8")
        raise GraphError(GraphErrorCode.SpeakerAttributionNotAllowed, transcript_id)


def test_error_codes_enumerated() -> None:
    assert {c.name for c in GraphErrorCode} == {
        "GraphAccessToTranscriptsDisabled",
        "SpeakerAttributionNotAllowed",
        "DeltaFilterNotAllowed",
        "ApplicationAccessPolicyMissing",
    }
    assert all(c.value == c.name for c in GraphErrorCode)
    assert set(GRAPH_ERROR_HINTS) == set(GraphErrorCode), "every code carries an IT hint"

    err = GraphError(GraphErrorCode.ApplicationAccessPolicyMissing, "organiser x")
    assert isinstance(err, PraktikaError)
    assert err.code is GraphErrorCode.ApplicationAccessPolicyMissing
    assert str(err).startswith("ApplicationAccessPolicyMissing: ")
    assert "Grant-CsApplicationAccessPolicy" in str(err) and "(organiser x)" in str(err)
    assert "never -Global" in str(err)

    # the tenant switches name the exact cmdlet and parameter an administrator sets
    hints = GRAPH_ERROR_HINTS
    assert "-Identity Global -EnableGraphTranscriptAccess $true" in " ".join(
        hints[GraphErrorCode.GraphAccessToTranscriptsDisabled].split()
    )
    assert (
        "-Identity Global -EnableAttributedTranscripts $true"
        in (hints[GraphErrorCode.SpeakerAttributionNotAllowed])
    )
    # a filter on a delta link is a defect in the poller, not a tenant setting
    assert "defect" in hints[GraphErrorCode.DeltaFilterNotAllowed]
    assert "ApplicationAccessPolicy" not in hints[GraphErrorCode.DeltaFilterNotAllowed]

    # a string code is coerced; an unknown code is rejected at construction
    assert (
        GraphError("SpeakerAttributionNotAllowed").code
        is GraphErrorCode.SpeakerAttributionNotAllowed
    )
    with pytest.raises(ValueError):
        GraphError("NotAGraphCode")  # type: ignore[arg-type]


def test_fixture_source_round_trip() -> None:
    source = FixtureGraphSource({ORGANISER})
    assert isinstance(source, GraphTranscriptSource)

    new, delta = source.list_new(ORGANISER, None)
    assert [d["transcript_id"] for d in new] == ["tr-en", "tr-ar"]
    assert all({"meeting_id", "transcript_id"} <= set(d) for d in new)
    assert delta == "delta-1"

    text = source.fetch_vtt(ORGANISER, new[0]["meeting_id"], new[0]["transcript_id"])
    transcript = parse_teams_vtt(text, new[0]["meeting_id"], ["AI Lab Meeting Room"])
    assert transcript.source == "vtt"
    assert transcript.segments[0].speaker == "F. Khalid"
    assert len(transcript.segments) > 50

    # a second poll from the returned delta link yields nothing new and the same link
    assert source.list_new(ORGANISER, delta) == ([], "delta-1")

    # tenant conditions surface as the enumerated errors, not as generic failures
    with pytest.raises(GraphError) as info:
        source.list_new("someone.else@acme.test", None)
    assert info.value.code is GraphErrorCode.ApplicationAccessPolicyMissing
    with pytest.raises(GraphError) as info:
        source.fetch_vtt(ORGANISER, "M-20260916-a1b2", "tr-missing")
    assert info.value.code is GraphErrorCode.SpeakerAttributionNotAllowed


def test_protocol_is_structural() -> None:
    class Incomplete:
        def list_new(self, organiser_upn: str, delta_link: str | None) -> tuple[list, str]:
            return [], ""

    assert not isinstance(Incomplete(), GraphTranscriptSource)
