"""Seam for Microsoft Graph transcript polling (not built yet).

Only the ``Protocol`` and the error vocabulary exist. A real implementation (certificate client
credentials, ``getAllTranscripts/delta``) needs the Microsoft 365 tenant changes named in
``GRAPH_ERROR_HINTS``; until one exists, tests use a fixture-backed fake that satisfies the same
Protocol. Nothing here performs network I/O.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from praktika.errors import PraktikaError


class GraphErrorCode(StrEnum):
    """Graph responses a poller must recognise: a tenant setting or permission that is missing,
    and one poller defect (``DeltaFilterNotAllowed``)."""

    GraphAccessToTranscriptsDisabled = "GraphAccessToTranscriptsDisabled"
    SpeakerAttributionNotAllowed = "SpeakerAttributionNotAllowed"
    DeltaFilterNotAllowed = "DeltaFilterNotAllowed"
    ApplicationAccessPolicyMissing = "ApplicationAccessPolicyMissing"


# Operator guidance per code: the tenant change an administrator must make, or for a poller
# defect, what to report.
GRAPH_ERROR_HINTS: dict[GraphErrorCode, str] = {
    GraphErrorCode.GraphAccessToTranscriptsDisabled: (
        "Teams meeting configuration blocks Graph transcript access. A Teams administrator "
        "must run Set-CsTeamsMeetingConfiguration -Identity Global "
        "-EnableGraphTranscriptAccess $true."
    ),
    GraphErrorCode.SpeakerAttributionNotAllowed: (
        "Attributed transcripts are disabled for the tenant. A Teams administrator must run "
        "Set-CsTeamsMeetingConfiguration -Identity Global -EnableAttributedTranscripts $true."
    ),
    GraphErrorCode.DeltaFilterNotAllowed: (
        "Graph refused a filter on a delta link. A poller must follow delta links exactly as "
        "Graph returns them, so this is a defect in the poller, not a tenant setting; "
        "restarting the organiser's delta from scratch clears it."
    ),
    GraphErrorCode.ApplicationAccessPolicyMissing: (
        "The app has no application access policy for this organiser. A Teams administrator "
        "must add the organiser to the policy with Grant-CsApplicationAccessPolicy (per user "
        "or pilot group, never -Global); changes take up to 30 minutes to reach Graph."
    ),
}


class GraphError(PraktikaError):
    """A Graph transcript source failed with one of the enumerated codes."""

    def __init__(self, code: GraphErrorCode, detail: str = "") -> None:
        self.code = GraphErrorCode(code)
        hint = GRAPH_ERROR_HINTS[self.code]
        super().__init__(f"{self.code.value}: {hint}" + (f" ({detail})" if detail else ""))


@runtime_checkable
class GraphTranscriptSource(Protocol):
    """Source of Teams transcripts for one organiser, polled via a delta link.

    ``list_new`` returns the transcript descriptors created since ``delta_link`` (``None`` for
    the first poll) together with the next delta link; each descriptor is a dict carrying at
    least ``meeting_id`` and ``transcript_id``. ``fetch_vtt`` returns the WebVTT text of one
    transcript, ready for ``ingest.vtt.parse_teams_vtt``. Implementations raise ``GraphError``
    for the enumerated tenant conditions.
    """

    def list_new(
        self, organiser_upn: str, delta_link: str | None
    ) -> tuple[list[dict[str, Any]], str]: ...

    def fetch_vtt(self, organiser_upn: str, meeting_id: str, transcript_id: str) -> str: ...
