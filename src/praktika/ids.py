"""Identifier generation for meetings and transcript segments."""

from __future__ import annotations

import secrets
from datetime import datetime

MEETING_ID_PREFIX = "M"
SEGMENT_ID_MAX = 99_999


def new_meeting_id(now: datetime) -> str:
    """Return a new meeting id of the form ``M-YYYYMMDD-xxxx``.

    ``now`` supplies the date part (its own timezone is used as given; callers pass an aware
    datetime). The suffix is four random hex characters from ``secrets``; uniqueness within a
    single day is expected, not guaranteed, and the store enforces uniqueness on insert.
    """
    return f"{MEETING_ID_PREFIX}-{now:%Y%m%d}-{secrets.token_hex(2)}"


def segment_id(i: int) -> str:
    """Return the segment id for 1-based position ``i``: ``segment_id(1) == "S0001"``.

    Ids are zero-padded to four digits and grow to five above 9999, matching the
    ``Segment.id`` pattern ``^S\\d{4,5}$``. Raises ``ValueError`` outside 1..99999.
    """
    if not 1 <= i <= SEGMENT_ID_MAX:
        raise ValueError(f"segment index out of range 1..{SEGMENT_ID_MAX}: {i}")
    return f"S{i:04d}"
