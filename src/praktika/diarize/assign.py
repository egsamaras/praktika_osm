"""Reconcile diarisation turns with transcript segments.

``assign_speakers`` gives each segment the label of the turn it overlaps most; when word times
exist a segment straddling a turn boundary is split there so a citation never spans two
speakers. ``apply_names`` is the reviewer's mapping from labels to roster names.
"""

from __future__ import annotations

from praktika.ids import segment_id
from praktika.models import Segment, Word

from .base import Turn

UNKNOWN = "unknown"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_turn(start: float, end: float, turns: list[Turn]) -> Turn | None:
    """The turn with the largest overlap; ties go to the earlier turn. None when no overlap."""
    best: Turn | None = None
    best_overlap = 0.0
    for turn in turns:  # sorted by start, so strict '>' keeps the earlier turn on ties
        ov = _overlap(start, end, turn.start, turn.end)
        if ov > best_overlap:
            best, best_overlap = turn, ov
    return best


def _label(seg: Segment, turn: Turn | None) -> Segment:
    if turn is None:
        return seg.model_copy(update={"speaker": UNKNOWN, "speaker_kind": "unknown"})
    return seg.model_copy(update={"speaker": turn.label, "speaker_kind": "label"})


def _group_words(
    words: list[Word], turns: list[Turn], fallback: Turn | None
) -> list[tuple[Turn | None, list[Word]]]:
    """Consecutive runs of words sharing the same best turn (segment-level turn as fallback)."""
    groups: list[tuple[Turn | None, list[Word]]] = []
    for w in words:
        turn = _best_turn(w.start, w.end, turns) or fallback
        if groups and groups[-1][0] is turn:
            groups[-1][1].append(w)
        else:
            groups.append((turn, [w]))
    return groups


def _split(seg: Segment, turns: list[Turn]) -> list[Segment]:
    seg_turn = _best_turn(seg.start, seg.end, turns)
    groups = _group_words(seg.words, turns, seg_turn)
    if len(groups) <= 1:
        return [_label(seg, seg_turn)]
    pieces: list[Segment] = []
    for i, (turn, words) in enumerate(groups):
        start = seg.start if i == 0 else words[0].start
        end = seg.end if i == len(groups) - 1 else words[-1].end
        piece = seg.model_copy(
            update={
                "start": start,
                "end": max(end, start),
                "text": " ".join(w.text for w in words).strip() or seg.text,
                "words": list(words),
            }
        )
        pieces.append(_label(piece, turn))
    return pieces


def assign_speakers(
    segments: list[Segment], turns: list[Turn], *, split_on_words: bool = True
) -> list[Segment]:
    """Label segments by maximum time overlap with ``turns``; split on boundaries when possible.

    Segments with no overlapping turn become ``unknown``. Segments already attributed to the
    organiser's own microphone (``speaker_kind == "self"``) are left untouched. With
    ``split_on_words`` and word timings, a segment whose words fall under different turns is cut
    into one segment per run of words. Ids are renumbered ``S0001..`` in list order.
    """
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    out: list[Segment] = []
    for seg in segments:
        if seg.speaker_kind == "self":
            out.append(seg)
        elif split_on_words and seg.words:
            out.extend(_split(seg, ordered))
        else:
            out.append(_label(seg, _best_turn(seg.start, seg.end, ordered)))
    return [s.model_copy(update={"id": segment_id(i)}) for i, s in enumerate(out, start=1)]


def apply_names(segments: list[Segment], mapping: dict[str, str]) -> list[Segment]:
    """Replace speaker labels found in ``mapping`` with names and mark them ``identity``.

    Idempotent: names are not keys of the mapping, so a second application changes nothing.
    Segments whose speaker is not in the mapping are returned unchanged.
    """
    out: list[Segment] = []
    for seg in segments:
        name = mapping.get(seg.speaker)
        if name is None or not name.strip():
            out.append(seg)
        else:
            out.append(seg.model_copy(update={"speaker": name.strip(), "speaker_kind": "identity"}))
    return out
