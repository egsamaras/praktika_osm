"""Quote-to-segment matching for the verifier and the assembler (C-07).

``quote_matches`` decides whether a model quote is a verbatim or near-verbatim part of a
transcript segment. Beyond the fuzzy score it guards two ways a near-verbatim quote can still
misreport what was said: the aligned window of the segment must carry exactly the same
negation tokens as the quote (a dropped or inserted "not"/"لا" flips the meaning while the
score stays above the threshold), and every digit run in the quote must occur in that window
(a swapped date or amount scores 96+ on characters). ``cited_runs`` finds the maximal runs of
consecutive cited segments so a quote that straddles a segment boundary — Whisper emits 5-10
word segments — can be checked against the joined text instead of failing on both halves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rapidfuzz import fuzz

from praktika.models import Segment
from praktika.redact.normalise import arabic_indic_to_western
from praktika.store.search import normalise_ar

QUOTE_MIN_CHARS = 20
QUOTE_MIN_TOKENS = 4
QUOTE_TOKEN_SLACK = 2
NEGATION_TOKENS = frozenset(
    {"not", "no", "never", "cannot", "لا", "لن", "لم", "ليس", "ليست", "ما", "غير"}
)
_DIGIT_RUN = re.compile(r"\d+")


def norm(text: str) -> str:
    """Comparison form for quotes and segments: tashkeel stripped, alef/yaa/taa-marbuta
    unified, Arabic-Indic digits Western, thousands separators dropped, lower-cased."""
    return normalise_ar(arabic_indic_to_western(text)).replace(",", "").replace("٬", "")


def _negations(tokens: Iterable[str]) -> list[str]:
    """The negation tokens among ``tokens`` (English contractions count as ``n't``)."""
    out = []
    for tok in tokens:
        bare = tok.strip(".,;:!?()[]\"'«»،؛")
        if bare in NEGATION_TOKENS:
            out.append(bare)
        elif bare.endswith(("n't", "n’t")):
            out.append("n't")
    return sorted(out)


def _aligned_window(q: str, t: str) -> str:
    """The part of ``t`` that ``partial_ratio`` matched ``q`` against, widened to whole
    tokens so a token cut at the edge of the alignment is not misread."""
    a = fuzz.partial_ratio_alignment(q, t)
    if a is None:
        return t
    start, end = a.dest_start, a.dest_end
    while start > 0 and not t[start - 1].isspace():
        start -= 1
    while end < len(t) and not t[end].isspace():
        end += 1
    return t[start:end]


def same_polarity_and_figures(quote: str, window: str) -> bool:
    """True unless ``quote`` and ``window`` differ in their negation tokens or ``quote`` has a
    digit run that ``window`` lacks. Both arguments are already in comparison form."""
    if _negations(quote.split()) != _negations(window.split()):
        return False
    present = set(_DIGIT_RUN.findall(window))
    return all(run in present for run in _DIGIT_RUN.findall(quote))


def quote_matches(quote: str, text: str, ratio: int = 85) -> bool:
    """True when ``quote`` is a verbatim or near-verbatim part of ``text``.

    Both sides are normalised (``norm``). A quote equal to the whole segment always matches.
    Otherwise it must (a) score ``partial_ratio >= ratio`` against the segment, (b) not have
    more tokens than the segment plus ``QUOTE_TOKEN_SLACK``, (c) be substantial — at least
    ``QUOTE_MIN_CHARS`` characters or ``QUOTE_MIN_TOKENS`` tokens — so "the budget" cannot
    cite a long segment and a paragraph cannot cite "Yes", and (d) agree with the aligned
    window of the segment on negation tokens and digit runs, so "we will proceed" never
    verifies against "we will not proceed" and "30 October" never against "20 October".
    """
    q, t = norm(quote).strip(), norm(text).strip()
    if not q or not t:
        return False
    if q == t:
        return True
    q_tokens, t_tokens = q.split(), t.split()
    if len(q_tokens) > len(t_tokens) + QUOTE_TOKEN_SLACK:
        return False
    if len(q) < QUOTE_MIN_CHARS and len(q_tokens) < QUOTE_MIN_TOKENS:
        return False
    if fuzz.partial_ratio(q, t) < ratio:
        return False
    return same_polarity_and_figures(q, _aligned_window(q, t))


def cited_runs(ids: Iterable[str], segments: list[Segment]) -> dict[str, str]:
    """Map each cited id that belongs to a run of two or more *consecutive* transcript
    segments (all cited) to the space-joined text of that run, in transcript order.

    Ids that are not in the transcript, or whose neighbours are not cited, are absent from
    the result: a quote can only be checked against text the model actually cited.
    """
    order = {s.id: i for i, s in enumerate(segments)}
    cited = sorted({i for i in (order.get(sid) for sid in ids) if i is not None})
    out: dict[str, str] = {}
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 2:
            text = " ".join(segments[i].text for i in run)
            for i in run:
                out[segments[i].id] = text

    for i in cited:
        if run and i != run[-1] + 1:
            flush()
            run = []
        run.append(i)
    flush()
    return out


def supports(quote: str, segment: Segment, run_text: str | None, ratio: int = 85) -> bool:
    """True when ``quote`` verifies against ``segment`` on its own or, for a segment inside a
    run of consecutive cited segments, against the joined text of that run."""
    if quote_matches(quote, segment.text, ratio):
        return True
    return run_text is not None and quote_matches(quote, run_text, ratio)


def neighbour_texts(segment_id: str, segments: list[Segment]) -> list[str]:
    """The cited segment joined with its previous neighbour and with its next neighbour.

    A pause splits one spoken sentence across two Whisper segments, and the model often cites
    only one of them; a quote that is verbatim across the boundary is still transcript text.
    Only immediate neighbours are joined (never further), so a quote can never verify against
    text more than one segment away from what was cited.
    """
    order = {s.id: i for i, s in enumerate(segments)}
    i = order.get(segment_id)
    if i is None:
        return []
    out: list[str] = []
    if i > 0:
        out.append(f"{segments[i - 1].text} {segments[i].text}")
    if i + 1 < len(segments):
        out.append(f"{segments[i].text} {segments[i + 1].text}")
    return out
