"""Bidirectional text for DOCX runs.

``add_text`` splits a string into script runs so only Arabic spans carry ``w:rtl`` and marks the
paragraph ``w:bidi`` only when Arabic letters are at least ``BIDI_SHARE`` of its letters; a
code-switched or mostly English sentence therefore keeps its left-to-right layout while an
Arabic paragraph reads right-to-left.
"""

from __future__ import annotations

import re
from typing import Any

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from praktika.render.markdown import ARABIC_RE

BIDI_SHARE = 0.5
# Arabic spans (Arabic letters/marks, with any digits, spaces and Arabic punctuation between).
_ARABIC_SPAN = re.compile(r"[؀-ۿﹰ-ﻼ][؀-ۿﹰ-ﻼ0-9٠-٩\s،؛؟.,:%]*")
_LETTER = re.compile(r"[^\W\d_]")


def _mark_rtl_run(run: Any) -> None:
    rpr = run._r.get_or_add_rPr()
    rtl = OxmlElement("w:rtl")
    rtl.set(qn("w:val"), "1")
    rpr.append(rtl)


def _mark_bidi(paragraph: Any) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    if ppr.find(qn("w:bidi")) is None:
        ppr.append(OxmlElement("w:bidi"))


def paragraph_is_rtl(text: str) -> bool:
    """Right-to-left base direction when Arabic letters are at least ``BIDI_SHARE`` of the
    letters. A code-switched quote that merely opens with an Arabic name stays LTR."""
    letters = _LETTER.findall(text)
    if not letters:
        return False
    arabic = sum(1 for ch in letters if ARABIC_RE.match(ch))
    return arabic / len(letters) >= BIDI_SHARE


def script_runs(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(span, is_arabic)`` pieces so only Arabic spans become RTL runs."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _ARABIC_SPAN.finditer(text):
        span = m.group(0).rstrip()
        end = m.start() + len(span)
        if m.start() > pos:
            out.append((text[pos : m.start()], False))
        if span:
            out.append((span, True))
        pos = end
    if pos < len(text):
        out.append((text[pos:], False))
    return out or [(text, False)]


def add_text(paragraph: Any, text: str, *, bold: bool = False, italic: bool = False) -> Any:
    """Append ``text`` as script runs; Arabic spans get ``rtl`` runs and the paragraph is
    ``bidi`` only when its text reads right-to-left (``paragraph_is_rtl``). Returns the last run."""
    run = None
    for span, arabic in script_runs(text):
        run = paragraph.add_run(span)
        run.bold, run.italic = bold, italic
        if arabic:
            _mark_rtl_run(run)
    if ARABIC_RE.search(text) and paragraph_is_rtl(text):
        _mark_bidi(paragraph)
    return run
