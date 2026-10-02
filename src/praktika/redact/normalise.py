"""Text normalisation used before identifier matching.

Every function here is a pure, length-preserving-where-stated transform so callers can match on
the normalised string and map spans back onto the original text.
"""

from __future__ import annotations

import re

# Arabic-Indic (U+0660..U+0669) and Eastern Arabic-Indic / Persian (U+06F0..U+06F9) digits, the
# Arabic decimal separator (U+066B) and the Arabic thousands separator (U+066C).
_DIGIT_MAP = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
        "٫": ".",
        "٬": ",",
    }
)

# Harakat U+064B..U+0652, superscript alef U+0670 and tatweel U+0640.
_TASHKEEL_RE = re.compile(r"[ً-ْٰـ]")

_SEPARATORS_RE = re.compile(r"[,\s'_  ]")


def arabic_indic_to_western(text: str) -> str:
    """Return ``text`` with Arabic-Indic and Eastern Arabic-Indic digits replaced by ASCII digits.

    The Arabic decimal separator ``٫`` becomes ``.`` and the thousands separator ``٬`` becomes
    ``,``. Every replacement is one character for one character, so string offsets are preserved
    and a span found in the result addresses the same characters in the input.
    """
    return text.translate(_DIGIT_MAP)


def strip_tashkeel(text: str) -> str:
    """Return ``text`` without Arabic diacritics (harakat, superscript alef) and tatweel.

    This is *not* length-preserving; use it for comparison and search, not for span mapping.
    """
    return _TASHKEEL_RE.sub("", text)


def normalise_number(s: str) -> str:
    """Return the digits of a spoken or written number in canonical form, e.g. ``"1200000.5"``.

    Arabic-Indic digits are converted first; thousands separators (``,``, spaces, non-breaking
    spaces, apostrophes, underscores) are removed; a decimal comma is unified to ``.`` when it is
    the only separator and is followed by one or two digits (``"3,5"`` -> ``"3.5"``). Trailing
    ``.0`` fractions are kept as written. A leading sign is preserved. The function never raises;
    input with no digits is returned stripped of separators.
    """
    text = arabic_indic_to_western(s).strip()
    if "." not in text and re.fullmatch(r"[-+]?\d+,\d{1,2}", text):
        return text.replace(",", ".")
    return _SEPARATORS_RE.sub("", text)
