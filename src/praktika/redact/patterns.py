"""Deterministic identifier detectors (control C-06).

The personal-name heuristics that qualify an amount (``name_near``) live in ``redact/names``.

``PATTERNS`` maps a kind to a compiled regex. Each regex is applied to text whose Arabic-Indic
digits have already been normalised (see ``normalise.arabic_indic_to_western``). A pattern may
mark the part to tokenise with a named group ``v`` (context before the value) or ``v2`` (context
after the value); ``value_span`` resolves which one matched. Numeric kinds are candidates only
until their validator (``iban_ok``, ``luhn_ok``) passes.
"""

from __future__ import annotations

import re

# Order matters: the tokeniser applies patterns in this order and later patterns never see text
# already replaced by a token, so context-bearing kinds run before bare-digit kinds.
KINDS: tuple[str, ...] = (
    "email",
    "iban",
    "account",
    "card",
    "cpr_bh",
    "iqama_sa",
    "phone",
    "amount_with_name",
)

TOKEN_PREFIX: dict[str, str] = {
    "iban": "IBAN",
    "card": "CARD",
    "cpr_bh": "CPR",
    "iqama_sa": "IQAMA",
    "phone": "PHONE",
    "email": "EMAIL",
    "account": "ACC",
    "amount_with_name": "AMT",
}

# Placeholder the tokeniser writes over existing tokens while scanning. Context gaps must not
# cross it, or the context word of an already-tokenised value would qualify the next number.
MASK = "\ufffc"
_CTX_GAP = rf"[^\d\n{MASK}]{{0,25}}"
# Arabic context words are morphology-aware: possessive suffixes (رقمه الشخصي, رقمها الشخصي,
# إقامته, هويتها) turn taa marbuta into taa, so a bare "إقامة" would never see them; the spoken
# "سي بي آر" is how Gulf speakers say CPR.
_CPR_CTX = r"(?:\bCPR\b|سي بي آر|(?:ال)?رقم(?:ه|ها|ي|ك|كم|هم)?\s+(?:ال)?شخصي|رقم الهوية البحرينية)"
_IQAMA_CTX = (
    r"(?:\biqama\b|\bnational id\b|\bNID\b|(?:ال)?إقام(?:ة|ت\w{0,2})|(?:ال)?هوي(?:ة|ت\w{0,2}))"
)
_ACC_CTX = (
    r"(?:\baccount(?: number| no\.?| num)?\b|\ba/c\b|\bacct\.?|رقم الحساب"
    r"|حساب(?:ه|ها|ي|ك|كم|هم)?)"
)
# ISO codes and symbols, the Arabic words, and the English spoken forms Whisper actually emits
# ("4,500 dinars", "BD 4,500", "800 fils", "Bahraini dinars"); the word forms are bounded so
# "abd" or "SRT" never qualify.
_CURRENCY_WORDS = (
    r"\b(?:(?:bahraini|saudi|kuwaiti|qatari|omani|emirati|us|british)\s+)?"
    r"(?:dinars?|riyals?|rials?|dollars?|pounds?|euros?|dirhams?|fils|halalas?|BD|SR|KD)\b"
)
_CURRENCY = (
    r"(?:BHD|SAR|USD|GBP|EUR|KWD|AED|QAR|OMR|\$|£|€|د\.ب\.?|دينار|ريال|دولار|"
    + _CURRENCY_WORDS
    + ")"
)
_MAGNITUDE = r"(?:\s?(?:k|m|bn|mn|thousand|million|billion|ألف|مليون|مليار))?"
_NUMBER = r"\d{1,3}(?:[,  ]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    # Two letters, two check digits, then 11..30 alphanumerics in optional groups of up to four.
    # Case-insensitive: Teams VTT and STT output do not guarantee upper case.
    "iban": re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ -]?[A-Za-z0-9]{1,4}){3,8}\b", re.IGNORECASE),
    "account": re.compile(
        rf"(?i){_ACC_CTX}{_CTX_GAP}(?P<v>\d(?:[ -]?\d){{7,15}})(?!\d)", re.UNICODE
    ),
    # 13..19 digits with optional single spaces or hyphens between them; Luhn-checked.
    "card": re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
    "cpr_bh": re.compile(
        rf"(?i)(?:{_CPR_CTX}{_CTX_GAP}(?<!\d)(?P<v>\d{{9}})(?!\d)"
        rf"|(?<!\d)(?P<v2>\d{{9}})(?!\d)(?=[^\d\n{MASK}]{{0,25}}{_CPR_CTX}))"
    ),
    "iqama_sa": re.compile(
        rf"(?i)(?:{_IQAMA_CTX}{_CTX_GAP}(?<!\d)(?P<v>[12]\d{{9}})(?!\d)"
        rf"|(?<!\d)(?P<v2>[12]\d{{9}})(?!\d)(?=[^\d\n{MASK}]{{0,25}}{_IQAMA_CTX}))"
    ),
    "phone": re.compile(
        r"(?<![\d+])(?:"
        r"\+(?:973|966|44)[ -]?\(?\d{1,4}\)?(?:[ -]?\d){5,9}"  # international: BH, SA, UK
        r"|00(?:973|966|44)[ -]?\d(?:[ -]?\d){6,10}"  # 00-prefixed international
        r"|\b0(?:5\d{8}|7\d{9})\b"  # SA mobile 05xxxxxxxx, UK mobile 07xxxxxxxxx
        r"|\b0(?:5\d|7\d{3}) ?\d{3} ?\d{3,4}\b"  # same with spoken grouping
        r"|(?<!BHD )(?<!SAR )(?<!USD )\b(?:1[367]\d{2}|3\d{3}|6\d{3}|77\d{2}) ?\d{4}\b"  # BH local
        r"(?! ?(?:million|thousand|k\b|m\b|dinar|riyal))"
        r")(?![\d-])"
    ),
    "amount_with_name": re.compile(
        rf"(?i)(?:{_CURRENCY}\s?(?:{_NUMBER}){_MAGNITUDE}"
        rf"|(?:{_NUMBER}){_MAGNITUDE}\s?{_CURRENCY})"
    ),
}


def value_span(m: re.Match[str]) -> tuple[int, int]:
    """Return the ``(start, end)`` of the part of ``m`` to tokenise.

    The first non-``None`` group among ``v`` and ``v2`` wins; a pattern without those groups
    contributes its whole match.
    """
    for name in ("v", "v2"):
        if name in m.re.groupindex and m.group(name) is not None:
            return m.span(name)
    return m.span()


def iban_ok(s: str) -> bool:
    """Return True when ``s`` is a syntactically valid IBAN (15..34 chars, ISO 7064 mod-97 == 1).

    Spaces and hyphens are ignored; letters are case-insensitive. No country-specific length or
    registry check is performed, so IBANs with test country codes validate.
    """
    compact = re.sub(r"[ -]", "", s).upper()
    if not 15 <= len(compact) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(digits) % 97 == 1


def luhn_ok(s: str) -> bool:
    """Return True when the digits of ``s`` (separators ignored) pass the Luhn check.

    Requires 13..19 digits, the range of payment card numbers; anything else is False.
    """
    digits = re.sub(r"[ -]", "", s)
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
