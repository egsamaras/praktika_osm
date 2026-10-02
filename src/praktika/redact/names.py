"""Personal-name heuristics that qualify an amount as ``amount_with_name`` (C-06).

``name_near`` looks for a probable personal name within ``WINDOW_TOKENS`` words of a span.
Latin names are title-case words outside ``NAME_STOPWORDS`` and the roster; Arabic names are a
word after an honorific or a run of two Arabic words outside ``ARABIC_STOPWORDS`` and the
roster, so "راتب فاطمة العلي 3,200 دينار" is tokenised exactly like its English counterpart.
"""

from __future__ import annotations

import re

# A title-case Latin word that may be a personal name, optionally possessive ("Omar's",
# "Layla’s"); roster tokens and stopwords are removed by the caller. All-caps words (BHD, USD)
# and CamelCase words (ManCom) do not match.
NAME_TOKEN = re.compile(r"(?<![A-Za-z])[A-Z][a-z]{1,}(?:[-'][A-Z][a-z]+)?(?:['’]s)?(?![a-zA-Z])")
_POSSESSIVE = re.compile(r"['’]s$")

NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        # months, days
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        # common sentence starters and function words
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Then",
        "So",
        "And",
        "But",
        "Or",
        "If",
        "We",
        "You",
        "It",
        "He",
        "She",
        "They",
        "I",
        "Our",
        "Your",
        "Its",
        "Let",
        "Yes",
        "No",
        "Ok",
        "Okay",
        "Right",
        "Well",
        "Now",
        "Also",
        "Just",
        "Please",
        "Thanks",
        "Thank",
        "Sorry",
        "Actually",
        "Agreed",
        "Approved",
        "Decision",
        "Action",
        "Noted",
        "Fine",
        "Good",
        "Morning",
        "Afternoon",
        "Hello",
        "Hi",
        "For",
        "With",
        "From",
        "About",
        "Budget",
        "Total",
        "Phase",
        "Option",
        "Board",
        "Committee",
        "Chair",
        "Bank",
        "Treasury",
        "Finance",
        "Legal",
        "Compliance",
        "Risk",
        "Audit",
        "Vendor",
        "Licence",
        "License",
        "Invoice",
        "Quote",
        "Cost",
        "Price",
        "Amount",
        "Fee",
        "Fees",
        "Million",
        "Thousand",
        "Billion",
        "Dinar",
        "Dinars",
        "Riyal",
        "Riyals",
        "Rial",
        "Rials",
        "Dollar",
        "Dollars",
        "Pound",
        "Pounds",
        "Euro",
        "Euros",
        "Dirham",
        "Dirhams",
        "Fils",
        "Halala",
        "Halalas",
        "Bahraini",
        "Saudi",
        "Kuwaiti",
        "Qatari",
        "Omani",
        "Emirati",
        "British",
        "American",
        "Chapter",
        "Item",
        "Paper",
        "Annex",
        "Team",
        "Function",
        "Council",
        "Group",
    }
)

WINDOW_TOKENS = 6

# Arabic personal-name heuristics for ``name_near``: an honorific (optionally with the ل/لل
# prefix) followed by an Arabic word, or a run of two Arabic words that are neither function
# words, currency/number words nor roster tokens.
ARABIC_HONORIFICS: frozenset[str] = frozenset(
    {"السيد", "السيدة", "الأستاذ", "الأستاذة", "الدكتور", "الدكتورة", "المهندس", "المهندسة",
     "أبو", "أم", "الشيخ", "الشيخة"}
)  # fmt: skip
ARABIC_STOPWORDS: frozenset[str] = frozenset(
    {
        # function words
        "في", "من", "إلى", "الى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي",
        "هو", "هي", "هم", "أن", "إن", "ان", "لا", "نعم", "ثم", "أو", "او", "بعد", "قبل", "عند",
        "كل", "بين", "حتى", "لكن", "كان", "كانت", "يكون", "تكون", "خلاص", "طيب", "يعني", "بس",
        "فقط", "أيضا", "أيضاً", "كذلك", "الآن", "اليوم", "غدا", "غداً", "أمس", "حول", "لدى",
        "عندي", "عندنا", "حقها", "حقه", "حقنا", "له", "لها", "لهم", "لنا", "لي", "بها", "به",
        # currency and number words (also in _CURRENCY / _MAGNITUDE)
        "دينار", "ريال", "دولار", "ألف", "الف", "مليون", "مليار", "بحريني", "سعودي", "أمريكي",
        "شهرياً", "شهريا", "سنوياً", "سنويا", "راتب", "ميزانية", "الميزانية", "فاتورة", "مبلغ",
        "المبلغ", "تحويل", "حوّل", "ادفع", "دفع", "الدفع", "رسوم", "تكلفة", "التكلفة",
        # months and days
        "يناير", "فبراير", "مارس", "أبريل", "ابريل", "مايو", "يونيو", "يوليو", "أغسطس", "اغسطس",
        "سبتمبر", "أكتوبر", "اكتوبر", "نوفمبر", "ديسمبر", "الأحد", "الاثنين", "الثلاثاء",
        "الأربعاء", "الخميس", "الجمعة", "السبت", "محرم", "صفر", "رجب", "شعبان", "رمضان", "شوال",
    }
)  # fmt: skip
_ARABIC_WORD = re.compile(r"^[\u0621-\u064A\u0671-\u06D3]{2,}$")

_WORD_RE = re.compile(r"\S+")
_PUNCT = ".,;:!?()[]\"'«»"


def _bare(w: str) -> str:
    """``w`` without a leading conjunction/preposition clitic (و, ف, ب, ك, ل, لل)."""
    if w.startswith("لل"):
        return "ال" + w[2:]
    return w[1:] if w[:1] in "لوفبك" and len(w) > 3 else w


def _arabic_name_word(word: str, known_tokens: frozenset[str]) -> bool:
    """An Arabic word that could be part of a personal name (not a stopword or roster token)."""
    w = word.strip(_PUNCT + "،؛")
    if not _ARABIC_WORD.match(w) or w in known_tokens:
        return False
    bare = _bare(w)
    return w not in ARABIC_STOPWORDS and bare not in ARABIC_STOPWORDS and bare not in known_tokens


def _arabic_given_name(word: str, known_tokens: frozenset[str]) -> bool:
    """A name-run must open with a given name, which carries no definite article (فاطمة,
    أحمد); "المرحلة الأولى" or "الميزانية المعتمدة" therefore never look like a person."""
    w = word.strip(_PUNCT + "،؛")
    return _arabic_name_word(w, known_tokens) and not _bare(w).startswith("ال")


def _arabic_honorific(word: str) -> bool:
    w = word.strip(_PUNCT + "،؛")
    return w in ARABIC_HONORIFICS or (
        w.startswith(("لل", "ل"))
        and w.lstrip("ل") in {h.lstrip("ا") for h in ARABIC_HONORIFICS}
        or w[1:] in ARABIC_HONORIFICS
    )


def name_near(text: str, start: int, end: int, known_tokens: frozenset[str]) -> bool:
    """Return True when a probable personal name occurs within ``WINDOW_TOKENS`` of the span.

    Latin: a title-case word (``NAME_TOKEN``) that is neither in ``NAME_STOPWORDS`` nor in
    ``known_tokens`` (the roster's names and aliases split into words). A candidate that opens a
    sentence counts only when the next word is also a candidate, which keeps "Approved. BHD
    5,000" from looking like a person while "Ahmed Yusuf" still does; a possessive ("Omar's
    salary is BHD 4,500") is a name wherever it stands.

    Arabic: a word that follows an honorific (السيد, الدكتور, ... with an optional ل/لل prefix),
    or two consecutive Arabic words that are neither function words, currency/number words,
    month names nor roster tokens ("فاطمة العلي", "أحمد يوسف").
    """
    words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
    inside = [i for i, (ws, we, _) in enumerate(words) if ws < end and we > start]
    if not inside:
        return False
    lo, hi = inside[0], inside[-1]
    window = list(range(max(0, lo - WINDOW_TOKENS), lo)) + list(
        range(hi + 1, min(len(words), hi + 1 + WINDOW_TOKENS))
    )

    def candidate(i: int) -> bool:
        w = words[i][2].strip(_PUNCT)
        base = _POSSESSIVE.sub("", w)
        return (
            bool(NAME_TOKEN.fullmatch(w))
            and base not in NAME_STOPWORDS
            and base not in known_tokens
        )

    def possessive(i: int) -> bool:
        return bool(_POSSESSIVE.search(words[i][2].strip(_PUNCT)))

    def arabic(i: int) -> bool:
        return _arabic_name_word(words[i][2], known_tokens)

    for i in window:
        if candidate(i):
            opens_sentence = i == 0 or words[i - 1][2].endswith((".", "!", "?", ":"))
            if not opens_sentence or possessive(i) or (i + 1 < len(words) and candidate(i + 1)):
                return True
        if _arabic_honorific(words[i][2]) and i + 1 < len(words) and arabic(i + 1):
            return True
        if _arabic_given_name(words[i][2], known_tokens) and i + 1 < len(words) and arabic(i + 1):
            return True
    return False
