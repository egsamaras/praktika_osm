#!/usr/bin/env python3
"""Convert one of the project's Markdown documents to a plain black-and-white Word file.

Written for circulation inside an organisation: the output carries no colour, no shading and no
theme of its own, and uses Word's built-in styles (``Heading 1``-``Heading 4``, ``List Bullet``,
``List Number``, ``Table Grid``) so that pasting the content into a house template remaps it to
the template's own fonts and colours instead of fighting them.

Supported Markdown: ATX headings, paragraphs, bullet and numbered lists (one nested level),
fenced code blocks, pipe tables, thematic breaks, and the inline spans ``**bold**``,
``*italic*``, `` `code` `` and ``[text](target)`` (rendered as "text (target)").

Usage:
    uv run python scripts/md_to_docx.py docs/DEPLOYMENT.md
    uv run python scripts/md_to_docx.py docs/DEPLOYMENT.md --out /tmp/deployment.docx --font Calibri
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from praktika.render.docx import set_app_properties

BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBER_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```+\s*(\S*)\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
RULE_RE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
INLINE_RE = re.compile(
    r"(\*\*.+?\*\*)"  # bold
    r"|(~~.+?~~)"  # strikethrough
    r"|(`+[^`]+`+)"  # code
    r"|(\[[^\]]+\]\([^)]+\))"  # link
    r"|(?<![\w*])(\*[^*\n]+\*)(?![\w*])"  # italic
)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass
class Options:
    body_font: str
    mono_font: str
    body_size: float
    code_size: float


def _set_font(run, name: str) -> None:
    """Set a run's font for Latin, complex-script and East-Asian text alike."""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def add_runs(paragraph, text: str, opts: Options, *, bold: bool = False) -> None:
    """Add ``text`` to ``paragraph``, honouring inline bold, italic, code and links.

    Colour is never set: runs inherit the destination document's theme when pasted.
    """
    text = LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    pos = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > pos:
            _plain(paragraph, text[pos : match.start()], opts, bold=bold)
        token = match.group(0)
        if token.startswith("**"):
            _plain(paragraph, token[2:-2], opts, bold=True)
        elif token.startswith("~~"):
            run = _plain(paragraph, token[2:-2], opts, bold=bold)
            run.font.strike = True
        elif token.startswith("`"):
            stripped = token.strip("`")
            run = paragraph.add_run(stripped)
            _set_font(run, opts.mono_font)
            run.font.size = Pt(opts.code_size)
            run.bold = bold
        else:
            run = _plain(paragraph, token.strip("*"), opts, bold=bold)
            run.italic = True
        pos = match.end()
    if pos < len(text):
        _plain(paragraph, text[pos:], opts, bold=bold)


def _plain(paragraph, text: str, opts: Options, *, bold: bool):
    run = paragraph.add_run(text)
    run.bold = bold
    return run


def add_code_block(document, lines: list[str], opts: Options) -> None:
    """One monospaced paragraph per line, no shading and no box: a house template can style it."""
    for line in lines or [""]:
        para = document.add_paragraph()
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.left_indent = Pt(18)
        run = para.add_run(line if line.strip() else " ")
        _set_font(run, opts.mono_font)
        run.font.size = Pt(opts.code_size)


def split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def add_table(document, rows: list[list[str]], opts: Options) -> None:
    """A plain grid: black borders, bold header, no shading, no banding."""
    width = max(len(r) for r in rows)
    table = document.add_table(rows=len(rows), cols=width)
    table.style = document.styles["Table Grid"]
    table.autofit = True
    for r, row in enumerate(rows):
        for c in range(width):
            cell = table.cell(r, c)
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(2)
            add_runs(para, row[c] if c < len(row) else "", opts, bold=(r == 0))
    document.add_paragraph()


BLACK = RGBColor(0, 0, 0)


def _blacken_styles(document) -> None:
    """Force every style's font to black.

    Word's default heading styles are blue, and a document circulated for pasting into a house
    template must carry no colour of its own: the destination template supplies the palette.
    Styles are made black rather than colour being stripped, so that "keep source formatting"
    and "merge formatting" both produce black text.
    """
    for style in document.styles:
        font = getattr(style, "font", None)
        if font is None:
            continue
        try:
            font.color.rgb = BLACK
        except (AttributeError, ValueError):  # styles without a character format
            continue


def _emit_block(document, first: str, cont: list[str], opts: Options) -> None:
    """Emit one logical block: a heading, a bullet, a numbered item or a paragraph.

    ``cont`` holds the continuation lines of a hard-wrapped block; they are joined with a
    single space so a wrapped list item stays one list item instead of becoming a paragraph.
    Numbered items carry their number as text in a hanging-indent paragraph rather than using
    Word's automatic numbering, because a document with many separate numbered lists would
    otherwise be renumbered as one long sequence when it is pasted elsewhere.
    """
    text = " ".join([first.strip(), *(c.strip() for c in cont)]).strip()
    if heading := HEADING_RE.match(first):
        body = " ".join([heading.group(2).strip(), *(c.strip() for c in cont)]).strip()
        para = document.add_paragraph(style=f"Heading {min(len(heading.group(1)), 4)}")
        add_runs(para, body, opts)
        return
    if bullet := BULLET_RE.match(first):
        body = " ".join([bullet.group(2).strip(), *(c.strip() for c in cont)]).strip()
        depth = 2 if len(bullet.group(1)) >= 2 else 1
        para = document.add_paragraph(style="List Bullet" if depth == 1 else "List Bullet 2")
        add_runs(para, body, opts)
        return
    if number := NUMBER_RE.match(first):
        body = " ".join([number.group(3).strip(), *(c.strip() for c in cont)]).strip()
        para = document.add_paragraph(style="List Paragraph")
        indent = Pt(36 if len(number.group(1)) >= 2 else 18)
        para.paragraph_format.left_indent = indent
        para.paragraph_format.first_line_indent = Pt(-18)
        add_runs(para, f"{number.group(2)}. {body}", opts)
        return
    para = document.add_paragraph()
    add_runs(para, text, opts)


def _is_block_start(line: str) -> bool:
    """True when ``line`` begins a new block rather than continuing the previous one."""
    return bool(
        HEADING_RE.match(line)
        or BULLET_RE.match(line)
        or NUMBER_RE.match(line)
        or RULE_RE.match(line)
        or line.lstrip().startswith("|")
    )


#: The OPC limit on a core property; python-docx raises ValueError above it.
CORE_TEXT_MAX = 255


def _first_heading(lines: list[str]) -> str:
    """The text of the first ATX heading outside a fenced code block, or ``""``.

    Fences toggle exactly as :func:`convert` reads them, so a ``# comment`` inside a shell
    block is never taken for the document's title.
    """
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and (heading := HEADING_RE.match(line)):
            return heading.group(2).strip()
    return ""


def _core_title(heading: str) -> str:
    """``heading`` as plain text on one line, within the core-property length limit."""
    plain = LINK_RE.sub(r"\1", heading).replace("**", "").replace("`", "")
    flat = " ".join(plain.split())
    return flat if len(flat) <= CORE_TEXT_MAX else flat[: CORE_TEXT_MAX - 1] + "…"


def _set_core_properties(document, lines: list[str]) -> None:
    """Replace python-docx's template metadata (author "python-docx", a 2013 date, a
    "generated by python-docx" description) with the document's own title and today's date.

    The title is the first heading outside a code fence, cut to the 255-character limit. The
    author is left blank: Word fills in whoever saves the document next.
    """
    core = document.core_properties
    now = datetime.now(UTC).replace(microsecond=0)
    core.title = _core_title(_first_heading(lines))
    core.author = core.last_modified_by = ""
    core.comments = core.subject = core.keywords = core.category = ""
    core.created = core.modified = now
    core.revision = 1


def convert(md_path: Path, out_path: Path, opts: Options) -> Path:
    """Write ``md_path`` to ``out_path`` as a colourless .docx; returns the output path."""
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = opts.body_font
    normal.font.size = Pt(opts.body_size)
    rpr = normal.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), opts.body_font)
    _blacken_styles(document)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if FENCE_RE.match(line):
            i += 1
            block: list[str] = []
            while i < len(lines) and not FENCE_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            i += 1
            add_code_block(document, block, opts)
            continue

        if not line.strip():
            i += 1
            continue

        if RULE_RE.match(line):
            document.add_paragraph()
            i += 1
            continue

        is_table = line.lstrip().startswith("|") and i + 1 < len(lines)
        if is_table and TABLE_SEP_RE.match(lines[i + 1]):
            rows = [split_row(line)]
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            add_table(document, rows, opts)
            continue

        first, i = line, i + 1
        cont: list[str] = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not _is_block_start(lines[i])
            and not FENCE_RE.match(lines[i])
        ):
            cont.append(lines[i])
            i += 1
        _emit_block(document, first, cont, opts)

    _set_core_properties(document, lines)
    set_app_properties(document)  # not the template's Word build in docProps/app.xml
    document.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--font", default="Calibri", help="Body font (default Calibri)")
    parser.add_argument("--mono", default="Consolas", help="Monospace font (default Consolas)")
    parser.add_argument("--size", type=float, default=10.5)
    parser.add_argument("--code-size", type=float, default=9.0)
    args = parser.parse_args()
    out = args.out or args.markdown.with_suffix(".docx")
    opts = Options(args.font, args.mono, args.size, args.code_size)
    convert(args.markdown, out, opts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
