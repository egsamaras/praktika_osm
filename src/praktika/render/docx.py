"""DOCX export.

Contract: ``render_docx`` writes a Word document with the classification and, for drafts, the
``DRAFT — NOT APPROVED`` text in the header and footer, a metadata table, the narrative sections,
decisions and actions tables with citations, reviewer flags (removed items with full text) and a
provenance footer. Text is split into script runs: only Arabic spans carry ``w:rtl``, and a
paragraph is marked bidirectional (``w:bidi``) only when Arabic letters make up at least half
of its letters, so code-switched or mostly English text keeps its left-to-right layout;
headings get the same treatment. The file is created 0600. Tokens are
restored only when policy allows (same rule as the Markdown renderer).

Core properties (``docProps/core.xml``) never carry python-docx's template values: the title is
the meeting title (as shown in the heading), author and last-modified-by are ``Praktika``,
created and modified are the export time in UTC, comments carry the classification line and
subject, keywords and category are empty. The extended properties (``docProps/app.xml``) name
``Praktika`` as the application and nothing else, and the template's thumbnail is dropped
(``set_app_properties``, used for every document Praktika saves).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part

from praktika.config import Settings
from praktika.models import Meeting, Minutes, Transcript
from praktika.render.bidi import add_text, paragraph_is_rtl, script_runs
from praktika.render.markdown import ARABIC_RE, build_view, detokeniser

__all__ = [
    "ARABIC_RE",
    "add_text",
    "paragraph_is_rtl",
    "render_docx",
    "script_runs",
    "set_app_properties",
]

Detok = Callable[[str], str]

AUTHOR = "Praktika"
APPLICATION = "Praktika"
_CORE_TEXT_MAX = 255  # OPC limit on a core-property string
_APP_XML = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n"
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
    f"<Application>{APPLICATION}</Application></Properties>"
).encode()


def set_app_properties(doc: Any) -> None:
    """Replace the template's extended properties with Praktika's own, before ``doc.save``.

    python-docx copies its template's ``docProps/app.xml`` verbatim, which names the Word build
    the template was last saved with (application, version, company, template name and page
    statistics of an empty page). The replacement names ``Praktika`` as the application and
    nothing else. The template's thumbnail, a picture of its blank page, is dropped as well.
    """
    package = doc.part.package
    for rid, rel in list(package.rels.items()):
        if rel.reltype in (RT.EXTENDED_PROPERTIES, RT.THUMBNAIL):
            package.rels.pop(rid)
    part = Part(PackURI("/docProps/app.xml"), CT.OFC_EXTENDED_PROPERTIES, _APP_XML, package)
    package.relate_to(part, RT.EXTENDED_PROPERTIES)


def _core_text(value: str) -> str:
    """``value`` on one line and within the core-property length limit."""
    flat = " ".join(value.split())
    return flat if len(flat) <= _CORE_TEXT_MAX else flat[: _CORE_TEXT_MAX - 1] + "…"


def _set_core_properties(doc: Any, view: dict[str, Any], detok: Detok, stamp: datetime) -> None:
    """Replace the template's core properties with the export's own (see module docstring).

    ``stamp`` is the export time; a naive value is taken as local time and stored in UTC
    (python-docx writes the value with a ``Z`` suffix without converting it).
    """
    utc = stamp.astimezone(UTC)
    cp = doc.core_properties
    cp.title = _core_text(detok(view["title"]))
    cp.author = AUTHOR
    cp.last_modified_by = AUTHOR
    cp.comments = _core_text(_banner_line(view))
    cp.subject = ""
    cp.keywords = ""
    cp.category = ""
    cp.created = utc
    cp.modified = utc
    cp.revision = 1


def _heading(doc: Any, text: str, level: int) -> Any:
    """A heading rendered through ``add_text`` so Arabic titles get the same treatment."""
    heading = doc.add_heading("", level=level)
    add_text(heading, text)
    return heading


def _para(doc: Any, text: str, *, style: str | None = None, bold: bool = False) -> Any:
    p = doc.add_paragraph(style=style)
    add_text(p, text, bold=bold)
    return p


def _cites(doc: Any, refs: list[dict[str, Any]]) -> None:
    for r in refs:
        p = doc.add_paragraph(style="List Bullet 2")
        add_text(p, f"[{r['id']} {r['time']} {r['speaker']}] ")
        add_text(p, f"“{r['quote']}”", italic=True)
        if r["gloss"]:
            g = doc.add_paragraph(style="List Bullet 3")
            add_text(g, f"Gloss (EN): {r['gloss']}")


def _cite_text(refs: list[dict[str, Any]]) -> str:
    return "\n".join(f"{r['id']} {r['time']} {r['speaker']}: “{r['quote']}”" for r in refs)


def _cell_lines(cell: Any, value: str) -> None:
    """One paragraph per line of ``value`` so each citation gets its own direction."""
    cell.text = ""
    lines = value.split("\n") or [""]
    add_text(cell.paragraphs[0], lines[0])
    for line in lines[1:]:
        add_text(cell.add_paragraph(), line)


def _table(doc: Any, headers: list[str], rows: list[list[str]]) -> Any:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, head in zip(table.rows[0].cells, headers, strict=True):
        cell.text = ""
        add_text(cell.paragraphs[0], head, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row, strict=True):
            _cell_lines(cell, value)
    return table


def _banner_line(view: dict[str, Any]) -> str:
    parts = [f"Classification: {view['classification']}"]
    if view["banner"]:
        parts.append(view["banner"])
    return " · ".join(parts)


def _provenance_lines(view: dict[str, Any]) -> list[str]:
    p, r = view["provenance"], view["review"]
    engines = ", ".join(f"{k}={v}" for k, v in p["stt_engines"].items())
    return [
        f"Generator: {p['generator_model']} ({p['model_digest']})"
        + (" — degraded model" if p["degraded"] else ""),
        f"Prompts {p['prompt_version']} sha256 {p['prompt_sha256']}; "
        f"glossary {p['glossary_sha256']}",
        f"Template {p['template']}; git {p['git_sha']}; transcript sha256 {p['transcript_sha256']}",
        f"STT: {engines or '—'}; generated {p['generated_at']}; rendered {view['rendered_at']}",
        f"Reviewer: {r['reviewer']} ({r['reviewer_source']}) at {r['reviewed_at']}",
    ]


def _write_sections(doc: Any, v: dict[str, Any], d: Detok) -> None:
    _heading(doc, d(v["title"]), 1)
    if v["banner"]:
        _para(doc, v["banner"], bold=True)
    meta = [
        ["Meeting id", f"{v['meeting_id']} (minutes v{v['version']})"],
        ["Date", v["date"]],
        ["Type", f"{v['meeting_type']} · {v['platform']}"],
        ["Chair", v["chair"]],
        ["Organiser", v["organiser"]],
        ["Classification", v["classification"]],
        ["Language profile", v["language_profile"]],
        ["Status", v["review"]["status"]],
    ]
    if "next_one_to_one" in v:
        meta.append(["Next one-to-one", v["next_one_to_one"]])
    _table(doc, ["Field", "Value"], meta)

    doc.add_heading("Attendees", level=2)
    for a in v["attendees"]:
        role = f" — {a['role']}" if a["role"] else ""
        org = f"{a['organisation']}, " if a["organisation"] else ""
        _para(doc, f"{a['name']}{role} ({org}{a['status']})", style="List Bullet")
    if v.get("agenda"):
        doc.add_heading("Agenda", level=2)
        for a in v["agenda"]:
            paper = f" (paper {a['paper_ref']})" if a["paper_ref"] else ""
            who = f" — {a['presenter']}" if a["presenter"] else ""
            _para(doc, f"{a['item_no']}. {d(a['title'])}{paper}{who}")
    if v.get("matters_arising"):
        doc.add_heading("Matters arising", level=2)
        for m in v["matters_arising"]:
            _para(
                doc,
                f"{m['previous_action_id']} — {m['status']}: {d(m['note'])}",
                style="List Bullet",
            )
            _cites(doc, m["refs"])

    doc.add_heading("Summary", level=2)
    _para(doc, d(v["summary"]))
    doc.add_heading("Topics", level=2)
    for t in v["topics"]:
        _heading(doc, d(t["title"]), 3)
        _para(doc, d(t["summary"]))
        for point in t["key_points"]:
            _para(doc, d(point), style="List Bullet")
        _cites(doc, t["refs"])

    doc.add_heading("Decisions", level=2)
    _table(
        doc,
        ["ID", "Decision", "Kind", "Decided by", "Evidence"],
        [
            [
                x["id"],
                d(x["statement"])
                + (f"\nDissent/conditions: {d(x['dissent'])}" if x["dissent"] else ""),
                x["kind"],
                x["decided_by"],
                d(_cite_text(x["refs"])),
            ]
            for x in v["decisions"]
        ],
    )
    action_groups = [("Actions", v["actions"])]
    if "my_commitments" in v:
        action_groups = [
            ("Organiser's commitments", v["my_commitments"]),
            ("Other party's commitments", v["their_commitments"]),
            ("Other actions", v["other_actions"]),
        ]
    for heading, items in action_groups:
        doc.add_heading(heading, level=2)
        _table(
            doc,
            ["ID", "Action", "Owner", "Due", "Evidence"],
            [
                [
                    a["id"],
                    d(a["description"]),
                    f"{a['owner']} ({a['owner_confidence']})",
                    a["due"],
                    d(_cite_text(a["refs"])),
                ]
                for a in items
            ],
        )

    doc.add_heading("Open questions", level=2)
    for q in v["questions"]:
        _para(
            doc,
            d(q["question"]) + (f" (raised by {q['raised_by']})" if q["raised_by"] else ""),
            style="List Bullet",
        )
        _cites(doc, q["refs"])
    doc.add_heading("Risks", level=2)
    for r in v["risks"]:
        mit = f" — mitigation: {d(r['mitigation'])}" if r["mitigation"] else ""
        _para(doc, f"[{r['severity']}] {d(r['description'])}{mit}", style="List Bullet")
        _cites(doc, r["refs"])
    if v.get("figures") is not None:
        doc.add_heading("Figures mentioned", level=2)
        for f in v["figures"]:
            _para(
                doc,
                f"{f['value']} — {d(f['context'])} [{', '.join(f['refs'])}]",
                style="List Bullet",
            )
        doc.add_heading("Escalations to the Board", level=2)
        for e in v["escalations"]:
            _para(doc, d(e), style="List Bullet")
    doc.add_heading("Follow-ups", level=2)
    for f in v["follow_ups"]:
        _para(doc, d(f), style="List Bullet")

    doc.add_heading("Reviewer flags", level=2)
    if v["removed"]:
        _para(doc, "Removed items (uncited; restore from the review page if correct):", bold=True)
        for f in v["removed"]:
            _para(doc, f"(priority {f['priority']}) {d(f['detail'])}", style="List Bullet")
    for f in v["flags"]:
        _para(doc, f"{f['kind']} (priority {f['priority']}): {d(f['detail'])}", style="List Bullet")
        _cites(doc, f["refs"])
    doc.add_heading("Provenance", level=2)
    for line in _provenance_lines(v):
        _para(doc, line)


def render_docx(
    minutes: Minutes,
    meeting: Meeting,
    transcript: Transcript | None,
    out: Path,
    *,
    vault: Any | None = None,
    settings: Settings,
    now: datetime | None = None,
) -> Path:
    """Write the DOCX export to ``out`` (mode 0600) and return the path.

    ``now`` is the export time (default: the current time); it stamps the rendered-at line and
    the document's created/modified core properties.
    """
    stamp = now or datetime.now().astimezone()
    view = build_view(minutes, meeting, transcript, settings=settings, now=stamp)
    detok = detokeniser(minutes, vault, settings)
    doc = Document()
    _set_core_properties(doc, view, detok, stamp)
    set_app_properties(doc)
    section = doc.sections[0]
    add_text(section.header.paragraphs[0], _banner_line(view), bold=True)
    footer = section.footer.paragraphs[0]
    add_text(footer, _banner_line(view) + " · ", bold=True)
    add_text(footer, _provenance_lines(view)[0])
    _write_sections(doc, view, detok)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    os.close(fd)
    doc.save(str(out))
    os.chmod(out, 0o600)
    return out
