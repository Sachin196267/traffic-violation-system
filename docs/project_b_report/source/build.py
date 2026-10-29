"""Build the TraffiSense Project-B report. Two passes: the second fills in the page numbers of TOC / LOF / LOT."""
import sys
from pathlib import Path

from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Spacer

import engine as E
import story as ST
from engine import S, DotLine, Mark
from front import FRONT
from refs import REFS
import ch12, ch3, ch45

E.REFS.update(REFS)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else E.HERE / "TraffiSense_Project_B_Report.pdf"


def body():
    ST.reset()
    s = []
    s += ch12.abstract()
    lof_at = len(s)
    s += ch12.ch1() + ch12.ch2() + ch3.ch3() + ch45.ch4() + ch45.ch5()
    s += ch45.references() + ch45.acknowledgement() + ch45.publications()
    return s, lof_at


def toc_pages(entries):
    out = [Paragraph("Table of Contents", S["fronttitle"])]
    out.append(DotLine("Abstract", entries.get("abstract", "i"), 0, 29))
    out.append(DotLine("List of Figures", entries.get("lof", "ii"), 0, 29))
    out.append(DotLine("List of Tables", entries.get("lot", "iii"), 0, 29))
    for key, text, level in ST.TOC:
        if key in ("abstract",):
            continue
        out.append(DotLine(text, entries.get(key, "0"), 18 * level, 29))
    return out


def lists(entries):
    out = [PageBreak(), Mark("entry", key="lof"), Paragraph("List of Figures", S["fronttitle"])]
    for key, text in ST.LOF:
        out.append(DotLine(text, entries.get(key, "0"), 0, 16.6))
    out += [PageBreak(), Mark("entry", key="lot"), Paragraph("List of Tables", S["fronttitle"])]
    for key, text in ST.LOT:
        out.append(DotLine(text, entries.get(key, "0"), 0, 16.6))
    return out


def build(entries):
    main, lof_at = body()                       # fills ST.TOC / LOF / LOT
    story = [Spacer(1, 1)]
    for name, _ in FRONT[1:]:
        story += [NextPageTemplate(name), PageBreak(), Spacer(1, 1)]
    story += [NextPageTemplate("main"), PageBreak()]
    story += toc_pages(entries)
    # abstract (sets roman numbering), then the lists, then the chapters
    abstract = main[:lof_at]
    story += [PageBreak()] + abstract + lists(entries) + main[lof_at:]
    doc = E.Doc(OUT, FRONT)
    doc.build(story)
    return doc.seen, doc.page


if __name__ == "__main__":
    seen, n = build({})
    seen, n = build(seen)
    seen2, n2 = build(seen)
    print("pages:", n2, "stable:", seen2 == seen)
    print(OUT)
