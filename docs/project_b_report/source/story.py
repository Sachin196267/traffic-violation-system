"""Builders shared by the chapter files."""
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib import colors

import engine as E
from engine import S, State, Mark, fmt

TOC = []      # (key, text, level)
LOF = []      # (key, text)
LOT = []


def reset():
    TOC.clear(); LOF.clear(); LOT.clear(); E.ORDER.clear()
    State.chapter = State.fig = State.tab = State.eq = 0
    E._eqn = 0


def chapter(n, title, toc_title=None):
    State.chapter, State.fig, State.tab, State.eq = n, 0, 0, 0
    key = f"ch{n}"
    TOC.append((key, f"{n}. {toc_title or title}", 0))
    out = [PageBreak()]
    if n == 1:
        out.append(Mark("mode", mode="arabic"))
    out += [Mark("entry", key=key), Paragraph(f"Chapter {n}", S["chap"]), Paragraph(title, S["chapT"])]
    return out


def unnumbered(title, key, mode=None):
    TOC.append((key, title, 0))
    out = [PageBreak()]
    if mode:
        out.append(Mark("mode", mode=mode))
    out += [Mark("entry", key=key), Paragraph(title, S["fronttitle"])]
    return out


def h(num, title, level=1):
    key = f"s{num}"
    TOC.append((key, f"{num} {title}", level))
    return [CondPageBreak(60), Mark("entry", key=key), Paragraph(f"{num} {title}", S["h1"])]


def h3(text):
    return [Paragraph(text, S["h3"])]


def p(text, style="body"):
    st = S[style]
    if text.rstrip().endswith(":"):          # a lead-in line stays with the list or equation after it
        from reportlab.lib.styles import ParagraphStyle
        st = ParagraphStyle(st.name + "K", parent=st, keepWithNext=1)
    return [Paragraph(fmt(text), st)]


def bullets(items):
    return [KeepTogether([Paragraph(fmt(t), S["bullet"], bulletText="•") for t in items])]


def numbered(items):
    return [KeepTogether([Paragraph(fmt(t), S["bullet"], bulletText=f"{i}.") for i, t in enumerate(items, 1)])]


def code(lines):
    return [KeepTogether([Spacer(1, 3)] + [Paragraph(l.replace(" ", "&nbsp;") or "&nbsp;", S["code"]) for l in lines] + [Spacer(1, 8)])]


def table(caption, header, rows, widths, small=False, align_first_center=False):
    State.tab += 1
    num = f"{State.chapter}.{State.tab}"
    key = f"t{num}"
    LOT.append((key, f"Table {num}  {caption}"))
    cs, hs = (S["cellS"], S["cellSB"]) if small else (S["cell"], S["cellB"])
    tw = E.TABLE_W
    data = [[Paragraph(fmt(h_), hs) for h_ in header]] + [[Paragraph(fmt(str(c)), cs) for c in r] for r in rows]
    t = Table(data, colWidths=[w * tw for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5)]
    t.setStyle(TableStyle(st))
    cap = Paragraph(f"Table {num}: {caption}", S["cap"])
    return [E.Float([Spacer(1, 4), Mark("entry", key=key), cap, t, Spacer(1, 12)])], f"Table {num}"


def figure(path, caption, width=440, max_h=430):
    State.fig += 1
    num = f"{State.chapter}.{State.fig}"
    key = f"f{num}"
    LOF.append((key, f"Fig {num}  {caption}"))
    from PIL import Image as PI
    w, hh = PI.open(path).size
    h_ = width * hh / w
    if h_ > max_h:
        width, h_ = width * max_h / h_, max_h
    img = Image(str(path), width=width, height=h_)
    return [E.Float([Spacer(1, 4), img, Mark("entry", key=key), Paragraph(f"Fig {num}: {caption}", S["figcap"])])], f"Figure {num}"


def eq(tex, size=15):
    t, num = E.equation(tex, size)
    return [t], num
