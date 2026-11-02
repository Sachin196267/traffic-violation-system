"""Layout engine that reproduces the reference report's page geometry and typography (Letter, Times New Roman)."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, NextPageTemplate, CondPageBreak)

HERE = Path(__file__).parent
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)

SUP = "/System/Library/Fonts/Supplemental/"
for name, f in [("TNR", "Times New Roman.ttf"), ("TNR-B", "Times New Roman Bold.ttf"),
                ("TNR-I", "Times New Roman Italic.ttf"), ("TNR-BI", "Times New Roman Bold Italic.ttf"),
                ("COUR", "Courier New.ttf"), ("COUR-B", "Courier New Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, SUP + f))
addMapping("TNR", 0, 0, "TNR"); addMapping("TNR", 1, 0, "TNR-B"); addMapping("TNR", 0, 1, "TNR-I"); addMapping("TNR", 1, 1, "TNR-BI")
addMapping("COUR", 0, 0, "COUR"); addMapping("COUR", 1, 0, "COUR-B"); addMapping("COUR", 0, 1, "COUR"); addMapping("COUR", 1, 1, "COUR-B")

PAGE_W, PAGE_H = 612, 792
LEFT, RIGHT_X = 84, 563
TEXT_W = RIGHT_X - LEFT          # 479
TABLE_W = 476
LEAD = 20.7                      # 12 pt at 1.5 line spacing (Word)

S = {
    "body": ParagraphStyle("body", fontName="TNR", fontSize=12, leading=LEAD, alignment=TA_JUSTIFY, spaceAfter=5.3),
    "bodyL": ParagraphStyle("bodyL", fontName="TNR", fontSize=12, leading=LEAD, alignment=TA_LEFT, spaceAfter=5.3),
    "chap": ParagraphStyle("chap", fontName="TNR-B", fontSize=15.9, leading=20, alignment=TA_CENTER, spaceAfter=11),
    "chapT": ParagraphStyle("chapT", fontName="TNR-B", fontSize=15.9, leading=20, alignment=TA_CENTER, spaceAfter=19),
    "h1": ParagraphStyle("h1", fontName="TNR-B", fontSize=12, leading=20, spaceBefore=6, spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="TNR-BI", fontSize=12, leading=20, spaceBefore=4, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="TNR-B", fontSize=12, leading=15, alignment=TA_CENTER, spaceBefore=6, spaceAfter=7, keepWithNext=1),
    "figcap": ParagraphStyle("figcap", fontName="TNR-B", fontSize=12, leading=15, alignment=TA_CENTER, spaceBefore=6, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=11, leading=12.6),
    "cellB": ParagraphStyle("cellB", fontName="TNR-B", fontSize=11, leading=12.6, alignment=TA_CENTER),
    "cellS": ParagraphStyle("cellS", fontName="TNR", fontSize=9.9, leading=11.6),
    "cellSB": ParagraphStyle("cellSB", fontName="TNR-B", fontSize=9.9, leading=11.6, alignment=TA_CENTER),
    "bullet": ParagraphStyle("bullet", fontName="TNR", fontSize=12, leading=LEAD, alignment=TA_JUSTIFY, leftIndent=36,
                             bulletIndent=18, spaceAfter=2.5, bulletFontName="TNR"),
    "code": ParagraphStyle("code", fontName="COUR", fontSize=10, leading=13, alignment=TA_LEFT, spaceAfter=0),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=12, leading=15, alignment=TA_JUSTIFY, leftIndent=30,
                          firstLineIndent=-30, spaceAfter=8),
    "fronttitle": ParagraphStyle("ft", fontName="TNR-B", fontSize=15.9, leading=20, alignment=TA_CENTER, spaceAfter=13),
}

# ---------------------------------------------------------------- citations (numbered by first appearance)
REFS = {}
ORDER = []


def cites(text):
    def rep(m):
        out = []
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in REFS:
                raise KeyError(k)
            if k not in ORDER:
                ORDER.append(k)
            out.append(f"[{list(REFS).index(k) + 1}]")
        return "".join(out)
    return re.sub(r"\[@([\w,\s]+)\]", rep, text)


def fmt(t):
    t = cites(t)
    t = re.sub(r"`([^`]+)`", r'<font name="COUR" size="10.5">\1</font>', t)
    return t


# ---------------------------------------------------------------- numbering state
class State:
    chapter = 0
    fig = 0
    tab = 0
    eq = 0


ENTRIES = {}          # key -> page label from the previous pass


class Mark(Flowable):
    """zero-size marker: page numbering mode switch or TOC/LOF/LOT entry"""
    def __init__(self, kind, key=None, text=None, level=0, mode=None):
        super().__init__()
        self.kind, self.key, self.text, self.level, self.mode = kind, key, text, level, mode
        self.width = self.height = 0

    def wrap(self, *a):
        return 0, 0

    def draw(self):
        pass


def roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s; n -= v
    return out


class Float(KeepTogether):
    """a figure or table kept whole; when it does not fit, the following text moves up to fill the page"""


def _pullable(f):
    if isinstance(f, (Float, PageBreak)) or isinstance(f, DotLine if "DotLine" in globals() else ()):
        return False
    if isinstance(f, Paragraph):
        return f.style.name not in ("chap", "chapT", "ft")
    return isinstance(f, (Spacer, Mark, CondPageBreak, Table, Image, KeepTogether))


def _heading(f):
    return isinstance(f, Paragraph) and getattr(f.style, "keepWithNext", 0)


class Doc(BaseDocTemplate):
    def __init__(self, path, front_pages):
        super().__init__(str(path), pagesize=(PAGE_W, PAGE_H), leftMargin=LEFT, rightMargin=PAGE_W - RIGHT_X,
                         topMargin=70, bottomMargin=66, title="TraffiSense: An Explainable, Digital-Twin-Driven Intelligent Traffic Management and Violation Detection System - Project-B Report",
                         author="Sachin Nair, Ranjith Iyer, Aditya Vijay Kumar")
        self.mode, self.base, self.seen = None, 0, {}
        frame = Frame(LEFT, 66, TEXT_W, PAGE_H - 70 - 66, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="f")
        tpls = [PageTemplate(id=k, frames=[frame], onPage=fn) for k, fn in front_pages]
        tpls.append(PageTemplate(id="main", frames=[frame], onPageEnd=self._footer))
        self.addPageTemplates(tpls)

    def handle_flowable(self, flowables):
        fl = flowables
        # a lead-in line before a kept-together block (list, code) joins that block
        if len(fl) > 1 and isinstance(fl[0], Paragraph) and _heading(fl[0]) and type(fl[1]) is KeepTogether:
            fl[0:2] = [KeepTogether([fl[0]] + list(fl[1]._content))]
        # heading directly followed by a float: let the first paragraph of the section come before the float
        if fl and _heading(fl[0]):
            j = 1
            while j < len(fl) and isinstance(fl[j], Mark):
                j += 1
            if j + 1 < len(fl) and isinstance(fl[j], Float) and isinstance(fl[j + 1], Paragraph) and not _heading(fl[j + 1]):
                fl[j], fl[j + 1] = fl[j + 1], fl[j]
        if fl and isinstance(fl[0], Float):
            fr = self.frame
            avail = fr._y - fr._y1p
            full = fr._height - fr._topPadding - fr._bottomPadding
            fl[0].canv = self.canv
            fl[0].wrap(fr._aW, avail); h = fl[0]._H
            if h > avail and not fr._atTop:
                imgs = [c for c in fl[0]._content if isinstance(c, Image)]
                if imgs:
                    im = imgs[0]
                    k = (avail - (h - im.drawHeight) - 2) / im.drawHeight
                    if k >= 0.85:
                        im.drawWidth *= k; im.drawHeight *= k
                        im._width, im._height = im.drawWidth, im.drawHeight
                        fl[0].wrap(fr._aW, avail); h = fl[0]._H
            if h > avail and h <= full and not fr._atTop and avail > 0.3 * full:
                # take the next text block (a heading travels with the item after it)
                j = 1
                take = []
                while j < len(fl) and isinstance(fl[j], (Mark, CondPageBreak, Float, Spacer)):
                    if not isinstance(fl[j], Float):
                        take.append(j)
                    j += 1
                if j < len(fl) and _pullable(fl[j]):
                    take.append(j)
                    if _heading(fl[j]):
                        k = j + 1
                        while k < len(fl) and isinstance(fl[k], Mark):
                            take.append(k); k += 1
                        if k < len(fl) and _pullable(fl[k]) and not _heading(fl[k]):
                            take.append(k)
                        else:
                            take = []
                    if take:
                        moved = [fl[i] for i in take]
                        for i in reversed(take):
                            del fl[i]
                        fl[0:0] = moved
        return super().handle_flowable(flowables)

    def label(self):
        n = self.page - self.base
        return roman(n) if self.mode == "roman" else str(n)

    def afterFlowable(self, f):
        if isinstance(f, Mark):
            if f.kind == "mode":
                self.mode = f.mode
                self.base = self.page - 1
            elif f.kind == "entry":
                self.seen[f.key] = self.label() if self.mode else ""

    def _footer(self, canv, doc):
        if self.mode in ("roman", "arabic"):
            canv.setFont("TNR", 12)
            canv.drawCentredString((LEFT + RIGHT_X) / 2, 38, self.label())


# ---------------------------------------------------------------- dotted TOC line
class DotLine(Flowable):
    def __init__(self, text, page, indent=0, height=29, bold=False):
        super().__init__()
        self.text, self.page, self.indent, self.h, self.bold = text, page, indent, height, bold

    def wrap(self, aw, ah):
        return TEXT_W, self.h

    def draw(self):
        c = self.canv
        font = "TNR-B" if self.bold else "TNR"
        c.setFont(font, 12)
        y = self.h - 12
        c.drawString(self.indent, y, self.text)
        tw = pdfmetrics.stringWidth(self.text, font, 12)
        pw = pdfmetrics.stringWidth(self.page, "TNR", 12)
        c.setFont("TNR", 12)
        c.drawRightString(TEXT_W - 1, y, self.page)
        dot = pdfmetrics.stringWidth(".", "TNR", 12)
        x0, x1 = self.indent + tw + 3, TEXT_W - 1 - pw - 3
        n = max(0, int((x1 - x0) / dot))
        c.drawString(x1 - n * dot, y, "." * n)


# ---------------------------------------------------------------- equations
_eqn = 0


def _render(tex, size):
    global _eqn
    _eqn += 1
    p = FIG / f"eq_{_eqn}.png"
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, "$" + tex.replace(r"\frac", r"\dfrac") + "$", fontsize=size)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    from PIL import Image as PI
    w, h = PI.open(p).size
    return Image(str(p), width=w * 72 / 300, height=h * 72 / 300)


def equation(tex, size=15):
    lines = tex if isinstance(tex, list) else [tex]
    imgs = [_render(t, size) for t in lines]
    State.eq += 1
    num = f"({State.chapter}.{State.eq})"
    nump = Paragraph(num, ParagraphStyle("eqn", fontName="TNR", fontSize=12, alignment=2))
    rows = [[im, nump if i == len(imgs) - 1 else ""] for i, im in enumerate(imgs)]
    t = Table(rows, colWidths=[TEXT_W - 50, 50])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, -1), "LEFT" if len(imgs) > 1 else "CENTER"),
                           ("LEFTPADDING", (0, 0), (0, -1), 60 if len(imgs) > 1 else 0),
                           ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("LEFTPADDING", (1, 0), (1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t, num
