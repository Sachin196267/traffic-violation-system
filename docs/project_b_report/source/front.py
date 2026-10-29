"""Cover, certificate, approval and declaration pages, placed at the reference report's measured positions."""
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

from engine import HERE, LEFT, TEXT_W, PAGE_H

LOGO = str(HERE / "logo.png")
PARA_OFF = 0.0
TITLE_LINES = ["TraffiSense: An Explainable, Digital-Twin-Driven", "Intelligent Traffic Management and", "Violation Detection System"]
TITLE = "TraffiSense: An Explainable, Digital-Twin-Driven Intelligent Traffic Management and Violation Detection System"
MEMBERS = [("Sachin Nair", "B737"), ("Ranjith Iyer", "B718"), ("Aditya Vijay Kumar", "B702")]
SUPERVISOR = "Dr. Siddharth Hariharan"
HOD = "Dr. Sharvari Govilkar"
PRINCIPAL = "Dr. Sandeep M. Joshi"


def sc(text, size=12):
    """small caps: capitals at full size, other letters upper-cased at 80 %"""
    out = []
    for ch in text:
        if ch.islower():
            out.append(f'<font size="{size * 0.8:.1f}">{ch.upper()}</font>')
        else:
            out.append(ch)
    return "".join(out)


def put_line(c, top, text, size=12, font="TNR", align="center", x=None):
    c.setFont(font, size)
    y = PAGE_H - top - 0.89 * size
    if align == "center":
        c.drawCentredString((LEFT + LEFT + TEXT_W) / 2, y, text)
    elif align == "left":
        c.drawString(x if x is not None else LEFT, y, text)
    else:
        c.drawRightString(x if x is not None else LEFT + TEXT_W, y, text)


def para(c, top, markup, size=12, leading=20.7, align=TA_JUSTIFY):
    st = ParagraphStyle("p", fontName="TNR", fontSize=size, leading=leading, alignment=align)
    p = Paragraph(markup, st)
    w, h = p.wrap(TEXT_W, 1000)
    # first baseline at top + 0.89*size: reportlab puts it at (box top - leading) + (leading - size)*... -> align by offset
    p.drawOn(c, LEFT, PAGE_H - top - h + PARA_OFF)
    return h


def sc_line(c, top, text, size=12, font="TNR"):
    """centered small-caps line drawn piecewise"""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    parts = []
    for ch in text:
        s = size * 0.8 if ch.islower() else size
        parts.append((ch.upper(), s))
    total = sum(stringWidth(ch, font, s) for ch, s in parts)
    x = LEFT + (TEXT_W - total) / 2
    y = PAGE_H - top - 0.89 * size
    for ch, s in parts:
        c.setFont(font, s); c.drawString(x, y, ch); x += stringWidth(ch, font, s)


def dept_header(c):
    c.drawImage(LOGO, 286, PAGE_H - 142, 72, 70, mask="auto")
    sc_line(c, 154, "Department of computer engineering")
    put_line(c, 175, "Pillai College of Engineering")
    put_line(c, 196, "New Panvel – 410 206")


def cover(c, doc):
    c.saveState()
    sc_line(c, 78, "A Project-b Report".replace("b R", "B R"))
    put_line(c, 102, "On")
    for i, t in enumerate(TITLE_LINES):
        put_line(c, 124 + 21 * i, t, 16, "TNR-B")
    put_line(c, 194, "Submitted in partial fulfillment of the requirement of")
    put_line(c, 216, "University of Mumbai for the Degree of")
    put_line(c, 239, "Bachelor of Technology", font="TNR-B")
    put_line(c, 261, "In", 11)
    put_line(c, 282, "Computer Engineering", font="TNR-B")
    put_line(c, 305, "Submitted By", 11)
    for i, (n, _) in enumerate(MEMBERS):
        put_line(c, 325 + 21 * i, n, font="TNR-B")
    put_line(c, 387, "Supervisor", 11)
    put_line(c, 407, SUPERVISOR, font="TNR-B")
    c.drawImage(LOGO, 289, PAGE_H - 516, 66, 64, mask="auto")
    put_line(c, 528, "Department Of Computer Engineering", font="TNR-B")
    # PILLAI COLLEGE OF ENGINEERING(AUTONOMOUS) with small-caps "Autonomous"
    from reportlab.pdfbase.pdfmetrics import stringWidth
    a, b = "PILLAI COLLEGE OF ENGINEERING(", "AUTONOMOUS"
    parts = [(a, 12)] + [(ch, 12 if i == 0 else 9.6) for i, ch in enumerate(b)] + [(")", 12)]
    total = sum(stringWidth(t, "TNR-B", s) for t, s in parts)
    x = LEFT + (TEXT_W - total) / 2; y = PAGE_H - 547 - 0.89 * 12
    for t, s in parts:
        c.setFont("TNR-B", s); c.drawString(x, y, t); x += stringWidth(t, "TNR-B", s)
    put_line(c, 566, "New Panvel – 410 206", font="TNR-B")
    put_line(c, 584, "UNIVERSITY OF MUMBAI", font="TNR-B")
    put_line(c, 603, "Academic Year 2026 - 27", font="TNR-B")
    c.restoreState()


def certificate(c, doc):
    c.saveState()
    dept_header(c)
    put_line(c, 217, "CERTIFICATE", 20)
    para(c, 251, f"This is to certify that the requirements for the project report entitled ‘<b>{TITLE}</b>’ "
                 "have been successfully completed by the following students:")
    put_line(c, 325, "Name", font="TNR-B", align="left", x=188)
    put_line(c, 325, "Roll No.", font="TNR-B", align="left", x=348)
    for i, (n, r) in enumerate(MEMBERS):
        put_line(c, 349 + 24.7 * i, n, align="left", x=188)
        put_line(c, 349 + 24.7 * i, r, align="left", x=347)
    para(c, 451, "In partial fulfillment of Bachelor of Technology, University of Mumbai in the Department of "
                 "Computer Engineering, Pillai College of Engineering, New Panvel – 410 206 during the Academic "
                 "Year 2026 – 2027.")
    def sign(cx, top, name, role):
        """signature block: ruled line, name (bold), designation - all centred on cx"""
        y = PAGE_H - top
        c.setLineWidth(0.6); c.line(cx - 85, y, cx + 85, y)
        c.setFont("TNR-B", 12); c.drawCentredString(cx, y - 18, name)
        c.setFont("TNR", 12); c.drawCentredString(cx, y - 34, role)
    mid = LEFT + TEXT_W / 2
    sign(mid, 545, SUPERVISOR, "Supervisor")
    sign(LEFT + TEXT_W / 4, 640, HOD, "Head, Dept. of Computer Engineering")
    sign(LEFT + 3 * TEXT_W / 4, 640, PRINCIPAL, "Principal")
    c.restoreState()


def approval(c, doc):
    c.saveState()
    dept_header(c)
    put_line(c, 236, "PROJECT APPROVAL FOR B.Tech", 20)
    names = ", ".join(n for n, _ in MEMBERS[:-1]) + f" and {MEMBERS[-1][0]}"
    para(c, 272, f"This project entitled “<b>{TITLE}</b>” by {names} is approved for the degree of "
                 "Bachelor of Technology in Computer Engineering.")
    for top, t, x in [(346, "Examiners:", 445), (375, "1. ________________", 452), (399, "2. ________________", 452),
                      (423, "Supervisors:", 450), (451, "1. ________________", 452), (475, "2. ________________", 452),
                      (499, "Chairman:", 449), (528, "1. ________________", 452)]:
        put_line(c, top, t, align="left", x=x)
    put_line(c, 568, "Date:", align="left", x=84)
    put_line(c, 596, "Place:", align="left", x=84)
    c.restoreState()


def declaration(c, doc):
    c.saveState()
    dept_header(c)
    put_line(c, 236, "DECLARATION", 20)
    para(c, 272, f"We declare that this written submission for the B.Tech project entitled “<b>{TITLE}</b>” "
                 "represents our ideas in our own words and where others' ideas or words have been included, we have "
                 "adequately cited and referenced the original sources. We also declare that we have adhered to all "
                 "principles of academic honesty and integrity and have not misrepresented or fabricated or falsified "
                 "any ideas / data / fact / source in our submission. We understand that any violation of the above "
                 "will cause disciplinary action by the institute and also evoke penal action from the sources which "
                 "have not been properly cited or from whom prior permission have not been taken when needed.")
    put_line(c, 448, "Project Group Members:", align="right", x=563)
    for i, (n, _) in enumerate(MEMBERS):
        put_line(c, 477 + 27.7 * i, f"{n} & Sign: __________________________", align="right", x=563)
    put_line(c, 577, "Date:", align="left", x=84)
    put_line(c, 606, "Place:", align="left", x=84)
    c.restoreState()


FRONT = [("cover", cover), ("certificate", certificate), ("approval", approval), ("declaration", declaration)]
