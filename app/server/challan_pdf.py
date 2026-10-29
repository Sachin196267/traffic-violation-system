"""Official-style e-challan PDF (reportlab). All content comes from the stored finding / review / evidence."""
import time
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

INK = colors.HexColor("#111827"); MUTED = colors.HexColor("#6b7280"); LINE = colors.HexColor("#d1d5db")
ACCENT = colors.HexColor("#1f3a5f"); SOFT = colors.HexColor("#f3f4f6"); RED = colors.HexColor("#b42318")
S = {
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT, spaceBefore=10, spaceAfter=4, leading=11),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, textColor=INK, leading=12),
    "m": ParagraphStyle("m", fontName="Helvetica", fontSize=7.4, textColor=MUTED, leading=10),
    "k": ParagraphStyle("k", fontName="Helvetica", fontSize=7.4, textColor=MUTED, leading=10),
    "v": ParagraphStyle("v", fontName="Helvetica-Bold", fontSize=8.8, textColor=INK, leading=12),
}


def _plate_text(pl):
    if not pl:
        return "Not available"
    if pl.get("plate"):
        return f"{pl['plate']}  (read with {pl.get('confidence', 0) * 100:.0f} % confidence, multi-frame vote)"
    return {"NOT SURE": "Not confirmed — plate seen but not readable with certainty",
            "UNREADABLE": "Unreadable in this video", "NOT LINKED": "Not linked to a plate-tracked vehicle"}.get(pl.get("decision"), "Not available")


def _kv(rows, widths):
    t = Table([[Paragraph(k, S["k"]), Paragraph(str(v), S["v"])] for k, v in rows], colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE)]))
    return t


def _header(canv, doc, ch):
    w, h = A4
    canv.saveState()
    canv.setFillColor(ACCENT); canv.rect(0, h - 26 * mm, w, 26 * mm, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 14); canv.drawString(18 * mm, h - 13 * mm, "e-CHALLAN  ·  TRAFFIC VIOLATION NOTICE")
    canv.setFont("Helvetica", 8.5); canv.drawString(18 * mm, h - 19 * mm, "AI-assisted traffic enforcement · evidence-based record")
    canv.setFont("Helvetica-Bold", 10); canv.drawRightString(w - 18 * mm, h - 13 * mm, ch["number"])
    canv.setFont("Helvetica", 8); canv.drawRightString(w - 18 * mm, h - 19 * mm, time.strftime("%d %b %Y, %H:%M", time.localtime(ch["created"])))
    canv.setFillColor(MUTED); canv.setFont("Helvetica", 6.8)
    canv.drawString(18 * mm, 10 * mm, f"Evidence SHA-256: {ch['evidence_hash']}")
    canv.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")
    canv.restoreState()


def build(path: Path, ch: dict, f: dict, a: dict, review: dict | None, reviewer: dict | None, ev_dir: Path, extra=()):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=34 * mm, bottomMargin=18 * mm,
                            title=f"e-Challan {ch['number']}", author="Traffic Violation Platform")
    W = A4[0] - 36 * mm
    ev = f.get("evidence") or {}
    story = []

    # summary block + QR
    q = qr.QrCodeWidget(f"{ch['number']}|{ch['evidence_hash'][:32]}"); b = q.getBounds()
    sz = 26 * mm
    d = Drawing(sz, sz, transform=[sz / (b[2] - b[0]), 0, 0, sz / (b[3] - b[1]), 0, 0]); d.add(q)
    status = "Issued automatically (high-confidence AI decision)" if ch["source"] == "automatic" else "Issued after human review"
    charged = ch.get("offences") or [f["violation"]]
    left = _kv([("Offence" + ("s" if len(charged) > 1 else ""), " + ".join(charged)), ("Analysed", time.strftime("%d %b %Y", time.localtime(a["created"])) + f" · violation at {f['t']:.1f} s into the video"),
                ("Vehicle", (f.get("vehicle") or "—").title() + f" · track {f['track']}"),
                ("Number plate", _plate_text(ev.get("plate")) if (ev.get("plate") or {}).get("plate") or not ch.get("plate")
                 else f"{ch['plate']}  (entered by the reviewer from the evidence; AI reading was not confident)"),
                ("AI confidence", f"{f['confidence'] * 100:.0f} %"), ("Status", status)], [36 * mm, W - 36 * mm - 32 * mm])
    top = Table([[left, d]], colWidths=[W - 30 * mm, 30 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [top]

    # evidence
    story.append(Paragraph("EVIDENCE", S["h"]))
    imgs = []
    for fr in ev.get("frames", []):
        name = fr.get("boxed") or fr.get("raw")
        p = ev_dir / name
        if p.exists():
            im = Image(str(p)); r = im.imageHeight / im.imageWidth
            im.drawWidth = (W - 8 * mm) / 3; im.drawHeight = im.drawWidth * r
            imgs.append([im, Paragraph(f"{fr['t']:.1f} s" + (" · detection box" if fr.get("boxed") else ""), S["m"])])
    if imgs:
        t = Table([[x[0] for x in imgs], [x[1] for x in imgs]], colWidths=[(W) / 3] * len(imgs))
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, 0), 0.3, LINE)]))
        story.append(t)
    pcrop = ev_dir / "plate.jpg"
    if pcrop.exists() and (ev.get("plate") or {}).get("plate"):
        im = Image(str(pcrop)); im.drawHeight = 14 * mm; im.drawWidth = 14 * mm * im.imageWidth / im.imageHeight
        story += [Spacer(1, 4), Table([[im, Paragraph("Number-plate crop used for the reading. " + "; ".join(ev["plate"].get("reasons", [])[:2]), S["m"])]],
                                      colWidths=[im.drawWidth + 6, W - im.drawWidth - 6])]
    crop = ev_dir / "crop.jpg"
    if crop.exists():
        im = Image(str(crop)); im.drawHeight = 40 * mm; im.drawWidth = 40 * mm * im.imageWidth / im.imageHeight
        story += [Spacer(1, 4), Table([[im, Paragraph("Close-up of the detected vehicle / rider at the evidence time.", S["m"])]],
                                      colWidths=[im.drawWidth + 6, W - im.drawWidth - 6])]
    story.append(Paragraph(f"Video segment {f['t_start']:.1f}–{f['t_end']:.1f} s of “{a['filename']}”. Original video preserved; "
                           f"evidence files fingerprinted (SHA-256 in footer).", S["m"]))

    # one challan per vehicle: the other offences charged, then every other detection of this vehicle
    def shots(g, gdir, h):
        gev = g.get("evidence") or {}
        fr = next((x for x in gev.get("frames", []) if abs(x["t"] - g["t"]) < 0.05), (gev.get("frames") or [None])[0])
        out = []
        for pth in ([gdir / (fr.get("boxed") or fr.get("raw"))] if fr else []) + [gdir / "crop.jpg"]:
            if pth.exists():
                im = Image(str(pth)); im.drawHeight = h; im.drawWidth = h * im.imageWidth / im.imageHeight
                out.append(im)
        return out

    def block(g, gdir, h, lines):
        row = shots(g, gdir, h) + [Paragraph("<br/>".join(lines), S["m"])]
        wimg = sum(x.drawWidth + 6 for x in row[:-1])
        return Table([row], colWidths=[x.drawWidth + 6 for x in row[:-1]] + [W - wimg])
    more = []
    for g, gdir, o in extra:
        if o and g["violation"] in charged and g["violation"] != f["violation"]:
            story.append(Paragraph(f"ALSO CHARGED — {g['violation'].upper()}", S["h"]))
            story.append(block(g, gdir, 40 * mm, [f"Same vehicle, detected at {g['t']:.1f} s · AI confidence {g['confidence'] * 100:.0f} %"]
                               + (g.get("reasons") or [])[:4]))
        else:
            more.append((g, gdir))
    if more:
        story.append(Paragraph(f"MORE EVIDENCE OF THIS VEHICLE ({len(more)} other detection{'s' if len(more) > 1 else ''}, same case)", S["h"]))
        for g, gdir in more:
            story += [Spacer(1, 3), block(g, gdir, 28 * mm, [f"<b>{g['violation']}</b> at {g['t']:.1f} s · AI confidence {g['confidence'] * 100:.0f} %",
                                                              g.get("why") or "Same vehicle"])]
    not_charged = [o["violation"] for o in f.get("offences") or [] if o["violation"] not in charged]
    if not_charged:
        story.append(Paragraph("Also flagged for this vehicle but not certain enough, so NOT charged: " + ", ".join(not_charged) + ".", S["m"]))

    # explainable AI
    xai = [Paragraph("EXPLAINABLE AI — WHY THIS IS A VIOLATION", S["h"])]
    xai += [Paragraph(f"• {r}", S["b"]) for r in (f.get("reasons") or [])]
    mi = f.get("model_info") or {}
    xai.append(Spacer(1, 3))
    xai.append(_kv([("Model", mi.get("model") or mi.get("signal_model") or f["detector"]),
                    ("Decision rule", mi.get("rule") or "multi-frame decision with thresholds chosen on held-out cameras"),
                    ("Model decision", f["model_decision"]),
                    ("Frames supporting", ", ".join(f"{x['t']:.1f} s" for x in ev.get("frames", [])) or "—")], [36 * mm, W - 36 * mm]))
    story.append(KeepTogether(xai))

    # review
    if review:
        story.append(Paragraph("HUMAN REVIEW", S["h"]))
        story.append(_kv([("Reviewer", (reviewer or {}).get("name", "—")), ("Decision", "Create challan"),
                          ("Reviewed at", time.strftime("%d %b %Y, %H:%M", time.localtime(review["created"]))),
                          ("Why it needed review", "; ".join(f.get("uncertain_reasons") or []) or "—"),
                          ("Reviewer note", review.get("note") or "—")], [36 * mm, W - 36 * mm]))

    story.append(Paragraph("SYSTEM", S["h"]))
    story.append(_kv([("Analysis ID", a["id"]), ("Finding ID", f["id"]), ("Detector", f["detector"]),
                      ("Evidence reference", f"analyses/{a['id']}/evidence/{f['id']}/")], [36 * mm, W - 36 * mm]))
    story += [Spacer(1, 8), Paragraph("This notice was generated from recorded video by an AI-assisted system. Every challan is backed by the "
                                       "stored evidence listed above. Lane position is used as context only and never, on its own, "
                                       "results in a challan.", S["m"])]
    doc.build(story, onFirstPage=lambda c, d: _header(c, d, ch), onLaterPages=lambda c, d: _header(c, d, ch))
    return path
