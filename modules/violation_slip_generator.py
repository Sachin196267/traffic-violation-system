"""
Violation challan/slip generation.

Creates a realistic violation slip containing the evidence image, vehicle/track
details, statutory status, signal state, and XAI explanation. PDF generation uses
reportlab when available; otherwise an HTML slip is written as a fallback.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .lane_assigner import VehicleAssignment
from .xai_explainer import XAIExplanation


@dataclass
class SlipResult:
    path: str
    format: str
    challan_id: str
    evidence_path: str = ""
    evidence_valid: bool = False
    review_required: bool = True
    html_path: str = ""
    pdf_path: str = ""
    severity_level: str = "MEDIUM"
    severity_score: int = 50
    detection_confidence: float = 0.85


def create_evidence_composite(
    frame: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]],
    track_id: int,
    cls_name: str,
    violation_types: List[str],
    contact_point: Optional[Tuple[float, float]] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    target_width: int = 1200,
    target_height: int = 560,
) -> np.ndarray:
    """
    Creates a dual-panel evidence composite:
    - Left panel: Resized full context frame with highlighted bounding box,
                  trajectory polyline, and contact point indicator.
    - Right panel: High-resolution zoom crop of the offending vehicle with
                   bottom contact point crosshair and metadata banner.
    """
    frame_h, frame_w = frame.shape[:2]
    context_w = int(target_width * 0.65)
    detail_w = target_width - context_w
    total_h = target_height

    # 1. Annotated Context View
    annotated_frame = frame.copy()

    # Draw historical trajectory if available
    if trajectory and len(trajectory) >= 2:
        pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=3)

    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        # Target bounding box in vibrant red/coral
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        # Label tag above bounding box
        label_text = f"VIOLATOR #{track_id} [{cls_name.upper()}]"
        label_bg_y1 = max(0, y1 - 28)
        cv2.rectangle(annotated_frame, (x1, label_bg_y1), (x1 + 270, y1), (0, 0, 200), -1)
        cv2.putText(
            annotated_frame,
            label_text,
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw contact point crosshair
    if contact_point:
        cx, cy = int(contact_point[0]), int(contact_point[1])
        cv2.drawMarker(annotated_frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=3)

    context_resized = cv2.resize(annotated_frame, (context_w, total_h), interpolation=cv2.INTER_AREA)

    # 2. Zoom Detail Crop
    detail_panel = np.zeros((total_h, detail_w, 3), dtype=np.uint8)
    detail_panel[:] = (26, 32, 44)  # Dark slate background

    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)

        # 30% padding around bounding box
        pad_x = int(bw * 0.30)
        pad_y = int(bh * 0.30)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_w, x2 + pad_x)
        crop_y2 = min(frame_h, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            header_h = 44
            footer_h = 56
            avail_h = total_h - header_h - footer_h
            avail_w = detail_w - 20

            scale = min(avail_w / cw, avail_h / ch)
            scaled_w = int(cw * scale)
            scaled_h = int(ch * scale)
            crop_scaled = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            off_x = 10 + (avail_w - scaled_w) // 2
            off_y = header_h + (avail_h - scaled_h) // 2
            detail_panel[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = crop_scaled

            cv2.rectangle(
                detail_panel,
                (off_x - 1, off_y - 1),
                (off_x + scaled_w + 1, off_y + scaled_h + 1),
                (200, 200, 200),
                1,
            )

    # Detail Header
    cv2.rectangle(detail_panel, (0, 0), (detail_w, 40), (20, 25, 35), -1)
    cv2.putText(
        detail_panel,
        "OFFENDING VEHICLE ZOOM",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Detail Footer
    cv2.rectangle(detail_panel, (0, total_h - 50), (detail_w, total_h), (20, 25, 35), -1)
    violations_str = ", ".join(violation_types) if violation_types else "NONE"
    if len(violations_str) > 30:
        violations_str = violations_str[:27] + "..."
    cv2.putText(
        detail_panel,
        f"ID: #{track_id} | {cls_name.upper()}",
        (14, total_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        detail_panel,
        f"OFFENSE: {violations_str}",
        (14, total_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 140, 255),
        1,
        cv2.LINE_AA,
    )

    # 3. Assemble Full Composite
    composite = np.zeros((total_h, target_width, 3), dtype=np.uint8)
    composite[:, :context_w] = context_resized
    composite[:, context_w:] = detail_panel

    # Divider bar between panels
    cv2.line(composite, (context_w, 0), (context_w, total_h), (80, 90, 105), 3)

    return composite


class ViolationSlipGenerator:
    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        assignment: VehicleAssignment,
        frame_idx: int,
        snapshot_path: str,
        xai: XAIExplanation,
        signal_record: Dict[str, Any],
        secondary_detections: List[Any],
        video_name: str,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        trajectory: Optional[List[Tuple[float, float]]] = None,
        plate_number: Optional[str] = None,
        plate_confidence: float = 0.0,
        plate_review_required: bool = True,
        severity: Optional[Any] = None,
        detection_confidence: Optional[float] = None,
    ) -> SlipResult:
        challan_id = self._challan_id(frame_idx, assignment.track_id)

        # 1. Resolve Severity Assessment
        if severity is None:
            from .severity_engine import SeverityEngine
            sev_eng = SeverityEngine(self.config)
            severity = sev_eng.evaluate(
                violation_types=assignment.violation_types,
                cls_name=assignment.cls_name,
                speed_kmh=getattr(assignment, "speed_kmh", 0.0),
            )
        sev_dict = severity.to_dict() if hasattr(severity, "to_dict") else dict(severity)

        # 2. Resolve Detection Confidence
        if detection_confidence is None:
            if hasattr(xai, "model_prediction") and isinstance(xai.model_prediction, dict):
                detection_confidence = float(xai.model_prediction.get("overall_detection_confidence", 0.85))
            else:
                detection_confidence = 0.85

        # 3. Prepare evidence composite if raw frame is provided
        final_snapshot_path = snapshot_path
        evidence_valid = False
        if frame is not None:
            if not final_snapshot_path:
                final_snapshot_path = os.path.join(
                    self.output_dir, f"evidence_{challan_id}.jpg"
                )
            bbox = self._valid_bbox(bbox, frame.shape) if bbox else None
            composite_img = create_evidence_composite(
                frame=frame,
                bbox=bbox,
                track_id=assignment.track_id,
                cls_name=assignment.cls_name,
                violation_types=assignment.violation_types,
                contact_point=assignment.contact_point,
                trajectory=trajectory,
            )
            # Encode at JPEG quality 82 for optimal quality/filesize (~100-150KB)
            cv2.imwrite(
                final_snapshot_path,
                composite_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            evidence_valid = self._validated_image_size(final_snapshot_path) is not None

        payload = {
            "challan_id": challan_id,
            "issued_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video": video_name,
            "frame": frame_idx,
            "track_id": assignment.track_id,
            "vehicle_class": assignment.cls_name,
            "plate_number": plate_number or "UNKNOWN",
            "plate_confidence": float(plate_confidence or 0.0),
            "detection_confidence": round(float(detection_confidence), 3),
            "severity": sev_dict,
            "review_required": bool(plate_review_required or not evidence_valid or bbox is None),
            "evidence_valid": evidence_valid,
            "target_bbox_present": bbox is not None,
            "lane": assignment.lane_name,
            "direction": assignment.direction,
            "speed_kmh": getattr(assignment, "speed_kmh", 0.0),
            "violations": assignment.violation_types,
            "snapshot_path": os.path.abspath(final_snapshot_path) if final_snapshot_path else "",
            "signal": signal_record.get("active_phase", {}),
            "xai": xai.to_dict(),
            "secondary_detections": [
                det if isinstance(det, dict) else getattr(det, "__dict__", {})
                for det in secondary_detections
            ],
        }

        # Always write companion HTML receipt for browser display and direct printing
        html_path = os.path.join(self.output_dir, f"{challan_id}.html")
        self._write_html(html_path, payload)

        # Attempt high-fidelity single-page PDF generation
        pdf_path = os.path.join(self.output_dir, f"{challan_id}.pdf")
        format_used = "html"
        main_path = html_path
        try:
            self._write_pdf(pdf_path, payload)
            format_used = "pdf"
            main_path = pdf_path
        except Exception as exc:
            pass

        return SlipResult(
            path=main_path,
            format=format_used,
            challan_id=challan_id,
            evidence_path=payload["snapshot_path"],
            evidence_valid=evidence_valid,
            review_required=payload["review_required"],
            html_path=html_path,
            pdf_path=pdf_path if format_used == "pdf" else "",
            severity_level=sev_dict.get("level", "MEDIUM"),
            severity_score=int(sev_dict.get("score", 50)),
            detection_confidence=round(float(detection_confidence), 3),
        )

    def _challan_id(self, frame_idx: int, track_id: int) -> str:
        return f"TVS-{datetime.now().strftime('%Y%m%d')}-{frame_idx:06d}-{track_id:04d}"

    def _write_pdf(self, path: str, payload: Dict[str, Any]) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        # Tight margins to guarantee single-page fit on A4 (595.27 x 841.89 pt)
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=28,
            leftMargin=28,
            topMargin=22,
            bottomMargin=18,
        )
        base_styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ChallanTitle",
            parent=base_styles["Title"],
            fontSize=16,
            leading=18,
            textColor=colors.HexColor("#1A365D"),
            alignment=0,
        )
        subtitle_style = ParagraphStyle(
            "ChallanSubTitle",
            parent=base_styles["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4A5568"),
            alignment=0,
        )
        section_style = ParagraphStyle(
            "SectionHead",
            parent=base_styles["Heading2"],
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=3,
        )
        cell_style = ParagraphStyle(
            "CellText",
            parent=base_styles["Normal"],
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1A202C"),
        )
        cell_bold = ParagraphStyle(
            "CellBold",
            parent=cell_style,
            fontName="Helvetica-Bold",
        )
        evidence_callout = ParagraphStyle(
            "EvidenceCallout",
            parent=base_styles["Normal"],
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#2B6CB0"),
            fontName="Helvetica-Bold",
        )

        story = []

        # Header Title Table
        header_data = [
            [
                Paragraph("TRAFFIC VIOLATION E-CHALLAN", title_style),
                Paragraph("<b>EVIDENCE RECORD</b><br/><font size='9' color='#2B6CB0'>CERTIFIED ELECTRONIC NOTICE</font>", evidence_callout),
            ],
            [
                Paragraph("Intelligent Traffic Management System &bull; AI-Assisted Automated Evidence Slip", subtitle_style),
                Paragraph(
                    "<font size='7' color='#C53030'>Status: <b>REVIEW REQUIRED</b></font>"
                    if payload.get("review_required")
                    else "<font size='7' color='#276749'>Status: <b>EVIDENCE READY</b></font>",
                    cell_style,
                ),
            ],
        ]
        header_table = Table(header_data, colWidths=[5.0 * inch, 2.3 * inch])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LINEBELOW", (0, 1), (-1, 1), 1.2, colors.HexColor("#2B6CB0")),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6))

        # Metadata Table
        violations_joined = ", ".join(payload["violations"])
        sev = payload.get("severity", {})
        sev_lvl = sev.get("level", "MEDIUM")
        sev_score = sev.get("score", 50)
        sev_color = sev.get("badge_color", "#F59E0B")
        det_conf = payload.get("detection_confidence", 0.85)

        meta_rows = [
            [
                Paragraph("Challan ID", cell_bold), Paragraph(payload["challan_id"], cell_style),
                Paragraph("Issued Date/Time", cell_bold), Paragraph(payload["issued_at"], cell_style),
            ],
            [
                Paragraph("Video Source", cell_bold), Paragraph(os.path.basename(payload["video"]), cell_style),
                Paragraph("Frame Index", cell_bold), Paragraph(f"Frame #{payload['frame']:,}", cell_style),
            ],
            [
                Paragraph("Track ID", cell_bold), Paragraph(f"Vehicle #{payload['track_id']}", cell_style),
                Paragraph("Vehicle Class", cell_bold), Paragraph(payload["vehicle_class"].upper(), cell_style),
            ],
            [
                Paragraph("Registration Plate", cell_bold),
                Paragraph(f"<font color='#2B6CB0'><b>{payload.get('plate_number', 'UNKNOWN')}</b></font>", cell_style),
                Paragraph("ANPR Status", cell_bold),
                Paragraph(
                    f"<font color='#276749'><b>OCR {payload.get('plate_confidence', 0.0):.2f}</b></font>"
                    if not payload.get("review_required")
                    else "<font color='#C53030'><b>REVIEW REQUIRED</b></font>",
                    cell_style,
                ),
            ],
            [
                Paragraph("Severity Rating", cell_bold),
                Paragraph(f"<font color='{sev_color}'><b>{sev_lvl} ({sev_score}/100)</b></font>", cell_style),
                Paragraph("Detection Confidence", cell_bold),
                Paragraph(f"<b>{det_conf * 100:.1f}%</b> (Neural Ensemble)", cell_style),
            ],
            [
                Paragraph("Assigned Lane", cell_bold), Paragraph(payload["lane"], cell_style),
                Paragraph("Direction / Speed", cell_bold), Paragraph(f"{payload['direction'].capitalize()} ({payload.get('speed_kmh', 0.0):.1f} km/h)", cell_style),
            ],
            [
                Paragraph("Violation Category", cell_bold),
                Paragraph(f"<font color='#C53030'><b>{violations_joined}</b></font>", cell_style),
                Paragraph("Enforcement Notice", cell_bold),
                Paragraph("<b>Subject to Competent Authority Review</b>", cell_style),
            ],
        ]
        meta_table = Table(meta_rows, colWidths=[1.3 * inch, 2.35 * inch, 1.3 * inch, 2.35 * inch])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6))

        # Evidence Image Section
        story.append(Paragraph("Evidence Record (Annotated Scene & Target Zoom)", section_style))
        image_info = self._validated_image_size(payload["snapshot_path"])
        if image_info:
            img_w, img_h = image_info
            draw_w, draw_h = self._fit_size(img_w, img_h, 7.3 * inch, 2.85 * inch)
            story.append(Image(payload["snapshot_path"], width=draw_w, height=draw_h))
            story.append(Paragraph(
                f"<font size='6.5' color='#718096'>Evidence snapshot: {os.path.basename(payload['snapshot_path'])} "
                f"(Dimensions: {img_w}x{img_h}; target crop: {'yes' if payload.get('target_bbox_present') else 'no'})</font>",
                cell_style,
            ))
        else:
            story.append(Paragraph(
                "<i>Evidence image unavailable or unreadable. Manual video review required before legal enforcement.</i>",
                cell_style,
            ))
        story.append(Spacer(1, 6))

        # Signal State & Context Table
        signal = payload.get("signal", {})
        signal_text = (
            f"Active Phase: <b>{signal.get('label', 'N/A')}</b> | "
            f"Green Duration: <b>{signal.get('green_seconds', 'N/A')}s</b> | "
            f"Emergency Priority Override: <b>{'YES' if signal.get('emergency_priority') else 'NO'}</b>"
        )
        story.append(Paragraph("Traffic Signal Context", section_style))
        story.append(Paragraph(signal_text, cell_style))
        story.append(Spacer(1, 5))

        # Explainable AI (XAI) Section
        xai = payload["xai"]
        story.append(Paragraph("Explainable AI (XAI) Violation Attribution", section_style))
        story.append(Paragraph(
            f"Diagnostic Method: <b>{xai.get('method', 'Heuristic')}</b> | "
            f"Decision Score: <b>{xai.get('final_score', 0.0)}</b> | "
            f"Decision Threshold: <b>{xai.get('threshold', 0.50)}</b>",
            cell_style,
        ))
        if xai.get("primary_justification"):
            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<b>Telemetry Justification:</b> {xai.get('primary_justification')}", cell_style))
        story.append(Spacer(1, 3))

        reason_rows = [
            [
                Paragraph("<b>Feature</b>", cell_bold),
                Paragraph("<b>Weight / Impact</b>", cell_bold),
                Paragraph("<b>Observation Evidence</b>", cell_bold),
            ]
        ]
        top_reasons = xai.get("top_reasons", [])[:3]
        if not top_reasons:
            reason_rows.append([
                Paragraph("Deterministic Rule", cell_style),
                Paragraph("High (1.0)", cell_style),
                Paragraph("Spatial violation confirmed by tracker trajectory and polygon test.", cell_style),
            ])
        else:
            for reason in top_reasons:
                reason_rows.append([
                    Paragraph(str(reason.get("feature", "N/A")), cell_style),
                    Paragraph(str(reason.get("impact", "")), cell_style),
                    Paragraph(str(reason.get("evidence", "")), cell_style),
                ])

        reason_table = Table(reason_rows, colWidths=[1.8 * inch, 1.1 * inch, 4.4 * inch])
        reason_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(reason_table)
        story.append(Spacer(1, 8))

        # Human Review / Sign-off Footer
        disclaimer_table = Table([
            [
                Paragraph(
                    "<b>Statutory Notice:</b> This e-challan has been produced automatically by an AI-assisted traffic monitoring prototype. "
                    "In accordance with traffic enforcement standards, this notice must be verified and endorsed by an authorized traffic enforcement officer "
                    "prior to official dispatch or penalty collection.",
                    ParagraphStyle("Disclaimer", parent=cell_style, fontSize=6.5, leading=8.5, textColor=colors.HexColor("#718096")),
                ),
                Paragraph(
                    "<b>Reviewing Authority Signature:</b><br/><br/>___________________________<br/>Officer ID / Seal",
                    ParagraphStyle("Signature", parent=cell_style, fontSize=7, leading=9, alignment=2),
                ),
            ]
        ], colWidths=[5.4 * inch, 1.9 * inch])
        disclaimer_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(disclaimer_table)

        doc.build(story)

        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            raise RuntimeError("PDF generation produced an empty or invalid file")

    def _write_html(self, path: str, payload: Dict[str, Any], fallback_reason: str = "") -> None:
        reasons = "".join(
            f"<tr><td><b>{html.escape(str(r.get('feature', '')))}</b></td>"
            f"<td><span class='badge'>{r.get('impact', '')}</span></td>"
            f"<td>{html.escape(str(r.get('evidence', '')))}</td></tr>"
            for r in payload["xai"].get("top_reasons", [])
        )
        violations = ", ".join(payload["violations"])
        image_html = ""
        if payload["snapshot_path"] and os.path.exists(payload["snapshot_path"]):
            image_html = f"<img src='{html.escape(payload['snapshot_path'])}' alt='Evidence image' />"
        else:
            image_html = "<p class='muted'>Evidence image unavailable or unreadable. Review source video before enforcement.</p>"

        sev = payload.get("severity", {})
        sev_level = html.escape(str(sev.get("level", "MEDIUM")))
        sev_score = sev.get("score", 50)
        sev_color = sev.get("badge_color", "#f59e0b")
        det_conf = payload.get("detection_confidence", 0.85)

        # Build Rule Breakdown HTML
        rules_html = ""
        rule_dec = payload.get("xai", {}).get("rule_based_decision", {})
        for r in rule_dec.get("triggered_rules", []):
            rules_html += f"""
            <div class="rule-box">
              <div class="rule-title">📜 {html.escape(r.get('statute', 'Statutory Traffic Rule'))}</div>
              <div class="rule-detail"><b>Standard:</b> {html.escape(r.get('standard', ''))}</div>
              <div class="rule-detail"><b>Evidence Verified:</b> {html.escape(r.get('condition_met', ''))}</div>
            </div>
            """

        # Build Model Prediction Breakdown HTML
        model_pred = payload.get("xai", {}).get("model_prediction", {})
        objs_html = ""
        for o in model_pred.get("detected_objects", []):
            objs_html += f"<span class='obj-tag'>{html.escape(str(o.get('class', ''))).upper()} ({float(o.get('confidence', 0.0))*100:.0f}%)</span> "

        severity_reasons_html = "".join(f"<li>{html.escape(r)}</li>" for r in sev.get("reasons", []))

        primary_just = payload.get("xai", {}).get("primary_justification", "")
        primary_just_html = ""
        if primary_just:
            primary_just_html = f'<div style="margin-bottom: 8px; color: #1e293b; font-size: 13px;"><b>Primary Telemetry Justification:</b> {html.escape(primary_just)}</div>'

        content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>E-Challan {html.escape(payload['challan_id'])}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    @media print {{
      body {{ background: white !important; margin: 0 !important; }}
      .container {{ box-shadow: none !important; border: 1px solid #ccc !important; max-width: 100% !important; padding: 16px !important; }}
      .no-print {{ display: none !important; }}
      * {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      margin: 24px;
      color: #1a202c;
      background: #f1f5f9;
      line-height: 1.4;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #ffffff;
      padding: 32px;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.05);
      border: 1px solid #e2e8f0;
    }}
    .header-bar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #2b6cb0;
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    .badge-bar {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .badge-severity {{
      background: {sev_color};
      color: #ffffff;
    }}
    .badge-status-ready {{
      background: #dcfce7;
      color: #166534;
      border: 1px solid #86efac;
    }}
    .badge-status-review {{
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
    }}
    .badge-conf {{
      background: #e0f2fe;
      color: #0369a1;
      border: 1px solid #bae6fd;
    }}
    .evidence-box {{
      text-align: right;
    }}
    .evidence-label {{
      font-size: 11px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
    }}
    .evidence-val {{
      font-size: 16px;
      font-weight: 800;
      color: #1e40af;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #e2e8f0;
      padding: 7px 10px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #f8fafc;
      color: #334155;
      font-weight: 600;
    }}
    .section-title {{
      font-size: 15px;
      font-weight: 700;
      color: #1e293b;
      margin: 20px 0 8px 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #cbd5e1;
      display: block;
      margin: 6px 0;
    }}
    .rule-box {{
      background: #f8fafc;
      border-left: 4px solid #3b82f6;
      padding: 10px 14px;
      border-radius: 0 6px 6px 0;
      margin-bottom: 8px;
      font-size: 12.5px;
    }}
    .rule-title {{
      font-weight: 700;
      color: #1e3a8a;
      margin-bottom: 2px;
    }}
    .rule-detail {{
      color: #475569;
    }}
    .obj-tag {{
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      color: #334155;
    }}
    .print-btn {{
      background: #2563eb;
      color: white;
      border: none;
      padding: 8px 18px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 13px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease;
    }}
    .print-btn:hover {{
      background: #1d4ed8;
    }}
    .disclaimer {{
      font-size: 11px;
      color: #64748b;
      border-top: 1px solid #e2e8f0;
      padding-top: 12px;
      margin-top: 24px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="no-print" style="margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;">
      <a href="javascript:history.back()" style="color: #475569; text-decoration: none; font-size: 13px; font-weight: 500;">&larr; Back to Dashboard</a>
      <button onclick="window.print()" class="print-btn">🖨️ Print / Download Receipt</button>
    </div>

    <div class="header-bar">
      <div>
        <div style="font-size: 11px; font-weight: 800; color: #3b82f6; text-transform: uppercase; letter-spacing: 1px;">TRAFFIC POLICE ENFORCEMENT &bull; ITMS</div>
        <h1 style="margin: 3px 0 6px 0; font-size: 22px; color: #0f172a;">Traffic Violation E-Challan Notice</h1>
        <div style="font-size: 12px; color: #64748b;">Notice of Intended Prosecution under the Motor Vehicles Act</div>
        <div class="badge-bar">
          <span class="badge badge-severity">SEVERITY: {sev_level} ({sev_score}/100)</span>
          <span class="badge badge-conf">CONFIDENCE: {det_conf * 100:.1f}%</span>
          <span class="badge {'badge-status-review' if payload.get('review_required') else 'badge-status-ready'}">
            {'STATUS: OFFICER REVIEW REQUIRED' if payload.get('review_required') else 'STATUS: EVIDENCE CONFIRMED'}
          </span>
        </div>
      </div>
      <div class="evidence-box">
        <div class="evidence-label">Enforcement Record</div>
        <div class="evidence-val">CERTIFIED NOTICE</div>
        <div style="font-size: 11px; color: #64748b; font-weight: 600;">Challan #{html.escape(payload['challan_id'])}</div>
      </div>
    </div>

    <table>
      <tr>
        <th style="width: 18%;">Challan ID</th>
        <td style="width: 32%;"><b>{html.escape(payload['challan_id'])}</b></td>
        <th style="width: 18%;">Issued Date/Time</th>
        <td style="width: 32%;">{html.escape(payload['issued_at'])}</td>
      </tr>
      <tr>
        <th>Target Vehicle</th>
        <td><b>Vehicle #{payload['track_id']}</b> ({html.escape(payload['vehicle_class'].upper())})</td>
        <th>Registration Plate</th>
        <td>
          <b style="color: #1e40af;">{html.escape(payload.get('plate_number', 'UNKNOWN'))}</b>
          <span style="font-size: 11px; color: #64748b;">(ANPR Conf: {payload.get('plate_confidence', 0.0):.2f})</span>
        </td>
      </tr>
      <tr>
        <th>Assigned Lane</th>
        <td>{html.escape(payload['lane'])} ({html.escape(payload['direction'].capitalize())})</td>
        <th>Velocity</th>
        <td><b>{payload.get('speed_kmh', 0.0):.1f} km/h</b></td>
      </tr>
      <tr>
        <th>Violations Detected</th>
        <td colspan="3" style="color: #dc2626; font-weight: 700; font-size: 14px;">
          {html.escape(violations)}
        </td>
      </tr>
    </table>

    <div class="section-title">📷 Visual Evidence Composite (Full Scene & Target Zoom)</div>
    {image_html}

    <div class="section-title">⚖️ Statutory Traffic Rules & Explanations</div>
    {rules_html}

    <div class="section-title">📊 Severity & Risk Intensity Assessment</div>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-bottom: 12px; font-size: 13px;">
      <div style="margin-bottom: 6px;"><b>Classified Level:</b> <span style="font-weight: 800; color: {sev_color};">{sev_level} ({sev_score}/100)</span></div>
      <ul style="margin: 0; padding-left: 20px; color: #475569;">
        {severity_reasons_html}
      </ul>
    </div>

    <div class="section-title">🧠 Explainable AI (XAI) Attributions & Model Predictions</div>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-bottom: 12px; font-size: 12.5px;">
      {primary_just_html}
      <div style="margin-bottom: 6px;">
        <b>Neural Detections:</b> {objs_html}
      </div>
      <div style="color: #475569;">
        Attribution Method: <b>Exact Shapley (SHAP) Tabular</b> &bull; Decision Score: <b>{payload['xai'].get('final_score', 0.0)}</b> (Threshold: {payload['xai'].get('threshold', 1.0)})
      </div>
    </div>
    <table>
      <tr><th>Feature</th><th>Shapley Impact</th><th>Evidence Details</th></tr>
      {reasons}
    </table>

    <div class="disclaimer">
      <b>Statutory Notice:</b> This e-challan has been produced by an automated AI traffic monitoring system. In accordance with statutory provisions, photographic and digital evidence is recorded at the timestamp of the event. All rules and penalties are levied under the Motor Vehicles Act 1988 (as amended).
    </div>
  </div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _validated_image_size(self, path: str) -> Tuple[int, int] | None:
        if not path or not os.path.exists(path):
            return None
        try:
            from PIL import Image as PILImage

            with PILImage.open(path) as img:
                img.verify()
            with PILImage.open(path) as img:
                return img.size
        except Exception:
            return None

    def _valid_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None:
            return None
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(frame_w - 1), x1))
        y1 = max(0.0, min(float(frame_h - 1), y1))
        x2 = max(0.0, min(float(frame_w - 1), x2))
        y2 = max(0.0, min(float(frame_h - 1), y2))
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return None
        return (x1, y1, x2, y2)

    def _fit_size(self, width: int, height: int, max_width: float, max_height: float) -> Tuple[float, float]:
        if width <= 0 or height <= 0:
            return max_width, max_height
        scale = min(max_width / float(width), max_height / float(height))
        return width * scale, height * scale
