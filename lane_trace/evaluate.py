#!/usr/bin/env python3
"""
Scoring + drawing.

Ground truth = painted lines, measured ONLY here (the detector never sees
them): the traffic-free background is resampled into (u, y) space where lane
paint becomes vertical stripes; a horizontal white top-hat averaged over rows
gives a 1-D "paint profile" whose peaks are the painted lines.
"""
from __future__ import annotations

import cv2
import numpy as np

from lane_model import LaneModel

PALETTE = [(80, 175, 76), (243, 150, 33), (0, 152, 255), (176, 39, 156), (59, 235, 255),
           (136, 150, 0), (99, 30, 233), (72, 85, 121)]


def y_range(road, H: int, frac_top: float = 0.0):
    g = road.geometry
    y0 = max(g.y_top, g.y_h + 0.06 * (H - g.y_h))
    return y0 + frac_top * (H - y0), H - 1


def rectify(img, road, u_lo, u_hi, nu=900, ny=260, y0=None):
    g = road.geometry
    H = img.shape[0]
    ya, yb = y_range(road, H)
    if y0 is not None:
        ya = y0
    ys = np.linspace(ya, yb, ny)
    us = np.linspace(u_lo, u_hi, nu)
    mapx = g.x_of(us[None, :], ys[:, None]).astype(np.float32)
    mapy = np.repeat(ys[:, None], nu, 1).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), us, ys


def paint_profile(bg, model: LaneModel, road, u_lo, u_hi, nu=900):
    R, us, ys = rectify(bg, road, u_lo, u_hi, nu)
    grey = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(R, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] > 12) & (hsv[..., 0] < 40) & (hsv[..., 1] > 70)).astype(np.uint8) * 255
    du = us[1] - us[0]
    k = max(3, int(0.22 * model.lane_width_prior_u / du) | 1)   # wider than paint, narrower than a lane
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    th = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kern).astype(np.float32)
    th = np.maximum(th, cv2.morphologyEx(yellow, cv2.MORPH_TOPHAT, kern).astype(np.float32) * 0.3)
    valid = grey > 0
    # fraction of rows where a bright thin stripe is present (dashed lines still score)
    hit = ((th > 18) & valid).sum(0) / np.maximum(valid.sum(0), 1)
    return us, hit, R


def paint_peaks(us, prof, model: LaneModel, min_frac=0.09):
    from scipy.signal import find_peaks
    du = us[1] - us[0]
    sm = np.convolve(prof, np.ones(3) / 3, "same")
    pk, _ = find_peaks(sm, height=min_frac, distance=max(1, int(0.35 * model.lane_width_prior_u / du)),
                       prominence=0.045)
    return us[pk], sm[pk]


def score(model: LaneModel, bg, road_idx=0, cw_idx=None):
    """Compare detected boundaries with painted lines (inside the observed road).
    cw_idx restricts the error statistics to one carriageway."""
    road = model.roads[road_idx]
    allb = road.all_boundaries()
    p = np.median([c.lane_width_u for c in road.carriageways])
    u_lo, u_hi = min(allb) - 0.6 * p, max(allb) + 0.6 * p
    us, prof, R = paint_profile(bg, model, road, u_lo, u_hi)
    gt, strength = paint_peaks(us, prof, model)
    rows = []
    cws = road.carriageways if cw_idx is None else [road.carriageways[cw_idx]]
    for cw in cws:
        pw = cw.lane_width_u
        bs = cw.boundaries_u
        for i, b in enumerate(bs):
            kind = "edge" if i in (0, len(bs) - 1) else "divider"
            if len(gt) == 0:
                rows.append(dict(kind=kind, u=b, err=None))
                continue
            j = int(np.argmin(np.abs(gt - b)))
            err = float(abs(gt[j] - b) / pw)
            # only paint within half a lane counts as "the same line"
            rows.append(dict(kind=kind, u=b, gt=float(gt[j]), err_lane=err if err <= 0.5 else None))
    div = [r["err_lane"] for r in rows if r["kind"] == "divider" and r.get("err_lane") is not None]
    edge = [r["err_lane"] for r in rows if r["kind"] == "edge" and r.get("err_lane") is not None]
    n_div = sum(r["kind"] == "divider" for r in rows)
    n_edge = sum(r["kind"] == "edge" for r in rows)
    # painted lines strictly inside a carriageway that we did not reproduce
    missed = 0
    for cw in cws:
        lo, hi = cw.boundaries_u[0] + 0.3 * cw.lane_width_u, cw.boundaries_u[-1] - 0.3 * cw.lane_width_u
        for gu in gt:
            if lo < gu < hi and np.min(np.abs(np.array(cw.boundaries_u) - gu)) > 0.35 * cw.lane_width_u:
                missed += 1
    return dict(rows=rows, gt_u=[float(g) for g in gt], gt_strength=[float(s) for s in strength],
                divider_err_mean=float(np.mean(div)) if div else None,
                divider_err_max=float(np.max(div)) if div else None,
                edge_err_mean=float(np.mean(edge)) if edge else None,
                missed_paint=missed, matched=f"{len(div)}/{n_div} dividers, {len(edge)}/{n_edge} edges",
                profile=(us, prof), rect=R)


# --------------------------------------------------------------------------- drawing

def boundary_poly(road, u, H, n=60, y_start=None):
    g = road.geometry
    ya, yb = y_range(road, H)
    if y_start is not None:
        ya = y_start
    ys = np.linspace(ya, yb, n)
    return np.stack([g.x_of(u, ys), ys], 1)


def draw_lanes(frame, model: LaneModel, alpha=0.28, label=True, thickness=None):
    H, W = frame.shape[:2]
    th = thickness or max(2, W // 640)
    over = frame.copy()
    ci = 0
    for road in model.roads:
        for cw in road.carriageways:
            bs = cw.boundaries_u
            for i in range(len(bs) - 1):
                L = boundary_poly(road, bs[i], H)
                Rr = boundary_poly(road, bs[i + 1], H)
                poly = np.vstack([L, Rr[::-1]]).astype(np.int32)
                cv2.fillPoly(over, [poly], PALETTE[ci % len(PALETTE)])
                ci += 1
    out = cv2.addWeighted(over, alpha, frame, 1 - alpha, 0)
    lane_no = 0
    for road, cw in ((r, c) for r in model.roads for c in r.carriageways):
        bs = cw.boundaries_u
        for i, b in enumerate(bs):
            pts = boundary_poly(road, b, H, 120)
            edge = i in (0, len(bs) - 1)
            col = (40, 40, 255) if edge else (255, 255, 255)
            if edge:
                cv2.polylines(out, [pts.astype(np.int32)], False, col, th + 1, cv2.LINE_AA)
            else:                                           # dashed divider
                for k in range(0, len(pts) - 1, 2):
                    cv2.line(out, tuple(pts[k].astype(int)), tuple(pts[k + 1].astype(int)), col, th, cv2.LINE_AA)
        if label:
            yl = int(H - 0.12 * H)
            for c in cw.centers_u:
                lane_no += 1
                x = int(road.geometry.x_of(c, yl))
                txt = f"L{lane_no}"
                fs = max(0.6, W / 1600)
                (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
                if 0 <= x < W:
                    cv2.rectangle(out, (x - tw // 2 - 6, yl - tht - 8), (x + tw // 2 + 6, yl + 6), (20, 20, 20), -1)
                    cv2.putText(out, txt, (x - tw // 2, yl), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def banner(img, lines, color=(255, 255, 255)):
    H, W = img.shape[:2]
    fs = max(0.55, W / 2200)
    lh = int(34 * fs / 0.8)
    h = lh * len(lines) + 16
    cv2.rectangle(img, (0, 0), (W, h), (18, 18, 18), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (14, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
    return img


def comparison_panel(model: LaneModel, sc: dict, road_idx=0, width=900):
    """Rectified (bird's-eye-like) background with detected boundaries (colour) and
    painted-line peaks (profile plot) side by side - the honest check."""
    us, prof = sc["profile"]
    R = sc["rect"].copy()
    nu = R.shape[1]
    to_px = lambda u: int((u - us[0]) / (us[-1] - us[0]) * (nu - 1))
    road = model.roads[road_idx]
    for cw in road.carriageways:
        for i, b in enumerate(cw.boundaries_u):
            x = to_px(b)
            cv2.line(R, (x, 0), (x, R.shape[0]), (40, 40, 255) if i in (0, len(cw.boundaries_u) - 1) else (0, 255, 0), 2)
    ph = 140
    plot = np.full((ph, nu, 3), 25, np.uint8)
    pts = np.stack([np.arange(nu), ph - 10 - prof / max(prof.max(), 1e-6) * (ph - 30)], 1).astype(np.int32)
    cv2.polylines(plot, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
    for gu in sc["gt_u"]:
        x = to_px(gu)
        cv2.line(plot, (x, 0), (x, ph), (0, 200, 255), 1)
    for cw in road.carriageways:
        for b in cw.boundaries_u:
            cv2.drawMarker(plot, (to_px(b), 12), (0, 255, 0), cv2.MARKER_TRIANGLE_DOWN, 14, 2)
    cv2.putText(plot, "grey: paint profile (GT only)   orange: painted lines   green: detected from traffic",
                (8, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    panel = np.vstack([R, plot])
    s = width / panel.shape[1]
    return cv2.resize(panel, None, fx=s, fy=s)
