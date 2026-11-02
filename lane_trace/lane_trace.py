#!/usr/bin/env python3
"""
LaneTrace - lane detection from traffic behaviour (paint is never used).

    python lane_trace/lane_trace.py --video videos/x.mp4 --seconds 30 --out lane_trace/output/x

Writes to --out:
    lanes_video.mp4      the observed clip with lanes drawn; the model is re-learnt
                         every few seconds so you can watch it lock in
    lanes_preview.jpg    final lanes on a live frame
    lanes_background.jpg final lanes on the traffic-free background
    paint_check.jpg      bird's-eye strip: detected lines vs painted lines (scoring only)
    lanes.json           geometry + lane boundaries (image polylines)
    report.json          timings, lane counts, confidence, paint agreement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import Segment, Stabiliser, collect, save_segment  # noqa: E402
from evaluate import banner, comparison_panel, draw_lanes, score  # noqa: E402
from lane_model import infer_lanes                              # noqa: E402

CHECKPOINTS = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 240, 300]


def summary_line(model):
    parts = []
    for r in model.roads:
        for c in r.carriageways:
            tag = "" if c.mode == "traffic-lanes" else "~"
            parts.append(f"{tag}{len(c.centers_u)}")
    conf = np.mean([c.confidence for r in model.roads for c in r.carriageways]) if parts else 0
    return " + ".join(parts) or "-", float(conf)


def scaled(model, s):
    """Return a copy of the model whose geometry is expressed in pixels scaled by s."""
    import copy
    m = copy.deepcopy(model)
    for r in m.roads:
        g = r.geometry
        # x = sum c_j d^p  + u d,  d = y - y_h.  Scaling x,y by s: c_j -> c_j * s^(1-p)
        g.gcoef = [c * s ** (1 - p) for c, p in zip(g.gcoef, g.powers)]
        g.y_h *= s
        g.y_top *= s
        g.y_bot *= s
    return m


def render_video(video, seg_t0, seg_t1, det, models, out_path, width=1280):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    s = width / W
    size = (width, int(round(H * s)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seg_t0 * fps)))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    stab = Stabiliser(W)
    ok, ref = cap.read()
    stab.reset(ref)
    frame = ref
    n = int((seg_t1 - seg_t0) * fps)
    cache = {}
    for i in range(n):
        if i:
            ok, frame = cap.read()
            if not ok:
                break
        t = seg_t0 + i / fps
        last = max(models) if models else 0
        # the final model covers the whole clip; show it for the last half second
        avail = [k for k in models if k <= t - seg_t0 + 1e-6 or (k == last and t - seg_t0 >= last - 0.5)]
        M = stab.to_ref(frame) if i % 3 == 0 or "M" not in cache else cache["M"]
        if M is None:
            M = cache.get("M", np.array([[1, 0, 0], [0, 1, 0]], np.float32))
        cache["M"] = M
        # draw in the reference frame, then map back into the current (swaying) frame
        Minv = cv2.invertAffineTransform(M)
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if avail:
            key = max(avail)
            if key not in cache:
                cache[key] = scaled(models[key], s)
            ms = cache[key]
            Ms = Minv.copy()
            Ms[:, 2] *= s
            if np.abs(Ms - np.eye(2, 3)).max() < 0.5 * s * 1e-3 + 0.4:   # camera still: draw directly
                drawn = draw_lanes(small, ms)
            else:
                ref_view = cv2.warpAffine(small, cv2.invertAffineTransform(Ms), size, borderMode=cv2.BORDER_REPLICATE)
                drawn = cv2.warpAffine(draw_lanes(ref_view, ms), Ms, size, borderMode=cv2.BORDER_REPLICATE)
            lanes, conf = summary_line(models[key])
            final = key == max(models)
            status = "FINAL MODEL" if final else "learning..."
            lines = [f"LaneTrace (traffic only, paint ignored) | t={t - seg_t0:5.1f}s | {status}",
                     f"model from {key:.0f}s of traffic | lanes per carriageway: {lanes} | confidence {conf:.2f}"]
        else:
            drawn = small
            lines = [f"LaneTrace (traffic only, paint ignored) | t={t - seg_t0:5.1f}s",
                     "watching traffic... (need ~3 s before a first guess)"]
        vw.write(banner(drawn, lines))
    vw.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--obs", help="reuse an observations .npz (skips detection/tracking)")
    ap.add_argument("--fast", action="store_true",
                    help="640 px detector on every 3rd frame: ~2x faster (30 s clip in ~20 s on an M4), "
                         "slightly less precise")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = dict(video=os.path.basename(a.video), start_s=a.start, seconds=a.seconds)

    t0 = time.time()
    if a.obs:
        o = np.load(a.obs)
        seg = Segment(t0=float(o["t0"]), t1=float(o["t1"]), ref_frame=o["ref"])
        seg.det = o["det"].tolist()
        seg.bg_frames = []
        bg = o["bg"]
        meta = dict(W=int(o["W"]), H=int(o["H"]), fps=float(o["fps"]))
        report["collect_s"] = float(o["elapsed"])
        report["segments"] = [[round(seg.t0, 1), round(seg.t1, 1)]]
    else:
        kw = dict(imgsz=640, stride=3) if a.fast else {}
        segs, meta = collect(a.video, a.start, a.seconds, a.model, **kw)
        report["collect_s"] = round(time.time() - t0, 1)
        if not segs:
            sys.exit("camera never stayed still for >= 3 s - cannot learn lanes from this clip")
        seg = max(segs, key=lambda s: s.t1 - s.t0)
        report["segments"] = [[round(s.t0, 1), round(s.t1, 1)] for s in segs]
        bg = seg.background()
        save_segment(os.path.join(a.out, "observations.npz"), seg, meta)
    report["used_segment"] = [round(seg.t0, 1), round(seg.t1, 1)]
    W, H = meta["W"], meta["H"]
    det = seg.array()

    # progressive models (what a live system would show after k seconds)
    t1 = time.time()
    models, prog = {}, []
    dur = seg.t1 - seg.t0
    for k in [c for c in CHECKPOINTS if c < dur] + [dur]:
        d = det[det[:, 0] < seg.t0 + k]
        try:
            m = infer_lanes(d, W, H)
        except Exception as e:                                # too little traffic yet
            prog.append(dict(t=round(k, 1), error=str(e)))
            continue
        if not m.roads:
            prog.append(dict(t=round(k, 1), error="no road found yet"))
            continue
        models[k] = m
        lanes, conf = summary_line(m)
        prog.append(dict(t=round(k, 1), lanes=lanes, confidence=round(conf, 2), tracks=m.n_tracks))
    report["fit_s_total"] = round(time.time() - t1, 1)
    report["fast_mode"] = bool(a.fast)
    if not models:
        sys.exit("not enough moving vehicles to learn lanes")
    final = models[max(models)]
    tf = time.time()
    infer_lanes(det, W, H)
    report["final_fit_s"] = round(time.time() - tf, 2)
    report["lanes_ready_s"] = round(report["collect_s"] + report["final_fit_s"], 1)
    report["progressive"] = prog
    lanes, conf = summary_line(final)
    report["final"] = dict(lanes_per_carriageway=lanes, confidence=round(conf, 2), tracks=final.n_tracks,
                           roads=[dict(degree=r.degree, n_tracks=r.n_tracks, carriageways=[
                               dict(direction=c.direction, lanes=len(c.centers_u), mode=c.mode,
                                    confidence=c.confidence, periodicity=round(c.periodicity, 2),
                                    support=c.support) for c in r.carriageways]) for r in final.roads])
    final.meta.update(video=report["video"], segment=report["used_segment"])
    final.to_json(os.path.join(a.out, "lanes.json"), W, H)

    # images
    cv2.imwrite(os.path.join(a.out, "lanes_background.jpg"), draw_lanes(bg, final))
    cv2.imwrite(os.path.join(a.out, "lanes_preview.jpg"), draw_lanes(seg.ref_frame, final))
    checks = []
    for i in range(len(final.roads)):
        sc = score(final, bg, i)
        checks.append(comparison_panel(final, sc, i))
        report.setdefault("paint_check", []).append(dict(
            road=i, matched_to_paint=sc["matched"],
            divider_err_lanes=sc["divider_err_mean"], divider_err_max=sc["divider_err_max"],
            edge_err_lanes=sc["edge_err_mean"], painted_lines_found=len(sc["gt_u"])))
    cv2.imwrite(os.path.join(a.out, "paint_check.jpg"), np.vstack(checks))

    if not a.no_video:
        t2 = time.time()
        render_video(a.video, seg.t0, seg.t1, det, models, os.path.join(a.out, "lanes_video.mp4"))
        report["render_s"] = round(time.time() - t2, 1)
    report["total_s"] = round(time.time() - t0 + (report["collect_s"] if a.obs else 0), 1)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("used_segment", "collect_s", "lanes_ready_s", "final", "total_s")
                      if k in report}, indent=1))


if __name__ == "__main__":
    main()
