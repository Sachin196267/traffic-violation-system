#!/usr/bin/env python3
"""
Stage 1 - observation.

One pass over the video that
  * detects + tracks vehicles (YOLO11 + ByteTrack),
  * stabilises every frame against a reference frame (so a swaying pole camera
    still produces clean trajectories) and starts a new "segment" when the
    camera pans / zooms / cuts,
  * keeps a sparse stack of frames to build a traffic-free background (used
    ONLY for rendering and for scoring against painted lines, never for
    detection).

Output (per static segment): ground-contact samples in reference-frame pixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [1, 2, 3, 5, 7]          # COCO bicycle, car, motorcycle, bus, truck
CAR = 2


@dataclass
class Segment:
    t0: float
    t1: float = 0.0
    det: list = field(default_factory=list)   # rows: t, tid, cls, x1, y1, x2, y2, conf (ref coords)
    bg_frames: list = field(default_factory=list)
    ref_frame: np.ndarray | None = None

    def array(self) -> np.ndarray:
        return np.array(self.det, np.float32).reshape(-1, 8)

    def background(self) -> np.ndarray:
        if not self.bg_frames:
            return self.ref_frame
        return np.median(np.stack(self.bg_frames), axis=0).astype(np.uint8)


class Stabiliser:
    """Maps the current frame to the reference frame of the current segment."""

    def __init__(self, width: int, max_shift_frac: float = 0.08):
        self.scale = 640.0 / width
        self.max_shift = max_shift_frac * width
        self.ref = self.ref_pts = None

    def reset(self, frame):
        self.ref = self._grey(frame)
        self.ref_pts = cv2.goodFeaturesToTrack(self.ref, 500, 0.01, 8)

    def _grey(self, f):
        return cv2.cvtColor(cv2.resize(f, None, fx=self.scale, fy=self.scale,
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def to_ref(self, frame):
        """Return 2x3 affine (full-res frame -> ref) or None if the camera moved too much."""
        g = self._grey(frame)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.ref, g, self.ref_pts, None,
                                              winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            return None
        M, inl = cv2.estimateAffinePartial2D(nxt[ok], self.ref_pts[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
        # scene cut = the background no longer matches (swaying trees keep the
        # inlier *fraction* low even on a fixed camera, so use an absolute count)
        if M is None or inl.sum() < max(30, 0.12 * len(self.ref_pts)):
            return None
        s = self.scale
        M = M.copy()
        M[:, 2] /= s                                      # thumbnail -> full-res translation
        h, w = frame.shape[:2]
        c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32)
        if np.abs(c @ M.T - c[:, :2]).max() > self.max_shift:
            return None                                   # pan / zoom (sway up to 8 % is compensated)
        return M


def collect(video: str, start_s: float = 0.0, seconds: float = 30.0, model_path: str = "yolo11s.pt",
            stride: int = 2, imgsz: int = 960, min_segment_s: float = 3.0, verbose: bool = True):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    n = int(seconds * fps)
    model = YOLO(model_path)
    stab = Stabiliser(W)
    segs: list[Segment] = []
    seg = None
    tic = time.time()
    for i in range(n):
        if not cap.grab():
            break
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = start_s + i / fps
        M = stab.to_ref(frame) if seg is not None else None
        if M is None:
            if seg is not None:
                seg.t1 = t
                segs.append(seg)
            seg = Segment(t0=t, ref_frame=frame.copy())
            stab.reset(frame)
            model.predictor = None                         # fresh tracker ids per segment
            M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        if int(t * 2) != int((t - stride / fps) * 2):      # ~2 bg samples / second
            seg.bg_frames.append(cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
            if len(seg.bg_frames) > 90:
                seg.bg_frames = seg.bg_frames[::2]
        r = model.track(frame, persist=True, imgsz=imgsz, conf=0.25, classes=VEHICLE_CLASSES,
                        tracker="bytetrack.yaml", verbose=False, device="mps")[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy()
        # transform box corners into the reference frame (similarity => box stays a box)
        p1 = np.c_[xyxy[:, :2], np.ones(len(xyxy))] @ M.T
        p2 = np.c_[xyxy[:, 2:], np.ones(len(xyxy))] @ M.T
        for k, (tid, c, cf) in enumerate(zip(b.id.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())):
            seg.det.append((t, tid, c, p1[k, 0], p1[k, 1], p2[k, 0], p2[k, 1], cf))
    if seg is not None:
        seg.t1 = start_s + n / fps
        segs.append(seg)
    cap.release()
    elapsed = time.time() - tic
    segs = [s for s in segs if s.t1 - s.t0 >= min_segment_s]
    if verbose:
        print(f"collected {seconds:.0f}s in {elapsed:.1f}s; static segments: "
              + ", ".join(f"{s.t0:.1f}-{s.t1:.1f}s ({len(s.det)} dets)" for s in segs))
    return segs, dict(fps=fps, W=W, H=H, elapsed=elapsed)


def save_segment(path: str, seg: Segment, meta: dict):
    np.savez_compressed(path, det=seg.array(), bg=seg.background(), ref=seg.ref_frame,
                        t0=seg.t0, t1=seg.t1, **meta)
