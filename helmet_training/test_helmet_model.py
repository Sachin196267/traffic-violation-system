#!/usr/bin/env python3
"""
Standalone helmet / no_helmet test script (NOT part of the application; only evaluates this model).

    .venv/bin/python helmet_training/test_helmet_model.py --source path/to/video.mp4
    .venv/bin/python helmet_training/test_helmet_model.py --source path/to/images_dir --conf 0.35

Writes an annotated video / images plus summary.json to helmet_training/reports/video_tests/<name>/.
Videos are tracked (ByteTrack) so we can report label-free quality signals:
  * track class-flip rate  : fraction of tracks whose majority class changes frame-to-frame (flicker)
  * per-track majority vote: helmet / no_helmet tracks counted once each
"""
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_W = ROOT / "runs/helmet_nohelmet_final/weights/best.pt"
COL = {0: (0, 200, 0), 1: (0, 0, 255)}


def draw(frame, boxes, names):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()); c = int(b.cls); cf = float(b.conf)
        tid = f"#{int(b.id)} " if b.id is not None else ""
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL[c], 2)
        cv2.putText(frame, f"{tid}{names[c]} {cf:.2f}", (x1, max(12, y1 - 4)), 0, 0.5, COL[c], 2)
    return frame


def run_video(model, src, out_dir, a):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25; W, H = int(cap.get(3)), int(cap.get(4))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, a.stride)
    wr = cv2.VideoWriter(str(out_dir / f"{src.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))
    tracks = defaultdict(list); per_frame = []; t0 = time.time(); i = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames): break
        if i % stride: i += 1; continue
        r = model.track(fr, imgsz=a.imgsz, conf=a.conf, iou=0.6, device=a.device, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        for b in r.boxes:
            if b.id is not None: tracks[int(b.id)].append(int(b.cls))
        per_frame.append(Counter(int(c) for c in r.boxes.cls.tolist()))
        wr.write(draw(fr, r.boxes, model.names)); i += 1; n += 1
    cap.release(); wr.release()
    dt = time.time() - t0
    long_tracks = {k: v for k, v in tracks.items() if len(v) >= 5}
    flips = {k: sum(1 for x, y in zip(v, v[1:]) if x != y) for k, v in long_tracks.items()}
    maj = Counter(Counter(v).most_common(1)[0][0] for v in long_tracks.values())
    mixed = sum(1 for v in long_tracks.values() if min(Counter(v).values()) / len(v) >= 0.2 and len(set(v)) > 1)
    return {"source": str(src), "resolution": f"{W}x{H}", "frames_processed": n, "frames_in_video": n_total, "stride": stride,
            "proc_fps": round(n / dt, 2), "detections_per_frame": round(sum(sum(c.values()) for c in per_frame) / max(n, 1), 3),
            "helmet_dets": sum(c[0] for c in per_frame), "no_helmet_dets": sum(c[1] for c in per_frame),
            "tracks(>=5 frames)": len(long_tracks), "tracks_majority_helmet": maj.get(0, 0), "tracks_majority_no_helmet": maj.get(1, 0),
            "tracks_with_>=20%_minority_class(unstable)": mixed,
            "class_flip_rate(per track-frame)": round(sum(flips.values()) / max(1, sum(len(v) - 1 for v in long_tracks.values())), 4)}


def run_images(model, src, out_dir, a):
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    cnt = Counter()
    for p in files:
        r = model.predict(str(p), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        cnt.update(int(c) for c in r.boxes.cls.tolist())
        cv2.imwrite(str(out_dir / (p.stem + ".jpg")), draw(r.orig_img.copy(), r.boxes, model.names))
    return {"source": str(src), "images": len(files), "helmet_dets": cnt[0], "no_helmet_dets": cnt[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=None, help="default: operating conf stored next to the weights")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "helmet_training/reports/video_tests"))
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    if a.conf is None:
        cfg = Path(a.weights).parent / "operating_point.json"
        a.conf = json.loads(cfg.read_text())["conf"] if cfg.exists() else 0.35
    src = Path(a.source)
    out_dir = Path(a.out) / src.stem.replace(" ", "_")[:60]; out_dir.mkdir(parents=True, exist_ok=True)
    res = run_video(model, src, out_dir, a) if src.is_file() else run_images(model, src, out_dir, a)
    res.update({"weights": a.weights, "conf": a.conf, "imgsz": a.imgsz})
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
