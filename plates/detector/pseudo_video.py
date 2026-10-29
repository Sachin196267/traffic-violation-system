"""
Video-domain training crops for the plate detector (self-training from a stronger, slower teacher).
For each training clip: 1 frame/s -> IDD vehicle boxes -> tight vehicle crop -> the CURRENT detector at 2x the
pipeline's resolution (1280 px). Plates it finds there but often misses at 640 px become labels for the new model.
A box is kept only if plate-shaped (w/h 1.4-6), in the lower part of the vehicle (bus boards, 'HORN OK PLEASE' and
roof signs are above it) and either confident (>= 0.6) or read as a valid Indian plate by the fine-tuned reader.
Held-out clips (never used): see HOLD_OUT.  Run with the plate Python (ultralytics + fast_plate_ocr).
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
# the models and the Indian-format check exactly as run_video.py sets them up (its code up to where processing starts)
_src = (ROOT / "plates/run_video.py").read_text()
_src = _src[:_src.index("\nvp = Path(args.video)")]
_argv, sys.argv = sys.argv, [sys.argv[0], "unused.mp4", "--no-video"]
rv = type(sys)("rv"); rv.__dict__["__file__"] = str(ROOT / "plates/run_video.py")
exec(compile(_src, "run_video.py", "exec"), rv.__dict__)
sys.argv = _argv

from build_det_data import put   # noqa: E402

HOLD_OUT = {"anpr-smartcow", "str-del1", "pov-chennai", "anpr-qapov"}
VEH = {"car", "bus", "truck", "autorickshaw", "motorcycle", "bicycle", "caravan", "trailer", "vehicle fallback"}
PER_CLIP = 450


def clip(path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n, kept, fi = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0, 0
    while fi < n and kept < PER_CLIP:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        r = rv.idd.predict(f, imgsz=960, conf=0.3, verbose=False)[0]
        for k, (b, c) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist())):
            if r.names[c] not in VEH or b[3] - b[1] < 40:
                continue
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3]))
            vc = f[y1:y2, x1:x2]
            s = 1280 / max(vc.shape[:2])
            big = cv2.resize(vc, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            res = rv.plate_det.predict(big, imgsz=1280, conf=0.35, verbose=False)[0]
            boxes = []
            for q, cf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                q = [v / s for v in q]
                w, h = q[2] - q[0], q[3] - q[1]
                if h <= 0 or not 1.4 <= w / h <= 6 or (q[1] + q[3]) / 2 < 0.35 * vc.shape[0]:
                    continue
                if cf < 0.6:
                    pc = vc[int(q[1]):int(q[3]), int(q[0]):int(q[2])]
                    if pc.size == 0:
                        continue
                    if pc.shape[0] < 48:
                        pc = cv2.resize(pc, None, fx=48 / pc.shape[0], fy=48 / pc.shape[0], interpolation=cv2.INTER_CUBIC)
                    if rv.normalise(rv.read(pc)[0])[0] is None:
                        continue
                boxes.append(q)
            if boxes:
                put("train", f"vid_{path.stem}_{fi}_{k}", vc, boxes)
                kept += 1
        fi += int(round(fps))
    cap.release()
    return kept


if __name__ == "__main__":
    total = 0
    for p in sorted((ROOT / "datasets/plate_videos").glob("*.mp4")):
        if p.stem in HOLD_OUT:
            continue
        k = clip(p); total += k
        print(f"{p.stem}: {k} vehicle crops with plates", flush=True)
    print("video crops:", total)
