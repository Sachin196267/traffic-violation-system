#!/usr/bin/env python3
"""
Build head-box candidates from the user's own videos for manual helmet / no_helmet labelling.

Per sampled frame:
  1. new IDD model finds two-wheelers (motorcycle/bicycle merged with overlapping rider boxes)
  2. an expanded crop around each two-wheeler becomes one training image
  3. head candidates inside the crop = pose-keypoint head boxes (+ current helmet model boxes, which are tighter)
Outputs helmet_training/label_own/{crops/, candidates.json, sheets/sheet_XXX.jpg}; sheets are numbered
patches for review (label each: h = helmet, n = no_helmet, x = reject/unclear).

Split: the first 70 % of each video -> train, last 30 % -> test (gap of 5 s); Delhi-4K is held out entirely (test).
"""
import argparse, glob, json, math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "helmet_training/label_own"
(OUT / "crops").mkdir(parents=True, exist_ok=True)
(OUT / "sheets").mkdir(parents=True, exist_ok=True)

VIDEOS = {  # tag: (glob, sample every N seconds, mode)
    "mumbai": ("videos/test/model2_test.mp4", 1.5, "temporal"),
    "bombay": ("videos/bombay traffic*", 2.5, "temporal"),
    "sdelhi": ("videos/Traffic congestion*", 1.5, "temporal"),
    "road4k": ("videos/4K Road traffic*", 3.0, "temporal"),
    "delhi4k": ("videos/Original Traffic Sound*", 2.0, "heldout"),
}
MIN_TW_H, MIN_HEAD = 50, 14
ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", help="video tags to process (default: all)")
ap.add_argument("--every", type=float, help="override sampling interval (s)")
ap.add_argument("--append", action="store_true", help="append to candidates.json, skipping frames already sampled")
ap.add_argument("--block-split", type=float, default=0, help="alternate train/test in blocks of N s (2 s gap) instead of 70/30")
args = ap.parse_args()

idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
pose = YOLO(str(ROOT / "yolo11s-pose.pt"))
helm = YOLO(str(ROOT / "runs/helmet_nohelmet_final/weights/best.pt"))


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1]))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def two_wheelers(r):
    motos = [list(map(float, b.xyxy[0])) for b in r.boxes if r.names[int(b.cls)] in ("motorcycle", "bicycle")]
    riders = [list(map(float, b.xyxy[0])) for b in r.boxes if r.names[int(b.cls)] == "rider"]
    out, used = [], set()
    for m in motos:
        box = m[:]
        for j, rd in enumerate(riders):
            if overlap(m, rd) > 0.2 or overlap(rd, m) > 0.2:
                box = [min(box[0], rd[0]), min(box[1], rd[1]), max(box[2], rd[2]), max(box[3], rd[3])]
                used.add(j)
        out.append(box)
    return out + [rd for j, rd in enumerate(riders) if j not in used]


def pose_heads(crop):
    """Head boxes (crop coords) from COCO keypoints: anchored at the person's box top, sized from shoulders/ears."""
    h0, w0 = crop.shape[:2]
    s = max(1.0, 320 / min(h0, w0))
    big = cv2.resize(crop, (round(w0 * s), round(h0 * s))) if s > 1 else crop
    r = pose.predict(big, imgsz=640, conf=0.25, verbose=False)[0]
    heads = []
    if r.keypoints is None:
        return heads
    for pb, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
        kp = [(x / s, y / s, c) for x, y, c in kp]
        pb = [v / s for v in pb]
        vis = [k for k in kp[:5] if k[2] > 0.3]
        sh = [kp[5], kp[6]] if kp[5][2] > 0.3 and kp[6][2] > 0.3 else None
        sw = abs(sh[0][0] - sh[1][0]) if sh else 0
        ears = abs(kp[3][0] - kp[4][0]) if kp[3][2] > 0.3 and kp[4][2] > 0.3 else 0
        size = max(0.8 * sw, 1.4 * ears, 0.35 * (pb[2] - pb[0]))
        if size < 4:
            continue
        cx = np.mean([k[0] for k in vis]) if vis else ((sh[0][0] + sh[1][0]) / 2 if sh else (pb[0] + pb[2]) / 2)
        y1 = pb[1]
        heads.append([cx - size / 2, y1, cx + size / 2, y1 + 1.1 * size])
    return heads


cands = json.loads((OUT / "candidates.json").read_text()) if args.append else []
done = {(c["video"], int(c["crop"].split("_")[1])) for c in cands}
first_new = len(cands)
n_crops = 0
for tag, (pat, every, mode) in VIDEOS.items():
    if args.only and tag not in args.only:
        continue
    every = args.every or every
    vp = glob.glob(str(ROOT / pat))[0]
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cut, gap = int(n * 0.7), int(5 * fps)
    for fi in range(0, n, int(fps * every)):
        if (tag, fi) in done:
            continue
        if args.block_split:
            blk, pos = divmod(fi / fps, args.block_split)
            if pos < 1 or pos > args.block_split - 1:  # 2 s gap around block edges
                continue
            split = "test" if int(blk) % 2 else "train"
        else:
            if mode == "temporal" and cut - gap < fi < cut + gap:
                continue
            split = "test" if mode == "heldout" or fi >= cut else "train"
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        tws = sorted([b for b in two_wheelers(idd.predict(f, imgsz=960, conf=0.3, verbose=False)[0]) if b[3] - b[1] >= MIN_TW_H],
                     key=lambda b: -(b[3] - b[1]))
        chosen = []
        for b in tws:
            bh, bw = b[3] - b[1], b[2] - b[0]
            c = [max(0, b[0] - 0.2 * bw), max(0, b[1] - 0.35 * bh), min(W, b[2] + 0.2 * bw), min(H, b[3] + 0.05 * bh)]
            if any(overlap(o, c) > 0.5 for o in chosen):
                continue
            chosen.append(c)
            x1, y1, x2, y2 = map(int, c)
            crop = f[y1:y2, x1:x2]
            name = f"{tag}_{fi:06d}_{x1}_{y1}.jpg"
            heads = pose_heads(crop)
            # helmet-model boxes are tighter: replace matching pose boxes, add unmatched ones
            ch, cw = crop.shape[:2]
            s = max(1.0, 320 / min(ch, cw))
            big = cv2.resize(crop, (round(cw * s), round(ch * s))) if s > 1 else crop
            for hb in helm.predict(big, imgsz=640, conf=0.2, verbose=False)[0].boxes.xyxy.tolist():
                hb = [v / s for v in hb]
                m = [i for i, p in enumerate(heads) if iou(p, hb) > 0.25]
                if m:
                    heads[m[0]] = hb
                else:
                    heads.append(hb)
            heads = [h for h in heads if min(h[2] - h[0], h[3] - h[1]) >= MIN_HEAD]
            if not heads:
                continue
            cv2.imwrite(str(OUT / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_crops += 1
            for hb in heads:
                hb = [max(0, hb[0]), max(0, hb[1]), min(cw, hb[2]), min(ch, hb[3])]
                cands.append({"id": len(cands), "crop": name, "video": tag, "split": split, "box": [round(v, 1) for v in hb]})
    print(tag, "crops so far", n_crops, "candidates", len(cands), flush=True)

(OUT / "candidates.json").write_text(json.dumps(cands, indent=0))

# review sheets: 48 numbered patches (3x head size context, head box drawn)
P, COLS, ROWS = 120, 8, 6
new = cands[first_new:]
prefix = "sheet_add_" if args.append else "sheet_"
for si in range(math.ceil(len(new) / (COLS * ROWS))):
    sheet = np.full((ROWS * (P + 16), COLS * P, 3), 255, np.uint8)
    for k, c in enumerate(new[si * COLS * ROWS:(si + 1) * COLS * ROWS]):
        im = cv2.imread(str(OUT / "crops" / c["crop"]))
        x1, y1, x2, y2 = c["box"]
        sz = max(x2 - x1, y2 - y1) * 1.5
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b_ = int(max(0, cx - sz)), int(max(0, cy - sz))
        patch = im[b_:int(cy + sz), a:int(cx + sz)].copy()
        cv2.rectangle(patch, (int(x1 - a), int(y1 - b_)), (int(x2 - a), int(y2 - b_)), (0, 255, 255), 1)
        patch = cv2.resize(patch, (P, P))
        r, col = divmod(k, COLS)
        sheet[r * (P + 16) + 16:(r + 1) * (P + 16), col * P:(col + 1) * P] = patch
        cv2.putText(sheet, str(c["id"]), (col * P + 3, r * (P + 16) + 12), 0, 0.45, (0, 0, 200), 1)
    cv2.imwrite(str(OUT / "sheets" / f"{prefix}{si:03d}.jpg"), sheet)
print("new crops", n_crops, "new candidates", len(new), "total", len(cands), "sheets", math.ceil(len(new) / (COLS * ROWS)))
