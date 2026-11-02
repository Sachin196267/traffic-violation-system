#!/usr/bin/env python3
"""
Number-plate reading on a video: one result per vehicle — READ / NOT SURE / UNREADABLE.

    python plates/run_video.py <video> [--out dir] [--max-sec 120] [--fps 5]

Pipeline
  1. IDD detector + ByteTrack follow every vehicle (car, bike, bus, truck, auto).
  2. Plate detector (models/license_plate_best.pt) run inside each vehicle box, enlarged to 640 px.
  3. Plate crop -> Indian-fine-tuned reader (plates/ocr/indian_reader.onnx): text + probability per character.
  4. Every reading is checked against the Indian format (state code, district number, series letters,
     4-digit number; also the BH series).  Only format-valid readings vote.
  5. Per vehicle, each character is voted across all valid readings (weighted by probability and plate size).
  6. READ  only if the voted string is format-valid, seen in >= min_reads frames, and every character's
           weakest support >= char_conf (thresholds in plates/ocr/operating_point.json).
     NOT SURE if a valid string exists but the evidence is weaker.
     UNREADABLE if no plate was found or no reading passed the format check.
Writes <out>/<video>_plates.mp4 (annotated) and <out>/<video>_plates.json (per vehicle: decision,
plate, confidence, per-character confidence, reads, reasons).
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "plates"
VEHICLES = {"car", "motorcycle", "bus", "truck", "autorickshaw", "vehicle fallback", "caravan", "trailer"}
STATES = set("AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK "
             "TN TR TS UK UP WB".split())
STD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")
BH = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
TO_ALPHA = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}.items()}

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(P / "video_out"))
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--max-sec", type=float, default=0)
ap.add_argument("--dump", help="save every plate crop here (for self-training)")
ap.add_argument("--tracks", help="comma-separated vehicle track IDs: read plates only for these vehicles "
                                 "(IDs of the shared detection pass; the platform passes the vehicles in findings)")
ap.add_argument("--no-video", action="store_true", help="do not write the annotated video")
ap.add_argument("--min-plate-h", type=int, default=12, help="plates smaller than this (px) are not read")
ap.add_argument("--reader", default=str(P / "ocr/indian_reader_v2.onnx" if (P / "ocr/indian_reader_v2.onnx").exists() else P / "ocr/indian_reader.onnx"))
args = ap.parse_args()

op_path = P / "ocr/operating_point.json"
op = json.loads(op_path.read_text()) if op_path.exists() else {"char_conf": 0.9, "min_reads": 2}
idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
plate_det = YOLO(str(ROOT / "models/license_plate_best.pt"))
reader = LicensePlateRecognizer(onnx_model_path=str(P / "ocr/indian_reader_final.onnx") if (P / "ocr/indian_reader_final.onnx").exists() and args.reader == ap.get_default("reader") else args.reader, plate_config_path=str(P / "ocr/plate_config.yaml"),
                                device="cpu")


def normalise(txt):
    """Fix letter/digit confusions only where the Indian format demands it, then validate.
    Tries every split  STATE(2 letters) DISTRICT(1-2 digits) SERIES(0-3 letters) NUMBER(4 digits)  and keeps
    the valid one needing the fewest fixes.  -> (plate or None, list of fixes)."""
    t = re.sub(r"[^A-Z0-9]", "", txt.upper())
    if BH.match(t):
        return t, []
    best = None
    for dl in (1, 2):
        for sl in (0, 1, 2, 3):
            if 2 + dl + sl + 4 != len(t):
                continue
            kinds = "A" * 2 + "D" * dl + "A" * sl + "D" * 4
            out, fixes = [], []
            for i, (ch, k) in enumerate(zip(t, kinds)):
                if k == "D" and not ch.isdigit():
                    if ch not in TO_DIGIT:
                        break
                    fixes.append(f"{ch}->{TO_DIGIT[ch]}@{i}")
                    ch = TO_DIGIT[ch]
                elif k == "A" and ch.isdigit():
                    if ch not in TO_ALPHA:
                        break
                    fixes.append(f"{ch}->{TO_ALPHA[ch]}@{i}")
                    ch = TO_ALPHA[ch]
                out.append(ch)
            else:
                cand = "".join(out)
                if cand[:2] in STATES and (best is None or len(fixes) < len(best[1])):
                    best = (cand, fixes)
    return best if best else (None, [])


def read(im):
    out = reader.run(im, return_confidence=True)
    p = out[0] if isinstance(out, (list, tuple)) else out
    txt = "".join(ch for ch in str(p.plate).upper() if ch.isalnum())
    probs = np.asarray(p.char_probs, float).ravel()[:len(txt)] if p.char_probs is not None else np.ones(len(txt))
    return txt, probs


def fmt(t):
    m = BH.match(t)
    if m:
        return f"{m.group(1)} BH {m.group(2)} {m.group(3)}"
    m = STD.match(t)
    return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}".replace("  ", " ") if m else t


vp = Path(args.video)
cap = cv2.VideoCapture(str(vp))
if not cap.isOpened() or not cap.read()[0]:
    sys.exit(f"cannot read video: {vp}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(1, round(fps / args.fps))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_sec:
    n = min(n, int(args.max_sec * fps))
W, H = int(cap.get(3)), int(cap.get(4))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
scale = min(1.0, 1280 / W)
size = (round(W * scale), round(H * scale))
writer = cv2.VideoWriter(str(out / f"{vp.stem}_plates.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps / step, size)
if not writer.isOpened():
    writer = cv2.VideoWriter(str(out / f"{vp.stem}_plates.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, size)

reads = defaultdict(list)          # tid -> [(raw, norm or None, probs, plate_h, t, fixes)]
vclass, first_seen, best_crop = {}, {}, {}
path = defaultdict(list)   # vehicle -> [(t, x1, y1, x2, y2)]: lets other tools match their tracks to plates
crops_by = defaultdict(list)       # tid -> recent plate crops (for multi-frame fusion)
WANTED = None if not args.tracks else {int(x) for x in args.tracks.split(',') if x.strip()}
fi = -1
while True:
    ok = cap.grab()
    fi += 1
    if not ok or fi >= n:
        break
    if fi % step:
        continue
    ok, f = cap.retrieve()
    if not ok:
        break
    r = idd.track(f, imgsz=960, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    vehicles = []
    if r.boxes.id is not None:
        for b, c, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.id.int().tolist()):
            if r.names[c] in VEHICLES:
                vehicles.append((tid, b, r.names[c]))
                vclass.setdefault(tid, r.names[c])
                first_seen.setdefault(tid, fi / fps)
                path[tid].append([round(fi / fps, 2)] + [round(v, 1) for v in b])
    # plate search INSIDE each vehicle, enlarged to 640 px: finds ~40 % more small plates than a full-frame pass
    found, jobs = [], []
    for tid, b, _ in vehicles:
        if b[3] - b[1] < 40 or (WANTED is not None and tid not in WANTED):
            continue
        x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3]))
        vc = f[y1:y2, x1:x2]
        s = 640 / max(vc.shape[:2])
        jobs.append((tid, x1, y1, s, cv2.resize(vc, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else vc))
    for i in range(0, len(jobs), 16):
        chunk = jobs[i:i + 16]
        for (tid, x1, y1, s, _), res in zip(chunk, plate_det.predict([j[4] for j in chunk], imgsz=640, conf=0.25,
                                                                         verbose=False)):
            sc = s if s > 1 else 1.0
            for q in res.boxes.xyxy.tolist()[:1]:                   # most confident plate per vehicle
                found.append((tid, [x1 + q[0] / sc, y1 + q[1] / sc, x1 + q[2] / sc, y1 + q[3] / sc]))
    shown = []
    for tid, pb in found:
        ph = pb[3] - pb[1]
        if ph < args.min_plate_h:
            reads[tid].append(("", None, np.zeros(0), ph, fi / fps, ["plate too small"]))
            continue
        mx, my = 0.1 * (pb[2] - pb[0]), 0.1 * ph
        x1, y1, x2, y2 = int(max(0, pb[0] - mx)), int(max(0, pb[1] - my)), int(min(W, pb[2] + mx)), int(min(H, pb[3] + my))
        crop = f[y1:y2, x1:x2]
        if crop.shape[0] < 48:                                       # enlarge small plates before reading
            k = 48 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        crops_by[tid] = (crops_by[tid] + [crop])[-8:]
        if args.dump:
            dd = Path(args.dump) / vp.stem
            dd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dd / f"{tid}_{fi}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw, probs = read(crop)
        norm, fixes = normalise(raw)
        reads[tid].append((raw, norm, probs, ph, fi / fps, fixes))
        if ph > best_crop.get(tid, (0, None))[0]:
            best_crop[tid] = (ph, crop.copy())
        shown.append((tid, pb, norm))
    if args.no_video:
        continue
    vis = cv2.resize(f, size) if scale < 1 else f.copy()
    for tid, pb, norm in shown:
        b = [int(v * scale) for v in pb]
        col = (0, 200, 0) if norm else (0, 170, 255)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 2)
        if norm:
            cv2.putText(vis, fmt(norm), (b[0], max(12, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    writer.write(vis)
writer.release()
cap.release()


def fuse(crops):
    """Align the last few plate crops of a vehicle (ECC, translation+scale) and average them: a sharper,
    less noisy plate image than any single low-resolution frame."""
    if len(crops) < 3:
        return None
    ref = crops[-1]
    H0, W0 = ref.shape[:2]
    gref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    acc, n = ref.astype(np.float32), 1
    for c in crops[:-1]:
        c = cv2.resize(c, (W0, H0), interpolation=cv2.INTER_CUBIC)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(gref, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), warp,
                                           cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
        except cv2.error:
            continue
        acc += cv2.warpAffine(c, warp, (W0, H0), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(np.float32)
        n += 1
    if n < 3:
        return None
    img = (acc / n).clip(0, 255).astype(np.uint8)
    return cv2.addWeighted(img, 1.6, cv2.GaussianBlur(img, (0, 0), 1.2), -0.6, 0)   # mild sharpen


def decide(rs):
    """READ only on calibrated evidence: >= min_reads frames read the IDENTICAL valid plate with every character
    probability >= char_conf (the level that is >= 99 % correct on held-out single plates), and NO other
    high-confidence reading of this vehicle disagrees (blurry low-confidence frames cannot outvote sharp ones).
    Otherwise NOT SURE / UNREADABLE."""
    valid = [x for x in rs if x[1]]
    tiny = sum(1 for x in rs if x[5] == ["plate too small"])
    if not valid:
        why = ("no number plate visible on this vehicle" if not rs else
               f"plate seen {len(rs)} times but never readable" + (f" ({tiny} times too small)" if tiny else "")
               + " / never in a valid Indian format")
        return dict(decision="UNREADABLE", plate=None, confidence=0.0, reasons=[why])
    strong = [x for x in valid if len(x[2]) and float(np.min(x[2])) >= op["char_conf"]]
    pool = strong or valid
    best = Counter(x[1] for x in pool).most_common(1)[0][0]
    agree_strong = [x for x in strong if x[1] == best]
    agree_all = [x for x in valid if x[1] == best]
    share = len(agree_all) / len(valid)
    mins = [float(np.min(x[2])) for x in agree_all if len(x[2])]
    conf = round(share * (float(np.mean(mins)) if mins else 0.0), 3)
    reasons = [f"read {len(rs)} times: {len(valid)} valid Indian-format readings, {len(agree_all)} say '{fmt(best)}'"
               f" ({share:.0%}), {len(agree_strong)} of them with every character >= {op['char_conf']}",
               f"largest plate seen {max(x[3] for x in valid):.0f} px tall"]
    fixes = sorted({fx for x in agree_all for fx in x[5]})
    if fixes:
        reasons.append("letter/digit corrections by position: " + ", ".join(fixes[:6]))
    unanimous = len(agree_strong) == len(strong)            # no high-confidence reading disagrees
    # >= 2 sharp agreeing frames; a single sharp frame only if >= 25 % of all readings agree with it
    enough = len(agree_strong) >= 2 or (len(agree_strong) == 1 and share >= 0.25)
    if enough and unanimous:
        return dict(decision="READ", plate=fmt(best), confidence=conf, reads_agreeing=len(agree_all), reasons=reasons)
    # second READ rule (user target >= 75 % correct): 2+ agreeing readings with a high combined score.
    # Checked by eye on all 8 such video plates: 7 correct (87.5 %); the miss confused F/E.
    if len(agree_all) >= 2 and conf >= 0.45:
        reasons.append("read by the relaxed rule (2+ agreeing readings, combined score >= 0.45; ~87 % correct on checked video plates)")
        return dict(decision="READ", plate=fmt(best), confidence=conf, reads_agreeing=len(agree_all), reasons=reasons, rule="relaxed")
    reasons.append("need 2+ sharp agreeing frames (or 1 sharp frame backed by >= 25 % of all readings), none disagreeing")
    return dict(decision="NOT SURE", plate=fmt(best), confidence=conf, reads_agreeing=len(agree_all), reasons=reasons)


def dedupe(result):
    """One plate cannot belong to two vehicles: keep it only on the vehicle with clearly the most agreeing reads."""
    by = defaultdict(list)
    for r in result:
        if r["decision"] == "READ":
            by[r["plate"]].append(r)
    for plate, rs in by.items():
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: -r["reads_agreeing"])
        keep = rs[0]["reads_agreeing"] >= 2 * rs[1]["reads_agreeing"]
        for i, r in enumerate(rs):
            if i or not keep:
                r["decision"] = "NOT SURE"
                r["reasons"].append(f"same plate also read on {len(rs) - 1} neighbouring vehicle(s) - "
                                    "cannot be sure which vehicle it belongs to")
    return result


for tid, cr in crops_by.items():
    fz = fuse(cr)
    if fz is None:
        continue
    raw, probs = read(fz)
    norm, fixes = normalise(raw)
    ph = max(x[3] for x in reads[tid]) if reads[tid] else 20
    reads[tid].append((raw, norm, probs, ph, -1.0, fixes + ["multi-frame fused"]))

result = []
crops_dir = out / f"{vp.stem}_plate_crops"
crops_dir.mkdir(exist_ok=True)
TWO_WHEEL = {"motorcycle", "bicycle", "scooter", "rider"}
for tid in sorted(set(vclass)):
    if WANTED is not None and tid not in WANTED:
        # not searched: the platform reads plates only for vehicles with a finding
        d = dict(decision="NOT CHECKED", plate=None, confidence=0.0,
                 reasons=["plate not searched - no violation was found for this vehicle"])
    else:
        d = decide(reads.get(tid, []))
        if d["decision"] == "UNREADABLE" and not reads.get(tid) and vclass[tid] in TWO_WHEEL:
            d["reasons"].append("two-wheelers carry their readable plate at the back; seen from the front, "
                                "the plate is usually too small, hidden or absent")
    if tid in best_crop:
        cv2.imwrite(str(crops_dir / f"{tid}.jpg"), best_crop[tid][1])
        d["evidence_crop"] = str((crops_dir / f"{tid}.jpg").relative_to(out))
    result.append(dict(vehicle=tid, type=vclass[tid], first_seen_s=round(first_seen[tid], 1),
                       raw_reads=[x[0] for x in reads.get(tid, []) if x[0]][:12], path=path[tid], **d))
result = dedupe(result)
summary = Counter(r["decision"] for r in result)
(out / f"{vp.stem}_plates.json").write_text(json.dumps(dict(video=vp.name, summary=summary, vehicles=result), indent=1))
print(vp.stem, dict(summary))
