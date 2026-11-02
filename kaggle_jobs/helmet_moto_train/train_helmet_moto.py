"""
Motorcycle-level helmet model from the OSF HELMET dataset (Myanmar roadside CCTV, CC BY 4.0).
Box = motorcycle incl. riders.  0 = helmet (every rider wears one), 1 = no_helmet (at least one rider does not).
Split = the dataset's own clip-level data_split.csv (no clip in two splits).
"""
import collections, csv, glob, json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)
import cv2
from ultralytics import YOLO

T0 = time.time()
SMOKE = False  # True = tiny end-to-end test run
ROOT = Path(glob.glob("/kaggle/input/**/helmet-dataset/data_split.csv", recursive=True)[0]).parent
DST = Path("/tmp/hm")
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
STRIDE = {"training": 3, "validation": 10, "test": 10}   # 10 fps clips: consecutive frames are near-duplicates
SPLIT = {"training": "train", "validation": "val", "test": "test"}
MAX_SIDE, MIN_BOX = 1280, 12

split_of = {r["video_id"]: r["Set"] for r in csv.DictReader(open(ROOT / "data_split.csv"))}
clip_dir = {}
for d in glob.glob(str(ROOT / "images" / "**"), recursive=True):
    if os.path.isdir(d) and os.path.basename(d) in split_of:
        clip_dir[os.path.basename(d)] = d
print("clips in split file:", len(split_of), "clip dirs found:", len(clip_dir), flush=True)

raw_labels = collections.Counter()
stats = collections.defaultdict(collections.Counter)
jobs = []
for ann in sorted(glob.glob(str(ROOT / "annotation" / "**" / "*.csv"), recursive=True)):
    clip = Path(ann).stem
    s = split_of.get(clip)
    if s not in STRIDE or clip not in clip_dir:
        continue
    by_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(ann)):
        raw_labels[r["label"]] += 1
        by_frame[int(r["frame_id"])].append(r)
    frames = sorted(os.listdir(clip_dir[clip]))
    for fname in frames:
        fid = int(Path(fname).stem)
        if (fid - 1) % STRIDE[s]:
            continue
        jobs.append((os.path.join(clip_dir[clip], fname), SPLIT[s], f"{clip}_{fid:03d}", by_frame.get(fid, [])))

(lambda: [ (DST / sp / d).mkdir(parents=True, exist_ok=True) for sp in SPLIT.values() for d in ("images", "labels")])()


def convert(job):
    src, sp, stem, rows = job
    im = cv2.imread(src)
    if im is None:
        return sp, None
    H, W = im.shape[:2]
    lines, cls_count = [], collections.Counter()
    for r in rows:
        x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
            cls_count["dropped_tiny"] += 1
            continue
        c = 1 if "NoHelmet" in r["label"] else 0
        cls_count[c] += 1
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    s = MAX_SIDE / max(H, W)
    if s < 1:
        im = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DST / sp / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (DST / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    cls_count["images"] += 1
    cls_count["background_images"] += not lines
    return sp, cls_count


if SMOKE:
    jobs = jobs[::40]
with Pool(4) as pool:
    for sp, c in pool.imap_unordered(convert, jobs, chunksize=32):
        if c:
            stats[sp].update(c)
print("raw label strings:", raw_labels.most_common(40), flush=True)
build = {sp: {("helmet" if k == 0 else "no_helmet" if k == 1 else k): v for k, v in c.items()} for sp, c in stats.items()}
print("built:", json.dumps(build), f"in {time.time() - T0:.0f}s", flush=True)

yaml = DST / "data.yaml"
yaml.write_text(f"path: {DST}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: helmet\n  1: no_helmet\n")


def traffic_albumentations():
    import albumentations as A
    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.15), A.GaussianBlur(blur_limit=(3, 7), p=0.08),
        A.ImageCompression(quality_range=(25, 75), p=0.25), A.GaussNoise(std_range=(0.02, 0.07), p=0.12),
        A.RandomGamma(gamma_limit=(130, 250), p=0.12),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25), contrast_limit=0.3, p=0.25),
        A.RandomShadow(p=0.08), A.Downscale(scale_range=(0.35, 0.75), p=0.12),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.03, 0.08), hole_width_range=(0.03, 0.08), fill=114, p=0.08),
        A.CLAHE(p=0.03), A.ToGray(p=0.03),
    ]


budget_h = round(7.0 - (time.time() - T0) / 3600, 2)
if SMOKE:
    budget_h = 0.05
print("training budget (h):", budget_h, flush=True)
model = YOLO("yolo11s.pt")
model.train(data=str(yaml), imgsz=960, batch=16, epochs=150, time=budget_h, patience=25, device=0, workers=4,
            optimizer="AdamW", lr0=0.001, cos_lr=True, close_mosaic=10, amp=True, seed=0,
            hsv_s=0.6, hsv_v=0.5, degrees=3.0, translate=0.1, scale=0.6, perspective=0.0005, fliplr=0.5, mosaic=1.0, erasing=0.4,
            augmentations=traffic_albumentations(),
            project="/kaggle/working/runs", name="helmet_moto_yolo11s_960", exist_ok=True, plots=True)

run = Path("/kaggle/working/runs/helmet_moto_yolo11s_960")
best = YOLO(str(run / "weights/best.pt"))
res = {"build": build}
for split in ("val", "test"):
    m = best.val(data=str(yaml), split=split, imgsz=960, batch=16, device=0, plots=True,
                 project="/kaggle/working/runs", name=f"eval_{split}", exist_ok=True)
    cm = m.confusion_matrix.matrix.tolist()  # rows = predicted, cols = true (helmet, no_helmet, background)
    res[split] = {"P": float(m.box.mp), "R": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                  "per_class": {["helmet", "no_helmet"][c]: {"P": float(m.box.p[i]), "R": float(m.box.r[i]), "mAP50": float(m.box.ap50[i]),
                                                              "mAP50-95": float(m.box.ap[i])} for i, c in enumerate(m.box.ap_class_index)},
                  "confusion_matrix(pred x true: helmet,no_helmet,bg)": cm}
    print(split, json.dumps({k: v for k, v in res[split].items() if k != "confusion_matrix(pred x true: helmet,no_helmet,bg)"}), flush=True)
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        p = Path(f"/kaggle/working/runs/eval_{split}") / f
        if p.exists():
            shutil.copy(p, OUT / f"{split}_{f}")
res["wall_hours"] = round((time.time() - T0) / 3600, 2)
(OUT / "helmet_moto_metrics.json").write_text(json.dumps(res, indent=1))
for f in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "results.png", "BoxPR_curve.png", "BoxF1_curve.png"]:
    if (run / f).exists():
        shutil.copy(run / f, OUT / Path(f).name)
# a few test images for local visual checks
for p in sorted(glob.glob(str(DST / "test/images/*.jpg")))[::200][:30]:
    (OUT / "test_samples").mkdir(exist_ok=True); shutil.copy(p, OUT / "test_samples")
print("DONE", res["wall_hours"], "h")
