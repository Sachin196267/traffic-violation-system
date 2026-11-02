"""
Two-stage helmet check, stage 2: classify a motorcycle+rider crop as helmet / no_helmet.
Crops come from OSF HELMET (CC BY 4.0) motorcycle boxes; split = dataset's clip-level data_split.csv.
At inference the crop comes from the IDD detector's motorcycle/rider box.
"""
import collections, csv, glob, json, os, random, shutil, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)
import cv2
from ultralytics import YOLO

T0 = time.time()
SMOKE = False
ROOT = Path(glob.glob("/kaggle/input/**/helmet-dataset/data_split.csv", recursive=True)[0]).parent
DST = Path("/tmp/hcls")
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
STRIDE = {"training": 3, "validation": 10, "test": 10}
SPLIT = {"training": "train", "validation": "val", "test": "test"}
MIN_SIDE = 24

split_of = {r["video_id"]: r["Set"] for r in csv.DictReader(open(ROOT / "data_split.csv"))}
clip_dir = {os.path.basename(d): d for d in glob.glob(str(ROOT / "images" / "**"), recursive=True)
            if os.path.isdir(d) and os.path.basename(d) in split_of}
jobs = []
for ann in sorted(glob.glob(str(ROOT / "annotation" / "**" / "*.csv"), recursive=True)):
    clip = Path(ann).stem
    s = split_of.get(clip)
    if s not in STRIDE or clip not in clip_dir:
        continue
    by_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(ann)):
        by_frame[int(r["frame_id"])].append(r)
    for fname in sorted(os.listdir(clip_dir[clip])):
        fid = int(Path(fname).stem)
        if (fid - 1) % STRIDE[s] or fid not in by_frame:
            continue
        jobs.append((os.path.join(clip_dir[clip], fname), SPLIT[s], f"{clip}_{fid:03d}", by_frame[fid]))
if SMOKE:
    jobs = jobs[::40]
for sp in SPLIT.values():
    for c in ("helmet", "no_helmet"):
        (DST / sp / c).mkdir(parents=True, exist_ok=True)


def crop(job):
    src, sp, stem, rows = job
    im = cv2.imread(src)
    cnt = collections.Counter()
    if im is None:
        return sp, cnt
    H, W = im.shape[:2]
    rnd = random.Random(stem)
    for i, r in enumerate(rows):
        x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
        if min(w, h) < MIN_SIDE:
            cnt["dropped_small"] += 1
            continue
        # jittered padding: IDD boxes at inference are looser/tighter than HELMET's
        px, py = w * rnd.uniform(0.0, 0.15), h * rnd.uniform(0.0, 0.15)
        x1, y1 = int(max(0, x - px)), int(max(0, y - py))
        x2, y2 = int(min(W, x + w + px)), int(min(H, y + h + py))
        c = "no_helmet" if "NoHelmet" in r["label"] else "helmet"
        cv2.imwrite(str(DST / sp / c / f"{stem}_{i}.jpg"), im[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        cnt[c] += 1
    return sp, cnt


stats = collections.defaultdict(collections.Counter)
with Pool(4) as pool:
    for sp, c in pool.imap_unordered(crop, jobs, chunksize=32):
        stats[sp].update(c)
build = {k: dict(v) for k, v in stats.items()}
print("built:", json.dumps(build), f"in {time.time() - T0:.0f}s", flush=True)

model = YOLO("yolo11s-cls.pt")
model.train(data=str(DST), imgsz=224, batch=128, epochs=2 if SMOKE else 60, patience=15, device=0, workers=4,
            optimizer="AdamW", lr0=0.001, cos_lr=True, amp=True, seed=0,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, fliplr=0.5, erasing=0.3, auto_augment="randaugment",
            project="/kaggle/working/runs", name="helmet_cls", exist_ok=True, plots=True)

run = Path("/kaggle/working/runs/helmet_cls")
best = YOLO(str(run / "weights/best.pt"))
res = {"build": build}
for sp in ("val", "test"):
    m = best.val(data=str(DST), split=sp, imgsz=224, batch=128, device=0, plots=True,
                 project="/kaggle/working/runs", name=f"eval_{sp}", exist_ok=True)
    cm = m.confusion_matrix.matrix  # rows = predicted, cols = true
    names = list(best.names.values())
    per = {}
    for i, n in enumerate(names):
        tp = cm[i, i]; fp = cm[i, :].sum() - tp; fn = cm[:, i].sum() - tp
        per[n] = {"P": round(float(tp / max(1, tp + fp)), 4), "R": round(float(tp / max(1, tp + fn)), 4), "n_true": int(cm[:, i].sum())}
    res[sp] = {"top1": round(float(m.top1), 4), "per_class": per, "confusion(pred x true)": cm.astype(int).tolist(), "names": names}
    print(sp, json.dumps(res[sp]), flush=True)
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        p = Path(f"/kaggle/working/runs/eval_{sp}") / f
        if p.exists():
            shutil.copy(p, OUT / f"{sp}_{f}")
res["wall_hours"] = round((time.time() - T0) / 3600, 2)
(OUT / "helmet_cls_metrics.json").write_text(json.dumps(res, indent=1))
for f in ["weights/best.pt", "results.csv", "args.yaml", "results.png"]:
    if (run / f).exists():
        shutil.copy(run / f, OUT / Path(f).name)
print("DONE", res["wall_hours"], "h")
