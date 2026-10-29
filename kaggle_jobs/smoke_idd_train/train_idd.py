"""IDD-D full retrain: YOLO11s @ 960 on the complete train split (Kaggle T4). Outputs -> /kaggle/working/out."""
import glob, json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)
import cv2
from ultralytics import YOLO

T0 = time.time()
SMOKE = True  # True = tiny end-to-end test run
SRC = Path(glob.glob("/kaggle/input/**/IDDDetectionsYOLODataset", recursive=True)[0])
DST = Path("/tmp/idd")
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
MAX_SIDE = 960  # == training imgsz; pre-resizing only removes JPEG-decode cost (labels are normalised)
NAMES = ["animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "person", "rider",
         "traffic light", "traffic sign", "trailer", "train", "truck", "vehicle fallback"]


def resize(args):
    src, dst = args
    im = cv2.imread(src)
    if im is None:
        return 0
    h, w = im.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 1


jobs, counts = [], {}
for split in ("train", "val", "test"):
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC / split / "labels", DST / split / "labels", dirs_exist_ok=True)
    ims = sorted(glob.glob(str(SRC / split / "images" / "*")))
    counts[split] = len(ims)
    jobs += [(p, str(DST / split / "images" / Path(p).name)) for p in ims]
print("images per split:", counts, flush=True)
if SMOKE:
    jobs = jobs[::40]
with Pool(4) as pool:
    ok = sum(pool.imap_unordered(resize, jobs, chunksize=64))
print(f"resized {ok}/{len(jobs)} in {time.time() - T0:.0f}s", flush=True)

yaml = DST / "data.yaml"
yaml.write_text(f"path: {DST}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))

budget_h = round(10.6 - (time.time() - T0) / 3600, 2)  # Kaggle hard limit is 12 h; leave room for eval/export
if SMOKE:
    budget_h = 0.05
print("training budget (h):", budget_h, flush=True)
model = YOLO("yolo11s.pt")
model.train(data=str(yaml), imgsz=960, batch=16, epochs=200, time=budget_h, patience=30, device=0, workers=4,
            optimizer="AdamW", lr0=0.001, cos_lr=True, close_mosaic=10, amp=True, seed=0,
            scale=0.5, degrees=0.0, fliplr=0.5, hsv_v=0.4, mosaic=1.0,
            project="/kaggle/working/runs", name="idd_yolo11s_960", exist_ok=True, plots=True)

run = Path("/kaggle/working/runs/idd_yolo11s_960")
best = YOLO(str(run / "weights/best.pt"))
res = {}
for split in ("val", "test"):
    m = best.val(data=str(yaml), split=split, imgsz=960, batch=16, device=0, plots=True,
                 project="/kaggle/working/runs", name=f"eval_{split}", exist_ok=True)
    res[split] = {"P": float(m.box.mp), "R": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                  "per_class": {NAMES[c]: {"P": float(m.box.p[i]), "R": float(m.box.r[i]), "mAP50": float(m.box.ap50[i]),
                                           "mAP50-95": float(m.box.ap[i])} for i, c in enumerate(m.box.ap_class_index)}}
    print(split, json.dumps({k: v for k, v in res[split].items() if k != "per_class"}), flush=True)
res["image_counts"] = counts
res["wall_hours"] = round((time.time() - T0) / 3600, 2)
(OUT / "idd_metrics.json").write_text(json.dumps(res, indent=1))
for f in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"]:
    if (run / f).exists():
        shutil.copy(run / f, OUT / Path(f).name)
print("DONE", res["wall_hours"], "h")
