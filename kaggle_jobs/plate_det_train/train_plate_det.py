"""
Plate detector v2: fine-tune the current detector (models/license_plate_best.pt, YOLO11s trained at 416 px on whole
close-up photos) on VEHICLE crops enlarged to 640 px - what plates/run_video.py actually feeds it.
Data: private dataset tvs-plate-det-data (plates/detector/build_det_data.py + pseudo_video.py).
Afterwards old and new are evaluated on the same held-out sets (test_street: two-wheelers in traffic, test_close).
"""
import glob, json, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)
from ultralytics import YOLO

SMOKE = False
T0 = time.time()
ROOT = Path(glob.glob("/kaggle/input/**/manifest.json", recursive=True)[0]).parent
OLD = glob.glob("/kaggle/input/**/license_plate_best.pt", recursive=True)[0]
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
print("data", ROOT, json.loads((ROOT / "manifest.json").read_text()), flush=True)

def yaml(name, val):
    p = Path(f"/kaggle/working/{name}.yaml")
    p.write_text(f"path: {ROOT}\ntrain: train/images\nval: {val}/images\nnames:\n  0: plate\n")
    return p

data = yaml("plates", "val")
model = YOLO(OLD)
model.train(data=str(data), imgsz=640, batch=32, epochs=3 if SMOKE else 80, time=None if SMOKE else 7.5, patience=20, device=0, workers=4,
            cos_lr=True, scale=0.7, degrees=3, translate=0.1, mosaic=1.0, close_mosaic=10, hsv_v=0.5, fliplr=0.5,
            fraction=0.1 if SMOKE else 1.0, project=str(OUT), name="plate_det", exist_ok=True, plots=True, verbose=False)
best = OUT / "plate_det/weights/best.pt"

res = {}
for split in ("test_street", "test_close", "val"):
    y = yaml(f"eval_{split}", split)
    for tag, w in (("old", OLD), ("new", str(best))):
        m = YOLO(w).val(data=str(y), split="val", imgsz=640, batch=32, device=0, conf=0.25, iou=0.6, plots=False, verbose=False)
        res[f"{split}/{tag}"] = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
        print(split, tag, res[f"{split}/{tag}"], flush=True)
(OUT / "eval.json").write_text(json.dumps(res, indent=1))
print(f"DONE in {(time.time() - T0) / 3600:.2f} h", flush=True)
