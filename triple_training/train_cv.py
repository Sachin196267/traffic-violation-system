#!/usr/bin/env python3
"""
Stage 4 - rider-count classifier (one / two / three_plus / not_bike) with leave-videos-out CV.

    python triple_training/train_cv.py --folds 0 1 2 3 4   # out-of-fold probabilities -> oof.json
    python triple_training/train_cv.py --final             # model on all labelled videos

Last epoch is used (no checkpoint picking on held-out videos).
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs/triple_rider"

ap = argparse.ArgumentParser()
ap.add_argument("--folds", nargs="*", type=int, default=[])
ap.add_argument("--final", action="store_true")
ap.add_argument("--model", default=str(ROOT.parent / "helmet_training/robust/yolo11s-cls.pt"))
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--device", default="mps")
ap.add_argument("--resume", action="store_true", help="continue interrupted fold runs from their last.pt")
args = ap.parse_args()


def train(data, name):
    last = RUNS / name / "weights/last.pt"
    if args.resume and name.startswith("fold") and last.exists():
        YOLO(str(last)).train(resume=True)
        return last
    m = YOLO(args.model)
    m.train(data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=64, device=args.device, workers=4,
            project=str(RUNS), name=name, exist_ok=True, optimizer="AdamW", lr0=1e-3, cos_lr=True,
            hsv_v=0.5, hsv_s=0.5, degrees=5, scale=0.3, translate=0.1, fliplr=0.5, erasing=0.2,
            auto_augment="randaugment", seed=0, plots=False, verbose=False)
    return RUNS / name / "weights/last.pt"


samples = json.loads((ROOT / "samples.json").read_text())
oof_path = ROOT / "oof.json"
oof = json.loads(oof_path.read_text()) if oof_path.exists() else {}
for k in args.folds:
    m = YOLO(str(train(ROOT / f"cls/fold{k}", f"fold{k}")))
    paths = [s["path"] for s in samples if s["fold"] == k]
    for i in range(0, len(paths), 256):
        for p, r in zip(paths[i:i + 256], m.predict(paths[i:i + 256], imgsz=args.imgsz, device=args.device,
                                                    verbose=False)):
            oof[p] = {r.names[j]: round(float(v), 5) for j, v in enumerate(r.probs.data.tolist())}
    oof_path.write_text(json.dumps(oof))
    print(f"fold {k}: {len(paths)} held-out crops predicted", flush=True)
if args.final:
    print("final weights:", train(ROOT / "cls/all", "final"), flush=True)
