#!/usr/bin/env python3
"""Read-only per-class validation of existing IDD detectors on the IDD val split (no weights modified)."""
import json, sys
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/kaggle/idd/IDDDetectionsYOLODataset/data.yaml"
MODELS = {"models_traffic.pt(production)": ROOT / "models/traffic.pt",
          "idd_traffic_v2(best, lane config)": ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"}
out = {}
for tag, w in MODELS.items():
    m = YOLO(str(w))
    r = m.val(data=str(DATA), split="val", imgsz=640, batch=8, device="mps", plots=True, verbose=False,
              project=str(ROOT / "helmet_training/audit/idd_eval"), name=tag.split("(")[0], exist_ok=True)
    per = {}
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, a50, a = r.box.class_result(i)
        per[m.names[int(c)]] = {"P": round(p, 3), "R": round(rc, 3), "mAP50": round(a50, 3), "mAP50-95": round(a, 3), "n_inst": int(r.nt_per_class[c])}
    out[tag] = {"P": round(r.box.mp, 3), "R": round(r.box.mr, 3), "mAP50": round(r.box.map50, 3), "mAP50-95": round(r.box.map, 3), "per_class": per}
    print(json.dumps(out[tag], indent=1), flush=True)
(ROOT / "helmet_training/audit/idd_eval/idd_val_metrics.json").write_text(json.dumps(out, indent=1))
