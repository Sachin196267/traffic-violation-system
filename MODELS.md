# Trained models

All trained weights used by TraffiSense are included in this repository at the paths the code loads them from, so the
pipelines run straight after `git clone`. YOLO weights load with Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("runs/idd_final/weights/best.pt")
results = model("frame.jpg")
```

| Task | File | Model | Trained on | Result (held-out data) |
|---|---|---|---|---|
| Road-scene detection (vehicles, riders, people, traffic lights) | `runs/idd_final/weights/best.pt` | YOLO11s, 960 px | Full Indian Driving Dataset IDD-D train split, 33,569 images (Kaggle GPU, 35 epochs) | mAP50 0.537 / mAP50-95 0.355 on the 4,197-image test split; metrics in `runs/idd_final/idd_metrics.json` |
| Helmet / no-helmet detection | `runs/helmet_nohelmet_final/weights/best.pt` (+ `operating_point.json`) | YOLO11, 640 px | Leakage-free helmet dataset with traffic augmentation | mAP50 0.767 / mAP50-95 0.396; wrong no-helmet calls 0.6 % |
| Helmet detection, tuned on Indian traffic video | `runs/helmet_robust/final/weights/last.pt` (+ `operating_point.json`) | YOLO11 | Public data + rider heads from Indian traffic videos | Best model on real CCTV / street footage (see `helmet_training/reports/FINAL_REPORT.md`) |
| Triple-riding classifier | `runs/triple_rider/final/weights/last.pt` (+ `runs/triple_rider/operating_point.json`) | YOLO11s-cls, 256 px | Rider crops from Indian traffic and police-camera videos | 0 false triple-riding calls on held-out cameras (5-fold leave-cameras-out) |
| Traffic-signal state (red / yellow / green / not a light) | `runs/signal_state/weights/last.pt` | YOLO11s-cls, 64 px | LISA Traffic Light + TVD + 216 hand-labelled Indian signal crops | 26 / 26 Indian red lights correct; false "red" on non-lights 1.1-3.8 % |
| Emergency vehicle (ambulance / police / fire) | `runs/ambulance_cls/probe.npz` (+ `operating_point.json`) | Logistic regression on CLIP ViT-B/16 features | 4,600 cleaned web images + 490 hand-labelled video vehicles | 0 of 2,450 ordinary vehicles wrongly exempted; 84 % of emergency vehicles exempted or sent to review |
| Number-plate detection | `models/license_plate_best.pt` | YOLO | Indian vehicle licence plates | 99.4 % plate recall |
| Number-plate reading (OCR) | `plates/ocr/indian_reader_v2.onnx` (+ `plates/ocr/operating_point.json`) | fast-plate-ocr CCT-XS, fine-tuned | 1,381 Indian plates + degraded copies | 81.6 % exact plates, 95.4 % characters on held-out plates |
| Earlier models used by the original demo pipeline | `models/traffic.pt`, `models/ambulance_best.pt`, `runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt` | YOLO | - | Kept so the README demo commands run; superseded by the models above |

Notes
- `models/traffic.pt` is an early 1-epoch checkpoint; use `runs/idd_final/weights/best.pt` for real results.
- The `helmet_nohelmet-2` model's original 92.7 % mAP50 came from a validation split that leaked into training; on clean data it scores 0.583. `runs/helmet_nohelmet_final` is the corrected model.
- Dataset licences carry over to the weights: the signal-state model uses LISA (CC BY-NC-SA 4.0, non-commercial) and the triple-riding model uses a Kaggle set with no stated licence, so these weights are for study and research use only.
