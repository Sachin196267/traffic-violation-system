#!/usr/bin/env python3
"""
Controlled helmet / no_helmet training experiments (training-only; not wired into the app).

    PYTHONPATH=helmet_training/_pydeps .venv/bin/python helmet_training/train.py --exp s960_traffic

Experiments (one variable changed at a time):
    n640_default   yolo11n  640  Ultralytics default augmentation      (baseline, same arch as old model)
    n640_traffic   yolo11n  640  + traffic augmentation                (isolates augmentation effect)
    s640_traffic   yolo11s  640  + traffic augmentation                (isolates model size)
    s960_traffic   yolo11s  960  + traffic augmentation                (isolates resolution / small objects)

Outputs: runs/helmet_nohelmet_final/<exp>/ (weights, curves, confusion matrices, results.csv, args.yaml)
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/helmet_nohelmet_final/data.yaml"
PROJECT = ROOT / "runs/helmet_nohelmet_final"

EXPS = {
    "n640_default": dict(model="yolo11n.pt", imgsz=640, batch=16, traffic_aug=False),
    "n640_traffic": dict(model="yolo11n.pt", imgsz=640, batch=16, traffic_aug=True),
    "s640_traffic": dict(model="yolo11s.pt", imgsz=640, batch=16, traffic_aug=True),
    "s960_traffic": dict(model="yolo11s.pt", imgsz=960, batch=8, traffic_aug=True),
}

# Realistic road-camera degradations only (no flips upside-down, no heavy rotation, no colour inversion)
TRAFFIC_HYP = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,       # colour / brightness / exposure variation
    degrees=3.0, translate=0.1, scale=0.6,   # mild tilt, strong scale jitter (distant riders)
    shear=0.0, perspective=0.0005,           # slight camera-height perspective change
    fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0, close_mosaic=15,
)


def traffic_albumentations():
    import albumentations as A
    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.15),                                   # moving riders / shutter
        A.GaussianBlur(blur_limit=(3, 7), p=0.08),                                  # defocus
        A.ImageCompression(quality_range=(25, 75), p=0.25),                         # CCTV / streaming artefacts
        A.GaussNoise(std_range=(0.02, 0.07), p=0.12),                               # sensor noise at night
        A.RandomGamma(gamma_range=(130, 250), p=0.12),                              # under-exposure / dusk
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25), contrast_limit=0.3, p=0.25),
        A.RandomShadow(p=0.08),                                                     # tree / building shadows
        A.Downscale(scale_range=(0.35, 0.75), p=0.12),                              # low-resolution cameras
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.03, 0.08),
                        hole_width_range=(0.03, 0.08), fill=114, p=0.08),           # partial occlusion
        A.CLAHE(p=0.03), A.ToGray(p=0.03),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    from ultralytics import YOLO
    cfg = EXPS[a.exp]
    kw = dict(data=str(DATA), epochs=a.epochs, patience=a.patience, imgsz=cfg["imgsz"], batch=cfg["batch"],
              device=a.device, workers=a.workers, project=str(PROJECT), name=a.exp, exist_ok=False,
              seed=0, deterministic=False, cos_lr=True, optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
              warmup_epochs=3, plots=True, cache=False, amp=True)
    if cfg["traffic_aug"]:
        kw.update(TRAFFIC_HYP)
        kw["augmentations"] = traffic_albumentations()
    YOLO(str(ROOT / cfg["model"])).train(**kw)


if __name__ == "__main__":
    main()
