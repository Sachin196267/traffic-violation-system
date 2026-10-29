#!/usr/bin/env python3
"""Fine-tune the head-level helmet model on old clean data + own-video crops. Final epoch (last.pt) is the result."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "helmet_training/_pydeps"))
from ultralytics import YOLO
import albumentations as A

aug = [
    A.MotionBlur(blur_limit=(3, 9), p=0.15), A.GaussianBlur(blur_limit=(3, 7), p=0.08),
    A.ImageCompression(quality_range=(25, 75), p=0.25), A.GaussNoise(std_range=(0.02, 0.07), p=0.12),
    A.RandomGamma(gamma_limit=(130, 250), p=0.12),
    A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25), contrast_limit=0.3, p=0.25),
    A.RandomShadow(p=0.08), A.Downscale(scale_range=(0.35, 0.75), p=0.12),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.03, 0.08), hole_width_range=(0.03, 0.08), fill=114, p=0.08),
    A.CLAHE(p=0.03), A.ToGray(p=0.03),
]
YOLO(str(ROOT / "runs/helmet_nohelmet_final/weights/best.pt")).train(
    data=str(ROOT / "datasets/helmet_nohelmet_ft_own/data.yaml"), imgsz=640, batch=16, epochs=40, patience=1000,
    optimizer="AdamW", lr0=0.0005, lrf=0.1, cos_lr=True, warmup_epochs=1, close_mosaic=10, amp=True, seed=0, device="mps", workers=4,
    hsv_s=0.6, hsv_v=0.5, degrees=3.0, translate=0.1, scale=0.6, perspective=0.0005, fliplr=0.5, mosaic=1.0, erasing=0.4,
    augmentations=aug, project=str(ROOT / "runs"), name="helmet_nohelmet_ft_own", exist_ok=False, plots=True)
