#!/usr/bin/env python3
"""Resume the interrupted final_aug classifier training from its last checkpoint."""
from pathlib import Path
from ultralytics import YOLO

w = Path(__file__).resolve().parents[2] / "runs/helmet_robust/final_aug/weights/last.pt"
YOLO(str(w)).train(resume=True)
