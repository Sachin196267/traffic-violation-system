#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", default="datasets/huggingface/helmet_india")
parser.add_argument("--model", default="yolov8s.pt")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--device", default="mps")
parser.add_argument("--name", default="helmet_detector")
args = parser.parse_args()

data_yaml = Path(args.dataset_root) / "data.yaml"
model = YOLO(args.model)

results = model.train(
    data=str(data_yaml),
    epochs=args.epochs,
    imgsz=640,
    batch=args.batch_size,
    device=args.device,
    name=args.name,
    project="runs/detect"
)

print(f"✓ Training complete. Model saved to runs/detect/{args.name}/weights/best.pt")
