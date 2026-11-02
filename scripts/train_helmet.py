"""
Train a YOLOv8 model on the merged helmet dataset.

Usage:
    python train_helmet.py

Before running:
    pip install ultralytics
"""

from ultralytics import YOLO


def main():
    # yolov8n.pt = fastest/smallest (good starting point on ~11k images)
    # yolov8s.pt = a bit slower, usually more accurate — swap in once
    #              you've confirmed the pipeline works end-to-end
    model = YOLO("yolov8n.pt")

    model.train(
        data="/Users/sachinnair/Desktop/traffic violation system/datasets/merged/helmet_traffic_10k/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device="mps",       # Apple Silicon GPU. Use "0" for NVIDIA GPU, "cpu" if neither works
        patience=15,        # stop early if val metrics stall for 15 epochs
        workers=4,
        project="runs/helmet",
        name="helmet_yolov8n_30ep",
        plots=True,
    )

    # After training, best weights land at:
    #   runs/helmet/helmet_yolov8n_30ep/weights/best.pt
    print("Training complete. Best weights saved under runs/helmet/helmet_yolov8n_30ep/weights/best.pt")


if __name__ == "__main__":
    main()
