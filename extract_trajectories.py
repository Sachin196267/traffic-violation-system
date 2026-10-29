from ultralytics import YOLO
import csv

model = YOLO("/Users/sachinnair/Desktop/traffic violation system/runs/detect/runs/detect/idd_traffic_v2/weights/best.pt")

video_path = "/Users/sachinnair/Desktop/traffic violation system/datasets/ua_detrac_combined.mp4"
output_csv = "/Users/sachinnair/Desktop/traffic violation system/datasets/trajectories.csv"

results = model.track(source=video_path, tracker="botsort.yaml", conf=0.25, persist=True, stream=True)

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "track_id", "class", "x_center", "y_center", "width", "height"])

    for frame_idx, result in enumerate(results):
        if result.boxes.id is None:
            continue
        boxes = result.boxes.xywh.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            x, y, w, h = box
            writer.writerow([frame_idx, int(track_id), int(cls), x, y, w, h])

print(f"Saved trajectory data to {output_csv}")
