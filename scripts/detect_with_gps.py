"""
Run the trained helmet model on a video (or webcam) feed and log every
"Without Helmet" detection together with GPS coordinates + timestamp.

Usage:
    python detect_with_gps.py --source 0                     # webcam
    python detect_with_gps.py --source path/to/video.mp4
    python detect_with_gps.py --source video.mp4 --mock-gps   # no GPS hardware

Before running:
    pip install ultralytics gpsd-py3 opencv-python
    (gpsd-py3 is only needed if you're NOT using --mock-gps)

Real-GPS setup (typical for a Raspberry Pi / vehicle rig):
    A GPS receiver (e.g. a USB/UART GPS puck) reports NMEA sentences.
    `gpsd` is a daemon that reads that stream and exposes it over a socket,
    which is what this script queries via the `gpsd` python client.
    On Linux:  sudo apt install gpsd gpsd-clients && sudo systemctl start gpsd
"""

import argparse
import csv
import os
from datetime import datetime

import cv2
from ultralytics import YOLO


class MockGPS:
    """Fallback GPS source for testing without hardware. Returns a fixed
    point by default — edit lat/lon or feed it a list of points to simulate
    movement."""

    def __init__(self, lat=19.0760, lon=72.8777):  # default: Mumbai
        self.lat = lat
        self.lon = lon

    def get_position(self):
        return self.lat, self.lon


class RealGPS:
    """Reads live position from gpsd."""

    def __init__(self):
        import gpsd
        gpsd.connect()
        self._gpsd = gpsd

    def get_position(self):
        packet = self._gpsd.get_current()
        if packet.mode >= 2:  # 2D or 3D fix
            return packet.lat, packet.lon
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/helmet/helmet_yolov8n_30ep/weights/best.pt")
    parser.add_argument("--source", default="0", help="0 for webcam, or path to video file")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--mock-gps", action="store_true", help="Use a fixed/simulated GPS position")
    parser.add_argument("--out-dir", default="violations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "violations_log.csv")
    log_exists = os.path.exists(log_path)

    gps = MockGPS() if args.mock_gps else RealGPS()
    model = YOLO(args.weights)

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp", "latitude", "longitude", "confidence", "snapshot_path"])

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, conf=args.conf, verbose=False)[0]

            for box in results.boxes:
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)

                if cls_name == "no_helmet":
                    lat, lon = gps.get_position()
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    snap_name = f"violation_{frame_idx}_{timestamp.replace(':', '-')}.jpg"
                    snap_path = os.path.join(args.out_dir, snap_name)
                    cv2.imwrite(snap_path, frame)

                    writer.writerow([timestamp, lat, lon, round(conf, 3), snap_path])
                    print(f"[VIOLATION] {timestamp} lat={lat} lon={lon} conf={conf:.2f} -> {snap_path}")

            annotated = results.plot()
            cv2.imshow("Helmet Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
