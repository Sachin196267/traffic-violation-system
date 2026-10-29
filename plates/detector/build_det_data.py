"""
Training data for the plate detector, built the way plates/run_video.py uses it: the detector sees one VEHICLE crop,
enlarged so its longer side is 640 px (cubic), and finds the plate inside it.

Sources (labels converted to YOLO 'plate' boxes inside each crop):
  street   datasets/external/devgurucodes_triple_riding  rider boxes (helmet/no-helmet/triple) that contain a Plate label
           datasets/external/pkdarabi_helmet             same   -- only the two-wheelers' plates are labelled there, so whole
                                                                   images would teach the model to ignore car plates
  close    datasets/kaggle/plates_india (plates/data/plates.json, keeps the reader's train/test split)
           datasets/kaggle/plates_extra/*  (deepakat002 CC0, paneraghanshyam Apache-2.0): whole image = one vehicle
Output: plates/detector/data/{train,val,test_street,test_close}/{images,labels} + manifest.json
"""
import glob
import json
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
EXT = ROOT / "datasets/external"
random.seed(0)
stats = {}


def put(split, name, img, boxes):
    """img = vehicle crop (BGR); boxes = [(x1,y1,x2,y2)] in crop pixels -> enlarged to 640 long side, YOLO labels"""
    h, w = img.shape[:2]
    s = 640 / max(h, w)
    if s != 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    H, W = img.shape[:2]
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1 * s), max(0, y1 * s), min(W, x2 * s), min(H, y2 * s)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    if not lines:
        return
    d = OUT / split
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (d / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    stats[split] = stats.get(split, 0) + 1


def yolo_boxes(lbl, W, H):
    out = []
    for l in open(lbl):
        p = l.split()
        if len(p) < 5:
            continue
        c, x, y, w, h = int(p[0]), *map(float, p[1:5])
        out.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def street(root, plate_cls, rider_cls, tag):
    for split_dir, split in (("train", "train"), ("valid", "val"), ("test", "test_street")):
        for lbl in glob.glob(str(root / split_dir / "labels" / "*.txt")):
            stem = Path(lbl).stem
            img_p = next((p for p in glob.glob(str(root / split_dir / "images" / (stem + ".*")))), None)
            im = cv2.imread(img_p) if img_p else None
            if im is None:
                continue
            H, W = im.shape[:2]
            B = yolo_boxes(lbl, W, H)
            plates = [b[1:] for b in B if b[0] == plate_cls]
            for k, (c, x1, y1, x2, y2) in enumerate(b for b in B if b[0] in rider_cls):
                inside = [p for p in plates if p[0] >= x1 - 5 and p[2] <= x2 + 5 and p[1] >= y1 - 5 and p[3] <= y2 + 5]
                if not inside:
                    continue
                mx, my = 0.04 * (x2 - x1), 0.04 * (y2 - y1)   # small margin, like a detector box
                cx1, cy1, cx2, cy2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my))
                crop = im[cy1:cy2, cx1:cx2]
                if crop.shape[0] < 24 or crop.shape[1] < 16:
                    continue
                put(split, f"{tag}_{split_dir}_{stem[:40]}_{k}", crop, [(p[0] - cx1, p[1] - cy1, p[2] - cx1, p[3] - cy1) for p in inside])


def close_up():
    for r in json.load(open(ROOT / "plates/data/plates.json")):
        im = cv2.imread(r["image"])
        if im is None:
            continue
        split = "test_close" if r["split"] == "test" else ("val" if random.random() < 0.1 else "train")
        put(split, f"pi_{r['id']:05d}", im, [tuple(r["box"])])
    for root in sorted((ROOT / "datasets/kaggle/plates_extra").glob("*")):
        for lbl in glob.glob(str(root / "**/*.txt"), recursive=True):
            stem = Path(lbl).stem
            img_p = next((p for p in glob.glob(str(Path(lbl).parent.parent / "images" / (stem + ".*")))
                          + glob.glob(str(Path(lbl).parent / (stem + ".*"))) if not p.endswith(".txt")), None)
            im = cv2.imread(img_p) if img_p else None
            if im is None:
                continue
            H, W = im.shape[:2]
            boxes = [b[1:] for b in yolo_boxes(lbl, W, H)]
            put("val" if random.random() < 0.1 else "train", f"kx_{root.name[:10]}_{stem[:40]}", im, boxes)


if __name__ == "__main__":
    street(EXT / "devgurucodes_triple_riding/master_traffic_violation_dataset", 0, {1, 2, 3}, "dev")
    street(EXT / "pkdarabi_helmet/HelmetViolations", 0, {1, 2}, "pk")
    close_up()
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=1))
    print(stats)
