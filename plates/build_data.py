#!/usr/bin/env python3
"""
Step 1 - clean list of Indian plates (image, box, text) from datasets/kaggle/plates_india (VOC xml,
the <name> of each object is the plate text) + a held-out test split.

Split is GROUPED so near-duplicates never cross train/test: same plate text = same group
(video frames of one car, OLX re-posts, Google re-uploads).  ~20 % of groups -> test.
Plate texts are normalised to A-Z0-9; objects whose text is not a plausible Indian plate are kept
for the DETECTOR (box is still right) but flagged ocr_ok=False for the reader.
Writes plates/data/plates.json and plates/data/crops/<id>.jpg (plate crop + 10 % margin).
"""
import glob
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

R = Path(__file__).resolve().parent
SRC = R.parent / "datasets/kaggle/plates_india"
OUT = R / "data"
(OUT / "crops").mkdir(parents=True, exist_ok=True)
STATES = set("AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK "
             "TN TR TS UK UP WB BH".split())
FULL = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")


def plausible(t):
    m = FULL.match(t)
    return bool(m) and m.group(1) in STATES and len(t) >= 6


rows = []
for xf in sorted(glob.glob(str(SRC / "**/*.xml"), recursive=True)):
    try:
        root = ET.parse(xf).getroot()
    except ET.ParseError:
        continue
    stem = Path(xf).with_suffix("")
    img_path = next((p for p in (str(stem) + e for e in (".jpg", ".jpeg", ".png", ".JPG")) if Path(p).exists()), None)
    if img_path is None:
        cands = glob.glob(glob.escape(str(stem)) + ".*")
        img_path = next((c for c in cands if not c.endswith(".xml")), None)
    if img_path is None:
        continue
    im = cv2.imread(img_path)
    if im is None:
        continue
    H, W = im.shape[:2]
    source = Path(xf).relative_to(SRC).parts[0]
    for obj in root.findall("object"):
        text = re.sub(r"[^A-Z0-9]", "", (obj.findtext("name") or "").upper())
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 4:
            continue
        rid = len(rows)
        mx, my = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
        crop = im[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]
        cv2.imwrite(str(OUT / "crops" / f"{rid:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=rid, image=img_path, W=W, H=H, box=[x1, y1, x2, y2], text=text, source=source,
                         ocr_ok=plausible(text), plate_h=y2 - y1, crop=f"crops/{rid:05d}.jpg"))

# grouped split: by plate text (or by image if the text is junk)
for r in rows:
    g = r["text"] if r["ocr_ok"] else r["image"]
    r["split"] = "test" if int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0 else "train"
(OUT / "plates.json").write_text(json.dumps(rows, indent=0))
n = len(rows)
print(f"{n} plates from {len({r['image'] for r in rows})} images; readable-format texts {sum(r['ocr_ok'] for r in rows)}")
for s in ("train", "test"):
    sub = [r for r in rows if r["split"] == s]
    print(s, len(sub), {src: sum(r["source"] == src for r in sub) for src in ("State-wise_OLX", "google_images", "video_images")})
