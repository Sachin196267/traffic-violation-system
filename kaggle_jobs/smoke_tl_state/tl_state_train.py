"""
Traffic-light state classifier: red / yellow / green / not_light (off, back of a signal head, tail lights, signs...).

Sources: LISA Traffic Light (US, day + night, CC BY-NC-SA 4.0 - study use), traffic-violation-detection-dataset
(red light / green light boxes), and - when attached - the Indian crops dataset (tvs-signal-crops: crops/*.jpg +
labels.json {file: {label, video}}), which also provides the held-out Indian test set (videos in TEST_VIDEOS).
Every crop is padded to a square (classification transforms centre-crop, which would cut off the top/bottom lamp)
and training copies are degraded (downscale to 6-30 px tall, blur, JPEG) to look like far-away CCTV lights.
Output (/kaggle/working/out): weights/last.pt, report.json (per-source confusion matrices), log.txt.
"""
import csv, glob, json, os, random, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)
import cv2, numpy as np

T0 = time.time()
import torch
DEV = 0 if torch.cuda.is_available() else "cpu"
SMOKE = True
EPOCHS = 2 if SMOKE else 25
LISA_EVERY = 4            # consecutive LISA frames are near-identical; keep every 4th frame per clip
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/tl"); CLASSES = ["red", "yellow", "green", "not_light"]
log = open(OUT / "log.txt", "a")
rng = random.Random(0)


def say(*a):
    m = f"[{time.time() - T0:6.0f}s] " + " ".join(map(str, a)); print(m, flush=True); log.write(m + "\n"); log.flush()


def square(crop, size=96):
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def degrade(crop):
    h, w = crop.shape[:2]
    th = rng.randint(6, 30)
    if h > th:
        sc = th / h
        crop = cv2.resize(crop, (max(2, int(w * sc)), th), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
    return cv2.imdecode(enc, 1)


def pad_box(x1, y1, x2, y2, W, H, f=0.25):
    pw, ph = (x2 - x1) * f, (y2 - y1) * f
    return max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph))


samples = []   # (crop image, label, source, group, split)


def add(img, box, label, source, group, split):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = pad_box(*box, W, H)
    if x2 - x1 < 3 or y2 - y1 < 5:
        return
    samples.append((img[y1:y2, x1:x2].copy(), label, source, group, split))


def random_negatives(img, boxes, n, source, group, split):
    """background crops shaped like lights (tall), not overlapping any light - poles, signs, tail lights, sky"""
    H, W = img.shape[:2]
    for _ in range(n):
        h = rng.randint(12, 90); w = int(h * rng.uniform(0.3, 0.6))
        x, y = rng.randint(0, W - w - 1), rng.randint(0, int(H * 0.8) - h - 1)
        if any(x < b[2] and b[0] < x + w and y < b[3] and b[1] < y + h for b in boxes):
            continue
        samples.append((img[y:y + h, x:x + w].copy(), "not_light", source, group, split))


# ---------------- LISA ----------------
TAG = {"stop": "red", "stopLeft": "red", "go": "green", "goLeft": "green", "goForward": "green",
       "warning": "yellow", "warningLeft": "yellow"}
frames = {p.split("/")[-1]: p for p in glob.glob("/kaggle/input/**/frames/*.jpg", recursive=True)}
say("LISA frames on disk", len(frames))
ann = defaultdict(list)
for c in glob.glob("/kaggle/input/**/Annotations/**/frameAnnotationsBOX.csv", recursive=True):
    for r in csv.DictReader(open(c), delimiter=";"):
        ann[r["Filename"].split("/")[-1]].append((TAG.get(r["Annotation tag"]), [int(r[k]) for k in (
            "Upper left corner X", "Upper left corner Y", "Lower right corner X", "Lower right corner Y")]))
names = sorted(n for n in ann if n in frames)
if SMOKE:
    names = names[::40]
VAL_CLIPS = {"dayClip6", "nightClip4", "daySequence2"}     # held out whole clips
for i, n in enumerate(names):
    clip = n.split("--")[0]
    if i % LISA_EVERY:
        continue
    img = cv2.imread(frames[n])
    if img is None:
        continue
    split = "val" if clip in VAL_CLIPS else "train"
    for lab, b in ann[n]:
        if lab:
            add(img, b, lab, "lisa", clip, split)
    if i % (LISA_EVERY * 3) == 0:
        random_negatives(img, [b for _, b in ann[n]], 2, "lisa", clip, split)
say("after LISA", Counter((s[1], s[4]) for s in samples))

# ---------------- red/green light boxes (traffic-violation-detection-dataset) ----------------
for lp in glob.glob("/kaggle/input/**/labels/*/*.txt", recursive=True):
    ip = lp.replace("/labels/", "/images/")[:-4] + ".jpg"
    if not os.path.exists(ip):
        continue
    rows = [l.split() for l in open(lp)]
    rows = [r for r in rows if r and r[0] in ("5", "6")]
    if not rows:
        continue
    img = cv2.imread(ip); H, W = img.shape[:2]
    split = "val" if "/val/" in lp else "train"
    for r in rows:
        cx, cy, w, h = map(float, r[1:5])
        add(img, [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
            "red" if r[0] == "5" else "green", "tvd", Path(ip).stem, split)

# ---------------- Indian crops (optional) ----------------
TEST_VIDEOS = set()
ind = glob.glob("/kaggle/input/**/tvs_signal_labels.json", recursive=True)
if ind:
    root = Path(ind[0]).parent
    meta = json.loads(Path(ind[0]).read_text())
    TEST_VIDEOS = set(meta["test_videos"])
    for f, r in meta["labels"].items():
        img = cv2.imread(str(root / "crops" / f))
        if img is None or r["label"] not in CLASSES:
            continue
        split = "test" if r["video"] in TEST_VIDEOS else "train"
        samples.append((img, r["label"], "india", r["video"], split))
        if split == "train":   # few Indian crops: repeat them
            for _ in range(4):
                samples.append((img, r["label"], "india", r["video"], split))
say("all samples", Counter((s[2], s[1], s[4]) for s in samples))

# ---------------- write folders ----------------
for sp in ("train", "val", "test"):
    for c in CLASSES:
        (WORK / sp / c).mkdir(parents=True, exist_ok=True)
index = []
for k, (crop, lab, src, grp, sp) in enumerate(samples):
    folder = "val" if sp == "val" else ("test" if sp == "test" else "train")
    im = degrade(crop) if (folder == "train" and rng.random() < 0.6) else crop
    fn = f"{src}_{k}.jpg"
    cv2.imwrite(str(WORK / folder / lab / fn), square(im))
    index.append({"file": fn, "label": lab, "source": src, "group": grp, "split": folder})
if not any((WORK / "test").rglob("*.jpg")):
    import shutil; shutil.rmtree(WORK / "test")
say("images written", Counter(r["split"] for r in index))

from ultralytics import YOLO
m = YOLO("yolo11s-cls.pt")
m.train(data=str(WORK), imgsz=64, epochs=EPOCHS, batch=256, device=DEV, workers=4, project=str(OUT), name="run",
        exist_ok=True, optimizer="AdamW", lr0=1e-3, cos_lr=True, fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
        degrees=3, scale=0.2, translate=0.1, erasing=0.1, auto_augment=None, seed=0, plots=False, verbose=False)
best = OUT / "run/weights/last.pt"
m = YOLO(str(best))
report = {}
for folder in ("val", "test"):
    rows = [r for r in index if r["split"] == folder]
    if not rows:
        continue
    conf = defaultdict(Counter)
    paths = [str(WORK / folder / r["label"] / r["file"]) for r in rows]
    for i in range(0, len(paths), 512):
        for r, p in zip(rows[i:i + 512], m.predict(paths[i:i + 512], imgsz=64, verbose=False)):
            top = p.names[int(p.probs.top1)]
            conf[f'{r["source"]}:{r["label"]}'][top] += 1
    report[folder] = {k: dict(v) for k, v in conf.items()}
(OUT / "report.json").write_text(json.dumps(report, indent=1))
(OUT / "index.json").write_text(json.dumps(index))
say("report", json.dumps(report))
say("done")
