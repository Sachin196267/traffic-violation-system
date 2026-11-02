"""
Triple-riding classifier (one / two / three_plus / not_bike) with leave-videos-out CV on Kaggle.

Input: private dataset tvs-triple-crops = crops/*.jpg (+ deg_*.jpg degraded copies) and
samples.json rows {key, video, label, crop_h, fold, file}; fold -1 = training-only source.
Runs the 5 CV folds and the final model two at a time (one per T4), same recipe as
triple_training/train_cv.py (last epoch, no checkpoint picking on held-out videos).
Output (/kaggle/working/out): oof.json (held-out probabilities per file), final/weights/last.pt,
fold*/results.csv, log.txt.
"""
import glob, json, os, random, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.153"], check=True)

T0 = time.time()
SMOKE = True
EPOCHS = 2 if SMOKE else 30
K, REPEAT_TRIPLE = 5, 4
IN = Path(glob.glob("/kaggle/input/**/samples.json", recursive=True)[0]).parent
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/tcls")
samples = json.loads((IN / "samples.json").read_text())
if SMOKE:
    samples = samples[::10]
log = open(OUT / "log.txt", "a")


def say(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(map(str, a))
    print(msg, flush=True)
    log.write(msg + "\n"); log.flush()


def build(name, held):
    rng = random.Random(0)
    base = WORK / name
    for s in samples:
        if name == "all":
            split = "val" if s["fold"] == 0 and rng.random() < 0.1 else "train"
        else:
            split = "val" if s["fold"] == held else "train"
        d = base / split / s["label"]
        d.mkdir(parents=True, exist_ok=True)
        src, deg = IN / "crops" / s["file"], IN / "crops" / f"deg_{s['file']}"
        os.symlink(src, d / s["file"])
        if split == "train":
            os.symlink(deg, d / deg.name)
            if s["label"] == "three_plus":
                for j in range(1, REPEAT_TRIPLE):
                    os.symlink(src, d / f"rep{j}_{s['file']}")
                    os.symlink(deg, d / f"rep{j}_{deg.name}")
    return base


TRAIN = r'''
import sys
from ultralytics import YOLO
data, name, dev, epochs, out = sys.argv[1:]
m = YOLO("yolo11s-cls.pt")
m.train(data=data, imgsz=224, epochs=int(epochs), batch=64, device=int(dev), workers=2,
        project=out, name=name, exist_ok=True, optimizer="AdamW", lr0=1e-3, cos_lr=True,
        hsv_v=0.5, hsv_s=0.5, degrees=5, scale=0.3, translate=0.1, fliplr=0.5, erasing=0.2,
        auto_augment="randaugment", seed=0, plots=False, verbose=False)
'''
(WORK).mkdir(parents=True, exist_ok=True)
(WORK / "train_one.py").write_text(TRAIN)
jobs = [(f"fold{k}", k) for k in range(K)] + [("final", None)]
say(len(samples), "samples;", "jobs:", [j[0] for j in jobs])
for i in range(0, len(jobs), 2):
    procs = []
    for dev, (name, held) in enumerate(jobs[i:i + 2]):
        data = build("all" if held is None else name, held)
        say("start", name, "on GPU", dev)
        procs.append((name, subprocess.Popen([sys.executable, str(WORK / "train_one.py"), str(data), name,
                                              str(dev), str(EPOCHS), str(OUT)],
                                             stdout=open(OUT / f"{name}.log", "w"), stderr=subprocess.STDOUT)))
    for name, p in procs:
        p.wait()
        say("done", name, "exit", p.returncode)

from ultralytics import YOLO
oof = {}
for k in range(K):
    w = OUT / f"fold{k}" / "weights" / "last.pt"
    if not w.exists():
        say("missing", w); continue
    m = YOLO(str(w))
    files = [s["file"] for s in samples if s["fold"] == k]
    for j in range(0, len(files), 256):
        for f, r in zip(files[j:j + 256], m.predict([str(IN / "crops" / f) for f in files[j:j + 256]],
                                                      imgsz=224, device=0, verbose=False)):
            oof[f] = {r.names[c]: round(float(v), 5) for c, v in enumerate(r.probs.data.tolist())}
    say(f"fold {k}: {len(files)} held-out crops predicted")
(OUT / "oof.json").write_text(json.dumps(oof))
for d in OUT.glob("fold*/weights"):
    for f in d.glob("best.pt"):
        f.unlink()                       # keep output small: last.pt only
say("all done")
