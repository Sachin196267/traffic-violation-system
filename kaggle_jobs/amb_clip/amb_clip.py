"""
Pre-sort the public emergency-vehicle images with CLIP (zero-shot), so only uncertain ones need a human look.

For every image under /kaggle/input: flip-invariant 64-bit dHash (to drop the Roboflow flipped/brightened copies
and cross-dataset duplicates), CLIP ViT-L/14 zero-shot probabilities over the prompts below, and the image
embedding. Output (/kaggle/working): clip_scores.csv (path, hash, prob per class), clip_emb.npy (float16,
same row order), log.txt.
"""
import csv, glob, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
import numpy as np, torch, open_clip
from PIL import Image

T0 = time.time()
OUT = Path("/kaggle/working")
CLASSES = {
    "ambulance": ["a photo of an ambulance", "a photo of an Indian ambulance van", "a photo of a 108 emergency ambulance"],
    "police": ["a photo of a police car", "a photo of a police jeep or police van"],
    "fire": ["a photo of a fire truck", "a photo of a fire engine"],
    "car": ["a photo of a car", "a photo of an SUV"],
    "van": ["a photo of a white van", "a photo of a minibus or passenger van", "a photo of a school van"],
    "bus": ["a photo of a bus"],
    "truck": ["a photo of a truck", "a photo of a pickup truck", "a photo of a water tanker truck"],
    "auto": ["a photo of an auto rickshaw"],
    "twowheeler": ["a photo of a motorcycle or scooter"],
    "other": ["a photo of a poster or text", "a photo of people", "a photo of a building", "a photo of a road without vehicles"],
}
paths = sorted(p for p in glob.glob("/kaggle/input/**/*", recursive=True)
               if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
print(len(paths), "images", flush=True)

dev = "cuda"
model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k", device=dev)
tok = open_clip.get_tokenizer("ViT-L-14")
model.eval()
with torch.no_grad():
    T = []
    for k, ps in CLASSES.items():
        e = model.encode_text(tok(ps).to(dev)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        T.append(e.mean(0))
    T = torch.stack(T); T = T / T.norm(dim=-1, keepdim=True)


def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    hs = []
    for a in (g, g[:, ::-1], g[::-1], g[::-1, ::-1]):     # flip-invariant: min over flips
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        hs.append(int("".join("1" if b else "0" for b in bits), 2))
    return min(hs)


rows, embs = [], []
B = 64
for i in range(0, len(paths), B):
    batch, keep = [], []
    for p in paths[i:i + B]:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print("bad", p, e); continue
        batch.append(prep(im)); keep.append((p, dhash(im), im.size))
    if not batch:
        continue
    with torch.no_grad(), torch.autocast("cuda"):
        e = model.encode_image(torch.stack(batch).to(dev)).float()
    e = e / e.norm(dim=-1, keepdim=True)
    pr = (100 * e @ T.T).softmax(-1).cpu().numpy()
    for (p, h, sz), q in zip(keep, pr):
        rows.append([p.replace("/kaggle/input/", ""), f"{h:016x}", sz[0], sz[1]] + [round(float(v), 4) for v in q])
    embs.append(e.half().cpu().numpy())
    if i % (B * 20) == 0:
        print(f"[{time.time() - T0:5.0f}s] {i}/{len(paths)}", flush=True)

with open(OUT / "clip_scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "hash", "w", "h"] + list(CLASSES)); w.writerows(rows)
np.save(OUT / "clip_emb.npy", np.concatenate(embs))
print("done", len(rows), f"{time.time() - T0:.0f}s", flush=True)
