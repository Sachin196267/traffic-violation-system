#!/usr/bin/env python3
"""
Build datasets/helmet_nohelmet_final/ — a clean, leakage-controlled, head-level
helmet / no_helmet YOLO dataset.

Final class mapping (fixed):  0 = helmet   1 = no_helmet
Boxes are HEAD-level (the rider's head / helmet), never whole-rider boxes.

Sources (originals are only READ, never modified):
  andrewmvd  datasets/external/andrewmvd_helmet-detection      CC0-1.0     VOC xml
             "With Helmet" -> helmet, "Without Helmet" -> no_helmet
  india      datasets/kaggle/helmet_india (Roboflow aryan_1)    CC BY 4.0   YOLO
             class meaning (Kaggle card aryanvaid13/indian-helmet-detection-dataset,
             verified visually): 0 numberPlate(drop) 1 faceWithNoHelmet -> no_helmet
             2 faceWithGoodHelmet -> helmet  3 faceWithBadHelmet -> AMBIGUOUS, region
             grey-masked (not forced into either class)  4 rider(drop)
             Images containing a class-1/2/3 box that is clearly not a head box
             (h > 0.40 of image or h/w > 2.5) are EXCLUDED and listed for review.
  anees      datasets/external/aneesarom_rider-...-number-plate Apache-2.0  YOLO
             0 with helmet -> helmet, 1 without helmet -> no_helmet, 2 rider / 3 plate dropped
  idd_bg     IDD images with no person/rider/motorcycle/bicycle labels -> background
             (hard-negative) images with empty label files. IDD train split feeds
             our train split only; IDD val split feeds our val/test only.

Leakage control: images are grouped (union-find) by
  * source-specific sequence keys (india: filename prefix + index chunk of 20), and
  * perceptual near-duplicates (16x16 dHash, Hamming <= 24/256) across ALL sources,
and whole groups are assigned to train/val/test (70/15/15).
"""
from __future__ import annotations

import csv, json, random, re, shutil, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "helmet_nohelmet_final"
SEED = 20260924
NAMES = ["helmet", "no_helmet"]
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
N_BG = {"train": 180, "val": 40, "test": 40}
MIN_BOX_PX = 3  # boxes whose shorter side is < 3 px are dropped (logged)
EXT = {".jpg", ".jpeg", ".png"}

LICENSES = {"andrewmvd": "CC0-1.0", "india": "CC BY 4.0", "anees": "Apache-2.0", "idd_bg": "IDD (Kaggle redzapdos123) — already used in project"}
URLS = {
    "andrewmvd": "https://www.kaggle.com/datasets/andrewmvd/helmet-detection",
    "india": "https://universe.roboflow.com/yolo-ftygl/aryan_1 (also kaggle aryanvaid13/indian-helmet-detection-dataset)",
    "anees": "https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate",
    "idd_bg": "https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11",
}


class Item:
    def __init__(self, source, img_path, boxes, group_key, mask_boxes=(), fixed_split=None):
        self.source, self.img_path, self.boxes = source, Path(img_path), boxes  # boxes: [(cls, xc, yc, w, h)] normalized
        self.group_key, self.mask_boxes, self.fixed_split = group_key, list(mask_boxes), fixed_split


def read_yolo(p: Path):
    if not p.exists():
        return None
    return [l.split() for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- loaders
def load_andrewmvd(flags):
    base = ROOT / "datasets/external/andrewmvd_helmet-detection"
    items = []
    cmap = {"With Helmet": 0, "Without Helmet": 1}
    for xp in sorted((base / "annotations").glob("*.xml")):
        r = ET.parse(xp).getroot()
        img = base / "images" / r.findtext("filename")
        if not img.exists():
            flags.append(("andrewmvd", str(xp), "image missing")); continue
        W, H = int(r.findtext("size/width")), int(r.findtext("size/height"))
        im = cv2.imread(str(img))
        if im is None:
            flags.append(("andrewmvd", str(img), "unreadable")); continue
        if (im.shape[1], im.shape[0]) != (W, H):
            flags.append(("andrewmvd", str(img), f"xml size {W}x{H} != real {im.shape[1]}x{im.shape[0]} (real used)"))
            W, H = im.shape[1], im.shape[0]
        boxes = []
        for o in r.findall("object"):
            n = o.findtext("name")
            if n not in cmap:
                flags.append(("andrewmvd", str(xp), f"unknown class {n}")); continue
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((cmap[n], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        items.append(Item("andrewmvd", img, boxes, f"andrewmvd:{img.stem}"))
    return items


def load_india(flags):
    base = ROOT / "datasets/kaggle/helmet_india"
    items = []
    for split in ("train", "valid"):
        for img in sorted((base / split / "images").iterdir()):
            if img.suffix.lower() not in EXT:
                continue
            lines = read_yolo(base / split / "labels" / (img.stem + ".txt")) or []
            boxes, masks, bad = [], [], False
            for l in lines:
                c = int(l[0]); x, y, w, h = map(float, l[1:5])
                if c in (1, 2, 3) and (h > 0.40 or h / max(w, 1e-6) > 2.5):
                    bad = True
                    flags.append(("india", str(img), f"class {c} box not head-shaped (w={w:.2f},h={h:.2f}) -> image excluded, REVIEW"))
                if c == 2: boxes.append((0, x, y, w, h))
                elif c == 1: boxes.append((1, x, y, w, h))
                elif c == 3: masks.append((x, y, w, h))
            if bad:
                continue
            stem = img.name.split(".rf.")[0]
            m = re.match(r"^(\d+)[-_]([A-Za-z]*)(\d+)", stem)
            if m:
                key = f"india:{m.group(1)}{m.group(2)}:{int(m.group(3)) // 20}"
            elif stem.startswith("Screenshot"):
                key = "india:" + stem[:len("Screenshot-2023-08-02-1")]  # same capture session (hour)
            else:
                key = f"india:{stem}"
            items.append(Item("india", img, boxes, key, masks))
    return items


def load_anees(flags):
    base = ROOT / "datasets/external/aneesarom_rider-with-helmet-without-helmet-number-plate"
    items = []
    for split in ("train", "val"):
        for img in sorted((base / split / "images").iterdir()):
            if img.suffix.lower() not in EXT:
                continue
            lines = read_yolo(base / split / "labels" / (img.stem + ".txt"))
            if lines is None:
                flags.append(("anees", str(img), "missing label")); continue
            boxes = [(int(l[0]), *map(float, l[1:5])) for l in lines if l[0] in ("0", "1")]
            items.append(Item("anees", img, boxes, f"anees:{img.stem}"))
    return items


def _screen_no_people(paths):
    """IDD labels miss some people; reject frames where the existing IDD detector (read-only use)
    sees a bicycle/motorcycle/person/rider at conf >= 0.15."""
    from ultralytics import YOLO
    det = YOLO(str(ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"))
    keep = []
    for i in range(0, len(paths), 16):
        for p, r in zip(paths[i:i + 16], det.predict([str(p) for p in paths[i:i + 16]], conf=0.15, classes=[2, 6, 7, 8],
                                                     imgsz=960, device="mps", verbose=False)):
            if len(r.boxes) == 0:
                keep.append(p)
    print(f"  idd_bg screening: kept {len(keep)}/{len(paths)} people-free frames")
    return keep


def load_idd_bg(rng):
    base = ROOT / "datasets/kaggle/idd/IDDDetectionsYOLODataset"
    people_vehicles = {"2", "6", "7", "8"}  # bicycle, motorcycle, person, rider
    out = []
    for idd_split, targets in (("train", ["train"]), ("val", ["val", "test"])):
        cands = []
        for lp in sorted((base / idd_split / "labels").glob("*.txt")):
            lines = read_yolo(lp)
            if lines and not any(l[0] in people_vehicles for l in lines) and len(lines) >= 3:
                img = next((base / idd_split / "images" / (lp.stem + e) for e in (".jpg", ".png", ".jpeg")
                            if (base / idd_split / "images" / (lp.stem + e)).exists()), None)
                if img: cands.append(img)
        rng.shuffle(cands)
        cands = _screen_no_people(cands[: 4 * sum(N_BG[t] for t in targets)])
        i = 0
        for t in targets:
            for img in cands[i:i + N_BG[t]]:
                out.append(Item("idd_bg", img, [], f"idd:{img.stem}", fixed_split=t))
            i += N_BG[t]
    return out


# ---------------------------------------------------------------- grouping
def dhash16(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(im, (17, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def main():
    rng = random.Random(SEED)
    flags: list[tuple] = []
    items = load_andrewmvd(flags) + load_india(flags) + load_anees(flags)
    bg = load_idd_bg(rng)
    print(f"labelled candidates: {len(items)}  background: {len(bg)}  flagged: {len(flags)}")

    # --- groups: sequence keys + near duplicates (across all labelled sources)
    uf = UF(len(items))
    first = {}
    for i, it in enumerate(items):
        if it.group_key in first: uf.u(i, first[it.group_key])
        else: first[it.group_key] = i
    H = np.stack([dhash16(it.img_path) for it in items])
    bits = np.unpackbits(H, axis=1)
    n_near = 0; cross_src = Counter()
    for i in range(len(items)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.where(d <= 24)[0] + i + 1:
            uf.u(i, j); n_near += 1
            if items[i].source != items[j].source: cross_src[(items[i].source, items[j].source)] += 1
    groups = defaultdict(list)
    for i in range(len(items)): groups[uf.f(i)].append(i)
    print(f"near-dup pairs merged: {n_near} (cross-source: {dict(cross_src)}), groups: {len(groups)}")

    # --- exact-duplicate removal inside a group (keep first), keeps the group intact otherwise
    seen_exact = {}
    drop = set()
    for i, it in enumerate(items):
        k = H[i].tobytes()
        if k in seen_exact:
            drop.add(i); flags.append((it.source, str(it.img_path), f"exact duplicate of {items[seen_exact[k]].img_path.name} -> dropped"))
        else:
            seen_exact[k] = i

    # --- assign groups to splits, per source, largest groups first (greedy fill)
    split_of = {}
    by_src = defaultdict(list)
    for g, idx in groups.items():
        srcs = Counter(items[i].source for i in idx)
        by_src[srcs.most_common(1)[0][0]].append(g)
    def gstat(g):  # (images, helmet boxes, no_helmet boxes) of a group
        idx = [i for i in groups[g] if i not in drop]
        return np.array([len(idx), sum(b[0] == 0 for i in idx for b in items[i].boxes),
                         sum(b[0] == 1 for i in idx for b in items[i].boxes)], dtype=float)
    for src, gl in by_src.items():
        rng.shuffle(gl)
        st = {g: gstat(g) for g in gl}
        gl.sort(key=lambda g: -st[g][0])
        total = sum(st.values()) + 1e-9
        fill = {s: np.zeros(3) for s in SPLIT_FRAC}
        for g in gl:
            # class-aware greedy: pick the split with the largest normalised deficit over
            # (images, helmet, no_helmet), no_helmet weighted x2 (critical class)
            w = np.array([1.0, 1.0, 2.0])
            s = max(SPLIT_FRAC, key=lambda s: float((w * (SPLIT_FRAC[s] * total - fill[s]) / total).sum()))
            split_of[g] = s; fill[s] += st[g]

    # --- write
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in SPLIT_FRAC:
        (OUT / "images" / s).mkdir(parents=True); (OUT / "labels" / s).mkdir(parents=True)
    rows, stats = [], defaultdict(Counter)
    tiny = 0
    all_items = [(i, it, split_of[uf.f(i)]) for i, it in enumerate(items) if i not in drop] + \
                [(None, it, it.fixed_split) for it in bg]
    for i, it, s in all_items:
        im = cv2.imread(str(it.img_path))
        h, w = im.shape[:2]
        for (x, y, bw, bh) in it.mask_boxes:  # ambiguous "bad helmet" regions -> neutral grey
            x1, y1 = max(0, int((x - bw / 2) * w)), max(0, int((y - bh / 2) * h))
            x2, y2 = min(w, int((x + bw / 2) * w) + 1), min(h, int((y + bh / 2) * h) + 1)
            im[y1:y2, x1:x2] = 114
        lines = []
        for c, x, y, bw, bh in it.boxes:
            x1, y1, x2, y2 = max(0., x - bw / 2), max(0., y - bh / 2), min(1., x + bw / 2), min(1., y + bh / 2)
            if (x2 - x1) * w < MIN_BOX_PX or (y2 - y1) * h < MIN_BOX_PX:
                tiny += 1; flags.append((it.source, str(it.img_path), f"box <{MIN_BOX_PX}px dropped")); continue
            lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            stats[s][NAMES[c]] += 1
        name = f"{it.source}__{it.img_path.stem.split('.rf.')[0]}"
        name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:120]
        k = 1
        while (OUT / "images" / s / f"{name}.jpg").exists():
            name = f"{name}_{k}"; k += 1
        ext = ".png" if it.img_path.suffix.lower() == ".png" else ".jpg"
        dst = OUT / "images" / s / f"{name}{ext}"
        if it.mask_boxes: cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else: shutil.copy2(it.img_path, dst)
        (OUT / "labels" / s / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[s]["images"] += 1; stats[s][f"img_{it.source}"] += 1
        if not lines: stats[s]["background_images"] += 1
        rows.append({"file": dst.name, "split": s, "source": it.source, "license": LICENSES[it.source],
                     "source_url": URLS[it.source], "original_path": str(it.img_path.relative_to(ROOT)),
                     "group": f"g{uf.f(i)}" if i is not None else it.group_key, "masked_ambiguous_regions": len(it.mask_boxes),
                     "n_helmet": sum(1 for l in lines if l[0] == "0"), "n_no_helmet": sum(1 for l in lines if l[0] == "1")})

    (OUT / "data.yaml").write_text(
        f"# helmet / no_helmet (HEAD-level boxes). Built by helmet_training/build_dataset.py seed={SEED}\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: helmet\n  1: no_helmet\n")
    with open(OUT / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(OUT / "flagged_for_review.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["source", "path", "reason"]); wr.writerows(flags)
    # leakage verification: no group appears in two splits
    g_split = defaultdict(set)
    for r in rows: g_split[r["group"]].add(r["split"])
    leaks = [g for g, ss in g_split.items() if len(ss) > 1]
    rep = {"splits": {s: dict(stats[s]) for s in SPLIT_FRAC}, "groups": len(groups), "near_dup_pairs": n_near,
           "cross_source_near_dups": {f"{a}|{b}": n for (a, b), n in cross_src.items()},
           "exact_dups_dropped": len(drop), "tiny_boxes_dropped": tiny, "flags": len(flags), "groups_spanning_splits": len(leaks)}
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    assert not leaks, "LEAKAGE: group in multiple splits"


if __name__ == "__main__":
    main()
