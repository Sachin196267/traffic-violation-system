#!/usr/bin/env python3
"""
Full evaluation of a helmet / no_helmet detector (read-only; never touches app code).

    .venv/bin/python helmet_training/evaluate.py --weights runs/helmet_nohelmet_final/s960_traffic/weights/best.pt \
        --imgsz 960 --tag s960_traffic

What it measures (all written to helmet_training/reports/eval/<tag>/):
  1. Ultralytics mAP50 / mAP50-95 / per-class P,R on clean val and test.
  2. Operating-point error accounting on test (conf chosen on VAL by max macro-F1, never on test):
       per-class P / R / F1, full confusion incl. background,
       FALSE NO-HELMET  (GT helmet  -> predicted no_helmet)   <- critical
       FALSE HELMET     (GT no_helmet -> predicted helmet)
       missed heads, spurious detections.
  3. Head-size buckets (sqrt box area in px): <16, 16-32, 32-64, >=64.
  4. Scale sweep (image downscaled x1 .. x0.2) -> recall vs head size -> min practical head size.
  5. Synthetic extreme conditions on the test set (night, dusk, over-exposure, shadows, motion blur,
     defocus, heavy JPEG, noise, low-res, rain, fog, occlusion).
  6. Real-condition proxies on the clean test set: dark vs normal luminance, blurry vs sharp,
     crowded (>=5 heads) vs sparse, elevated-CCTV subset.
  7. False positives per image on 150 people-free IDD *test*-split frames (never used anywhere).
"""
from __future__ import annotations

import argparse, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/helmet_nohelmet_final"
NAMES = ["helmet", "no_helmet"]
CACHE = ROOT / "helmet_training/_eval_cache"
BUCKETS = [(0, 16, "xs(<16px)"), (16, 32, "s(16-32px)"), (32, 64, "m(32-64px)"), (64, 1e9, "l(>=64px)")]


# ------------------------------------------------------------------ io helpers
def load_split(split):
    items = []
    for p in sorted((DS / "images" / split).iterdir()):
        lp = DS / "labels" / split / (p.stem + ".txt")
        gt = [list(map(float, l.split())) for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        items.append((p, gt))
    return items


def to_xyxy(g, w, h):
    c, x, y, bw, bh = g
    return int(c), np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def predict(model, images, imgsz, conf=0.01, batch=8):
    """images: list of np arrays (BGR). returns list of (N,6) arrays xyxy,conf,cls."""
    out = []
    for i in range(0, len(images), batch):
        for r in model.predict(images[i:i + batch], imgsz=imgsz, conf=conf, iou=0.6, device=DEVICE, verbose=False):
            b = r.boxes
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
                       if len(b) else np.zeros((0, 6)))
    return out


# ------------------------------------------------------------------ matching
def keep(P, thr):
    """per-class confidence filter; thr = (conf_helmet, conf_no_helmet)"""
    if not len(P): return P
    return P[P[:, 4] >= np.where(P[:, 5] == 0, thr[0], thr[1])]


def match(gts, preds, thr, iou_t=0.5):
    """Class-agnostic greedy matching so class swaps are visible.
    gts: list of (cls, xyxy). preds: (N,6). Returns list of records."""
    P = keep(preds, thr)
    P = P[np.argsort(-P[:, 4])] if len(P) else P
    used = set(); rec = []
    gt_hit = [None] * len(gts)
    for pi, p in enumerate(P):
        best, bj = iou_t, -1
        for j, (c, g) in enumerate(gts):
            if j in used: continue
            v = iou(p[:4], g)
            if v >= best: best, bj = v, j
        if bj >= 0:
            used.add(bj); gt_hit[bj] = int(p[5])
        else:
            rec.append(("fp", None, int(p[5]), None))
    for j, (c, g) in enumerate(gts):
        size = float(np.sqrt(max((g[2] - g[0]) * (g[3] - g[1]), 0)))
        rec.append(("gt", c, gt_hit[j], size))
    return rec


def summarize(records):
    """records from match(). Confusion rows = GT (helmet, no_helmet, background), cols = pred (helmet, no_helmet, missed)."""
    cm = np.zeros((3, 3), int)
    for kind, g, p, _ in records:
        if kind == "fp": cm[2, p] += 1
        else: cm[g, 2 if p is None else p] += 1
    res = {}
    for c in (0, 1):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        res[NAMES[c]] = {"P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / (P + R) if P + R else 0.0, 4),
                         "tp": int(tp), "fp": int(fp), "fn": int(fn), "n_gt": int(cm[c, :].sum())}
    n_h, n_n = cm[0].sum(), cm[1].sum()
    res["macro_F1"] = round((res["helmet"]["F1"] + res["no_helmet"]["F1"]) / 2, 4)
    res["false_no_helmet"] = {"count": int(cm[0, 1]), "rate_of_helmet_gt": round(cm[0, 1] / n_h, 4) if n_h else None}
    res["false_helmet"] = {"count": int(cm[1, 0]), "rate_of_no_helmet_gt": round(cm[1, 0] / n_n, 4) if n_n else None}
    res["missed"] = {"helmet": int(cm[0, 2]), "no_helmet": int(cm[1, 2])}
    res["spurious"] = {"as_helmet": int(cm[2, 0]), "as_no_helmet": int(cm[2, 1])}
    res["confusion_rows_gt[helmet,no_helmet,background]_cols_pred[helmet,no_helmet,missed]"] = cm.tolist()
    return res


def size_table(records):
    out = {}
    for lo, hi, name in BUCKETS:
        r = [x for x in records if x[0] == "gt" and lo <= x[3] < hi]
        if not r: continue
        n = len(r); det = sum(x[2] is not None for x in r); ok = sum(x[2] == x[1] for x in r)
        fnh = sum(x[1] == 0 and x[2] == 1 for x in r); fh = sum(x[1] == 1 and x[2] == 0 for x in r)
        out[name] = {"n_gt": n, "detected(any class)": round(det / n, 3), "correct_class_recall": round(ok / n, 3),
                     "false_no_helmet": fnh, "false_helmet": fh}
    return out


# ------------------------------------------------------------------ degradations
def _rain(img, rng):
    h, w = img.shape[:2]; layer = np.zeros((h, w), np.uint8)
    for _ in range(int(h * w / 600)):
        x, y = rng.integers(0, w), rng.integers(0, h); l = rng.integers(8, 20)
        cv2.line(layer, (x, y), (x + 2, y + l), 200, 1)
    layer = cv2.blur(layer, (3, 3))
    out = cv2.addWeighted(img, 0.85, cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR), 0.35, 0)
    return cv2.GaussianBlur(out, (3, 3), 0)


def _motion(img, k):
    ker = np.zeros((k, k)); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)


def _gamma(img, g):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _night(img, rng):
    d = _gamma(img, 2.6).astype(np.float32) * 0.55
    d[..., 0] *= 1.1; d[..., 2] *= 0.9  # slight blue cast
    d += rng.normal(0, 7, d.shape)
    return np.clip(d, 0, 255).astype(np.uint8)


def _streetlight(img, rng):
    d = _gamma(img, 2.2).astype(np.float32) * 0.6
    d[..., 2] *= 1.25; d[..., 1] *= 1.05; d[..., 0] *= 0.7  # sodium-vapour orange
    return np.clip(d + rng.normal(0, 5, d.shape), 0, 255).astype(np.uint8)


def _shadow(img, rng):
    h, w = img.shape[:2]; m = np.ones((h, w), np.float32)
    pts = np.array([[rng.integers(0, w), 0], [rng.integers(0, w), 0], [rng.integers(0, w), h], [rng.integers(0, w), h]])
    cv2.fillPoly(m, [pts], 0.45)
    return (img * cv2.GaussianBlur(m, (31, 31), 0)[..., None]).astype(np.uint8)


def _fog(img):
    return cv2.addWeighted(img, 0.55, np.full_like(img, 200), 0.45, 0)


def _lowres(img, s):
    h, w = img.shape[:2]
    return cv2.resize(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), (w, h),
                      interpolation=cv2.INTER_LINEAR)


def _jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _occlude(img, gts, rng):
    out = img.copy()
    for c, g in gts:  # cover 25-45 % of each head box with a grey bar (partial occlusion)
        x1, y1, x2, y2 = g.astype(int); f = rng.uniform(0.25, 0.45)
        if rng.random() < 0.5: out[y1:y1 + int((y2 - y1) * f), x1:x2] = 90
        else: out[y1:y2, x1:x1 + int((x2 - x1) * f)] = 90
    return out


CONDITIONS = {
    "clean": lambda im, g, r: im,
    "night(dark+noise)": lambda im, g, r: _night(im, r),
    "streetlight(orange, dark)": lambda im, g, r: _streetlight(im, r),
    "dusk(gamma1.8)": lambda im, g, r: _gamma(im, 1.8),
    "underexposed(x0.35)": lambda im, g, r: (im * 0.35).astype(np.uint8),
    "overexposed(sun)": lambda im, g, r: np.clip(im.astype(np.float32) * 1.7 + 40, 0, 255).astype(np.uint8),
    "hard_shadow": lambda im, g, r: _shadow(im, r),
    "motion_blur(k9)": lambda im, g, r: _motion(im, 9),
    "motion_blur(k15)": lambda im, g, r: _motion(im, 15),
    "defocus(gauss7)": lambda im, g, r: cv2.GaussianBlur(im, (7, 7), 0),
    "jpeg_q20": lambda im, g, r: _jpeg(im, 20),
    "jpeg_q8": lambda im, g, r: _jpeg(im, 8),
    "sensor_noise(s=20)": lambda im, g, r: np.clip(im + r.normal(0, 20, im.shape), 0, 255).astype(np.uint8),
    "lowres(x0.5)": lambda im, g, r: _lowres(im, 0.5),
    "lowres(x0.25)": lambda im, g, r: _lowres(im, 0.25),
    "rain(synthetic)": lambda im, g, r: _rain(im, r),
    "fog/haze": lambda im, g, r: _fog(im),
    "partial_occlusion(25-45%)": lambda im, g, r: _occlude(im, g, r),
}


# ------------------------------------------------------------------ main
def evaluate(weights, imgsz, tag, device="mps"):
    global DEVICE
    DEVICE = device
    from ultralytics import YOLO
    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    assert names == NAMES, f"class mapping mismatch: {names}"
    out_dir = ROOT / "helmet_training/reports/eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"weights": str(weights), "imgsz": imgsz}

    # 1. ultralytics mAP on val / test
    for split in ("val", "test"):
        r = model.val(data=str(DS / "data.yaml"), split=split, imgsz=imgsz, batch=8, device=device, verbose=False,
                      plots=True, project=str(out_dir), name=f"ultra_{split}", exist_ok=True, conf=0.001)
        per = {}
        for i, c in enumerate(r.box.ap_class_index):
            p, rc, a50, a = r.box.class_result(i)
            per[NAMES[int(c)]] = {"P": round(p, 4), "R": round(rc, 4), "mAP50": round(a50, 4), "mAP50-95": round(a, 4)}
        report[f"ultralytics_{split}"] = {"mAP50": round(r.box.map50, 4), "mAP50-95": round(r.box.map, 4),
                                          "P": round(r.box.mp, 4), "R": round(r.box.mr, 4), "per_class": per}

    # 2. pick operating threshold on VAL (max macro-F1)
    val = load_split("val")
    val_imgs = [cv2.imread(str(p)) for p, _ in val]
    val_gts = [[to_xyxy(g, im.shape[1], im.shape[0]) for g in gt] for (p, gt), im in zip(val, val_imgs)]
    val_pred = predict(model, val_imgs, imgsz)
    sweep = {}
    grid = [round(float(x), 3) for x in np.arange(0.10, 0.80, 0.05)]
    for th in grid:  # per-class thresholds, chosen on VAL only
        for tn in grid:
            recs = [x for g, pr in zip(val_gts, val_pred) for x in match(g, pr, (th, tn))]
            sweep[f"{th},{tn}"] = summarize(recs)["macro_F1"]
    best = max(sweep, key=sweep.get)
    thr = tuple(float(x) for x in best.split(","))
    single = {k: v for k, v in sweep.items() if k.split(",")[0] == k.split(",")[1]}
    report["operating_conf(chosen_on_val)"] = list(thr)
    report["val_macroF1_best_per_class_conf"] = sweep[best]
    report["val_macroF1_best_single_conf"] = {"conf": max(single, key=single.get), "macro_F1": max(single.values())}

    # 3. test at that threshold + conditions
    test = load_split("test")
    imgs = [cv2.imread(str(p)) for p, _ in test]
    gts = [[to_xyxy(g, im.shape[1], im.shape[0]) for g in gt] for (p, gt), im in zip(test, imgs)]
    cond_res = {}
    for name, fn in CONDITIONS.items():
        rng = np.random.default_rng(0)
        deg = [fn(im, g, rng) for im, g in zip(imgs, gts)]
        pred = predict(model, deg, imgsz)
        recs = [x for g, pr in zip(gts, pred) for x in match(g, pr, thr)]
        s = summarize(recs)
        cond_res[name] = s
        if name == "clean":
            report["test_operating_point"] = s
            report["test_size_buckets"] = size_table(recs)
            clean_pred = pred
            # save a few annotated failure examples
            _save_failures(test, imgs, gts, pred, thr, out_dir / "failure_examples")
        if name.startswith("lowres") or name.startswith("night"):
            report.setdefault("condition_size_buckets", {})[name] = size_table(recs)
    report["extreme_conditions(test, synthetic)"] = {
        k: {"helmet_P": v["helmet"]["P"], "helmet_R": v["helmet"]["R"], "no_helmet_P": v["no_helmet"]["P"],
            "no_helmet_R": v["no_helmet"]["R"], "macro_F1": v["macro_F1"],
            "false_no_helmet": v["false_no_helmet"]["count"], "false_helmet": v["false_helmet"]["count"],
            "missed": v["missed"], "spurious": v["spurious"]} for k, v in cond_res.items()}

    # 4. scale sweep: shrink whole image (heads shrink proportionally)
    sweep_res = {}
    for s in (1.0, 0.75, 0.5, 0.35, 0.25, 0.18):
        small = [cv2.resize(im, (max(8, int(im.shape[1] * s)), max(8, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA) for im in imgs]
        sg = [[(c, g * s) for c, g in gl] for gl in gts]
        pred = predict(model, small, imgsz)
        recs = [x for g, pr in zip(sg, pred) for x in match(g, pr, thr)]
        sweep_res[f"x{s}"] = size_table(recs)
    report["scale_sweep_size_buckets"] = sweep_res
    agg = defaultdict(lambda: [0, 0, 0])
    for s in sweep_res.values():
        for b, v in s.items():
            agg[b][0] += v["n_gt"]; agg[b][1] += round(v["correct_class_recall"] * v["n_gt"]); agg[b][2] += round(v["detected(any class)"] * v["n_gt"])
    report["size_vs_recall(pooled over scale sweep)"] = {b: {"n_gt": n, "correct_class_recall": round(ok / n, 3), "detected": round(d / n, 3)}
                                                         for b, (n, ok, d) in agg.items()}

    # 5. real-condition proxies on clean test predictions
    prox = defaultdict(list)
    for (p, _), im, g, pr in zip(test, imgs, gts, clean_pred):
        lum = float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())
        sharp = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        recs = match(g, pr, thr)
        prox["dark(mean_luma<80)" if lum < 80 else "normal_light"].extend(recs)
        prox["blurry(lapvar<100)" if sharp < 100 else "sharp"].extend(recs)
        prox["crowded(>=5 heads)" if len(g) >= 5 else ("single(1 head)" if len(g) == 1 else "2-4 heads")].extend(recs)
        if p.name.startswith("india__036"): prox["elevated_CCTV(india 036 cam)"].extend(recs)
        prox[f"source:{p.name.split('__')[0]}"].extend(recs)
    report["real_condition_subsets(test, clean)"] = {k: {**{kk: vv for kk, vv in summarize(v).items() if kk in ("helmet", "no_helmet", "macro_F1", "false_no_helmet", "false_helmet")}} for k, v in prox.items()}

    # 5b. enforcement-relevant subset: GT heads that belong to a two-wheeler rider (per IDD detector)
    report["riders_only(test, clean)"] = rider_only(test, imgs, gts, clean_pred, thr)

    # 6. false positives on unseen people-free IDD test frames
    report["background_false_positives"] = background_fp(model, imgsz, thr)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


def _save_failures(test, imgs, gts, preds, thr, d, n=24):
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    for (p, _), im, g, pr in zip(test, imgs, gts, preds):
        recs = match(g, pr, thr)
        bad = [r for r in recs if r[0] == "fp" or r[2] != r[1]]
        if not bad or k >= n: continue
        vis = im.copy()
        for c, b in g:
            cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), (0, 255, 0) if c == 0 else (0, 0, 255), 1)
        for x in keep(pr, thr):
            col = (255, 255, 0) if int(x[5]) == 0 else (255, 0, 255)
            cv2.rectangle(vis, tuple(x[:2].astype(int)), tuple(x[2:4].astype(int)), col, 2)
            cv2.putText(vis, f"{NAMES[int(x[5])]} {x[4]:.2f}", (int(x[0]), max(10, int(x[1]) - 3)), 0, 0.4, col, 1)
        cv2.imwrite(str(d / p.name.replace(".png", ".jpg")), vis); k += 1
    (d / "LEGEND.txt").write_text("thin green/red = GT helmet/no_helmet; thick cyan/magenta = predicted helmet/no_helmet\n")


def rider_regions(test, imgs):
    """Rider regions from the existing IDD detector (read-only use): rider boxes, and motorcycle/bicycle
    boxes extended upward by 1x their height (where the rider's head is). Cached per image."""
    cache = CACHE / "rider_regions.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(p, im) for (p, _), im in zip(test, imgs) if p.name not in data]
    if todo:
        from ultralytics import YOLO
        det = YOLO(str(ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"))
        for p, im in todo:
            r = det.predict(im, conf=0.25, classes=[2, 6, 8], imgsz=960, device=DEVICE, verbose=False)[0]
            regs = []
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                x1, y1, x2, y2 = b
                regs.append([x1, y1 - 0.15 * (y2 - y1), x2, y1 + 0.6 * (y2 - y1)] if int(c) == 8 else [x1, y1 - 1.0 * (y2 - y1), x2, y1 + 0.3 * (y2 - y1)])
            data[p.name] = regs
        CACHE.mkdir(exist_ok=True); cache.write_text(json.dumps(data))
    return [data[p.name] for p, _ in test]


def rider_only(test, imgs, gts, preds, thr):
    regs = rider_regions(test, imgs)
    inside = lambda b, R: any(r[0] <= (b[0] + b[2]) / 2 <= r[2] and r[1] <= (b[1] + b[3]) / 2 <= r[3] for r in R)
    recs = []
    for g, pr, R in zip(gts, preds, regs):
        gr = [(c, b) for c, b in g if inside(b, R)]
        prr = keep(pr, thr)
        prr = prr[[inside(x[:4], R) for x in prr]] if len(prr) else prr
        recs += match(gr, prr, (0.0, 0.0))
    s = summarize(recs)
    s["note"] = ("GT heads whose centre lies in an IDD-detected rider region (rider box upper part, or above a motorcycle/bicycle). "
                 "Predictions are restricted to the same regions. Region detection itself can miss riders.")
    return s


def background_fp(model, imgsz, thr, n=150):
    base = ROOT / "datasets/kaggle/idd/IDDDetectionsYOLODataset/test"
    lst = CACHE / "idd_test_bg.txt"
    if lst.exists():
        paths = [Path(x) for x in lst.read_text().splitlines() if x.strip()]
    else:
        from ultralytics import YOLO
        det = YOLO(str(ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"))
        paths = []
        for lp in sorted((base / "labels").glob("*.txt")):
            L = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if len(L) >= 3 and not any(l[0] in {"2", "6", "7", "8"} for l in L):
                ip = base / "images" / (lp.stem + ".jpg")
                if ip.exists() and len(det.predict(str(ip), conf=0.15, classes=[2, 6, 7, 8], imgsz=960, device=DEVICE, verbose=False)[0].boxes) == 0:
                    paths.append(ip)
            if len(paths) >= n: break
        CACHE.mkdir(exist_ok=True); lst.write_text("\n".join(map(str, paths)))
    imgs = [cv2.imread(str(p)) for p in paths]
    pred = predict(model, imgs, imgsz)
    fps = [len(keep(p, thr)) for p in pred]
    cls = Counter(int(c) for p in pred for c in keep(p, thr)[:, 5])
    return {"n_images": len(paths), "fp_per_image": round(float(np.mean(fps)), 4), "images_with_any_fp": int(sum(f > 0 for f in fps)),
            "fp_as_helmet": cls.get(0, 0), "fp_as_no_helmet": cls.get(1, 0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    rep = evaluate(ROOT / a.weights if not Path(a.weights).is_absolute() else Path(a.weights), a.imgsz, a.tag, a.device)
    print(json.dumps({k: rep[k] for k in ("ultralytics_test", "operating_conf(chosen_on_val)", "test_operating_point", "background_false_positives")}, indent=1))
