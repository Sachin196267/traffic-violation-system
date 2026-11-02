"""Charts for chapter 4, all from the project's own measured results."""
import csv
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/sachinnair/Desktop/traffic violation system")
OUT = Path(__file__).parent / "fig"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6})
C1, C2, C3, C4 = "#2f5d8a", "#c0504d", "#7a9a3a", "#8064a2"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)


def training_curve():
    rows = list(csv.DictReader(open(ROOT / "runs/idd_final/results.csv")))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], color=C1, lw=1.6, label="mAP@0.5")
    ax.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in rows], color=C2, lw=1.6, label="mAP@0.5:0.95")
    ax.plot(ep, [float(r["metrics/precision(B)"]) for r in rows], color=C3, lw=1.1, ls="--", label="Precision")
    ax.plot(ep, [float(r["metrics/recall(B)"]) for r in rows], color=C4, lw=1.1, ls="--", label="Recall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation score"); ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", frameon=False)
    save(fig, "c_train")


def per_class():
    d = json.load(open(ROOT / "runs/idd_final/idd_metrics.json"))["test"]["per_class"]
    cls = ["car", "bus", "truck", "autorickshaw", "motorcycle", "rider", "person", "bicycle", "traffic light"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.27
    xs = range(len(cls))
    for k, (m, c) in enumerate([("P", C1), ("R", C3), ("mAP50", C2)]):
        ax.bar([x + (k - 1) * w for x in xs], [d[n][m] for n in cls], w, color=c,
               label={"P": "Precision", "R": "Recall", "mAP50": "mAP@0.5"}[m])
    ax.set_xticks(list(xs)); ax.set_xticklabels([c.replace("autorickshaw", "auto-\nrickshaw").replace("traffic light", "traffic\nlight") for c in cls], fontsize=8.5)
    ax.set_ylim(0, 1); ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper right")
    save(fig, "c_class")


def helmet():
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    labels = ["'No helmet' calls", "Wrong 'no helmet' calls", "Too blurry to verify"]
    first, final = [137, 20, 42], [29, 1, 1]
    xs = range(3); w = 0.36
    b1 = ax.bar([x - w / 2 for x in xs], first, w, color="#bfbfbf", label="First version (3 % not-sure setting)")
    b2 = ax.bar([x + w / 2 for x in xs], final, w, color=C1, label="Final precision-first rule")
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(int(b.get_height())), ha="center", fontsize=8.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels); ax.set_ylabel("Riders (7 unseen videos)")
    ax.legend(fontsize=8.5, frameon=False); ax.set_ylim(0, 160)
    save(fig, "c_helmet")


def emergency():
    s = json.load(open(ROOT / "runs/ambulance_cls/eval.json"))["sweeps"]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for key, c, m in [("emergency (probe)", C1, "o"), ("emergency (zero-shot)", C2, "s")]:
        pts = sorted((float(t), v) for t, v in s[key].items())
        ax.plot([100 * v["false"] / v["of_neg"] for _, v in pts], [100 * v["caught"] / v["of"] for _, v in pts],
                marker=m, color=c, lw=1.4, ms=4, label="Trained probe" if "probe" in key else "Zero-shot CLIP")
        for t, v in pts:
            if key.endswith("(probe)") and t in (0.5, 0.9, 0.98):
                ax.annotate(f"t={t}", (100 * v["false"] / v["of_neg"], 100 * v["caught"] / v["of"]),
                            textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Ordinary vehicles wrongly flagged (%) - 2,450 held-out tracks")
    ax.set_ylabel("Emergency vehicles caught (%)\n170 held-out tracks", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    save(fig, "c_emerg")


def timing():
    c = sqlite3.connect(ROOT / "app/data/platform.db")
    pts = []
    for cr, fi, m in c.execute("select created, finished, meta from analyses where status='done'"):
        m = json.loads(m or "{}")
        pts.append((m.get("duration") or 0, fi - cr, (m.get("original") or m).get("width", m.get("width"))))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for d, t, w in pts:
        ax.scatter(d, t / 60, s=28, color=C2 if (w or 0) > 1920 else C1 if (w or 0) >= 1280 else C3, zorder=3)
    ax.plot([0, 320], [0, 320 / 60], color="#888", ls="--", lw=1)
    ax.text(250, 250 / 60 + 0.6, "real time (1x)", fontsize=8, color="#666")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C2, label="4K source"),
                       Line2D([], [], marker="o", ls="", color=C1, label="720p-1080p"),
                       Line2D([], [], marker="o", ls="", color=C3, label="640x480")], fontsize=8.5, frameon=False)
    ax.set_xlabel("Video length (s)"); ax.set_ylabel("Processing time (min)")
    save(fig, "c_time")


def outcomes():
    c = sqlite3.connect(ROOT / "app/data/platform.db")
    det = {"helmet": "No helmet", "triple": "Triple riding", "redlight": "Red light"}
    data = {k: {"confirmed": 0, "review": 0} for k in det}
    for route, d, n in c.execute("select route, detector, count(*) from findings group by route, detector"):
        if d in data and route in data[d]:
            data[d][route] = n
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ys = range(len(det))
    conf = [data[k]["confirmed"] for k in det]; rev = [data[k]["review"] for k in det]
    ax.barh(list(ys), conf, color=C2, label="Confirmed by the model's own rule")
    ax.barh(list(ys), rev, left=conf, color="#f0c05a", label="Sent to human review")
    for y, a, b in zip(ys, conf, rev):
        ax.text(a + b + 2, y, f"{a} + {b}", va="center", fontsize=8.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(list(det.values())); ax.set_xlabel("Raw findings (before duplicate merging)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right"); ax.set_xlim(0, 175)
    save(fig, "c_outcome")


if __name__ == "__main__":
    for f in (training_curve, per_class, helmet, emergency, timing, outcomes):
        f()
    print("charts done")
