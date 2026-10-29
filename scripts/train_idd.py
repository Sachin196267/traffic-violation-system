#!/usr/bin/env python3
"""
Phase 2 training: base vehicle/person detector on IDD-D (Indian Driving Dataset).

Run this on your own machine (with a GPU) inside the project repo, from the
repo root — it needs the IDD files you already downloaded to
datasets/kaggle/idd/, not anything on this server. It cannot run here because
this sandbox has no GPU and none of the actual image files were uploaded,
only the JSON/README metadata.

What it does:
  1. Locates data.yaml inside datasets/kaggle/idd/ (searches a couple of
     likely nesting levels since Kaggle zips sometimes add an extra folder).
  2. Sanity-checks the split folders actually contain images before wasting
     a training run on a broken path.
  3. Fine-tunes a YOLO model (default: yolo11s.pt) starting from COCO
     pretrained weights — this matches the roadmap's Phase 2 instruction to
     "use the existing YOLO model first, fine-tune with Indian data only if
     required" rather than training from scratch.
  4. Copies the best weights to models/traffic.pt, the deliverable named in
     the roadmap.

Usage:
    pip install ultralytics
    python scripts/train_idd.py                     # sensible defaults
    python scripts/train_idd.py --epochs 100 --imgsz 640 --batch 16
    python scripts/train_idd.py --model yolo11n.pt   # smaller/faster
    python scripts/train_idd.py --data path/to/data.yaml   # override search

Notes:
  - You downloaded a capped subset (10,000 of 41,962 images) to keep disk/
    bandwidth sane. That's plenty to validate the pipeline end-to-end and get
    a real mAP baseline. If results look promising, re-run the downloader
    without --only idd's cap (or increase max_files) to train on the full
    set later.
  - Default epochs is intentionally modest (60) for a first pipeline-
    validation run on the 10k subset. Bump it once you're past "does this
    work" and into "how good can this get".
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent

CANDIDATE_DATA_YAMLS = [
    REPO_ROOT / "datasets" / "kaggle" / "idd" / "IDDDetectionsYOLODataset" / "data.yaml",
    REPO_ROOT / "datasets" / "kaggle" / "idd" / "data.yaml",
]


def find_data_yaml(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            sys.exit(f"--data given but not found: {p}")
        return p

    for candidate in CANDIDATE_DATA_YAMLS:
        if candidate.exists():
            return candidate

    # Fall back to a search in case Kaggle's zip nested things differently.
    idd_root = REPO_ROOT / "datasets" / "kaggle" / "idd"
    if idd_root.exists():
        hits = list(idd_root.rglob("data.yaml"))
        if hits:
            return hits[0]

    sys.exit(
        "Could not find data.yaml under datasets/kaggle/idd/.\n"
        "Checked:\n  " + "\n  ".join(str(c) for c in CANDIDATE_DATA_YAMLS) + "\n"
        "Pass the correct location explicitly with --data path/to/data.yaml"
    )


def sanity_check(data_yaml: Path) -> None:
    import yaml  # provided by ultralytics' deps

    cfg = yaml.safe_load(data_yaml.read_text())
    base = data_yaml.parent
    root = base / cfg.get("path", ".") if not str(cfg.get("path", ".")).startswith(("/", "C:", "c:")) else Path(cfg["path"])

    problems = []
    for split_key in ("train", "val", "test"):
        rel = cfg.get(split_key)
        if not rel:
            continue
        split_dir = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not split_dir.exists():
            problems.append(f"  {split_key}: {split_dir} (missing)")
            continue
        n_images = sum(1 for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"  {split_key}: {split_dir}  ->  {n_images} images")
        if n_images == 0:
            problems.append(f"  {split_key}: {split_dir} (0 images found)")

    if problems:
        sys.exit(
            "data.yaml points at split folders that are missing or empty:\n"
            + "\n".join(problems)
            + "\n\nRe-check your download (datasets/kaggle/idd/) before training."
        )

    print(f"  classes ({cfg.get('nc')}): {cfg.get('names')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="Explicit path to IDD's data.yaml (auto-detected if omitted)")
    ap.add_argument("--model", default="yolo11s.pt", help="Starting weights (COCO-pretrained). e.g. yolo11n.pt, yolo11s.pt, yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for first GPU, 'cpu' to force CPU (slow)")
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "detect"))
    ap.add_argument("--name", default="idd_traffic")
    ap.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs with no val improvement)")
    ap.add_argument("--resume", action="store_true", help="Resume the most recent interrupted run with this --name")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install ultralytics")

    data_yaml = find_data_yaml(args.data)
    print(f"Using dataset config: {data_yaml}")
    print("Checking split folders...")
    sanity_check(data_yaml)

    model = YOLO(args.model)

    print(f"\nStarting training: model={args.model} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        plots=True,
    )

    # Ultralytics stores the run's weights under <project>/<name>/weights/best.pt
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"\nTraining finished but couldn't find best.pt at {best} — check {run_dir}/weights/ manually.")
        return

    dest_dir = REPO_ROOT / "models"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / "traffic.pt"
    shutil.copy2(best, dest)
    print(f"\nDone. Best weights copied to: {dest}")
    print(f"Full run artifacts (curves, confusion matrix, val predictions): {run_dir}")
    print("\nNext: evaluate this against your existing best.pt on Indian traffic clips before deciding whether")
    print("further fine-tuning (e.g. on the full 41,962-image IDD set, or UVH-26) is actually needed — per the")
    print("roadmap's Phase 2 instruction not to fine-tune unless the base model underperforms.")


if __name__ == "__main__":
    main()
