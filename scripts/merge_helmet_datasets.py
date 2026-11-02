#!/usr/bin/env python3
"""
Merge multiple Roboflow-exported YOLO helmet datasets into one unified
dataset with a consistent 2-class scheme (helmet / no_helmet), and keep
a manifest logging exactly where every image came from and where it
was saved.

USAGE
-----
1. Download each source dataset yourself from Roboflow (YOLOv8 export format)
   and extract each into its own folder, e.g.:

     datasets/roboflow/helmet_ph/          (Motorcycle Helmet Detection Dataset PH)
     datasets/roboflow/helmet_bike_v2/     (Bike Helmet Detection v2)
     datasets/roboflow/helmet_rider/       (Helmet and no helmet rider detection)

   Each of those folders should contain train/valid/test subfolders with
   images/ and labels/ inside, plus a data.yaml.

2. Edit the SOURCES list below so each entry's "path" points to the
   folder you extracted, and "class_map" maps that dataset's class
   names (from its own data.yaml) to either "helmet" or "no_helmet".
   Any class not listed in class_map (e.g. "Licence Plate") is dropped.

3. Run:
     python merge_helmet_datasets.py

   Output goes to datasets/merged/helmet_traffic_10k/ with:
     images/train, images/val, labels/train, labels/val, data.yaml
   and a manifest.csv logging every single file's original source.
"""

from pathlib import Path
import csv
import random
import shutil
import sys
import yaml

# ---------------------------------------------------------------------------
# EDIT THIS SECTION to match your actual downloaded folders and class names
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "name": "helmet_ph",
        "path": Path("/Users/sachinnair/Desktop/traffic violation system/Motorcycle Helmet Detection Dataset PH.v2i.yolov8"),
        "source_url": "https://universe.roboflow.com/mapua-university/motorcycle-helmet-detection-dataset-ph",
        # map THIS dataset's class names -> unified class name
        # NOTE: double check these against the actual data.yaml "names:" list
        # inside this folder before running -- the script will print a warning
        # if a class name in data.yaml has no entry here (it just gets dropped).
        "class_map": {
            "rider_full_face": "helmet",
            "rider_half_face": "helmet",
            "rider_helmet_invalid": "no_helmet",
            "rider_no_helmet": "no_helmet",
        },
    },
    {
        "name": "helmet_bike_v2",
        "path": Path("/Users/sachinnair/Desktop/traffic violation system/Bike Helmet Detection.v2-more-preprocessing-augmentation.yolov8"),
        "source_url": "https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo",
        "class_map": {
            "With Helmet": "helmet",
            "Without Helmet": "no_helmet",
        },
    },
    {
        "name": "helmet_rider",
        "path": Path("/Users/sachinnair/Desktop/traffic violation system/Helmet and no helmet rider detection.v5-balanced-version-augumented.yolov8"),
        "source_url": "https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection",
        "class_map": {
            "With Helmet": "helmet",
            "Without Helmet": "no_helmet",
            # "Licence Plate" intentionally not mapped -> dropped
        },
    },
]

OUTPUT_ROOT = Path("datasets/merged/helmet_traffic_10k")
VAL_SPLIT = 0.15  # fraction of merged images held out for validation
UNIFIED_CLASSES = ["helmet", "no_helmet"]  # index 0, 1
SEED = 42

# ---------------------------------------------------------------------------
# Implementation (shouldn't need editing below this line)
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_source_class_names(data_yaml_path: Path) -> list[str]:
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        # some exports use {0: "a", 1: "b"} instead of a list
        names = [names[i] for i in sorted(names)]
    return names


def find_split_dirs(dataset_root: Path) -> list[Path]:
    """Roboflow exports usually have train/, valid/, test/ subfolders,
    each containing images/ and labels/. Grab whichever exist."""
    found = []
    for split in ("train", "valid", "test", "val"):
        d = dataset_root / split
        if (d / "images").is_dir():
            found.append(d)
    if not found:
        # fallback: maybe images/labels sit directly at the root
        if (dataset_root / "images").is_dir():
            found.append(dataset_root)
    return found


def remap_label_file(
    src_label: Path, class_names: list[str], class_map: dict[str, str]
) -> list[str] | None:
    """Read a YOLO .txt label file, drop unmapped classes, remap indices
    to the unified scheme. Returns new lines, or None if the file ends up
    empty after filtering (image with only dropped classes)."""
    new_lines = []
    with open(src_label) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_idx = int(parts[0])
            if cls_idx >= len(class_names):
                continue
            src_name = class_names[cls_idx]
            unified_name = class_map.get(src_name)
            if unified_name is None:
                continue  # dropped class (e.g. Licence Plate)
            new_idx = UNIFIED_CLASSES.index(unified_name)
            new_lines.append(" ".join([str(new_idx)] + parts[1:]))
    return new_lines if new_lines else None


def main():
    random.seed(SEED)
    img_out = OUTPUT_ROOT / "images"
    lbl_out = OUTPUT_ROOT / "labels"
    for split in ("train", "val"):
        (img_out / split).mkdir(parents=True, exist_ok=True)
        (lbl_out / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_kept = 0
    total_dropped_empty = 0

    for source in SOURCES:
        root = source["path"]
        if not root.exists():
            print(f"[SKIP] {source['name']}: path not found: {root}")
            print("       -> download + extract this dataset first, or fix the path above")
            continue

        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            print(f"[SKIP] {source['name']}: no data.yaml found at {data_yaml}")
            continue

        class_names = load_source_class_names(data_yaml)
        unmapped = [c for c in class_names if c not in source["class_map"]]
        if unmapped:
            print(f"[WARN] {source['name']}: data.yaml has classes not in class_map "
                  f"(will be dropped): {unmapped}")
            print(f"        full class list in this dataset's data.yaml: {class_names}")
        split_dirs = find_split_dirs(root)
        if not split_dirs:
            print(f"[SKIP] {source['name']}: couldn't find images/ under train/valid/test")
            continue

        source_items = []
        for split_dir in split_dirs:
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                label_path = labels_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    continue
                source_items.append((img_path, label_path))

        print(f"[{source['name']}] found {len(source_items)} labeled images")

        random.shuffle(source_items)
        n_val = int(len(source_items) * VAL_SPLIT)
        val_set = set(source_items[:n_val])

        for img_path, label_path in source_items:
            new_lines = remap_label_file(label_path, class_names, source["class_map"])
            if new_lines is None:
                total_dropped_empty += 1
                continue

            split = "val" if (img_path, label_path) in val_set else "train"
            # prefix filenames with source name to avoid collisions across datasets
            new_stem = f"{source['name']}__{img_path.stem}"
            dest_img = img_out / split / f"{new_stem}{img_path.suffix.lower()}"
            dest_lbl = lbl_out / split / f"{new_stem}.txt"

            shutil.copy2(img_path, dest_img)
            with open(dest_lbl, "w") as f:
                f.write("\n".join(new_lines) + "\n")

            manifest_rows.append({
                "unified_filename": dest_img.name,
                "split": split,
                "source_dataset": source["name"],
                "source_url": source["source_url"],
                "original_image_path": str(img_path),
                "saved_to": str(dest_img),
            })
            total_kept += 1

    # write data.yaml for the merged dataset
    merged_yaml = {
        "train": str((img_out / "train").resolve()),
        "val": str((img_out / "val").resolve()),
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    with open(OUTPUT_ROOT / "data.yaml", "w") as f:
        yaml.safe_dump(merged_yaml, f, sort_keys=False)

    # write manifest so you can always trace an image back to its source
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "unified_filename", "split", "source_dataset",
            "source_url", "original_image_path", "saved_to",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Done. {total_kept} images merged ({total_dropped_empty} skipped: no labels left after class filtering)")
    print(f"Saved to: {OUTPUT_ROOT.resolve()}")
    print(f"  images/train, images/val, labels/train, labels/val, data.yaml")
    print(f"Manifest (source tracking): {manifest_path.resolve()}")

    if total_kept < 10000:
        print()
        print(f"NOTE: only reached {total_kept} images, below the 10,000 target.")
        print("Check the [SKIP] lines above -- likely one of the source paths wasn't found.")


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        print("This script uses `list[str] | None` type hints, needs Python 3.10+")
        sys.exit(1)
    main()
