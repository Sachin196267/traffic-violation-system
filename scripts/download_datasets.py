#!/usr/bin/env python3
"""Download only the four datasets the Dataset Plan PDF says to download first.

Runs unattended: resumes partial files, skips finished datasets, retries
network failures, and keeps going until everything that can be fetched is
done. There is no time cap.

Exact URLs and identifiers come from:
docs/references/India_Centric_Traffic_AI_Dataset_Plan.pdf (sections 7 and 9).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Must be set before huggingface_hub is imported.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "20"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
LOG_PATH = DATASETS_DIR / "download.log"
STATE_PATH = DATASETS_DIR / ".download_state.json"
LOCK_PATH = DATASETS_DIR / ".download.lock"
COMPLETE_MARK = ".download_complete"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# PDF "Download only these four first"
APPROVED = [
    {
        "id": "uvh26",
        "name": "UVH-26 (IISc AIM)",
        "platform": "huggingface",
        "url": "https://huggingface.co/datasets/iisc-aim/UVH-26",
        "repo_id": "iisc-aim/UVH-26",
        "local_dir": DATASETS_DIR / "huggingface" / "uvh26",
        "notes": "Public. Images + COCO JSON. Hub listing is ~84 GB / 26,652 files.",
        "expected_files": 26652,
        "expected_gb": 83.81,
        "optional_payload": False,
    },
    {
        "id": "helmet_india",
        "name": "Indian Helmet Detection Dataset (YOLO format)",
        "platform": "kaggle",
        "url": "https://www.kaggle.com/datasets/aryanvaid13/indian-helmet-detection-dataset",
        "kaggle_id": "aneesarom/rider-with-helmet-without-helmet-number-plate",
        "local_dir": DATASETS_DIR / "huggingface" / "helmet_india",
        "notes": "Kaggle API credentials required. 942 images, YOLOv8 format, 5 classes: numberPlate, faceWithNoHelmet, faceWithGoodHelmet, faceWithBadHelmet, rider. India-specific.",
        "expected_files": 20000,
        "expected_gb": 0.062,
        "optional_payload": False,
    },
    {
        "id": "plates_india",
        "name": "Indian Vehicle License Plate Dataset",
        "platform": "kaggle",
        "url": "https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset",
        "kaggle_id": "saisirishan/indian-vehicle-dataset",
        "local_dir": DATASETS_DIR / "kaggle" / "plates_india",
        "notes": "Kaggle API credentials required.",
        "expected_files": None,
        "expected_gb": None,
        "optional_payload": False,
    },
    {
        "id": "idd",
        "name": "IDD-D YOLOv11",
        "platform": "kaggle",
        "url": "https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11",
        "kaggle_id": "redzapdos123/indian-driving-dataset-detections-yolov11",
        "local_dir": DATASETS_DIR / "kaggle" / "idd",
        "notes": "Kaggle API credentials required. Zip is ~20.5 GB (still downloaded in full; "
                 "only max_files image/label pairs are kept on disk after extraction).",
        "expected_files": 41962,
        "expected_gb": 20.5,
        "optional_payload": False,
        # Cap what's kept on disk after extraction. The full ~20.5 GB zip is still
        # downloaded (Kaggle doesn't support partial zip fetch), but only this many
        # image+label pairs are extracted, sampled proportionally across whatever
        # train/val/test split folders exist in the archive. Set to None to keep all.
        "max_files": 10000,
    },
]


def _default_workers() -> int:
    cpus = os.cpu_count() or 8
    return max(48, min(96, cpus * 8))


DEFAULT_WORKERS = _default_workers()
PROGRESS_INTERVAL_SEC = 5.0
HF_RETRIES = 5
_print_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}" if not msg.startswith("[") else msg
    with _print_lock:
        print(msg, flush=True)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line if line.endswith("\n") else line + "\n")
        except OSError:
            pass


def _iter_payload_files(path: Path):
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file() and ".cache" not in p.parts and p.name not in {COMPLETE_MARK}:
            yield p


def _scan(path: Path) -> tuple[int, float]:
    files = list(_iter_payload_files(path))
    size = sum(p.stat().st_size for p in files)
    return len(files), size


def _summarize(path: Path) -> str:
    if not path.exists():
        return "missing"
    n, size = _scan(path)
    if size >= 1024**3:
        return f"{n} files, {size / (1024 ** 3):.2f} GB"
    return f"{n} files, {size / (1024 ** 2):.1f} MB"


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _kaggle_zip_path(item: dict) -> Path:
    slug = item["kaggle_id"].split("/")[-1]
    return item["local_dir"] / f"{slug}.zip"


def _kaggle_ready() -> tuple[bool, str]:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    default_json = Path.home() / ".kaggle" / "kaggle.json"
    local_json = ROOT / "kaggle.json"
    if username and key:
        return True, "KAGGLE_USERNAME / KAGGLE_KEY"
    if default_json.is_file():
        return True, str(default_json)
    if local_json.is_file():
        os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT)
        return True, str(local_json)
    return False, (
        "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
        "in .env, or place kaggle.json at ~/.kaggle/kaggle.json."
    )


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _ensure_hf_xet() -> None:
    try:
        import hf_xet  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "hf_xet is required for fast Hugging Face downloads. "
            "Run: pip install -r requirements.txt"
        ) from exc


def _remote_size(entry) -> int:
    if getattr(entry, "lfs", None) is not None and getattr(entry.lfs, "size", None):
        return int(entry.lfs.size)
    return int(entry.size or 0)


def _mark_complete(item: dict, status: str) -> None:
    dest: Path = item["local_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / COMPLETE_MARK
    marker.write_text(f"{status}\n{_now()}\n{_summarize(dest)}\n", encoding="utf-8")


def _load_state() -> dict:
    if not STATE_PATH.is_file():
        return {"datasets": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"datasets": {}}


def _write_state(results: list[tuple[str, str, str]]) -> None:
    state = _load_state()
    state["updated"] = _now()
    datasets = state.setdefault("datasets", {})
    by_id = {item["id"]: item for item in APPROVED}
    for dataset_id, status, detail in results:
        item = by_id.get(dataset_id)
        n, size = _scan(item["local_dir"]) if item else (0, 0)
        datasets[dataset_id] = {
            "status": status,
            "detail": detail,
            "files": n,
            "bytes": size,
            "expected_files": item.get("expected_files") if item else None,
        }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(force: bool) -> None:
    if LOCK_PATH.is_file():
        try:
            old = int(LOCK_PATH.read_text(encoding="utf-8").strip().split()[0])
        except (OSError, ValueError):
            old = None
        if old and _pid_alive(old) and old != os.getpid():
            if not force:
                raise SystemExit(
                    f"Downloader already running as pid {old}. "
                    "Leave it running, or pass --force to replace it."
                )
            _log(f"Replacing previous downloader pid {old}")
            try:
                os.kill(old, 15)
                time.sleep(2)
                if _pid_alive(old):
                    os.kill(old, 9)
            except OSError:
                pass
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")

    def _release() -> None:
        try:
            if LOCK_PATH.is_file() and LOCK_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_PATH.unlink()
        except OSError:
            pass

    atexit.register(_release)


class ProgressMonitor:
    def __init__(self, item: dict, interval: float = PROGRESS_INTERVAL_SEC) -> None:
        self.item = item
        self.path = item["local_dir"]
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_n, self._start_bytes = _scan(self.path)
        self._t0 = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"progress-{self.item['id']}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        dataset_id = self.item["id"]
        expected_files = self.item.get("expected_files")
        expected_bytes = None
        if self.item.get("expected_gb"):
            expected_bytes = float(self.item["expected_gb"]) * (1024**3)

        while not self._stop.wait(self.interval):
            n, size = _scan(self.path)
            elapsed = max(time.monotonic() - self._t0, 1e-6)
            delta_bytes = max(size - self._start_bytes, 0)
            rate = delta_bytes / elapsed
            rate_mb = rate / (1024 * 1024)

            file_part = f"{n:,} files"
            if expected_files:
                file_part = f"{n:,} / ~{expected_files:,} files"

            if size >= 1024**3:
                size_part = f"{size / (1024 ** 3):.2f} GB"
            else:
                size_part = f"{size / (1024 ** 2):.1f} MB"

            eta = "unknown"
            if expected_bytes and rate > 0:
                remaining = max(expected_bytes - size, 0)
                eta = _fmt_eta(remaining / rate)
            elif expected_files and n > self._start_n:
                files_per_sec = (n - self._start_n) / elapsed
                if files_per_sec > 0:
                    eta = _fmt_eta((expected_files - n) / files_per_sec)

            _log(
                f"    [{dataset_id}] {file_part} | {size_part} | "
                f"{rate_mb:.1f} MB/s | ETA {eta}"
            )


def _list_hf_files(repo_id: str):
    from huggingface_hub import HfApi, RepoFile

    api = HfApi(token=_hf_token())
    entries = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    files = []
    for entry in entries:
        if not isinstance(entry, RepoFile):
            continue
        if entry.path.endswith("/") or entry.path.startswith(".git"):
            continue
        files.append(entry)
    return files


def _download_one_hf(repo_id: str, filename: str, local_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    last_exc: Exception | None = None
    for attempt in range(1, HF_RETRIES + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=_hf_token(),
                force_download=False,
            )
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(min(16, 0.75 * (2 ** (attempt - 1))))
    raise RuntimeError(f"{filename}: {last_exc}") from last_exc


def download_huggingface(item: dict, workers: int) -> None:
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install -r requirements.txt") from exc

    _ensure_hf_xet()
    dest: Path = item["local_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    _log("    listing remote files (skip anything already complete)...")
    remote_files = _list_hf_files(item["repo_id"])
    item["expected_files"] = len(remote_files)
    total_bytes = sum(_remote_size(f) for f in remote_files)
    if total_bytes:
        item["expected_gb"] = total_bytes / (1024**3)

    pending = []
    skipped = 0
    skipped_bytes = 0
    for entry in remote_files:
        local_path = dest / entry.path
        size = _remote_size(entry)
        if local_path.is_file() and local_path.stat().st_size == size:
            skipped += 1
            skipped_bytes += size
            continue
        pending.append(entry)

    _log(
        f"    remote={len(remote_files):,} files ({total_bytes / (1024 ** 3):.2f} GB) | "
        f"already local={skipped:,} ({skipped_bytes / (1024 ** 3):.2f} GB) | "
        f"to download={len(pending):,}"
    )
    if not pending:
        _log("    nothing left to download")
        return

    workers = max(8, min(workers, len(pending)))
    _log(f"    downloading with {workers} parallel workers")
    failures: list[str] = []

    def _job(entry) -> str | None:
        try:
            _download_one_hf(item["repo_id"], entry.path, dest)
            return None
        except Exception as exc:
            return f"{entry.path}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, entry) for entry in pending]
        for fut in as_completed(futs):
            err = fut.result()
            if err:
                failures.append(err)

    if failures:
        sample = "; ".join(failures[:5])
        raise RuntimeError(f"{len(failures)} file(s) failed this round. First errors: {sample}")


def _zip_is_valid(zip_path: Path) -> bool:
    """Return True only if the zip's central directory is intact and every
    member passes CRC verification. A partial/interrupted download will
    fail this even though the file exists on disk."""
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def _pick_yolo_subset(names: list[str], max_files: int, seed: int = 42) -> set[str] | None:
    """Choose a capped subset of a YOLO-style archive to extract: up to
    max_files image+label pairs, sampled proportionally across whichever
    train/val/test split folders exist, plus every non-image file (data.yaml,
    classes.txt, README, licenses, ...) so the dataset config stays intact.

    Returns None if the archive doesn't look like an images/labels layout,
    so the caller can fall back to extracting everything rather than
    silently producing a broken subset.
    """
    import random

    def is_image(n: str) -> bool:
        return not n.endswith("/") and n.lower().endswith(_IMAGE_EXTS)

    images = [n for n in names if is_image(n)]
    if not images:
        return None

    image_set = set(images)

    def is_label_file(n: str) -> bool:
        return "/labels/" in n.lower() and n.lower().endswith(".txt")

    # "Config" files are kept unconditionally (data.yaml, classes.txt, README,
    # LICENSE, ...). Per-image label files are NOT config files — they're only
    # kept when they match a selected image, handled below.
    config_files = [
        n for n in names
        if n not in image_set and not is_label_file(n) and not n.endswith("/")
    ]

    def split_of(n: str) -> str:
        for part in n.lower().split("/"):
            if part in _SPLIT_NAMES:
                return part
        return "_root"

    by_split: dict[str, list[str]] = {}
    for n in images:
        by_split.setdefault(split_of(n), []).append(n)

    total = len(images)
    rng = random.Random(seed)
    selected_images: list[str] = []
    for split_names in by_split.values():
        share = min(len(split_names), max(1, round(max_files * (len(split_names) / total))))
        pool = split_names[:]
        rng.shuffle(pool)
        selected_images.extend(pool[:share])
    selected_images = selected_images[:max_files]

    name_set = set(names)

    def label_for(image_name: str) -> str | None:
        if "/images/" not in image_name:
            return None
        candidate = image_name.replace("/images/", "/labels/")
        stem, _ext = candidate.rsplit(".", 1)
        candidate = stem + ".txt"
        return candidate if candidate in name_set else None

    selected = set(selected_images)
    for img in selected_images:
        lbl = label_for(img)
        if lbl:
            selected.add(lbl)

    selected.update(config_files)  # data.yaml, classes.txt, README, LICENSE, etc.
    return selected


def _extract_zip_fast(zip_path: Path, dest: Path, max_files: int | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not _zip_is_valid(zip_path):
        _log(f"    zip failed integrity check, deleting and forcing re-download: {zip_path.name}")
        try:
            zip_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"{zip_path.name} was corrupt/incomplete and has been deleted. "
            "Re-run the downloader to fetch it again."
        )

    if max_files:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            subset = _pick_yolo_subset(names, max_files)
            if subset is None:
                _log(
                    f"    max_files={max_files} requested but archive layout wasn't "
                    "recognized (no image files found under an images/ folder); "
                    "extracting everything instead."
                )
            else:
                n_img = sum(1 for n in subset if n.lower().endswith(_IMAGE_EXTS))
                _log(f"    extracting capped subset: {n_img:,} images (+ matching labels/config) of {len(names):,} archive entries")
                for name in subset:
                    zf.extract(name, dest)
                return

    ditto = shutil.which("ditto")
    if ditto:
        _log(f"    extracting with ditto: {zip_path.name}")
        try:
            subprocess.run([ditto, "-xk", str(zip_path), str(dest)], check=True)
            return
        except subprocess.CalledProcessError:
            _log("    ditto failed on a zip that passed integrity check; falling back to zipfile module")
    _log(f"    extracting with zipfile: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def download_kaggle(item: dict) -> None:
    ready, detail = _kaggle_ready()
    if not ready:
        raise PermissionError(detail)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("kaggle is not installed. Run: pip install -r requirements.txt") from exc

    api = KaggleApi()
    api.authenticate()
    dest = item["local_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = _kaggle_zip_path(item)
    payload = [p for p in _iter_payload_files(dest) if p.suffix.lower() != ".zip"]
    if payload and not zip_path.exists():
        _log("    already extracted; skipping Kaggle fetch")
        return

    # If a zip is already on disk but is corrupt/partial, force a clean
    # re-download instead of silently skipping it (force=False would keep
    # re-extracting the same broken file forever).
    force_download = zip_path.exists() and not _zip_is_valid(zip_path)
    if force_download:
        _log("    existing zip failed integrity check; forcing re-download")

    _log("    downloading zip (Kaggle API resumes an existing zip)")
    api.dataset_download_files(
        item["kaggle_id"],
        path=str(dest),
        unzip=False,
        quiet=False,
        force=force_download,
    )
    if not zip_path.exists():
        zips = list(dest.glob("*.zip"))
        if not zips:
            raise RuntimeError(f"Kaggle zip not found in {dest}")
        zip_path = zips[0]

    _extract_zip_fast(zip_path, dest, max_files=item.get("max_files"))
    try:
        zip_path.unlink()
    except OSError as exc:
        _log(f"    could not delete zip after extract: {exc}")


def _helmet_has_payload(dest: Path) -> bool:
    return any(
        p.is_file()
        and ".cache" not in p.parts
        and p.name not in {".gitattributes", "README.md", "data.yaml", COMPLETE_MARK}
        for p in dest.rglob("*")
    )


def _looks_complete(item: dict) -> bool:
    dest = item["local_dir"]
    marker = dest / COMPLETE_MARK
    if marker.is_file():
        return True
    if not dest.exists():
        return False
    if item["id"] == "helmet_india" and dest.exists() and not _helmet_has_payload(dest):
        card = any((dest / name).exists() for name in ("README.md", "data.yaml"))
        return card
    if item["platform"] == "kaggle":
        zip_path = _kaggle_zip_path(item)
        payload = [p for p in _iter_payload_files(dest) if p.suffix.lower() != ".zip"]
        if zip_path.exists():
            return False
        # If capped, "complete" means we hit the cap (roughly), not the full
        # archive count — otherwise the downloader would never stop retrying.
        expected = item.get("max_files") or item.get("expected_files")
        if expected:
            return len(payload) >= int(expected) * 0.90
        return bool(payload)
    expected = item.get("expected_files")
    n, _ = _scan(dest)
    if expected:
        return n >= int(expected) * 0.98
    return False


def list_datasets() -> None:
    print("Approved datasets (PDF section 9 only):\n")
    for item in APPROVED:
        ident = item.get("repo_id") or item.get("kaggle_id")
        print(f"  [{item['id']}] {item['name']}")
        print(f"      platform : {item['platform']}")
        print(f"      url      : {item['url']}")
        print(f"      id       : {ident}")
        print(f"      dest     : {item['local_dir'].relative_to(ROOT)}")
        print(f"      notes    : {item['notes']}")
        print(f"      local    : {_summarize(item['local_dir'])}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the four datasets listed as 'download first' in the Dataset Plan PDF."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="Download a subset: uvh26 helmet_india plates_india idd",
    )
    parser.add_argument("--list", action="store_true", help="Print approved datasets and exit.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets that already look complete (default on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-check every dataset even if a completion marker exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel Hugging Face file downloads (default {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Download datasets one after another instead of in parallel.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass and exit (default is keep retrying until done).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace another downloader process if one is already running.",
    )
    return parser.parse_args()


def _process_item(item: dict, workers: int) -> tuple[str, str, str]:
    dest = item["local_dir"]
    _log(f"==> {item['name']}")
    _log(f"    {item['url']}")
    _log(f"    -> {dest}")

    monitor = ProgressMonitor(item)
    monitor.start()
    try:
        if item["platform"] == "huggingface":
            download_huggingface(item, workers)
        elif item["platform"] == "kaggle":
            download_kaggle(item)
        else:
            raise RuntimeError(f"Unsupported platform: {item['platform']}")
    except PermissionError as exc:
        monitor.stop()
        _log(f"    FAIL_AUTH: {exc}\n")
        return item["id"], "FAIL_AUTH", str(exc)
    except Exception as exc:
        monitor.stop()
        _log(f"    FAIL: {exc}\n")
        return item["id"], "FAIL", str(exc)

    monitor.stop()
    summary = _summarize(dest)
    if item["id"] == "helmet_india" and not _helmet_has_payload(dest):
        msg = (
            "Hub snapshot contains no image/annotation payload "
            "(only card files). Marked complete for automation."
        )
        _log(f"    WARN: {msg}")
        _log(f"    local: {summary}\n")
        _mark_complete(item, "REQUIRES REVIEW")
        return item["id"], "REQUIRES REVIEW", msg

    _mark_complete(item, "OK")
    _log(f"    OK ({summary})\n")
    return item["id"], "OK", summary


def _run_pass(wanted: list[dict], workers: int, serial: bool, skip_existing: bool) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    to_fetch: list[dict] = []
    for item in wanted:
        if skip_existing and _looks_complete(item):
            summary = _summarize(item["local_dir"])
            _log(f"==> {item['name']}")
            _log(f"    SKIP (already complete: {summary})\n")
            results.append((item["id"], "SKIPPED", summary))
            continue
        to_fetch.append(item)

    if not to_fetch:
        return results

    if serial or len(to_fetch) <= 1:
        for item in to_fetch:
            results.append(_process_item(item, workers))
        return results

    _log("Fetching incomplete datasets in parallel (Hugging Face + Kaggle).\n")
    with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
        futs = {pool.submit(_process_item, item, workers): item for item in to_fetch}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def _all_settled(results: list[tuple[str, str, str]]) -> bool:
    retryable = {"FAIL"}
    return not any(status in retryable for _, status, _ in results)


def main() -> int:
    args = parse_args()
    if args.list:
        list_datasets()
        return 0

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    _acquire_lock(force=args.force)

    workers = max(8, args.workers)
    selected_ids = args.only or [item["id"] for item in APPROVED]
    known = {item["id"] for item in APPROVED}
    unknown = [d for d in selected_ids if d not in known]
    if unknown:
        print(f"Unknown dataset id(s): {', '.join(unknown)}", file=sys.stderr)
        print("Allowed ids: " + ", ".join(item["id"] for item in APPROVED), file=sys.stderr)
        return 2

    wanted = [item for item in APPROVED if item["id"] in selected_ids]
    _log("Downloading only PDF-approved datasets. No other sources will be used.")
    _log("Unattended mode: resume + retry until finished. No time cap.")
    _log(f"Hugging Face workers: {workers}")
    _log(f"Log file: {LOG_PATH}")
    _log("")

    round_no = 0
    auth_failed = False
    while True:
        round_no += 1
        _log(f"---- pass {round_no} ----")
        results = _run_pass(wanted, workers, args.serial, args.skip_existing)
        order = {item["id"]: i for i, item in enumerate(APPROVED)}
        results.sort(key=lambda row: order.get(row[0], 99))
        _write_state(results)

        _log("Summary")
        _log("-" * 72)
        for dataset_id, status, detail in results:
            _log(f"  {dataset_id:16} {status:16} {detail}")
        _log("")

        if any(status == "FAIL_AUTH" for _, status, _ in results):
            auth_failed = True

        if _all_settled(results):
            _log("All fetchable datasets are complete. Helmet metadata-only is recorded as REQUIRES REVIEW.")
            return 1 if auth_failed else 0

        if args.once:
            _log("Single pass requested; remaining files will continue next run.")
            return 1

        delay = min(180, 15 * (2 ** min(round_no - 1, 4)))
        _log(f"Retrying incomplete downloads in {delay}s (already-saved files are kept).")
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(main())