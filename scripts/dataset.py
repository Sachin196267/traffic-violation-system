#!/usr/bin/env python3
"""Run the dataset downloader: python scripts/dataset.py"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("download_datasets.py")), run_name="__main__")
