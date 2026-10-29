#!/usr/bin/env python3
"""List the crop files shown in sheet cells (same pick as make_sheets.py): cell_crops.py <sheet> <cell> ..."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
sheet = sys.argv[1]
video = sheet.rsplit("_", 1)[0]
keys = json.loads((R / "sheets" / f"{video}_index.json").read_text())[sheet]
units = json.loads((R / "data" / video / "units.json").read_text())


def pick3(rows):
    rows = sorted(rows, key=lambda r: r["t"])
    if len(rows) <= 3:
        return rows
    big = sorted(rows, key=lambda r: -r["crop_h"])[:max(3, len(rows) // 2)]
    big.sort(key=lambda r: r["t"])
    return [big[i] for i in np.linspace(0, len(big) - 1, 3).round().astype(int)]


for c in map(int, sys.argv[2:]):
    k = keys[c - 1]
    rows = [r for r in units if f"{r['video']}:{r['track']}" == k]
    print(c, k, [r["crop"] for r in pick3(rows)])
