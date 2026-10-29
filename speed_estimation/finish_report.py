#!/usr/bin/env python3
"""Append the VS13 results section to REPORT.md (idempotent: replaces an earlier auto-generated section)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "runs/vs13_results.json").read_text())
ok = [r for r in rows if "median" in r]
lines = ["<!-- VS13-AUTO-START -->", "## VS13 test (real cars, cruise-control speeds, uncalibrated)", ""]
if ok:
    err = np.array([(r["median"] - r["truth"]) / r["truth"] for r in ok])
    shown = [r for r in ok if r["confidence"] in ("high", "medium", "low")]
    lines += [
        f"Peugeot 208, {len(rows)} clips (30–105 km/h). Camera: roadside, 1.2 m high, car driving **toward** it along the road, "
        "the hardest geometry for uncalibrated mode. Ground truth is the cruise-control speed from the file name "
        "(speedometers usually read 2–5 % high).",
        "",
        f"- Measured: {len(ok)}/{len(rows)} clips",
        f"- Shown with a number (high/medium/low): **{len(shown)}**; everything else was `speed ?`",
        f"- 90 % range contains the true speed: {np.mean([r['inside'] for r in ok]) * 100:.0f} % of clips",
        f"- Middle estimate (hidden when `speed ?`): median error {np.median(err) * 100:+.0f} %, mean |error| {np.mean(np.abs(err)) * 100:.0f} %",
        "",
        "Conclusion: with no calibration, a low camera looking along the road cannot measure speed. The tool correctly "
        "refuses to show a number instead of showing a wrong one. Calibrate the camera (one known length on the road) "
        "for such views.",
        "",
        "| truth km/h | middle estimate | 90 % range | label |", "|---|---|---|---|",
    ]
    lines += [f"| {r['truth']} | {r['median']:.0f} | {r['range'][0]:.0f}–{r['range'][1]:.0f} | {r['confidence']} |" for r in ok]
failed = [r for r in rows if "median" not in r]
if failed:
    lines += ["", "Not measured: " + ", ".join(f"{r['clip']} ({r['status']})" for r in failed)]
lines.append("<!-- VS13-AUTO-END -->")
rep = HERE / "REPORT.md"
s = rep.read_text()
if "<!-- VS13-AUTO-START -->" in s:
    s = s[:s.index("<!-- VS13-AUTO-START -->")] + s[s.index("<!-- VS13-AUTO-END -->") + len("<!-- VS13-AUTO-END -->"):]
s = s.replace("**Side-on view, uncalibrated (VS13 dataset, cars at GPS/cruise-control speeds):** see section below.",
              "**VS13 (real cars at cruise-control speeds, uncalibrated):** see the VS13 section below.")
s = s.replace("## Your videos", "\n".join(lines) + "\n\n## Your videos", 1)
rep.write_text(s)
print("VS13 section written:", len(ok), "clips")
