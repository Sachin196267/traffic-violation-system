#!/usr/bin/env python3
"""
Vehicle speed ranges for a video, in one command (track -> calibrate -> speed).

  python speed_estimation/run.py <video> [--detector idd|coco] [--calib calibrations/<file>.json]
                                 [--start S] [--max-sec N] [--fps 10]

Without --calib the focal length comes from GeoCalib only; that is reliable enough for side-on views, but for
cameras looking along the road the ranges get wide and many vehicles show "speed ?". A one-time calibration
(one known length along the road, e.g. standard lane dashes) gives usable ranges there.
Output: speed_estimation/runs/<video>/ -> <video>_speed.mp4, speeds.json
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--detector", default="idd", choices=["idd", "coco"], help="idd for Indian roads, coco elsewhere")
ap.add_argument("--calib")
ap.add_argument("--start", default="0")
ap.add_argument("--max-sec", default="0")
ap.add_argument("--fps", default="10")
a = ap.parse_args()
py = sys.executable
steps = [
    [py, str(HERE / "track.py"), a.video, "--detector", a.detector, "--start", a.start, "--max-sec", a.max_sec, "--fps", a.fps],
    [py, str(HERE / "calibrate.py"), a.video],
    [py, str(HERE / "speed.py"), a.video] + (["--calib", a.calib] if a.calib else []),
]
for cmd in steps:
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    print(f"[{Path(cmd[1]).stem}] " + (last[-1] if last else r.stdout.strip()[-300:]))
    if r.returncode:
        sys.exit(r.stderr[-2000:])
print("->", HERE / "runs" / Path(a.video).stem)
