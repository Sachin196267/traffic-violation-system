"""
Download the CC0 photo-scanned textures and sky HDRIs the simulator uses (Poly Haven, https://polyhaven.com, CC0)
into Assets/Resources/PH/ and prepare them for Unity:
  <name>_albedo.png  colour + smoothness (1 - roughness) in alpha, as URP/Standard Lit expects
  <name>_normal.png  OpenGL normal map
  sky_<tod>.hdr      equirectangular sky; sky.json records where its sun is so the scene's sun matches it
Run once:  python3 signal_sim/Tools/fetch_assets.py
"""
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "Assets/Resources/PH"
TEXTURES = {   # simulator name: (Poly Haven id, resolution)
    "asphalt": ("asphalt_02", "2k"),
    "pavers": ("concrete_pavers_02", "1k"),
    "plaster": ("plastered_wall_02", "1k"),
    "concrete": ("concrete_floor_02", "1k"),
    "ground": ("dry_ground_01", "1k"),
}
SKIES = {"day": "kloofendal_48d_partly_cloudy_puresky", "evening": "qwantani_dusk_2_puresky", "night": "qwantani_night_puresky",
         "rain": "kloofendal_overcast_puresky"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "signal-sim-asset-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def img(data, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def texture(name, pid, res):
    files = json.loads(get(f"https://api.polyhaven.com/files/{pid}"))
    url = lambda key: files[key][res]["jpg"]["url"]
    albedo = img(get(url("Diffuse")), cv2.IMREAD_COLOR)
    rough = img(get(url("Rough")), cv2.IMREAD_GRAYSCALE)
    normal = img(get(url("nor_gl")), cv2.IMREAD_COLOR)
    rough = cv2.resize(rough, (albedo.shape[1], albedo.shape[0]))
    rgba = np.dstack([albedo, 255 - rough])
    cv2.imwrite(str(OUT / f"{name}_albedo.png"), rgba)
    cv2.imwrite(str(OUT / f"{name}_normal.png"), normal)
    print(f"{name}: {pid} {res} {albedo.shape[1]}x{albedo.shape[0]}", flush=True)


def sky(tod, pid):
    files = json.loads(get(f"https://api.polyhaven.com/files/{pid}"))
    data = get(files["hdri"]["2k"]["hdr"]["url"])
    (OUT / f"sky_{tod}.hdr").write_bytes(data)
    hdr = img(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    lum = cv2.GaussianBlur(hdr.mean(axis=2), (0, 0), 3)
    y, x = np.unravel_index(np.argmax(lum), lum.shape)
    h, w = lum.shape
    sun = {"azimuth": round(x / w * 360.0, 2), "elevation": round(90.0 - y / h * 180.0, 2), "peak": float(lum[y, x])}
    print(f"sky {tod}: {pid} sun {sun}", flush=True)
    return sun


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for name, (pid, res) in TEXTURES.items():
        if not only or name in only:
            texture(name, pid, res)
    suns = {tod: sky(tod, pid) for tod, pid in SKIES.items() if not only or tod in only}
    meta = OUT / "sky.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    old.update({tod: {"id": SKIES[tod], **s} for tod, s in suns.items()})
    meta.write_text(json.dumps(old, indent=1))
    (OUT / "LICENSE.txt").write_text("Textures and HDRIs from Poly Haven (https://polyhaven.com), licensed CC0 1.0. "
                                      "Ids: " + ", ".join(p for p, _ in TEXTURES.values()) + ", " + ", ".join(SKIES.values()) + "\n")
