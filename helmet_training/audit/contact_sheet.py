#!/usr/bin/env python3
"""Draw YOLO labels on sample images and tile them into a contact sheet (for manual label review)."""
import sys, random, cv2, numpy as np
from pathlib import Path
COL = [(0,200,0),(0,0,255),(255,128,0),(0,200,255),(255,0,255),(128,128,128)]
def sheet(img_dir, out, names, n=24, seed=0, only_cls=None, tile=320, cols=6):
    img_dir = Path(img_dir); lbl_dir = Path(str(img_dir).replace("images", "labels"))
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg",".png",".jpeg"})
    random.Random(seed).shuffle(files); tiles = []
    for p in files:
        lp = lbl_dir / (p.stem + ".txt")
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        if only_cls is not None and not any(l[0] == str(only_cls) for l in lines): continue
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        for l in lines:
            c = int(l[0]); x, y, bw, bh = map(float, l[1:5])
            x1, y1, x2, y2 = int((x-bw/2)*w), int((y-bh/2)*h), int((x+bw/2)*w), int((y+bh/2)*h)
            thick = 2 if only_cls is None or c == only_cls else 1
            cv2.rectangle(im, (x1,y1), (x2,y2), COL[c % 6], thick)
            cv2.putText(im, names[c] if c < len(names) else str(c), (x1, max(10,y1-3)), 0, 0.4, COL[c % 6], 1)
        im = cv2.resize(im, (tile, tile)); cv2.putText(im, p.name[:40], (2, tile-4), 0, 0.33, (255,255,255), 1)
        tiles.append(im)
        if len(tiles) >= n: break
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows)); print(out, len(tiles))
if __name__ == "__main__":
    a = sys.argv; sheet(a[1], a[2], a[3].split(","), n=int(a[4]), only_cls=None if a[5] == "-" else int(a[5]))
