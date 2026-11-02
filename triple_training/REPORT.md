# Triple-riding detector — report (2026-09-27)

Standalone experiment, **not** wired into the website.

```bash
python triple_training/run_video.py <video.mp4> [--max-sec 120] [--out dir]
# -> <video>_triple.mp4 (red TRIPLE RIDING / orange not sure) + <video>_triple.json
```

## How it works
1. IDD detector + ByteTrack follow every motorcycle; rider/person boxes sitting on it are attached.
2. Each sampled frame: crop of bike + riders → YOLO11s-cls classifier: `one / two / three_plus / not_bike`.
3. Per motorcycle, all frames are combined (size-weighted vote) into one decision:
   **TRIPLE** only if p(three_plus) ≥ 0.997, **not triple** if ≤ 0.681, otherwise **not sure**
   (`runs/triple_rider/operating_point.json`).

## Data
| Source | Triple examples | Use |
|---|---|---|
| 33 Indian traffic videos (labelled by eye, track + crop level) | 13 | train + test |
| IDD dashcam stills (on disk) | 30 | train + test |
| Roboflow staged set (on disk) | 62 | training only |
| Kaggle `devgurucodes/...triple-riding...` originals (downloaded with your OK, **no license stated → project/study use only**) | 415 | training only |
| Negatives: 1 rider / 2 riders / not a two-wheeler | 616 / 371 / 174 units | train + test |

Tests use **leave-cameras-out** 5-fold cross-validation: every score below comes from a model that
never saw that camera. Downloaded / staged sets are never scored.

## Results (held-out cameras, 714 two-wheelers, 43 real triples)

| Setting | Triples caught | Wrong TRIPLE calls | Not sure |
|---|---|---|---|
| **Chosen: never wrong** (t_hi 0.997) | **7 / 43 (16 %)** | **0** | 3.3 % (≈1 in 30) |
| t_hi 0.99 | 9 / 43 | 1 | 3.3 % |
| t_hi 0.95 | 11 / 43 | 5 | 3.3 % |
| t_hi 0.90 | 16 / 43 | 8 | 3.3 % |

End-to-end on videos (each with the fold model that never saw it): Indore, Pune, Jaipur, Mumbai —
~700 motorcycles, **0 wrong calls, 1 not-sure**, but the ~5 real triples in those clips were **all missed**.

## Honest verdict
- **Meets:** "say triple only if sure" (0 wrong calls on unseen cameras) and "not sure rarely" (≤1 in 30).
- **Does not meet:** usefulness — it misses ~5 of every 6 triple riders on a new camera.
- **Why:** triple riding is rare in real footage (~1 bike in 150); only 43 real examples on 8 cameras.
  The main error is **overlap**: a rider on the bike behind looks like a passenger in a 2-D crop
  (the one wrong call at t_hi 0.99 was exactly that). Children sandwiched between adults are the
  most common missed case.

## What would improve it most
1. **More real triples from real CCTV/dashcam angles** — a few hundred, labelled as boxes. This is the
   single biggest lever; the loop here (`rank_unlabelled.py` → `label.py`) finds candidates fast.
2. **Count people, not crops:** train a head/person *detector* on the rider crops and count heads
   whose body sits on the bike seat — separates neighbours better than whole-crop classification.
3. **Longer training** (30 epochs on GPU; Kaggle job `tvs-triple-cls-yolo11s` is ready but was stuck
   in Kaggle's queue — weekly GPU quota).

## Files
`extract_units.py` (tracks → crops), `make_sheets.py` / `crop_sheets.py` / `label.py` / `zoom*.py`
(labelling), `import_idd.py` / `import_roboflow.py` / `import_devguru.py`, `build_dataset.py`,
`train_cv.py`, `eval_tracks.py`, `rank_unlabelled.py`, `run_video.py`, `focus.py` (tried, no gain).
Weights: `runs/triple_rider/final/weights/last.pt` (all data), `runs/triple_rider/fold*/` (CV).
Test videos: `triple_training/video_out/`.

---
# v3 update (2026-09-27)

Data: 114 real triple tracks (39 police-camera clips + IDD + own videos), overlays erased; Tirupati clips split by camera site.
Model: YOLO11s-cls 256 px, 15 epochs, 5-fold leave-cameras-out. Decision (`combine.py`): classifier + heads-on-seat
count -> score; TRIPLE only if score >= 0.897 AND >= 3 frames agree; "not sure" band sized to 1 in 12 bikes.
Every bike in `run_video.py` output has score, confidence, heads on seat, frames agreeing and plain-language reasons.

| Test | Triples caught | Wrong TRIPLE calls | Not sure |
|---|---|---|---|
| Held-out cameras, 980 bikes / 125 triples | **50 (40 %)** (+40 flagged not-sure = 72 %) | **0** | 8.3 % |
| Full videos Indore / Pune / Jaipur, 5 fps, ~700 bikes | 0 of ~5 visible triples | **1** (man + side-saddle woman, Indore 81 s) | 3 bikes |
| Pune at 10 fps (159 bikes) | 0 (7 s triple moved to not-sure) | 0 | 1 |

Honest conclusion: good on clear police-CCTV views; weak on busy street-level hand-held video (families,
child hidden between adults, bikes visible < 1 s). 10 fps did not fix it - the model's score for those
triples is low (0.1-0.55), not just too few frames. Biggest remaining lever: more labelled street-level
family triples, then retrain (fix "c").
