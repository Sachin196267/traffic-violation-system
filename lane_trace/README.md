# LaneTrace — lane detection from traffic, not paint

Standalone experiment (not wired into the website). It finds lanes on a road
**without looking at lane markings**: vehicles are detected and tracked, and the
lanes are inferred from where and how they drive. Painted lines are used only
afterwards, in `evaluate.py`, to score the result.

```bash
# full run: detect + track + learn lanes + annotated video (30 s of video)
python lane_trace/lane_trace.py --video "videos/clip.mp4" --seconds 30 --out lane_trace/output/clip

# about 2x faster (30 s clip -> lanes ready in ~20 s on this M4)
python lane_trace/lane_trace.py --video "videos/clip.mp4" --seconds 30 --fast --out lane_trace/output/clip_fast

# how much video is needed? (re-learns from the first 3, 5, 8 ... s)
python lane_trace/convergence.py lane_trace/output/clip/observations.npz --out lane_trace/output/convergence
```

Each output folder has `lanes_video.mp4` (the model re-learns every few seconds so
you can watch it lock in), `lanes_preview.jpg`, `lanes_background.jpg`,
`paint_check.jpg` (detected lines vs painted lines in a bird's-eye strip),
`lanes.json` (boundaries as image polylines) and `report.json`.

## Why the earlier attempts were inaccurate

| Problem | Where | Effect |
|---|---|---|
| No camera-motion check | `scripts/run_lane_detection.py`, `generalized_lane_system.py` | The HTC clip pans/cuts every few seconds, Bombay has cuts, the Delhi 4K pole sways ~200 px. Tracks from different views were mixed → the fan of random lines in `outputs/lane_detection/lane_detected_preview.jpg`. |
| Geometry hard-coded, not detected | `robust_lane_delineator.py:295-310` | Road edges fixed at x=276/818 px, vanishing point y=-180, lane ratios 0.38/0.31/0.31 — tuned to one DETRAC frame, so any other video is wrong. |
| Lane count given by hand or by pixels | `scripts/lane_detect_flow.py --n_lanes`, `modules/trajectory_clustering.py:111` (`width_px / 380`) | Lane count cannot adapt to the road. |
| Clustering raw pixel x-positions | `scripts/lane_detect_flow.py:255` (k-means per image strip) | Perspective makes far lanes narrower than near ones, so clusters drift; quadratic fits (`degree=2`, line 340) then balloon (the red/green/blue "lens" lanes). |
| Wrong ground-contact point | `modules/tracker_pipeline.py:28` (box bottom-centre) | On a diagonal view the box bottom-centre is not on the road under the car; measured on the highway clip it smears every lane across the whole road. |
| Moving + queued traffic treated the same | all | In a jam the road width seen from moving tracks is wrong (a full lane waiting at a red light disappears). |

## How LaneTrace works

1. **Observe** (`collect.py`): YOLO11s + ByteTrack. Every frame is registered to a
   reference frame (background corners + RANSAC similarity), so a swaying pole camera
   still gives clean tracks; a pan, zoom or cut starts a new segment.
2. **Road coordinates** (`lane_model.py`): for a fixed camera over flat ground a point
   at lateral offset X projects to `x = g(y) + u·(y − y_h)` with `u ∝ X`. In `(u, y)`
   every lane line is vertical and every lane has the same width, near or far.
   * horizon `y_h` from how car width grows with image row (refined with track convergence),
   * ground contact = centre of the car's front/rear edge (the box corner on the road's
     downhill side minus half a car width), not the box bottom-centre,
   * each road = one smooth `g(y)` found by RANSAC over whole tracks, per travel direction
     (so a curving service road next to the main road gets its own geometry),
   * `g(y)` is straight unless ≥25 tracks clearly need a curve.
3. **Lanes** per carriageway: a 1-D density of `u` over all vehicles.
   * Lane width prior = 1.95 × median car width (3.5 m lane / 1.8 m car).
   * If traffic forms clear streams, lane width = stream spacing and lane positions =
     phase of the evenly spaced comb that fits the streams best ("traffic-lanes").
   * If there is no lane discipline (typical Indian traffic), lanes = used road width
     (including stopped/queued vehicles) ÷ standard lane width ("virtual", shown as `~N`).
4. **Confidence** per carriageway from the number of tracks and how strongly traffic
   forms lanes. Lines are drawn as far up the road as vehicles (moving or not) are seen.

## Results (lanes from traffic only, scored against the paint afterwards)

Error = distance between a detected line and the matching painted line, in lane
widths (1.0 = one full lane ≈ 3.5 m, so 0.1 ≈ 35 cm).

| Output folder | Scene | Video used | Lanes found | Confidence | Divider error | Edge error |
|---|---|---|---|---|---|---|
| `highway_3min` | quiet US highway, 2 carriageways, diagonal view | 180 s | 2 + 2 | 1.00 | 0.14 | 0.12 |
| `highway_30s` | same, only 22 usable vehicles | 30 s | 2 (near side only) | 0.30 | 0.17 | 0.16 |
| `curved_road_4k_60s` | curving road, steep view | 60 s | 3 | 1.00 | 0.08 | 0.04 |
| `far_highway_4k_22s` | far highway, small vehicles | 22 s | 4 + 4 | 0.57 | 0.22 (3 of 6 matched) | 0.19 |
| `delhi_arterial_4k_60s` | Delhi signal queue, swaying pole camera | 60 s | ~5 (virtual) | 0.35 | 0.07 | 0.16 |
| `delhi_metro_jam_54s` | Delhi jam, road narrowed by metro works + service road | 54 s | ~4 + 3 | 0.66 | 0.28 (4 of 5 matched) | 0.23 |
| `delhi_metro_jam_30s_fast` | same, `--fast`, fresh 30 s run | 30 s | ~4 + 3 | 0.61 | 0.23 | 0.21 |

`~N` = virtual lanes (no lane discipline, so lanes = road width ÷ standard lane).
Confidence is deliberately capped at 0.35 for virtual lanes.

## How much video is needed?

`convergence.py` re-learns the lanes from the first 3, 5, 8, … seconds and compares
with the final model and with the paint (`output/convergence/convergence.json`):

| Clip | Traffic | Lane count right from | Within 0.15 lane of final from | Fully stable |
|---|---|---|---|---|
| Curved 4K road | ~2 vehicles/s, follows lanes | 3 s (one blip at 8 s) | 15 s | 15 s / 54 vehicles |
| Delhi metro jam | ~5 vehicles/s, merging | 8 s | 15 s | 40 s / 229 vehicles |
| Quiet highway | ~1.5 vehicles/s total | 10 s (busy side), 120 s (quiet far side) | 120 s | 180 s / 272 vehicles |
| Delhi 4K signal queue | stop-go, no discipline | flips 5 ↔ 6 | positions wobble ±0.3 lane | — |

**What matters is vehicles, not seconds:** a carriageway is reliable once about
**40 moving vehicles** have been tracked on it (that is where confidence reaches ~0.8–1.0).

Recommendation for a future site:

* **Minimum 30 s** of steady (not panning) footage. On busy Indian city roads that is
  already enough, and `--fast` processes a 30 s 1080p clip in **~20 s**.
* **Recommended 60 s** — covers signal cycles and gives some margin.
* **Quiet roads (< 1 vehicle/s per direction): 2–3 minutes.**
* Better than a fixed length: keep watching until every carriageway reaches confidence
  ≥ 0.8 (≈ 40 vehicles), stop at 3 minutes, and show "learning…" meanwhile (what the
  video banner already does).

Speed on this Mac (M4, MPS):

| Setting | 1080p | 4K |
|---|---|---|
| default (YOLO11s @ 960 px, every 2nd frame) | 30 s clip → 39 s | 60 s clip → 113–129 s (4K decoding) |
| `--fast` (YOLO11s @ 640 px, every 3rd frame) | 30 s clip → **19–21 s** | — |
| lane fitting only | 1–5 s | 1–5 s |

Rendering the annotated video is extra (~1.3× the clip length) and is not needed if a
site only needs `lanes.json`. YOLO11n was tested and was not faster (decoding and
stabilisation dominate) but was less accurate.

## Limitations

* Needs a **fixed camera** (sway is compensated; pans/zooms/cuts split the clip and only
  the longest steady part is used). Dash-cam / moving-camera video is out of scope.
* A **junction or open plaza** (Bombay clip) has no lanes to find; the output there is meaningless.
* Where a **concrete barrier hides the bottom of the cars** on the far carriageway, that
  carriageway can shift by ~0.3 lane.
* With no lane discipline the lane *count* is road width ÷ 3.5 m, so a road that is
  ~4.5 lanes wide can come out as 4 or 5 depending on the window; positions stay within ~0.3 lane.
* The car-width correction for strongly diagonal views (`car_width_u`) was calibrated on
  one clip.

## Files

| File | Role |
|---|---|
| `lane_trace.py` | command-line entry point (video in → lanes, video, JSON, report out) |
| `collect.py` | detection + tracking + stabilisation + static-segment split + background |
| `lane_model.py` | road geometry, road/carriageway separation, lanes, confidence |
| `evaluate.py` | drawing, and the paint-based scoring (never used for detection) |
| `convergence.py` | "how much video is needed" study |
| `camera_motion.py` | quick check: which parts of a video have a steady camera |
| `cache/` | cached observations used for the results above (`cache/run_collect.py` rebuilds them) |
