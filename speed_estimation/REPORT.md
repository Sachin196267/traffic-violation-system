# Vehicle speed from video — speed ranges with confidence

Date: 2026-09-27 · Apple M4 (MPS) · standalone tool in `speed_estimation/` (not wired into the app)

## What it does
For every tracked vehicle it shows a **90 % speed range** (e.g. `72–84 km/h`) and a **confidence level**:

| label on video | meaning |
|---|---|
| green **high** | 90 % range within ±15 % of the middle value |
| yellow **medium** | within ±25 % |
| orange **low** | within ±40 % |
| grey **speed ?** | range wider than ±40 %; no number shown |
| **stopped** | whole range below 5 km/h |

"90 % range" means the true speed should fall inside it 9 times out of 10. The tests below check that.

## How it works (`track.py` → `calibrate.py` → `speed.py`, or `run.py` for all three)
1. **Tracking and stabilisation.** Vehicles are detected and tracked with the IDD model (Indian roads) or COCO YOLO11s. Every frame is aligned to the first frame using background features (vehicles masked out), so hand-held or swaying cameras work. A scene cut or large pan ends the analysed part, and the output says so.
2. **Camera geometry.**
   - *Horizon:* taken from the road's vanishing point, where straight vehicle paths meet. This is the parameter speed depends on most. If there are no straight paths, it comes from GeoCalib.
   - *Lens (focal length):* from GeoCalib, a published single-image calibration network, or from a one-time calibration (below).
   - *Camera height:* every vehicle is modelled as a 3-D box of its class's typical size (car 1.8 × 4.4 × 1.5 m, bike with rider, auto, bus, truck). The height is the one that makes these boxes as wide as the detected boxes, taking the median over vehicles.
3. **Speed and its range.** Box bottoms are converted to positions on the road plane, and a velocity is fitted per vehicle. The range comes from repeating this about 300 times while varying the uncertain inputs:
   - lens and roll uncertainty;
   - bootstrap over vehicles (vanishing point and camera height re-estimated);
   - fleet-average vehicle size (±5 %);
   - per-vehicle size (±8 %/√n);
   - tracking noise;
   - measured model error (±6 % per scene, ±4 % per vehicle).

### Calibrated vs uncalibrated
- **Why calibration matters:** from a camera looking **along** the road (bridges, most CCTV), nothing in the image reveals the along-road scale. A car's length is squeezed almost flat at that angle, and the speed scales directly with the lens estimate.
- **What went wrong without it:** GeoCalib's lens estimate was 4 % off on one test scene but **2× off on another**, which gave −48 % speeds.
- **Uncalibrated mode:** assumes the lens could be off by up to about 2×, so these views honestly come out as `speed ?`. Side-on views depend much less on the lens.
- **Calibrated mode:** one known length along the road, the way real speed cameras are set up at installation, gives usable ranges. Two ways:
  - **Lane dashes** of a known standard: `{"type": "dashes", "cycle_m": …, "dash_m": …, "lines": …}`. Find the dashes with `lane_reference.py detect <video>`.
  - **Two points** on the road a known distance apart: `{"type": "distance", "points": [[x1, y1], [x2, y2]], "metres": d}`.
  - Examples are in `calibrations/`.

## Tests against independent reference speeds
There were no public videos with measured speeds from the kind of cameras used here, so references were built from **lane markings of legally fixed spacing**:
- **UK M6** motorway: TSRGD diagram 1005.1, 2 m mark / 7 m gap.
- **German Autobahn** (`highway2`): 6 m / 12 m.

In both, the standard was confirmed by the fit: dash ends match to within 7–26 cm, while other patterns fit 3–8× worse. Each vehicle's reference speed is its distance along the dash-derived road scale over time (`lane_reference.py`). The estimator itself never sees these references.

**Calibrated mode.** Calibrated from one carriageway's lane lines, tested on vehicles on the **other** carriageway, whose references use different lines:

| test | vehicles | mean error | 90 % range contains truth | typical range |
|---|---|---|---|---|
| M6, calibrated on left, tested on right | 33 | 6.3 km/h (8.4 %) | 97 % | ±17.6 km/h |
| M6, calibrated on right, tested on left (slow lanes) | 27 | 4.3 km/h (11.4 %) | 85 % | ±6.6 km/h |
| Autobahn, calibrated on left, tested on right | 39 | 10.7 km/h (8.8 %) | 100 % | ±32 km/h |
| Autobahn, calibrated on right, tested on left | 38 | 10.3 km/h (7.6 %) | 100 % | ±44 km/h |
| **all** | **137** | **≈ 9 %** | **96 %** | |

With all four M6 lines used for calibration (the demo video), vehicles on both carriageways show 9 high / 51 medium labels. Mean error is 9.1 %, and the 90 % range contains the truth for 90 % of vehicles.

**Uncalibrated mode (GeoCalib lens only), same scenes:** every vehicle was labelled `speed ?`, so **no wrong number was shown**. Underneath, the middle estimates were 13.6 % off on M6 and 53 % off on the Autobahn; the wide ranges correctly flag this.

**VS13 (real cars at cruise-control speeds, uncalibrated):** see the VS13 section below.

<!-- VS13-AUTO-START -->
## VS13 test (real cars, cruise-control speeds, uncalibrated)

Peugeot 208, 29 clips (30–105 km/h). Camera: roadside, 1.2 m high, car driving **toward** it along the road, the hardest geometry for uncalibrated mode. Ground truth is the cruise-control speed from the file name (speedometers usually read 2–5 % high).

- Measured: 29/29 clips
- Shown with a number (high/medium/low): **0**; everything else was `speed ?`
- 90 % range contains the true speed: 100 % of clips
- Middle estimate (hidden when `speed ?`): median error +4 %, mean |error| 38 %

Conclusion: with no calibration, a low camera looking along the road cannot measure speed. The tool correctly refuses to show a number instead of showing a wrong one. Calibrate the camera (one known length on the road) for such views.

| truth km/h | middle estimate | 90 % range | label |
|---|---|---|---|
| 30 | 28 | 8–148 | unknown |
| 32 | 9 | 1–164 | unknown |
| 34 | 34 | 20–210 | unknown |
| 37 | 67 | 3–322 | unknown |
| 40 | 36 | 23–206 | unknown |
| 43 | 33 | 2–326 | unknown |
| 45 | 45 | 26–307 | unknown |
| 47 | 47 | 6–258 | unknown |
| 50 | 42 | 26–286 | unknown |
| 51 | 62 | 9–1466 | unknown |
| 54 | 100 | 33–711 | unknown |
| 57 | 85 | 27–345 | unknown |
| 60 | 85 | 35–670 | unknown |
| 62 | 42 | 2–373 | unknown |
| 64 | 98 | 37–790 | unknown |
| 67 | 2 | 0–578 | unknown |
| 68 | 113 | 36–953 | unknown |
| 71 | 68 | 30–487 | unknown |
| 73 | 154 | 41–1268 | unknown |
| 76 | 34 | 9–463 | unknown |
| 77 | 40 | 8–549 | unknown |
| 79 | 126 | 42–1481 | unknown |
| 82 | 63 | 12–505 | unknown |
| 84 | 100 | 47–1026 | unknown |
| 87 | 126 | 43–6559 | unknown |
| 90 | 101 | 42–1631 | unknown |
| 92 | 115 | 43–1477 | unknown |
| 95 | 99 | 46–1565 | unknown |
| 96 | 136 | 48–2451 | unknown |
<!-- VS13-AUTO-END -->

## Your videos (uncalibrated, first 60 s)
| video | result |
|---|---|
| Bombay (hand-held, sways up to 112 px, stabilised) | 54 low-confidence ranges (mostly 10–30 km/h at the junction), 109 stopped, 236 `speed ?` |
| Bangalore signal (camera about 0.7 m high; scene cut at 52.6 s, analysis stops there) | 14 stopped, 36 `speed ?` |
| South Delhi | 24 stopped, 434 `speed ?` |

These views look along the road and have no calibration, so most vehicles honestly show `speed ?`. For useful numbers on a fixed camera, calibrate it once:
- lane dashes (Indian IRC markings, or measure one dash cycle on site);
- or two road points whose distance you know or measure;
- zebra-crossing stripes also work as a known length.

## Run it
```bash
.venv/bin/python speed_estimation/run.py "<video>.mp4" --max-sec 60                        # uncalibrated
.venv/bin/python speed_estimation/run.py "<video>.mp4" --calib speed_estimation/calibrations/<file>.json
# -> speed_estimation/runs/<video>/<video>_speed.mp4 (annotated) and speeds.json (per vehicle: median, range, confidence)
.venv/bin/python speed_estimation/evaluate.py speed_estimation/runs/<video>   # if a reference.json exists
```
Speed on the M4: about 7 s of processing per second of 1080p video at 12.5 fps sampling (measured: 15 s of video in 101 s; detection and tracking dominate). Use `--fps 5` to halve it.

## Limits
- **Calibrated error is about 9 % on average** (worst single vehicles about 20 %). Treat the range, not the middle value, as the answer. It is not accurate enough for legal speeding challans without a certified device.
- **Flat, straight road assumed.** Curves, hills and junction turns add error; the ranges only partly cover this.
- **Fast or tiny vehicles:** very fast vehicles seen for under 0.8 s, and very small or occluded ones, get no speed.
- **Tested on 2 reference scenes (137 vehicles), plus 29 VS13 clips (uncalibrated).** The model-error allowance (±6 %/±4 %) was set from these same tests, so more scenes would make the ranges more trustworthy.
- **Crowded junctions:** the detector sometimes draws one box around a group of bikes/autos (seen on Bombay). In every test these boxes came out `stopped` or `speed ?`, never with a number. A box-size check was tried but rejected: on the 0.7 m-high Bangalore camera it also rejected normal cars.
- **Video labels:** when labels would overlap, the nearest vehicles and vehicles with a speed number are labelled first. The rest get a thin box with no label, but their values are still in `speeds.json`.
- **Lane-dash references are only as good as the painted markings.** The fits show they match the standard to within a few cm here.
