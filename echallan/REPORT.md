# Red-light violations, emergency-vehicle exemption and a combined e-challan tool

Date: 2026-09-27 · Apple M4 + Kaggle (CPU; the GPU queue did not start) · **standalone, not connected to the website**

## What was asked
1. Detect the traffic-signal state and red-light jumping.
2. Challan every vehicle that breaks a rule, **except emergency vehicles** (ambulance, police, fire).
3. One combined tool for all violations.
4. Precision first: no wrong challans; when unsure, send the case to a person.

## What was built

| Part | Where | What it does |
|---|---|---|
| Red-light checker | `red_light/observe.py`, `red_light/decide.py` | Signal state over time, stop line, crossing on red, decision + reasons |
| Signal-state model | `runs/signal_state/weights/last.pt` | Classifies each signal crop as red / yellow / green / not a light |
| Emergency-vehicle model | `runs/ambulance_cls/` (`ambulance_training/train_probe.py`) | Classifies each vehicle as ambulance / police / fire / other (CLIP features + trained classifier) |
| e-challan tool | `echallan/run.py` | Runs helmet + triple riding + red light + speed; every finding becomes CHALLAN / REVIEW / EXEMPT |

### Run it
```bash
.venv/bin/python echallan/run.py <video> [--camera red_light/cameras/<cam>.json] \
      [--speed-calib speed_estimation/calibrations/<cam>.json --speed-limit 50] [--max-sec 120]
# -> echallan/runs/<video>/challans.json, challans.md, evidence/*.jpg
.venv/bin/python echallan/grade_sheet.py <video>        # contact sheet of the challans, for checking by eye
```

### Decisions
| Decision | Meaning |
|---|---|
| **CHALLAN** | The tool is sure, using its own strict thresholds. |
| **REVIEW** | The tool is not sure; a person must look before any challan. |
| **EXEMPT** | Ambulance, police vehicle or fire engine. Never challaned; the reason is recorded. |

Every item carries a confidence score, plain-language reasons (for the explainable-AI part of the future site) and an evidence image.

## 1. Signal state

**Model:** YOLO11s classifier, 64 px, trained on Kaggle.

**Training data:**
- LISA Traffic Light: US footage, day and night; CC BY-NC-SA 4.0, so non-commercial use only.
- Red/green boxes from the traffic-violation-detection dataset (CC BY 4.0).
- Every crop is padded to a square so no lamp is cut off, and shrunk and blurred so it looks like a small far-away CCTV signal.

**Held-out LISA/TVD clips:**

| Class | Correct |
|---|---|
| Red | 2,252 / 2,252 |
| Yellow | 72 / 72 |
| Green | 1,114 / 1,132 |
| Not a light | 456 / 461 |
| TVD red | 72 / 73 |
| TVD green | 76 / 78 |

**Indian signals:** I labelled 216 crops by eye from the downloaded clips (`signal_training/head_labels.json`).
- Real red lights: 26 / 26 correct.
- Signboards, red vehicles and the red circles police draw on videos: **57 of 149 wrongly called red.**

**Fine-tuned with Indian crops** (`signal_training/finetune.py`). The clips were split into two halves; each half was tested with a model that never saw it.

| Held-out half | Real reds correct | Non-lights confidently called red | Non-lights called green |
|---|---|---|---|
| A | 7 / 7 | 26 / 679 (3.8 %) | 55 |
| B | 18 / 19 | 6 / 548 (1.1 %) | 35 |

The final model uses all the data.

**Weak spot: yellow.** There were very few Indian yellow examples, and several were misread.

Because single-frame errors remain, the decision step adds three safeguards:
1. **Smoothing:** a lamp keeps its colour for seconds and changes in a fixed order, so one misread frame cannot flip the state.
2. **Visible lit face:** a signal head is used only if its lit face is seen.
3. **Colour change seen:** a challan needs the head to be **seen changing colour** in the shot (at least 2 s red and at least 2 s green/yellow, or dark for red-only heads). A signboard or red car never "turns green", so it can at most cause REVIEW.

**Finding signals:**
- **Detectors (IDD + COCO):** often miss close-up LED signal heads.
- **Colour finder** (fills that gap): a red lamp and a green lamp next to each other that are never lit together, or a red lamp going on and off in phases lasting seconds.
- **Too-short shots:** clips cut into very short shots cannot be checked; the report says so per shot.

## 2. Red-light crossing
- **Crossing:** a vehicle's ground point crosses the stop line in the controlled direction while the smoothed signal is red.
  - Red must hold both at the last sighting before the line and at the first after it (vehicles hidden behind poles).
  - The light must have been red for at least 1 s.
- **Sent to REVIEW instead:**
  - just turned red;
  - weak signal reading;
  - signal not seen changing colour;
  - short or unclear track;
  - the vehicle turned more than 50° after the line (it may be a free-left turn);
  - it might be an emergency vehicle.
- **Stop line and direction, two ways:**
  - **Automatic:** the direction whose traffic moves on green and stops on red, and the stop line at the front of the red queue. On the downloaded clips (short YouTube compilations, busy Indian junctions with many signal jumpers) it was **never confident enough** on any clip. So automatic mode made no challans, which is safe but not yet useful.
  - **One-time camera setup** (`red_light/cameras/*.json`): draw the stop line and the direction arrow once per camera, as real red-light cameras are set up at installation. Setups can be per shot.

**Test.** There is no public Indian video with marked red-light violations, so I set up 3 cameras and graded by eye.

| Clip | Result |
|---|---|
| sig-03 (Tirupati CCTV) | 2 violations, **both correct**: a car and a motorcycle entering on red, red for at least 2.0 s and 4.8 s; 11 crossings on green correctly marked OK |
| sig-13, sig-21 | The camera sits beyond the stop line, so almost no vehicle crosses it on screen; nothing to judge |

Six more junction clips were downloaded and checked automatically (sig-30 to sig-35). None shows a signal face the camera can read, and I confirmed this by eye:
- sig-30: the signal heads are seen from the side;
- sig-31, sig-32: drone and timelapse views, too far away;
- sig-33: no signal in view;
- sig-34: US junction, signals face away;
- sig-35: night phone video.

The tool correctly reported "red-light check not possible" and made no challans on them.

My first direction arrow for sig-03 was about 25° off. This shows a camera setup must be checked on a few crossings before use.

**Honest status:** the red-light chain works end to end and was right in the one real test, but that test is tiny. It needs a longer fixed-camera clip of one junction with real traffic before it can be trusted.

## 3. Emergency vehicles (ambulance / police / fire)
**Training data:**
- Web photos: Indian and Pakistani emergency-vehicle datasets. Their own labels are wrong (most "other" images are ambulances, and some "ambulances" are police jeeps), so the images were sorted with CLIP and only confident ones kept: 1,633 ambulance, 522 police, 642 fire, 1,795 other.
- Vehicle crops from 44 videos, labelled by eye per vehicle (`ambulance_training/labels.json`, 490 vehicles checked):
  - 23 ambulance clips;
  - 4 police clips (convoys and traffic police);
  - 4 fire-engine clips (Kerala, Delhi, Mumbai);
  - 13 ordinary traffic clips.

**Model:** logistic regression on CLIP ViT-B/16 image features. A vehicle's score is the mean of its 3 best crops.

**Test:** videos held out from training in 5 groups. Operating point: exempt at score ≥ 0.96, review at score ≥ 0.54.

| Type | Vehicles | Exempted (no challan) | Exempted or sent to review |
|---|---|---|---|
| Ambulance | 103 | 66 | 90 |
| Police | 20 | 5 | 15 |
| Fire engine | 47 | 27 | 38 |
| **All emergency** | **170** | **98 (58 %)** | **143 (84 %)** |
| Ordinary vehicles | 2,450 | **0 wrongly exempted** | 49 sent to review (2 %) |

**Reading the results:**
- **No ordinary vehicle was wrongly let off.**
- **Missed emergency vehicles:** 27 of 170 score like ordinary vehicles (mostly blurred, far away or half hidden). If one of them breaks a rule it would be challaned, so challans should still be checked by a person before they are sent.
- **Police is the weakest type:** plain white Boleros and cars look like private vehicles unless the beacon or the "POLICE" text is visible.
- **Plain CLIP, no training:** at a similar setting it wrongly exempted 2 ordinary vehicles; the trained model did better.
- **Test-set rule:** tracks that were not reviewed count as "ordinary" only if no crop looks like any emergency type. 741 unreviewed middle-score tracks are left out of the test.

## 4. Combined e-challan tool: test

| Clip | Helmet challans | Triple-riding challans | Notes |
|---|---|---|---|
| tpt-26 (Tirupati CCTV) | 12 right, **1 wrong** (person standing by the road) | 13 plausible; 4 evidence crops miss the bike | 31 items sent to review |
| tpt-15 | 7 right, **2 wrong** (the same pedestrian next to a bike, twice) | 3 right; 1 crop misses the bike | |
| sig-03 | 2 no-helmet challans (not graded) | – | 2 red-light violations, both correct |
| amb-14 (ambulance clip) | 0 challans, 9 to review | – | no red-light check possible (no signal visible) |

**Found problems:**
1. **Helmet tool (wrong challans): a pedestrian standing next to a motorcycle can be taken as its rider.** This happened 3 times in 22 no-helmet challans. It needs fixing before any real use.
2. **Triple-riding evidence:** the tool records when a bike was first seen but its box from the last sighting, so some evidence crops miss the bike. The decision itself is not affected. The fix belongs in `triple_training/run_video.py`, which another session was changing at the time, so it was left untouched.
3. **Speeding:** challans only with a calibrated camera and a speed limit. The combined path was tested on the calibrated M6 motorway clip (section 5).

## 5. Speed path check
M6 motorway, 20 s, calibrated from lane dashes, limit 100 km/h. Result: 0 challans, 1 REVIEW.
- The car's speed range was 86–121 km/h, so a legal speed is still possible.
- A speeding challan needs **even the low end** of the 90 % range to be over the limit.
- Without a calibration file the tool does not check speed and says so.

## Automation used (so work continued unattended)
- `red_light/autopilot.sh`, then `echallan/autopilot2.sh`: waited for Kaggle jobs, installed models, restarted stopped jobs, ran all clips and wrote status files.
- **Incidents:**
  - A download script looked for its tool in the wrong folder.
  - zsh did not split a file list into separate names.
  - The Kaggle GPU queue did not start, so jobs moved to CPU.
  - The disk was 98 % full; UVH-26 and UA-DETRAC were deleted at the user's request.
  - A crash-test showed colour-found signals could never be used; fixed.
  - A symlink was resolved to the real M6 file, which could have overwritten earlier speed results; caught before it did.

## Next steps (in order of value)
1. Fix the pedestrian-as-rider false no-helmet challans (reject heads whose body is standing beside the bike, not on the seat).
2. Record a proper test video for red-light checking: one fixed camera looking at a stop line for 10+ minutes, with a known signal cycle.
3. More Indian yellow-signal examples, and police and fire vehicles in videos.
4. Store a box with a matching time for triple-riding evidence.
5. Licence check before any commercial use (LISA is non-commercial; one ambulance dataset has no stated licence).

## 6. Extra test: fixed camera, US intersection (Wikimedia Commons, 66 s, 8K scaled to 1080p)
File: `datasets/signal_long/long-md355.mp4`; run folder `red_light/runs/long-md355/`.

**Signal colour** (graded by eye, 64 frames from 8 real signal heads):
- Red: 47 / 47 correct.
- Green: 9 / 10 correct (one green arrow read as red).
- Overall: 56 / 57 = 98 %; 7 frames were too small or dark to judge.
- One false "signal" (red brick paving) was read as red. It never changes colour, so it cannot cause a challan.

**Red-light crossings:** the signals facing the camera stay red for the whole clip, and the moving traffic is cross traffic whose signals face away from the camera. So this clip cannot test crossings.

**Emergency vehicles:** 109 ordinary vehicles scored; 0 wrongly exempted and 0 sent to review. There were no emergency vehicles in the clip.

**YouTube** (for longer clips) blocks downloads from this Mac with a bot check, even with browser cookies. `red_light/autopilot4.sh` retries every 30 min.

## 7. Extra test: Las Vegas fixed traffic camera, night (Internet Archive, 5 min used)
File: `datasets/signal_long/long-lv2.mp4`; run folder `red_light/runs/long-lv2/`. Graded by eye: 50 frames from 5 vehicle signal heads.

| True colour | Read correctly | Wrong colour | No call ("unknown") |
|---|---|---|---|
| Red (43) | 32 | 3 (called yellow) | 8 |
| Yellow (5) | 4 | 0 | 1 |
| Green (2) | 2 | 0 | 0 |

- **When it makes a call:** 38 / 41 = 93 % correct.
- **Of all frames:** 38 / 50 = 76 %; misses are over-exposed red lamps at night.
- **No wrong-challan risk:** green or yellow was never read as red, so errors can only miss a violation, not cause a wrong challan.
- **Pedestrian signal:** a pedestrian "hand" signal was also found and read as yellow or unknown. It is never red, so it cannot cause a challan.

**Red-light crossings:** not testable. At night the detector tracked only 60 vehicles in 5 minutes.

**Emergency vehicles:** 42 vehicles scored.
- 0 wrongly exempted.
- **8 (19 %) sent to review as "possible police"** — checked by eye, none is police. Most are taxis with lit roof-advert boxes, which at night look like police light bars; the rest are white SUVs.
- These would cost reviewer time but would never wrongly cancel a challan.
- Night footage and roof signs are a weak spot; the model has no night-time non-Indian training data.
