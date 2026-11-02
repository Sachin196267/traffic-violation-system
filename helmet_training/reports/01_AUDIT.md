# Phase 1–6 Audit (completed before any training)

Date: 2026-09-24/25 · Hardware: Apple M4, 16 GB unified memory, MPS (no CUDA) · Ultralytics 8.4.153, torch 2.14 (`.venv`)

Everything in this folder is read-only with respect to the application. No app code, config or production weight was changed.

---
## A. Existing IDD traffic-object model

| | `models/traffic.pt` (**used by `config/integrated_system_config.yaml` + `traffic_violation/final_video_testing/config.yaml`**) | `runs/detect/runs/detect/idd_traffic_v2/weights/best.pt` (used by `config/lane_system_config.yaml`) |
|---|---|---|
| Identity | **byte-identical (md5 f6e920f9…) to `runs/detect/idd_traffic-2/weights/best.pt`** | fine-tune of `runs/detect/train/weights/best.pt` |
| Arch | YOLO11s | YOLOv8n |
| Training | **1 epoch, batch 4**, 640 px, COCO-pretrained | 20 ep (yolov8n) + 60 ep continuation, 640 px, batch 16 |
| Data | IDD-D YOLO subset: 8 000 train / 1 000 val / 1 000 test (of 41 962), 15 classes | same |
| Val P / R | 0.595 / 0.257 | 0.665 / 0.413 |
| Val mAP50 / 50-95 | **0.269 / 0.166** | **0.451 / 0.306** |

Per-class on IDD val (re-measured by `helmet_training/audit/eval_idd.py`; confusion matrices in `helmet_training/audit/idd_eval/*/`):

| class (val instances) | traffic.pt P / R / mAP50 | idd_traffic_v2 P / R / mAP50 / mAP50-95 |
|---|---|---|
| motorcycle (2228) | 0.60 / 0.47 / 0.483 | 0.73 / 0.55 / 0.595 / 0.350 |
| rider (2089) | 0.69 / 0.36 / 0.414 | 0.73 / 0.44 / 0.492 / 0.282 |
| person (1924) | 0.57 / 0.34 / 0.361 | 0.72 / 0.35 / 0.419 / 0.224 |
| bicycle (81) | 0.89 / 0.10 / 0.154 | 0.63 / 0.39 / 0.394 / 0.235 |
| car (1889) | 0.65 / 0.52 / 0.529 | 0.80 / 0.56 / 0.626 / 0.444 |
| bus (425) | 0.42 / 0.52 / 0.444 | 0.78 / 0.56 / 0.610 / 0.471 |
| truck (609) | 0.41 / 0.43 / 0.344 | 0.64 / 0.48 / 0.549 / 0.386 |
| autorickshaw (711) | 0.60 / 0.46 / 0.482 | 0.76 / 0.58 / 0.626 / 0.438 |
| traffic light (83) | 0.68 / 0.04 / 0.076 | 0.67 / 0.22 / 0.244 / 0.119 |
| traffic sign (306) | 0.20 / 0.12 / 0.070 | 0.55 / 0.22 / 0.224 / 0.118 |

Class mapping is consistent (15 IDD names, identical in yaml and both checkpoints). Class imbalance is strong (motorcycle/rider/person/car ≈ 17–19 k train boxes each; bicycle 630, traffic light 668, caravan 30, trailer 3, train 13).

**Verdict (Phase 2):** the *production* IDD weight is seriously under-trained: it is a 1-epoch smoke-test checkpoint, so recall is 0.26 overall and 0.04 for traffic lights. However, a properly trained IDD model **already exists** (`idd_traffic_v2`, mAP50 0.451). That is reasonable for a nano model on a 10 k-image IDD subset. Rider/motorcycle recall is still modest (0.44/0.55), and traffic light/sign are weak. **I did not retrain IDD**: a better existing weight is available, and IDD is not what this task is about. **Recommendation, not applied:** point `models/traffic.pt` consumers at `idd_traffic_v2` (or a longer YOLO11s run on the full 41 962-image IDD) in a future, separate change.

---
## B. Existing helmet / no-helmet model

Production helmet model (referenced by `config/integrated_system_config.yaml`, `traffic_violation/final_video_testing/config.yaml`, `violation_detector.py`):
`runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt`, YOLO11n, 512 px, 30 epochs, batch 16, classes `{0: helmet, 1: no_helmet}`, trained on `datasets/merged/helmet_traffic_10k` (13 217 train / 1 844 val, boxes 24 168 helmet / 13 163 no_helmet).

Reported val: P 0.879, R 0.873, **mAP50 0.927**, mAP50-95 0.640. **These numbers are invalid because of leakage (see below).**
Other helmet runs: `runs/detect/runs/helmet/helmet_yolov8n_30ep{,-2}` stopped after 2 epochs (mAP50 0.573); `runs/detect/runs/detect/helmet_detector` and `model2/helmet_nohelmet` have no weights.

Re-measured on the new leakage-free test split (`helmet_training/reports/eval/OLD_helmet_nohelmet-2/`): **mAP50 0.583, mAP50-95 0.268**, macro-F1 0.62. Indian images score macro-F1 0.49, and the elevated-CCTV camera scores 0.06.

---
## C. Helmet data found in the project

| dataset | licence | images | classes | box type | findings |
|---|---|---|---|---|---|
| `Motorcycle Helmet Detection Dataset PH.v2i.yolov8` | CC BY 4.0 | 8 324 (416², stretched) | full_face, half_face, helmet_invalid, no_helmet | **whole rider + motorcycle** | frames of **only 3 videos**; Roboflow 2× augmented copies; 54 % empty labels while riders are visible (missing labels); 981 exact-dup groups, 379 span train/valid/test |
| `Bike Helmet Detection.v2-more-preprocessing-augmentation.yolov8` | Public Domain | 3 735 | With/Without Helmet | head | only **630 unique source images** (andrewmvd subset) × baked-in mosaic, ±30° rotation, shear |
| `datasets/kaggle/helmet_india` (Roboflow aryan_1) | CC BY 4.0 | 942 (640² stretched) | names `'0'..'4'` undocumented | head + rider + plate | meaning confirmed from the Kaggle card and visually: 0 plate, 1 faceWithNoHelmet, 2 faceWithGoodHelmet, 3 faceWithBadHelmet, 4 rider. 18 exact-dup groups (7 span splits). 244 class-2 / 80 class-1 boxes are **not head boxes** (whole rider or vehicle) |
| `datasets/merged/helmet_traffic_10k` (used for the production model) | mixed | 15 061 | helmet, no_helmet | **mixed head + rider boxes** | see below |

`helmet_traffic_10k` defects (built by `scripts/merge_helmet_datasets.py`):
1. **Train/val leakage.** A random per-file split of pre-augmented Roboflow copies means 72 % (PH), 93 % (helmet_rider) and 94 % (bike) of val source images also appear in train. There are 802 exact-duplicate groups across train/val.
2. **Bike dataset included twice** (3 546 + 126 `bikev2_*` files outside the manifest, plus the `helmet_bike_v2__*` copies).
3. **Inconsistent box semantics.** PH contributes whole-rider boxes, while Bike and helmet_rider contribute head boxes, all under the same two class ids.
4. **`helmet_rider` source is 94 % tiny crops** (15–30 px images with a full-image box `0.5 0.5 1 1`), i.e. classification data inside a detection set. The crops come from BikesHelmets images (cross-source leakage), and the original folder is no longer on disk.
5. `rider_helmet_invalid` was silently mapped to `no_helmet`, which is an ambiguous class.

---
## D. Data sufficiency decision (Phase 5): external data WAS needed

After removing leakage, duplicates and incompatible box types, only ≈ 1 700 usable head-level images remained, few of them Indian, with no night, dense-CCTV or low-resolution coverage. Searched Kaggle and Hugging Face; licences were checked from Kaggle metadata.

| candidate | licence | decision |
|---|---|---|
| andrewmvd/helmet-detection (764 originals behind the Bike set) | CC0-1.0 | **USED** (replaces the augmented Roboflow copies) |
| aneesarom/rider-with-helmet-without-helmet-number-plate (124 Indian road images) | Apache-2.0 | **USED** (head boxes, clean) |
| pkdarabi/helmet | CC BY 4.0 (Roboflow says "Private") | rejected: "helmet" boxes are whole-rider, mostly plate close-ups |
| abuzarkhaaan/helmetbehncode | CDLA-P / CC BY 4.0 | rejected: only a *Helmet* class; bare heads unlabeled → would teach "head = background" |
| hquan345/helmet-person-motorbike (Vietnam, elevated) | CDLA-Sharing-1.0 | rejected for training/metrics: ~48 % of helmets unlabeled (verified visually); used for qualitative testing only |
| irfanmohammad1729/two-wheeler-violation (night) | CC BY-ND 4.0 | rejected for metrics: boxes visibly misaligned; ND licence → images used unmodified for qualitative night testing only |
| ayushraj2349/sample-videos-for-helmet-detection | MIT | video testing only |
| devgurucodes/… | DbCL (re-aggregation of unknown sources) | rejected: provenance/licence unclear |
| lcsc00, aryanvaid13, yashjindal, lilflerkin, sarthakmodi, … | unknown / copyright | rejected: licence unclear |
| OSF HELMET (Myanmar, 91 k frames) | CC BY 4.0 | not downloaded: 9.4 GB vs 9.9 GB free disk; rider-level boxes |
| HF UniqueData/helmet_detection | CC BY-NC-ND | rejected (NC-ND) |

Nothing was downloaded from Roboflow Universe because no API key is configured.
