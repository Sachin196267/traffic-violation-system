# TraffiSense: AI Traffic Violation Detection & E-Challan System

TraffiSense is a computer-vision research project for Indian roads. It analyses traffic video, detects
**helmet, triple-riding and red-light violations**, tracks every vehicle, reads number plates, and produces
e-challans with photo evidence and plain-language reasons. It also includes **PRAVAH**, an adaptive
traffic-signal controller tested in a Unity simulation of an Indian junction.

The design is **precision first**: when the system is not sure, a case goes to human review instead of becoming
a challan. Every finding is labelled **CHALLAN**, **REVIEW** or **EXEMPT** (ambulance, police, fire engine).

> This is a student research prototype. Its outputs are not legal documents and must be checked by an authorised
> officer before any action.

---

## Results (held-out data)

| Component | Model | Result |
|---|---|---|
| Road-scene detection (vehicles, riders, people, traffic lights) | YOLO11s fine-tuned on the full Indian Driving Dataset (33,569 images) | mAP50 0.537 on the 4,197-image test split (previous model 0.40 on the same 1,000-image check) |
| Helmet / no-helmet | YOLO11, leakage-free splits | mAP50 0.767; wrong no-helmet calls 0.6 %; better macro-F1 in all 17 stress tests (night, fog, blur, low resolution) |
| Number-plate reading | Plate detector + fine-tuned OCR (fast-plate-ocr) | 99.4 % plate recall; 81.6 % exact plates (95.4 % characters) vs 45.5 % pretrained and 13.7 % EasyOCR |
| Triple riding | YOLO11s classifier + heads-on-seat count | 5-fold leave-cameras-out: 0 false triple-riding calls on 980 bikes; 72 % of real triples sent to challan or review |
| Emergency-vehicle exemption | Logistic regression on CLIP ViT-B/16 features | 0 of 2,450 ordinary vehicles wrongly exempted; 84 % of emergency vehicles exempted or reviewed |
| Traffic-signal state | YOLO11s classifier fine-tuned on Indian signal crops | 26 / 26 Indian red lights correct; false "red" on non-lights 1.1-3.8 % |
| Speed from video | Camera calibration + Monte Carlo / bootstrap uncertainty | About 9 % mean error; true speed inside the 90 % range for 90 % of reference vehicles |
| PRAVAH signal control | Person-weighted adaptive controller (C# / Python) | 52 % lower average wait than fixed-time and 37 % lower than Webster, 6 scenarios x 3 seeds |

Detailed reports with methods, failure cases and limitations:
`helmet_training/reports/FINAL_REPORT.md`, `triple_training/REPORT.md`, `echallan/REPORT.md`,
`speed_estimation/REPORT.md`, `signal_sim/README.md`, `signal_sim/Results/`.

**Lesson learned:** an early helmet model reported 92.7 % mAP50, but its validation images had leaked into training.
On clean data it scored 0.583. The splits were rebuilt and the model retrained (0.767 above).

---

## Trained models

All trained weights are in this repository at the paths the code loads them from. See **[MODELS.md](MODELS.md)**
for every model, its training data and its held-out score.

---

## Repository structure

```
├── app/                     # Web platform: FastAPI server (app/server) + React/TypeScript UI (app/web)
├── echallan/                # Combined e-challan tool: helmet + triple riding + red light + speed
├── helmet_training/         # Helmet detector training, leakage-free evaluation, stress tests
├── triple_training/         # Triple-riding classifier, leave-cameras-out cross-validation
├── plates/                  # Number-plate reader (detector + fine-tuned OCR, frame voting)
├── red_light/               # Signal-state tracking and red-light crossing logic
├── ambulance_training/      # Emergency-vehicle classifier (CLIP features + logistic regression)
├── speed_estimation/        # Speed from video with uncertainty ranges
├── signal_sim/, signal_bridge/  # Unity junction simulator, PRAVAH controller, Python-in-the-loop bridge
├── lane_trace/              # Lane estimation from traffic flow (context only, never creates fines)
├── modules/                 # Shared pipeline modules (tracking, severity, rule-based XAI, PDF slips)
├── webapp/                  # Earlier web interface (FastAPI + Jinja)
├── config/                  # Fines, severity and temporal rules (Motor Vehicles Act sections)
├── models/, runs/           # Trained weights (see MODELS.md)
└── MODELS.md                # Model card
```

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Videos are not included in the repository; use your own traffic clip.

**Combined e-challan tool**

```bash
python echallan/run.py <video.mp4> [--camera red_light/cameras/<cam>.json] [--max-sec 120]
# -> echallan/runs/<video>/challans.json, challans.md, evidence/*.jpg
```

**Web platform** (FastAPI + React)

```bash
./app/start.sh            # http://127.0.0.1:8600
```

**Original integrated pipeline and dashboard**

```bash
python integrated_traffic_system.py --video <video.mp4> --max_frames 150 --output_dir outputs/run1
python backend_api.py --output-dir outputs/run1 --port 8765   # open http://127.0.0.1:8765/dashboard.html
```

---

## How decisions are made

- **Multi-frame decisions:** a vehicle must show the violation over several frames; a cooldown per track prevents
  duplicate challans for the same event.
- **Operating points chosen on held-out data:** each model's thresholds were picked to avoid wrong challans first.
- **Explainability:** each finding stores the model confidences and the rule checks that fired, as plain-language
  reasons and weighted rule-based attributions (not exact Shapley values).
- **Lanes are context only** and never create a fine.
- **Fines** follow the Motor Vehicles Act sections configured in `config/integrated_system_config.yaml`.

---

## Data and licences

| Dataset | Used for |
|---|---|
| [IDD-D (YOLO format)](https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11) | Road-scene detector |
| [UVH-26 (IISc AIM)](https://huggingface.co/datasets/iisc-aim/UVH-26) | Downloaded as additional Indian vehicle data |
| [Indian Vehicle License Plate Dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) | Plate detector and OCR |
| Indian helmet datasets (Kaggle / Hugging Face / Roboflow) + labelled frames from test videos | Helmet models |
| LISA Traffic Light (CC BY-NC-SA 4.0), TVD (CC BY 4.0) + labelled Indian signal crops | Signal-state classifier |
| Public emergency-vehicle image sets + labelled video crops | Emergency-vehicle classifier |

Dataset licences carry over to the trained weights. The signal-state model (LISA, non-commercial) and the
triple-riding model (a Kaggle set with no stated licence) are for study and research use only. Downloader:
`python scripts/download_datasets.py --list`.

---

## Known limitations

- Triple riding is reliable on clear CCTV views but misses many cases in busy street-level video
  (children between adults, overlapping riders).
- The helmet model can mistake a pedestrian standing next to a bike for its rider; challans need human review.
- Red-light checks need a one-time camera setup (stop line and direction) and were validated on only a few clips.
- The road detector is weak on steep overhead camera angles, which are rare in its training data.
- Speed estimates assume a flat, straight road and are not accurate enough for legal speeding challans.

---

## Author

**Sachin Nair**, B.Tech Computer Science, Pillai College of Engineering, Panvel
[LinkedIn](https://www.linkedin.com/in/n-sachin-nair) · [GitHub](https://github.com/Sachin196267)
