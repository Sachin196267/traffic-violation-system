# Intelligent Traffic Violation Enforcement & Road Perception System (ITMS)

An enterprise-grade, India-centric Intelligent Traffic Management System (ITMS) featuring deep-learning multi-violation detection, standalone road lane perception, explainable AI (XAI), hazard severity assessment, automated fine calculation, and court-admissible printable receipts (PDF and HTML).

---

## 🚀 Key Highlights & Capabilities

1. **High-Accuracy Helmet & Triple-Riding Detection**
   - **Targeted Two-Wheeler ROI Inference**: Motorcycle/scooter detections from the primary IDD YOLO model (`models/traffic.pt`) are cropped with vertical head padding and evaluated using the dedicated 92.7% mAP50 helmet detector (`runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt`).
   - **Multi-Cue Triple Riding**: Evaluates overlapping person bounding boxes, head/helmet spatial clusters, and vertical aspect ratios to reliably detect triple riding without spurious false alarms.
   - **Temporal Hysteresis & Anti-Duplication**: Requires consecutive frame validation (configurable `min_frames`, default: 3) and imposes a debouncing cooldown (configurable `cooldown_frames`, default: 50) per vehicle track ID to guarantee strictly zero duplicate challans.

2. **Standalone Road Lane Perception (Strictly ZERO Fines)**
   - Lane detection operates exclusively as a **perception and situational awareness** module.
   - Generates drivable corridor boundaries, vanishing point estimation, and overlay visualization videos.
   - **Guaranteed Isolation**: Lane boundaries and vehicle lane positions **NEVER** generate fines or violation records.

3. **Configurable Fine Calculation & Compound Policies**
   - All statutory violations and penalties are configured in [`config/integrated_system_config.yaml`](config/integrated_system_config.yaml) under statutory Motor Vehicles Act schedules:
     - `NO_HELMET`: INR 1,000 (Sec 129 / Rule 138)
     - `TRIPLE_RIDING`: INR 1,000 (Sec 128 / Rule 123)
     - `SPEEDING`: INR 2,000 (Sec 183(1))
     - `RED_LIGHT_VIOLATION`: INR 5,000 (Sec 184 / 119)
     - `WRONG_WAY_DRIVING`: INR 5,000 (Sec 184)
     - `EMERGENCY_OBSTRUCTION`: INR 10,000 (Sec 194E)
   - Supports configurable multi-violation compound policies:
     - `sum`: Total sum of all applicable violation fines.
     - `max`: Maximum fine among active violations.
     - `highest_plus_half`: Highest fine plus 50% of each secondary violation.

4. **Professional Court-Admissible Receipts (PDF & HTML)**
   - **PDF Receipts**: High-resolution, strictly single-page documents generated with ReportLab (`outputs/<dir>/violation_slips/*.pdf`).
   - **HTML Receipts**: Companion responsive, standalone receipts with `@media print` styling, dual evidence images, severity badges, and an integrated "Print Challan" button (`outputs/<dir>/violation_slips/*.html`).

5. **Dual-Layer Explainable AI (XAI)**
   - Explicitly segregates:
     - **Layer 1 (Neural Model Prediction)**: YOLO classes, bounding boxes, and raw neural confidence scores.
     - **Layer 2 (Statutory Rule Decision)**: Motor Vehicles Act sections, threshold conditions met, and exact Shapley feature attributions.

6. **Hazard Severity & Intensity Engine**
   - Computes a dynamic risk hazard score (0–100) combining violation baseline severity with real-time operational modifiers (speed over limit, red signal elapsed time, multi-passenger risks).
   - Categorizes violations into **LOW**, **MEDIUM**, **HIGH**, and **CRITICAL** with transparent natural-language reasoning.

7. **Interactive Web Dashboard & Command Center**
   - Modern Tailwind CSS glassmorphism UI featuring live KPI telemetry, Three.js 3D digital twin, filterable violation log, and interactive evidence inspector with direct PDF and HTML receipt links.

---

## 📁 Repository Structure

```
├── config/
│   └── integrated_system_config.yaml   # Central configuration: fines, severity, temporal rules
├── models/
│   ├── traffic.pt                      # Primary IDD multi-class detector
│   ├── license_plate_best.pt           # ANPR license plate detector
│   └── ambulance_best.pt               # Emergency vehicle detector
├── runs/detect/runs/model2/
│   └── helmet_nohelmet-2/weights/best.pt  # Dedicated 92.7% mAP50 helmet detector
├── modules/
│   ├── severity_engine.py              # 0-100 hazard scoring & intensity engine
│   ├── temporal_violation_analyzer.py  # Spatio-temporal voting, helmet ROI crops, triple riding
│   ├── violation_slip_generator.py     # PDF & printable HTML receipt generator
│   ├── xai_explainer.py                # Dual-layer Shapley XAI explainer
│   ├── lane_assigner.py                # Lane assignment & diagnostic status (0 fines)
│   ├── road_model.py                   # Roadway corridor modeling & vanishing point
│   ├── tracker_pipeline.py             # Multi-object tracking (ByteTrack/BoT-SORT)
│   └── signal_manager.py               # Junction signal timing & queue pressure
├── scripts/
│   ├── run_lane_detection.py           # Dedicated standalone lane perception runner
│   ├── verify_all_requirements.py      # Automated 9-part test & verification suite
│   └── download_datasets.py            # Dataset downloader utility
├── integrated_traffic_system.py        # End-to-end multi-violation enforcement pipeline
├── backend_api.py                      # Local HTTP API & dashboard server
└── dashboard.html                      # Interactive web dashboard & command center
```

---

## 🛠️ Installation & Setup

### 1. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Models & Test Suite

Run the automated verification suite to validate all 9 critical components:

```bash
python scripts/verify_all_requirements.py
```

---

## 🚦 Usage Guide

### 1. Run Standalone Lane Detection (Zero Fines Guaranteed)

To inspect lane boundaries, drivable corridors, and road perception on a video without issuing tickets:

```bash
python scripts/run_lane_detection.py \
  --video "videos/test/model2_test.mp4" \
  --max_frames 120 \
  --output_dir outputs/lane_detection
```

**Outputs generated:**
- `outputs/lane_detection/lane_detected_video.mp4`: Video with lane overlays.
- `outputs/lane_detection/lane_detected_preview.jpg`: High-resolution preview image.
- `outputs/lane_detection/lane_boundaries.json`: Geometric boundary coordinates.
- *Fines issued: Strictly 0.*

### 2. Run Integrated Traffic Violation Pipeline

Process video surveillance streams, detect infractions, and automatically generate e-Challan records and printable receipts:

```bash
python integrated_traffic_system.py \
  --video "videos/test/model2_test.mp4" \
  --max_frames 150 \
  --output_dir outputs/final_verify
```

**Outputs generated in `--output_dir`:**
- `integrated_annotated.mp4`: Full annotated video with bounding boxes and telemetry.
- `integrated_report.json`: Complete execution report with lane model, XAI, and violations.
- `violations.csv` & `challans.json`: Structured violation records with severity scores and fine amounts.
- `violation_slips/`: Official court-admissible PDF receipts (`.pdf`) and companion responsive printable HTML receipts (`.html`).
- `snapshots/`: Dual-panel cropped and full-frame visual evidence.

### 3. Launch Web Dashboard & Command Center

Start the lightweight local server to view the interactive dashboard:

```bash
python backend_api.py --output-dir outputs/final_verify --port 8765
```

Open `http://127.0.0.1:8765/dashboard.html` in your browser:
- Filter infractions by category (`NO_HELMET`, `TRIPLE_RIDING`, `SPEEDING`, etc.).
- Inspect violation telemetry and dual-layer XAI attributions.
- Click **HTML** to preview the printable receipt or **PDF** to download the official e-Challan.

---

## ⚙️ Configuration Reference

All settings can be customized in [`config/integrated_system_config.yaml`](config/integrated_system_config.yaml):

```yaml
fines:
  currency: "INR"
  compound_policy: "highest_plus_half"  # 'sum', 'max', or 'highest_plus_half'
  rules:
    NO_HELMET:
      amount: 1000
      statute: "Motor Vehicles Act Section 129 / Rule 138"
      description: "Driving or riding a two-wheeler without protective headgear conforming to BIS standards"
    TRIPLE_RIDING:
      amount: 1000
      statute: "Motor Vehicles Act Section 128 / Rule 123"
      description: "Carrying more than one pillion rider on a two-wheeled motorcycle"
    SPEEDING:
      amount: 2000
      statute: "Motor Vehicles Act Section 183(1)"
      description: "Driving at a speed exceeding statutory corridor speed limit"
    # ... additional violations

severity:
  levels:
    CRITICAL: 80
    HIGH: 65
    MEDIUM: 40
    LOW: 0
  base_scores:
    EMERGENCY_OBSTRUCTION: 85
    WRONG_WAY_DRIVING: 80
    RED_LIGHT_VIOLATION: 75
    TRIPLE_RIDING: 60
    NO_HELMET: 55
    SPEEDING: 50
```

---

## 🛡️ Git Restore Point & Preservation

The original baseline state has been preserved. To review or revert:

- **Baseline Tag**: `v0.1-baseline`
- **Baseline Branch**: `baseline-original`
- **Revert Command**:
  ```bash
  git checkout baseline-original
  ```

---

## 📦 Dataset Download Setup (Task 1 Reference)

The Dataset Acquisition Plan PDF designates the following approved datasets:

| Dataset | Platform | Exact PDF URL |
| --- | --- | --- |
| UVH-26 (IISc AIM) | Hugging Face | https://huggingface.co/datasets/iisc-aim/UVH-26 |
| Indian Traffic Helmet Violation | Hugging Face | https://huggingface.co/datasets/thundarstrom/traffic-helmet-violation |
| Indian Vehicle License Plate Dataset | Kaggle | https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset |
| IDD-D YOLOv11 | Kaggle | https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11 |

Full notes and verification status: `datasets/APPROVED_DATASETS.md`.

### Download Commands

```bash
# List approved datasets
python scripts/download_datasets.py --list

# Download specific dataset
python scripts/download_datasets.py --only helmet_india
python scripts/download_datasets.py --only uvh26
python scripts/download_datasets.py --only plates_india idd
```
