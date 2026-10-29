# End-to-End Traffic Violation Detection, Digital Twin Visualizer & e-Challan Pipeline

A specialized, enterprise-grade traffic enforcement and 3D digital twin system built to analyze complex, high-density urban Indian intersection footage from Mumbai ([`https://youtu.be/zxxfvP8-lrU`](https://youtu.be/zxxfvP8-lrU)).

---

## 🚀 Key System Features

1. **Video Ingestion & Standardization (`download_video.py`)**
   - Automatically downloads the surveillance footage from YouTube via `yt-dlp` into `raw_input.mp4`.
   - Standardizes the stream to 1280x720 at 30 fps (7,895 total frames, 4m 23s).

2. **Spatial Mapping & 2.5D Digital Twin Visualizer (`digital_twin_engine.py`)**
   - **Imaginary Virtual Lane Corridor**: 4-point ground-plane polygon calibrated to the junction corridor under the flyover where painted lanes are absent.
   - **Planar Homography ($H$)**: Maps camera pixel coordinates into top-down / 2.5D digital twin coordinate space.
   - **Digital Twin Simulation**:
     - Synthetic roadway asphalt with lane dividers, curb boundaries, and virtual stop lines.
     - Real-time mirroring of vehicle positions with class-specific 2.5D avatars (motorcycles, auto-rickshaws, cars, buses, trucks).
     - Fading trajectory motion trails tracking vehicle path history.
     - Animated pulsing red warning halos for active violators.
     - 3-aspect Traffic Signal Head (Green, Yellow, Red) with countdown timer.
     - Side-by-side synchronized composite video output (`processed_traffic_output.mp4`, 1880x720).

3. **Deep Learning Detection & Multi-Violation Logic (`violation_detector.py`)**
   - **Multi-Object Tracking**: ByteTrack coupled with `models/traffic.pt` for persistent vehicle and rider IDs and ground velocity estimation.
   - **No-Helmet Violation**: Two-wheeler detections are cropped with vertical head padding and classified using the dedicated 92.7% mAP50 helmet model (`runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt`).
   - **Triple Riding Violation**: Detects and clusters human instances along the two-wheeler chassis, flagging when $\ge 3$ occupants are present.
   - **Signal Light Violation**: Junction traffic light cycle state machine; flags vehicles crossing the virtual stop line into the intersection during the RED phase.
   - **ANPR Engine (`anpr_engine.py`)**: License plate detection (`models/license_plate_best.pt`), image preprocessing (bilateral filter + CLAHE + Otsu adaptive thresholding), and EasyOCR alphanumeric extraction.

4. **Explainable AI (XAI) & Grad-CAM Heatmaps (`xai_and_gradcam.py`)**
   - **Grad-CAM Visual Attention Maps**: Computes 2D spatial feature activation heatmaps (Jet colormap) indicating the focal region (rider head for helmet, rider cluster for triple riding, stop line for red light).
   - **XAI Reasoning Matrix**: Clear separation of neural model confidence from statutory Motor Vehicles Act criteria.
   - **Severity & Intensity Index**: 0–100 dynamic hazard score categorizing infractions into `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`.

5. **Court-Admissible e-Challan PDFs (`echallan_pdf_generator.py`)**
   - Single-page ReportLab PDF notices stored in `traffic_violation/final_video_testing/challans/`.
   - Includes 4-panel visual evidence (Scene context, Vehicle/rider crop, ANPR plate crop, Grad-CAM attention heatmap), XAI reasoning matrix, severity assessment, statutory penalty, and digital signature block.

---

## 📁 Directory Structure

```
traffic_violation/final_video_testing/
├── raw_input.mp4                     # Ingested 1280x720 source video (7,895 frames)
├── processed_traffic_output.mp4      # Rendered video (Real video + Digital Twin side-by-side)
├── config.yaml                       # Master configuration file (geometry, signal, fines, models)
├── download_video.py                 # Automated YouTube ingestion script
├── calibrate_roi.py                  # Imaginary lane & stop line calibration tool
├── digital_twin_engine.py            # 2.5D Digital Twin renderer & homography engine
├── detection_and_tracking.py         # YOLO multi-class detector + ByteTrack associator
├── violation_detector.py             # Multi-violation detection & temporal hysteresis logic
├── anpr_engine.py                    # License plate detector + image preprocessing + EasyOCR
├── xai_and_gradcam.py                # Grad-CAM attention maps & XAI reasoning matrix
├── echallan_pdf_generator.py         # Court-admissible ReportLab PDF generator
├── run_pipeline.py                   # Master end-to-end execution pipeline
├── challans/                         # Generated e-Challan PDF notices
│   ├── MH-TVS-*.pdf
└── evidence/                         # Visual evidence snapshots, crops, and Grad-CAM maps
    ├── *_scene.jpg
    ├── *_vehicle.jpg
    ├── *_plate.jpg
    └── *_gradcam.jpg
```

---

## 🛠️ Setup & Model Weights

### 1. Dependencies

```bash
source .venv/bin/activate
pip install opencv-python ultralytics reportlab easyocr pyyaml tqdm
```

### 2. Model Weights Setup

The pipeline utilizes the following trained weights from the parent project:
- **Primary Multi-Class Traffic Detector**: `models/traffic.pt` (classes include `motorcycle`, `person`, `rider`, `car`, `autorickshaw`, `bus`, `truck`, `traffic light`).
- **Dedicated Helmet Classifier**: `runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt` (92.7% mAP50 for `helmet` and `no_helmet`).
- **License Plate Detector**: `models/license_plate_best.pt`.

---

## 🚦 Execution Commands

### 1. Download & Prepare Source Video

```bash
python traffic_violation/final_video_testing/download_video.py
```

### 2. Preview & Verify Virtual Lane Calibration

```bash
python traffic_violation/final_video_testing/calibrate_roi.py
# View generated preview at: traffic_violation/final_video_testing/roi_preview.jpg
```

### 3. Run Master End-to-End Pipeline

```bash
# Fast verification run (150 frames):
python traffic_violation/final_video_testing/run_pipeline.py --max_frames 150

# Red light violation evaluation (starting frame 620 during Red signal phase):
python traffic_violation/final_video_testing/run_pipeline.py --start_frame 620 --max_frames 60

# Full video processing (entire 7,895 frames):
python traffic_violation/final_video_testing/run_pipeline.py --max_frames -1
```

---

## ⚖️ Statutory Penalties & Motor Vehicles Act Reference

| Violation Category | Statute Reference | Statutory Penalty | Baseline Severity |
|:---|:---|:---:|:---:|
| **NO_HELMET** | Motor Vehicles Act Section 129 / Rule 138 | **INR 1,000** | 55 / 100 (MEDIUM) |
| **TRIPLE_RIDING** | Motor Vehicles Act Section 128 / Rule 123 | **INR 1,000** | 60 / 100 (HIGH) |
| **SIGNAL_LIGHT_VIOLATION** | Motor Vehicles Act Section 184 / Section 119 | **INR 5,000** | 75 / 100 (HIGH) |

*Compound Policy*: `highest_plus_half` (Maximum fine + 50% of each concurrent violation).
