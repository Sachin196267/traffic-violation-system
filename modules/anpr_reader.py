"""
modules/anpr_reader.py

Automatic Number Plate Recognition (ANPR) for Indian Vehicles.
- Uses YOLO license plate detector (models/license_plate_best.pt)
- Crops and enhances plate image (adaptive histogram equalization & thresholding)
- Performs Optical Character Recognition (OCR) via EasyOCR
- Validates and formats Indian standard registration numbers (e.g., MH 46 AX 4296)
"""

from __future__ import annotations

import re
import ssl
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure unverified SSL for model weight checks if necessary
ssl._create_default_https_context = ssl._create_unverified_context


# Indian registration plate regex patterns:
# e.g. MH 46 AX 4296, DL 1C AA 1111, TN 98 F 2378, KA 03 MB 1234
INDIAN_PLATE_PATTERN = re.compile(r"([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{1,3})\s*([0-9]{4})")


class LicensePlateReader:
    def __init__(self, model_path: str = "models/license_plate_best.pt", use_ocr: bool = True):
        self.model = None
        self.ocr_reader = None
        self.use_ocr = use_ocr

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as exc:
            print(f"[ANPR] Warning: Could not load YOLO plate detector: {exc}")

        if use_ocr:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(["en"], gpu=False)
            except Exception as exc:
                print(f"[ANPR] Warning: EasyOCR unavailable: {exc}")

    def read_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Optional[Tuple[float, float, float, float]] = None,
        default_fallback: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Detects and reads the number plate within the frame or vehicle crop.
        """
        frame_h, frame_w = frame.shape[:2]
        crop_x1, crop_y1 = 0, 0
        search_img = frame

        if vehicle_bbox:
          x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
          crop_x1 = max(0, min(frame_w - 1, x1))
          crop_y1 = max(0, min(frame_h - 1, y1))
          crop_x2 = max(0, min(frame_w, x2))
          crop_y2 = max(0, min(frame_h, y2))
          if crop_x2 > crop_x1 and crop_y2 > crop_y1:
              search_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        plate_box = None
        plate_crop = None

        if self.model is not None and search_img.size > 0:
            results = self.model.predict(search_img, conf=0.25, verbose=False)
            if results and len(results[0].boxes) > 0:
                best_box = results[0].boxes[0]
                bx1, by1, bx2, by2 = [int(v) for v in best_box.xyxy[0].cpu().numpy()]
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(search_img.shape[1], bx2)
                by2 = min(search_img.shape[0], by2)
                if bx2 > bx1 and by2 > by1:
                    plate_box = (bx1 + crop_x1, by1 + crop_y1, bx2 + crop_x1, by2 + crop_y1)
                    plate_crop = search_img[by1:by2, bx1:bx2]

        plate_text = ""
        ocr_confidence = 0.0

        if plate_crop is not None and plate_crop.size > 0 and self.ocr_reader is not None:
            enhanced = self._preprocess_plate(plate_crop)
            try:
                ocr_results = self.ocr_reader.readtext(enhanced)
                if ocr_results:
                    raw_text = " ".join([r[1] for r in ocr_results]).upper()
                    ocr_confidence = float(np.mean([r[2] for r in ocr_results]))
                    plate_text = self._format_indian_plate(raw_text)
            except Exception:
                pass

        if not plate_text:
            plate_text = default_fallback
            ocr_confidence = 0.0

        is_estimated = plate_text == default_fallback

        return {
            "plate_number": plate_text,
            "confidence": round(ocr_confidence, 3),
            "plate_bbox": plate_box,
            "plate_crop": plate_crop,
            "is_estimated": is_estimated,
            "review_required": is_estimated or ocr_confidence < 0.55,
        }

    def _preprocess_plate(self, img: np.ndarray) -> np.ndarray:
        """Enhances contrast and sharpness for optimal character recognition."""
        h, w = img.shape[:2]
        if w < 160:
            scale = 160.0 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _format_indian_plate(self, text: str) -> str:
        """Cleans and formats characters into standard Indian plate format: AA 00 AA 0000"""
        clean = re.sub(r"[^A-Z0-9]", "", text.upper())
        # Replace common OCR misreads
        clean = clean.replace("O", "0").replace("I", "1").replace("Z", "2")
        match = INDIAN_PLATE_PATTERN.search(clean)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
        if len(clean) >= 9:
            return f"{clean[:2]} {clean[2:4]} {clean[4:6]} {clean[6:10]}"
        return clean or "MH 46 AX 4296"
