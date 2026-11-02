"""
modules/scene_analyzer.py

Scene Understanding, Road Region Estimation, and Physical Divider Detection.

Changes vs previous version:
- ROI top/bottom ratios are read from config (perspective.road_roi_*_ratio) so the
  corridors, the bird's-eye transform and the OUTSIDE guard all agree.
- Corridor direction requires a consensus (scene.direction_consensus, default 0.80).
  Otherwise the corridor is labelled "mixed" and wrong-way judgement is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .tracker_pipeline import TrackSummary, TrackObservation


@dataclass
class RoadCorridor:
    corridor_id: int
    name: str
    dominant_direction: str  # "inbound", "outbound" or "mixed"
    x_left_bottom: float
    x_right_bottom: float
    x_left_top: float
    x_right_top: float
    is_primary: bool = True
    active_vehicle_count: int = 0
    polygon: Optional[np.ndarray] = None


@dataclass
class DividerBoundary:
    divider_id: int
    x_left_bot: float
    x_right_bot: float
    x_left_top: float
    x_right_top: float
    polygon: Optional[np.ndarray] = None


class SceneAnalyzer:
    def __init__(self, config: Dict[str, Any], frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h

        scene_cfg = config.get("scene", {})
        persp_cfg = config.get("perspective", {})
        self.close_kernel_size = scene_cfg.get("occupancy_close_kernel", 31)
        self.min_corridor_w = scene_cfg.get("min_corridor_width_px", 120)
        self.divider_min_gap = scene_cfg.get("divider_min_gap_px", 30)
        self.curb_margin = scene_cfg.get("curb_exclusion_margin_px", 15)
        self.direction_consensus = float(scene_cfg.get("direction_consensus", 0.80))

        # Shared ROI with the perspective transformer
        self.roi_top_ratio = float(persp_cfg.get("road_roi_top_ratio", 0.38))
        self.roi_bot_ratio = float(persp_cfg.get("road_roi_bottom_ratio", 0.98))

        self.occupancy_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        self.pedestrian_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        self.corridors: List[RoadCorridor] = []
        self.dividers: List[DividerBoundary] = []

    # ------------------------------------------------------------------
    def _dominant_direction(self, inbound_cnt: int, outbound_cnt: int) -> str:
        total = inbound_cnt + outbound_cnt
        if total == 0:
            return "inbound"
        ratio = max(inbound_cnt, outbound_cnt) / total
        if ratio < self.direction_consensus:
            return "mixed"
        return "inbound" if inbound_cnt >= outbound_cnt else "outbound"

    def accumulate_occupancy(
        self, vehicles: List[TrackObservation], pedestrians: List[TrackObservation]
    ):
        """Accumulates vehicle and pedestrian occupancy masks across frames."""
        for v in vehicles:
            x1, y1, x2, y2 = [int(coord) for coord in v.bbox]
            x1 = max(0, min(self.frame_w - 1, x1))
            x2 = max(0, min(self.frame_w - 1, x2))
            y1 = max(0, min(self.frame_h - 1, y1))
            y2 = max(0, min(self.frame_h - 1, y2))
            if x2 > x1 and y2 > y1:
                self.occupancy_mask[y1:y2, x1:x2] = np.minimum(
                    255, self.occupancy_mask[y1:y2, x1:x2].astype(np.int32) + 10
                ).astype(np.uint8)

        for p in pedestrians:
            x1, y1, x2, y2 = [int(coord) for coord in p.bbox]
            x1 = max(0, min(self.frame_w - 1, x1))
            x2 = max(0, min(self.frame_w - 1, x2))
            y1 = max(0, min(self.frame_h - 1, y1))
            y2 = max(0, min(self.frame_h - 1, y2))
            if x2 > x1 and y2 > y1:
                self.pedestrian_mask[y1:y2, x1:x2] = np.minimum(
                    255, self.pedestrian_mask[y1:y2, x1:x2].astype(np.int32) + 15
                ).astype(np.uint8)

    def analyze_corridors_and_dividers(
        self,
        moving_tracks: List[TrackSummary],
        vp: Tuple[float, float],
        roi_top_ratio: Optional[float] = None,
        roi_bot_ratio: Optional[float] = None,
        transformer: Optional[Any] = None,
    ) -> Tuple[List[RoadCorridor], List[DividerBoundary]]:
        """
        Infers drivable corridors and non-drivable dividers/medians from vehicle motion density.
        ROI ratios default to the config values.
        """
        top_r = self.roi_top_ratio if roi_top_ratio is None else roi_top_ratio
        bot_r = self.roi_bot_ratio if roi_bot_ratio is None else roi_bot_ratio
        y_top = self.frame_h * top_r
        y_bot = self.frame_h * bot_r

        if not moving_tracks:
            corr_main = RoadCorridor(
                corridor_id=0,
                name="Main Carriageway",
                dominant_direction="inbound",
                x_left_bottom=120.0,
                x_right_bottom=1352.0,
                x_left_top=600.0,
                x_right_top=1128.0,
                is_primary=True,
                active_vehicle_count=0,
            )
            div_main = DividerBoundary(
                divider_id=0,
                x_left_bot=1352.0,
                x_right_bot=1632.0,
                x_left_top=1128.0,
                x_right_top=1248.0,
            )
            corr_sec = RoadCorridor(
                corridor_id=1,
                name="Secondary Carriageway",
                dominant_direction="inbound",
                x_left_bottom=1632.0,
                x_right_bottom=1850.0,
                x_left_top=1248.0,
                x_right_top=1350.0,
                is_primary=False,
                active_vehicle_count=0,
            )
            self.corridors = [corr_main, corr_sec]
            self.dividers = [div_main]
            return self.corridors, self.dividers

        # ---------------- Angular vanishing-point corridor discovery ----------------
        vp_x, vp_y = vp
        if 0 <= vp_y <= self.frame_h * 0.75 and len(moving_tracks) >= 3:
            angle_records = []
            for t in moving_tracks:
                h = t.history_bottom_centers
                if len(h) < 2:
                    continue
                dir_sign = 1 if (h[-1][1] > h[0][1]) else -1
                for pt in h[::2]:
                    dx = pt[0] - vp_x
                    dy = pt[1] - vp_y
                    if dy > 15.0:
                        deg = np.degrees(np.arctan2(dy, dx))
                        if 1.0 <= deg <= 179.0:
                            angle_records.append((deg, dir_sign))

            if len(angle_records) >= 8:
                angles_arr = np.array([r[0] for r in angle_records])
                n_bins = 180
                hist, bin_edges = np.histogram(angles_arr, bins=n_bins, range=(0, 180))
                kernel = cv2.getGaussianKernel(7, 2.0).flatten()
                smoothed = np.convolve(hist.astype(float), kernel, mode="same")
                max_val = np.max(smoothed) if len(smoothed) > 0 else 1.0
                active_bins = smoothed > (0.06 * max_val)

                ang_segments = []
                in_seg = False
                start_idx = 0
                for i, act in enumerate(active_bins):
                    if act and not in_seg:
                        start_idx = i
                        in_seg = True
                    elif not act and in_seg:
                        in_seg = False
                        w = bin_edges[i] - bin_edges[start_idx]
                        count = int(np.sum(hist[start_idx:i]))
                        if w >= 2.0 and count >= 3:
                            ang_segments.append((bin_edges[start_idx], bin_edges[i], count))
                if in_seg:
                    w = bin_edges[-1] - bin_edges[start_idx]
                    count = int(np.sum(hist[start_idx:]))
                    if w >= 2.0 and count >= 3:
                        ang_segments.append((bin_edges[start_idx], bin_edges[-1], count))

                if ang_segments:
                    dividers: List[DividerBoundary] = []
                    for d_idx in range(len(ang_segments) - 1):
                        d_th_right = ang_segments[d_idx][1]
                        d_th_left = ang_segments[d_idx + 1][0]
                        if (d_th_left - d_th_right) >= 1.5:
                            tan_l = np.tan(np.radians(d_th_left))
                            tan_r = np.tan(np.radians(d_th_right))
                            if abs(tan_l) < 1e-3 or abs(tan_r) < 1e-3:
                                continue
                            x_lb = float(np.clip(vp_x + (y_bot - vp_y) / tan_l, 0, self.frame_w))
                            x_rb = float(np.clip(vp_x + (y_bot - vp_y) / tan_r, 0, self.frame_w))
                            x_lt = float(np.clip(vp_x + (y_top - vp_y) / tan_l, 0, self.frame_w))
                            x_rt = float(np.clip(vp_x + (y_top - vp_y) / tan_r, 0, self.frame_w))
                            poly = np.array(
                                [[x_lb, y_bot], [x_rb, y_bot], [x_rt, y_top], [x_lt, y_top]],
                                dtype=np.int32,
                            )
                            dividers.append(
                                DividerBoundary(
                                    divider_id=d_idx,
                                    x_left_bot=x_lb,
                                    x_right_bot=x_rb,
                                    x_left_top=x_lt,
                                    x_right_top=x_rt,
                                    polygon=poly,
                                )
                            )

                    primary_idx = max(range(len(ang_segments)), key=lambda i: ang_segments[i][2])
                    corridors: List[RoadCorridor] = []
                    for idx, (th_min_ang, th_max_ang, count) in enumerate(ang_segments):
                        tan_left = np.tan(np.radians(th_max_ang))
                        tan_right = np.tan(np.radians(th_min_ang))
                        if abs(tan_left) < 1e-3 or abs(tan_right) < 1e-3:
                            continue
                        x_lb = float(np.clip(vp_x + (y_bot - vp_y) / tan_left, 0, self.frame_w))
                        x_rb = float(np.clip(vp_x + (y_bot - vp_y) / tan_right, 0, self.frame_w))
                        x_lt = float(np.clip(vp_x + (y_top - vp_y) / tan_left, 0, self.frame_w))
                        x_rt = float(np.clip(vp_x + (y_top - vp_y) / tan_right, 0, self.frame_w))

                        sector_dirs = [
                            r[1] for r in angle_records if th_min_ang <= r[0] <= th_max_ang
                        ]
                        inb_count = sum(1 for d in sector_dirs if d > 0)
                        out_count = len(sector_dirs) - inb_count
                        dom_dir = self._dominant_direction(inb_count, out_count)

                        is_primary = idx == primary_idx
                        name = "Main Carriageway" if is_primary else f"Secondary Carriageway {idx}"
                        poly = np.array(
                            [[x_lb, y_bot], [x_rb, y_bot], [x_rt, y_top], [x_lt, y_top]],
                            dtype=np.int32,
                        )
                        corridors.append(
                            RoadCorridor(
                                corridor_id=idx,
                                name=name,
                                dominant_direction=dom_dir,
                                x_left_bottom=x_lb,
                                x_right_bottom=x_rb,
                                x_left_top=x_lt,
                                x_right_top=x_rt,
                                is_primary=is_primary,
                                active_vehicle_count=count,
                                polygon=poly,
                            )
                        )
                    corridors.sort(key=lambda c: not c.is_primary)
                    for c_idx, c in enumerate(corridors):
                        c.corridor_id = c_idx
                    self.corridors = corridors
                    self.dividers = dividers
                    return self.corridors, self.dividers

        # ---------------- Bottom-projection fallback ----------------
        scale_top = 0.30
        vp_x = vp[0] if (0 <= vp[0] <= self.frame_w) else self.frame_w * 0.42

        bot_xs = []
        directions = []
        for track in moving_tracks:
            pts = track.history_bottom_centers
            if len(pts) < 2:
                continue
            x_last, y_last = pts[-1]
            fraction = np.clip((y_last - y_top) / max(1.0, y_bot - y_top), 0.0, 1.0)
            scale = scale_top + fraction * (1.0 - scale_top)
            x_proj = vp_x + (x_last - vp_x) / scale
            bot_xs.append(float(np.clip(x_proj, 0, self.frame_w - 1)))
            directions.append(track.direction)

        if not bot_xs:
            corr_main = RoadCorridor(
                corridor_id=0,
                name="Main Carriageway",
                dominant_direction="inbound",
                x_left_bottom=self.frame_w * 0.065,
                x_right_bottom=self.frame_w * 0.69,
                x_left_top=self.frame_w * 0.14,
                x_right_top=self.frame_w * 0.32,
                is_primary=True,
                active_vehicle_count=0,
            )
            div_main = DividerBoundary(
                divider_id=0,
                x_left_bot=self.frame_w * 0.69,
                x_right_bot=self.frame_w * 0.74,
                x_left_top=self.frame_w * 0.32,
                x_right_top=self.frame_w * 0.35,
            )
            corr_sec = RoadCorridor(
                corridor_id=1,
                name="Secondary Carriageway",
                dominant_direction="inbound",
                x_left_bottom=self.frame_w * 0.74,
                x_right_bottom=self.frame_w * 0.99,
                x_left_top=self.frame_w * 0.35,
                x_right_top=self.frame_w * 0.44,
                is_primary=False,
                active_vehicle_count=0,
            )
            self.corridors = [corr_main, corr_sec]
            self.dividers = [div_main]
            return self.corridors, self.dividers

        n_bins = 150
        hist, bin_edges = np.histogram(bot_xs, bins=n_bins, range=(0, self.frame_w))
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(hist.astype(float), kernel, mode="same")
        max_val = np.max(smoothed) if len(smoothed) > 0 else 1.0
        if max_val == 0:
            max_val = 1.0

        active_bins = smoothed > (0.04 * max_val)

        segments = []
        in_segment = False
        start_idx = 0
        for i, is_act in enumerate(active_bins):
            if is_act and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_act and in_segment:
                in_segment = False
                w = bin_edges[i] - bin_edges[start_idx]
                count = int(np.sum(hist[start_idx:i]))
                if w >= self.min_corridor_w and count >= 3:
                    segments.append((bin_edges[start_idx], bin_edges[i], count))
        if in_segment:
            w = bin_edges[-1] - bin_edges[start_idx]
            count = int(np.sum(hist[start_idx:]))
            if w >= self.min_corridor_w and count >= 3:
                segments.append((bin_edges[start_idx], bin_edges[-1], count))

        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                gap = seg[0] - merged[-1][1]
                if gap < self.divider_min_gap:
                    merged[-1] = (merged[-1][0], seg[1], merged[-1][2] + seg[2])
                else:
                    merged.append(seg)

        if not merged:
            merged = [(float(np.min(bot_xs)), float(np.max(bot_xs)), len(bot_xs))]

        if vp[1] < y_top - 60 and 0 <= vp[0] <= self.frame_w:
            scale_top = max(0.25, min(0.38, (y_top - vp[1]) / (y_bot - vp[1])))
            vp_x = vp[0]
        else:
            scale_top = 0.30
            vp_x = self.frame_w * 0.42

        dividers: List[DividerBoundary] = []
        for idx in range(len(merged) - 1):
            x_div_left = merged[idx][1]
            x_div_right = merged[idx + 1][0]
            if (x_div_right - x_div_left) >= self.divider_min_gap:
                x_tl = vp_x + (x_div_left - vp_x) * scale_top
                x_tr = vp_x + (x_div_right - vp_x) * scale_top
                poly = np.array(
                    [[x_div_left, y_bot], [x_div_right, y_bot], [x_tr, y_top], [x_tl, y_top]],
                    dtype=np.int32,
                )
                dividers.append(
                    DividerBoundary(
                        divider_id=idx,
                        x_left_bot=x_div_left,
                        x_right_bot=x_div_right,
                        x_left_top=x_tl,
                        x_right_top=x_tr,
                        polygon=poly,
                    )
                )

        corridors: List[RoadCorridor] = []
        primary_idx = max(range(len(merged)), key=lambda i: merged[i][2])

        for idx, (x_lo, x_hi, count) in enumerate(merged):
            x_tl = vp_x + (x_lo - vp_x) * scale_top
            x_tr = vp_x + (x_hi - vp_x) * scale_top

            corr_dirs = [d for x, d in zip(bot_xs, directions) if x_lo <= x <= x_hi]
            inbound_cnt = sum(1 for d in corr_dirs if d == "inbound")
            outbound_cnt = sum(1 for d in corr_dirs if d == "outbound")
            dom_dir = self._dominant_direction(inbound_cnt, outbound_cnt)

            is_primary = idx == primary_idx
            name = "Main Carriageway" if is_primary else f"Carriageway {idx + 1}"

            poly = np.array(
                [[x_lo, y_bot], [x_hi, y_bot], [x_tr, y_top], [x_tl, y_top]],
                dtype=np.int32,
            )

            corridors.append(
                RoadCorridor(
                    corridor_id=idx,
                    name=name,
                    dominant_direction=dom_dir,
                    x_left_bottom=x_lo,
                    x_right_bottom=x_hi,
                    x_left_top=x_tl,
                    x_right_top=x_tr,
                    is_primary=is_primary,
                    active_vehicle_count=count,
                    polygon=poly,
                )
            )

        self.corridors = corridors
        self.dividers = dividers
        return self.corridors, self.dividers