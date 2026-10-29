#!/usr/bin/env python3
"""
Video Ingestion Script for Traffic Violation & Digital Twin Pipeline.
Downloads and standardizes YouTube video stream.
"""
import os
import sys
import subprocess
import argparse
import cv2

DEFAULT_URL = "https://youtu.be/zxxfvP8-lrU"
DEFAULT_OUTPUT = "traffic_violation/final_video_testing/raw_input.mp4"

def download_video(url: str, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[Ingestion] Fetching stream from: {url}")
    print(f"[Ingestion] Target destination: {output_path}")

    # Use yt-dlp binary if available
    yt_dlp_candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.14/bin/yt-dlp",
        "yt-dlp"
    ]
    yt_binary = None
    for c in yt_dlp_candidates:
        if os.path.exists(c) or subprocess.run(["which", c], capture_output=True).returncode == 0:
            yt_binary = c
            break

    if not yt_binary:
        raise RuntimeError("yt-dlp binary not found. Please install yt-dlp.")

    cmd = [
        yt_binary,
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--force-overwrites",
        url
    ]

    print(f"[Ingestion] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        # Fallback to single format without merge
        print("[Ingestion] Retrying with best single mp4 stream...")
        fallback_cmd = [
            yt_binary,
            "-f", "best[ext=mp4]/best",
            "-o", output_path,
            "--force-overwrites",
            url
        ]
        res2 = subprocess.run(fallback_cmd)
        if res2.returncode != 0:
            raise RuntimeError(f"Failed to download video with code {res2.returncode}")

    # Validate downloaded video
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Expected video at {output_path} not found.")

    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open downloaded video at {output_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[Ingestion] SUCCESS! Video validated: {w}x{h} @ {fps:.1f} fps, {total_frames} frames ({total_frames/fps:.1f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTube video URL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output file path")
    args = parser.parse_args()
    download_video(args.url, args.output)
