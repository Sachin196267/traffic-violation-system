"""
Background job queue — ThreadPoolExecutor-based worker pool.
"""
from __future__ import annotations
import os
import sys
import uuid
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from pipeline_runner import run_job

# Two concurrent pipeline workers max (each is CPU/GPU heavy)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline-worker")
_url_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="url-download-worker")

# Track futures: {job_id: Future}
_futures: dict[str, object] = {}


def submit_job(job_id: str, video_path: str, max_frames: int, start_frame: int) -> str:
    """Submit a pipeline job to the worker pool. Returns job_id."""
    future = _executor.submit(run_job, job_id, video_path, max_frames, start_frame)
    _futures[job_id] = future
    return job_id


def submit_url_job(job_id: str, url: str, video_path: str, max_frames: int, start_frame: int) -> str:
    """Download video in background via yt-dlp or direct fetch then queue for processing."""
    from database import update_job_status

    def _worker():
        try:
            update_job_status(job_id, "downloading", progress=5)
            job_dir = os.path.dirname(video_path)

            yt_dlp_cmd = [
                sys.executable, "-m", "yt_dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
                "-o", video_path,
                "--force-overwrites",
                "--no-playlist",
                "--merge-output-format", "mp4",
                "--no-check-certificates",
                url,
            ]
            res = subprocess.run(yt_dlp_cmd, capture_output=True, text=True, timeout=300)

            # Fallback download attempt with broadest format matching
            if (not os.path.exists(video_path) or os.path.getsize(video_path) == 0):
                fb_cmd = [
                    sys.executable, "-m", "yt_dlp",
                    "-f", "b/best",
                    "-o", os.path.join(job_dir, "downloaded.%(ext)s"),
                    "--force-overwrites",
                    "--no-playlist",
                    "--no-check-certificates",
                    url,
                ]
                subprocess.run(fb_cmd, capture_output=True, text=True, timeout=300)

                # Look for downloaded file
                for fname in os.listdir(job_dir):
                    if fname.startswith("downloaded."):
                        src_f = os.path.join(job_dir, fname)
                        if os.path.getsize(src_f) > 0:
                            if fname.endswith(".mp4"):
                                shutil.move(src_f, video_path)
                            else:
                                from pipeline_runner import transcode_to_h264
                                transcode_to_h264(src_f, video_path)
                                try:
                                    os.remove(src_f)
                                except Exception:
                                    pass
                            break

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                err_msg = res.stderr.strip() if res.stderr else "Failed to download video from URL"
                # Keep error message concise for UI
                short_err = err_msg.split("\n")[-1] if err_msg else "Download failed"
                update_job_status(job_id, "failed", error_message=f"Download error: {short_err}")
                return

            update_job_status(job_id, "processing", progress=10)
            run_job(job_id, video_path, max_frames, start_frame)
        except Exception as err:
            update_job_status(job_id, "failed", error_message=str(err))

    future = _url_executor.submit(_worker)
    _futures[job_id] = future
    return job_id


def generate_job_id() -> str:
    return str(uuid.uuid4())[:12]


def get_job_future(job_id: str):
    return _futures.get(job_id)
