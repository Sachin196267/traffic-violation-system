# Triple-rider v3 — progress checkpoint (read this first when resuming)

Project root: `/Users/sachinnair/Desktop/traffic violation system` (all paths below relative to it).
User rules: standalone only — **never** wire into the website; do not commit unless asked;
"not sure" allowed up to ~1 in 12 bikes; TRIPLE only when sure; every decision must carry a
confidence score + reasons (for explainable AI on a future site). Post a short ✅/⏳/⬜ status list
after each step. Background: `triple_training/REPORT.md` (v2 = 7/43 caught, 0 wrong).

## Steps
- [x] 1a Download 39 enforcement clips -> `datasets/triple_videos/` (tpt-*, hydpol-*, vjw-01, chn-01, misc-*)
- [x] 1b Extract crops: `triple_training/data/<clip>/` (misc-* may still be running: `pgrep -f extract_units.py`)
- [x] 1c Overlay removal: `python3 triple_training/deoverlay.py` -> `crops_clean/` (build_dataset uses it)
- [x] 2a Head-on-seat counter: `triple_training/heads_seat.py` (tested on idd_4)
- [x] 1d LABEL new clips (done; hydpol-05_02..07, misc-05_01, misc-06, misc-01/04/07 skipped - low value). Sheets: `triple_training/sheets/<clip>_NN.jpg` + `<clip>_index.json`.
      Record: `python3 triple_training/label.py <sheet> "<labels in cell order>"`.
      Labels 1/2/3/x/u. Overlay frames: label by CONTENT (overlays are erased in training).
      Hidden third rider not visible in the frame -> `u`, never guess `3`.
      Duplicates tpt-19 (=tpt-05) and tpt-24 (=tpt-22) moved to triple_training/excluded/ - skip them.
      Done so far: tpt-01..tpt-18. Next: tpt-20_00 onwards, then hydpol-*, vjw-01, chn-01, misc-*
      (misc-* need `make_sheets.py --video <v> --b-frac 0.3 --c-frac 0.1` first).
      Check which are done: `ls triple_training/labels/`.
- [x] 2b Run heads_seat.py on all videos (77/77 done).
- [x] 3  (RUNNING in background since 05:04, detached; DONE when triple_training/logs/train_cv_v3.log contains 'final weights' and no train_cv process is running) `python3 triple_training/build_dataset.py` then train v3 locally:
      `train_cv.py --folds 0 1 2 3 4 --epochs 15 --imgsz 256` then `--final` (log: triple_training/logs/train_cv_v3.log).
      Tirupati clips split into 3 camera sites (TPT_SITES in build_dataset.py). NOTE: group all tpt-* clips into ONE fold (same CCTV network) - build_dataset groups by name before '-'.
- [x] 4  (DONE 13:5x: combine.py = classifier + heads logistic combiner, LOFO-fitted; TRIPLE needs score>=t_hi AND >=min_votes frames agreeing; 4-fold result 43/105 caught, 0 wrong, not-sure 8.2%; run_video.py outputs score, confidence, heads, reasons. autopilot re-runs combine.py on all 5 folds.) Combiner + thresholds: classifier p3 + heads-on-seat -> confidence; tune t_hi (0 wrong),
      t_lo so not-sure <= 1/12; write runs/triple_rider/operating_point.json; run_video.py outputs
      confidence + reasons per bike.
- [ ] 5  End-to-end tests with held-out fold models; 6 update REPORT.md; status to user.

## Autopilot (runs unattended)
`triple_training/autopilot.sh` (nohup, started 05:1x) restarts crashed training, then runs thresholds
(not-sure <= 1/12) -> logs/eval_v3.log, and video tests -> triple_training/video_out_v3 + logs/video_tests_v3.log.
Log: triple_training/logs/autopilot.log. When it says AUTOPILOT DONE, only step 4b (head-count combiner:
add heads_seat evidence to the decision + reasons) and step 6 (REPORT.md) remain.
If autopilot.sh is not running and not done: restart it with `nohup triple_training/autopilot.sh &`.

## Resume checklist
1. `pgrep -fl "extract_units|train_cv|heads_seat"` — anything still running? let it finish.
2. Look at the newest log in `triple_training/logs/` for Traceback; fix and re-run that step only.
3. Continue from the first unchecked step above; tick steps here as they finish.


## Incident log
- 13:20 Mac rebooted during fold 4 (epoch 10/15). Resume task restarted autopilot 13:23; train_cv.py --resume continued fold 4 from epoch 10.
- 15:2x user asked: stop remaining autopilot tests, re-test indore/pune/jaipur at --fps 10 (triple_training/video_out_10fps)
- 15:55 user chose Pune only for the 10fps test; Indore/Jaipur 10fps runs stopped
