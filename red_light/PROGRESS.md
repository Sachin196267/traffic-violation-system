# Red-light violation + ambulance exemption — progress

Task (user, 2026-09-27): detect traffic-signal state and red-light jumping; detect ambulances so they are never
challaned; train on Kaggle/Colab. Standalone like the helmet / triple tools — **do not wire into the website**.
Precision first: no wrong challans; "NOT SURE" is fine. Every decision carries a confidence + reasons.
Estimate given to user: 8–11 h.

## Layout
- `red_light/observe.py` — pass 1 per video (tracks, signal heads via IDD + COCO detectors + colour pre-pass,
  head crops, ambulance crops) -> `red_light/runs/<video>/obs.json, heads.npz, amb_crops/`.
- `red_light/decide.py` — pass 2 (HMM signal state, controlled flow + auto stop line, crossings, decisions,
  `--render` video). Camera config json optional (stop_line / direction / light).
- `signal_training/` — light-state data; Kaggle job `kaggle_jobs/tl_state_train` (id sachin196267/tvs-tl-state,
  CPU because GPU queue is stuck) trains red/yellow/green/not_light on LISA + TVD crops, outputs weights +
  tl_crops.zip. Target weights path: `runs/signal_state/weights/last.pt`.
- `ambulance_training/` — datasets in `datasets/external/*` (ganeshmohane Indian emergency vehicles: labels are
  WRONG — "others" are mostly ambulances, some "ambulance" are police; dedupe by flip-invariant dHash);
  Kaggle job `kaggle_jobs/amb_clip_cpu` (sachin196267/tvs-amb-clip-cpu) = CLIP pre-sort of those images.
  `extract_crops.py` + `run_extract.sh` -> `ambulance_training/crops/<video>/` (CLIP-scored vehicle crops from
  `datasets/ambulance_videos/amb-*` and Indian negatives). Plan: CLIP ViT-B/16 linear probe (clip_features.py)
  vs yolo11s-cls; pick on held-out videos. Target: `runs/ambulance_cls/` + operating_point.json.
- Videos: `datasets/signal_videos/sig-*` (skip 17-20, 24: talk/slides), `datasets/ambulance_videos/amb-*`.

## Steps
- [x] datasets downloaded; signal + ambulance clips downloaded
- [x] observe.py / decide.py written; colour pre-pass (red+green alternating or red lamp in phases)
- [x] light-state model from Kaggle -> runs/signal_state (LISA/TVD held-out 99 %); Indian crops labelled
      (signal_training/head_labels.json, 216): reds 26/26 but many non-lights called red -> fine-tune
      signal_training/finetune.py (A/B held-out clip groups) -> runs/signal_state_ft, installed by autopilot2
- [x] emergency probe: ambulance_training/train_probe.py -> runs/ambulance_cls (held-out videos: 68/105 exempt,
      0/2563 wrong exempt; 95/105 reach review). Labels ambulance_training/labels.json (294 tracks).
- [x] echallan/run.py written (helmet + triple + red light + calibrated speed, emergency exemption, evidence)
- [ ] USER SCOPE (2026-09-27): exempt ambulance + police + fire; build BOTH red-light and combined e-challan tool,
      total within ~10 h; automate everything that doesn't need manual review.
- [ ] run decide.py on all clips, grade by eye (ground truth json), tune; render demo videos
- [ ] REPORT.md in red_light/, status to user

## Autopilot 2 (current): `echallan/autopilot2.sh` -> echallan/AUTOPILOT2_STATUS.md
Waits for fine-tune, installs it, runs decide.py --render on all sig clips, runs echallan/run.py on 6 test clips.
If not running and no "AUTOPILOT2 DONE": `nohup caffeinate -i echallan/autopilot2.sh &`.

## Autopilot 1 (stopped, superseded) (started 2026-09-27 15:50)
`red_light/autopilot.sh` (nohup + caffeinate): waits for Kaggle tvs-tl-state -> runs/signal_state/weights/last.pt,
downloads CLIP web pre-sort -> ambulance_training/clip_web/, waits for/restarts observe_all.sh + run_extract.sh,
writes red_light/logs/ambulance_summary.txt, runs decide.py --render on every clip. Status: red_light/AUTOPILOT_STATUS.md.
If it is not running and has no "AUTOPILOT DONE": `nohup caffeinate -i red_light/autopilot.sh &` (re-run safe).
Ambulance score currently = CLIP zero-shot (t_exempt 0.9, t_review 0.5) until a trained probe exists.
Labels so far: ambulance_training/labels.json (amb-01 done). Next: sheets.py video amb-02..; web sheets.

## Incidents (for the summary)
- yt-dlp script resolved its own path after `cd` -> all downloads failed instantly; fixed.
- zsh did not word-split a file list again -> scan failed; moved to bash scripts.
- Kaggle GPU job sat QUEUED 20+ min (weekly quota likely used) -> switched jobs to Kaggle CPU.
- Disk 98 % full -> user had UVH-26 + UA-DETRAC (26 GB) deleted; 29 GB free now.
- A second session appears to be running triple-rider video tests in parallel (3 run_video.py) — left alone.

## Status 2026-09-27 18:35 (user: focus on signal violation + ambulance/police/fire)
Done: echallan/REPORT.md (both tasks, honest numbers), camera configs red_light/cameras/sig-03|13|21.json,
decide.py per-shot configs + colour-change safeguard + conservative crossing time, emergency probe in both tools,
echallan/grades.json. Helmet pedestrian-as-rider false challans: cause found (rider box union in
helmet_training/robust/run_video.py ~line 190) - NOT fixed (user moved focus), listed in report next steps.
Running: red_light/autopilot3.sh -> red_light/AUTOPILOT3_STATUS.md (new junction clips sig-30..35, police pol-01..04,
fire fire-01..04; crops + label sheets + auto red-light runs). Also echallan speed-path test on datasets/speed/m6/m6_echallan.mp4.
Remaining (manual): label police/fire sheets (ambulance_training/sheets/vids_*.jpg; p=police f=fire), rerun
train_probe.py, camera configs + grading for new junctions, update echallan/REPORT.md.

## Status 2026-09-27 19:25 - DONE (reported to user)
Emergency probe final: 170 emergency tracks (103 amb / 20 police / 47 fire) held out: 98 exempt, 143 exempt-or-review,
0/2450 ordinary wrongly exempted (runs/ambulance_cls). New junctions sig-30..35: no readable signal face (checked by eye).
Red-light real test is still only sig-03 (2/2 correct). Open items: longer fixed-camera junction clip for testing,
helmet pedestrian-as-rider fix, triple evidence box. Report: echallan/REPORT.md.

## Status 2026-09-27 ~20:00 (session limit near)
Extra test long-md355 (Commons, 66 s): signal colour 56/57 graded frames; crossings untestable; 0/109 false emergency.
Running unattended: red_light/autopilot4.sh (YouTube retries every 30 min, observe+decide) + red_light/after4.sh
(e-challan + review sheet per downloaded clip). Mac notification when a clip downloads.
Next manual step: if long-0*.mp4 exist -> camera config (red_light/cameras/<clip>.json) for the approach facing the
signal, decide.py --camera, grade crossings by eye, add numbers to echallan/REPORT.md section 6.
