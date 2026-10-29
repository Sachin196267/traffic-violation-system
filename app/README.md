# Sentinel — AI traffic-violation platform

`./start.sh` → http://127.0.0.1:8600 (sign up → verify → upload a video).

- **server/** FastAPI + SQLite. `ml/detectors.py` wraps the trained tools (helmet, triple riding, red light,
  speed); `ml/plate.py` the number-plate reader; lanes are context only. Add a model = one Detector class.
- **web/** React + TypeScript (Vite). `npm run dev` for development (proxies /api to :8600), `npm run build` for dist/.
- Data: `data/platform.db`, per-analysis folders in `data/analyses/<id>/` (original video kept, evidence + hashes).
- Verification codes: without SMTP settings (`TVP_SMTP_HOST/USER/PASS/FROM`) the app runs in developer mode —
  codes are shown in the UI and written to `data/dev_outbox.log`. SMS needs a provider hook in `security.send_code`.
- Rules built in: multi-frame decisions only; uncertain → human review; extra persons on a bike always reviewed;
  plates printed only when read with certainty (reviewers may type them); emergency vehicles exempt; lanes never
  create a challan; every challan has fingerprinted evidence and an explanation; PDF at /api/challans/<id>/pdf.

## Signal timing (SIMULATED)
Analysis page → "Simulate signal timing" (`/signal/<id>`). Flow and vehicle mix of the filmed approach are measured from the
shared tracking pass (`server/signal_sim.py`, fragments merged with `pipeline.vehicle_chains`); the other approaches are the
user's assumptions. The Unity simulator (`signal_sim/`, headless, on a copy in `data/signal_sim_runner/`) runs Fixed 30 s,
Webster, Adaptive v1 and PRAVAH on identical traffic (3 seeds). Results are labelled SIMULATED and never create findings.
Needs the Unity editor (found under /Applications/Unity/Hub/Editor or `TVP_UNITY`); one run takes ~1-3 min and ~1.6 GB disk for the copy.
