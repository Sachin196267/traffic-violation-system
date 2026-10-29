"""Chapter 4 (Result and Discussion), Chapter 5 (Summary) and back matter."""
import json
import sqlite3

from PIL import Image as PI

from story import *
from engine import FIG, HERE
import engine as E
from refs import REFS

F = FIG
SHOTS = HERE.parent / "shots"
DB = "/Users/sachinnair/Desktop/traffic violation system/app/data/platform.db"
SURVEY_B = "/Users/sachinnair/Downloads/Survey_Paper_B.pdf"


def shot(name):
    return SHOTS / f"{name}.png"


def auth_composite():
    out = F / "s_auth.png"
    a, b = PI.open(shot("login")), PI.open(shot("signup"))
    w, h = a.size
    a, b = a.crop((w // 2, 0, w, h)), b.crop((w // 2, 0, w, h))
    c = PI.new("RGB", (w + 20, h), "white")
    c.paste(a, (0, 0)); c.paste(b, (w // 2 + 20, 0))
    c.save(out)
    return out


def pdf_shot():
    import pymupdf
    out = F / "s_pdf.png"
    d = pymupdf.open(SHOTS / "challan.pdf")
    pix = d[0].get_pixmap(dpi=130)
    pix.save(out)
    return out


def ch4():
    s = chapter(4, "Result and Discussion")
    s += h("4.1", "Sample of Inputs / Dataset / Database Used / Outputs / Screen Shots")
    s += p("This section describes the data that flow through TraffiSense: the videos and options supplied by the "
           "operator, the datasets used to develop the models, the database in which results are stored, the "
           "outputs produced for each finding, and the screens through which these are presented.")
    s += h("4.1.1", "Sample Inputs", 2)
    s += p("The main input is a recorded traffic video (MP4, MOV, AVI, MKV or WEBM, up to 2,048 MB). The dashboard "
           "shows the guidance in Table 4.1, because the quality of the video decides how many cases can be "
           "settled automatically and how many must be reviewed.")
    t, _ = table("Guidance for input videos shown on the dashboard", ["Property", "Recommendation and reason"], [
        ["Fixed camera", "CCTV pole or tripod; hand-held or moving footage breaks vehicle tracking."],
        ["One continuous shot", "No cuts or compilations; every cut restarts tracking and short shots cannot be judged."],
        ["720p or better, sharp", "1080p is ideal; riders' heads should be at least 40 px tall for helmet checks."],
        ["Angle 4–8 m high, looking along the road", "Tilted down 20–45°; avoids occlusion between riders and keeps plates visible."],
        ["Daylight works best", "At night lamps glare and fewer vehicles are detected, so more cases go to review."],
        ["20 seconds or longer", "Each vehicle is judged over many frames, so it needs to stay in view for a few seconds."],
        ["Red light / speed", "Red light is automatic (signal must be visible and seen changing colour); speed needs a calibration and a limit."],
    ], [0.3, 0.7])
    s += t
    s += p("With the video, the operator selects the checks and options listed in Table 4.2. Defaults can be "
           "saved on the settings page.")
    t, _ = table("Options of an analysis", ["Option", "Type", "Description"], [
        ["checks", "List", "Any of helmet, triple, redlight, speed."],
        ["max_sec", "Number", "Analyse only the first N seconds (for example the first 2 minutes)."],
        ["plates", "Boolean", "Read number plates of flagged vehicles (default on)."],
        ["lanes", "Boolean", "Estimate lane structure afterwards (context only, default on)."],
        ["speed_calib, speed_limit", "File, number", "Camera calibration and limit in km/h; speed is skipped without them."],
        ["camera", "File (optional)", "Manual stop line and direction for the red-light check; automatic otherwise."],
    ], [0.27, 0.18, 0.55])
    s += t
    s += h("4.1.2", "Dataset Used", 2)
    s += p("Each model was trained and evaluated on the data in Table 4.3. Wherever possible, scores come from "
           "cameras or whole videos that the model never saw during training (leave-cameras-out or "
           "leave-videos-out cross-validation), and downloaded or staged sets were used for training only. Some "
           "labels for the helmet and triple-riding sets were produced visually with AI assistance and have not "
           "yet been re-checked by a person; this is noted as a limitation.")
    t, _ = table("Datasets used for model development and testing", ["Model", "Data", "Size and split"], [
        ["Vehicle detector", "IDD-Detection (Indian Driving Dataset, YOLO export)", "33,569 train / 4,196 val / 4,197 test images; YOLO11s at 960 px, 35 epochs (10.7 h)"],
        ["Helmet classifier", "28 Indian traffic videos (15+ cities, day/night/rain, CCTV/street/dashcam) + public head crops", "4,212 persons (1,228 helmet, 873 no helmet, 1,265 can't tell, 846 not a head); 5-fold leave-videos-out; 7 unseen test videos (367 riders)"],
        ["Triple-riding classifier", "Police-camera clips, IDD stills, own videos; Kaggle and Roboflow sets for training only", "114 real triple tracks; held-out test of 980 bikes with 125 triples, leave-cameras-out"],
        ["Signal-state classifier", "LISA Traffic Light, traffic-violation dataset, 216 Indian signal crops", "Held-out LISA/TVD clips; Indian crops split in two halves, each tested by a model that never saw it"],
        ["Emergency probe", "Vehicle tracks from Indian clips (ambulance, police, fire) and ordinary traffic", "170 emergency tracks (103 ambulance, 20 police, 47 fire) and 2,450 ordinary tracks held out"],
        ["Plate reader", "Indian vehicle licence-plate dataset (box + text) + degraded copies", "1,696 plates; train 1,381 / test 315, grouped by plate text; 3,636 augmented training crops"],
        ["Speed estimator", "Two reference scenes with known speeds (M6, Autobahn); VS13 clips", "137 vehicles with ground truth; 29 uncalibrated clips"],
    ], [0.19, 0.38, 0.43], small=True)
    s += t
    s += h("4.1.3", "Database Used", 2)
    s += p("All records are stored in one SQLite database (`data/platform.db`) in write-ahead-logging mode, which "
           "allows the pages to read while the worker writes. Evidence files are kept outside the database in one "
           "folder per analysis (`data/analyses/&lt;id&gt;/`), next to the untouched original video. Table 4.4 "
           "lists the tables.")
    t, _ = table("Tables of the TraffiSense database", ["Table", "Main fields", "Purpose"], [
        ["users", "id, name, email, phone, pw_hash, verified, role, created", "Accounts (password stored only as PBKDF2 hash)"],
        ["sessions", "token_hash, user_id, created, expires", "Active sessions (only the HMAC of the token)"],
        ["codes", "target, purpose, code_hash, expires, attempts, used", "Verification and reset codes"],
        ["analyses", "id, user_id, filename, status, stage, progress, options, meta", "One row per uploaded video"],
        ["findings", "detector, violation, track, t, confidence, model_decision, route, status, reasons, uncertain_reasons, evidence, offences, duplicate_of", "One row per detection, with evidence and links"],
        ["reviews", "finding_id, user_id, decision, note, created", "Reviewer decisions"],
        ["challans", "number, finding_id, source, reviewer_id, status, plate, offences, evidence_hash", "Issued challans (unique per case)"],
    ], [0.15, 0.53, 0.32], small=True)
    s += t
    s += h("4.1.4", "Outputs", 2)
    s += p("Each analysis produces the outputs listed in Table 4.5. Together they answer what was detected, why, "
           "how sure the model was, and what happens next.")
    t, _ = table("Outputs of an analysis", ["Output", "Description"], [
        ["Findings", "Violation, vehicle, time, confidence, model decision and route for each detection."],
        ["Reasons", "Plain-language evidence from the model, e.g. “no-helmet probability 0.99, vote over 14 frames”."],
        ["Uncertain reasons", "Why a case went to review, e.g. “seen in only 2 frames” or “just turned red”."],
        ["Evidence", "Frames before, at and after the moment (raw and boxed), crop, original clip, annotated clip, plate crop."],
        ["Plate result", "READ with confidence, NOT SURE with a candidate for the reviewer, UNREADABLE or NOT LINKED."],
        ["Evidence fingerprint", "SHA-256 over all evidence files of the finding."],
        ["Challan", "Number, offences, plate, source (automatic or reviewer), evidence hash; PDF e-challan."],
        ["Annotated videos", "Helmet, triple-riding, signal and lane overlays, playable in the browser."],
        ["Notes", "Checks skipped or not possible (for example no visible signal), duplicates merged."],
    ], [0.24, 0.76])
    s += t
    s += h("4.1.5", "Screen Shots", 2)
    s += p("The screens below show the main pages of the implemented system, captured from the running platform "
           "with the team's own analyses. Each figure is paired with a short description of what the page presents.")
    shots = [
        (auth_composite(), "Sign-in and account creation pages",
         "Figure 4.1 shows the sign-in and account-creation pages. Accounts are created with an e-mail address or phone number, a password that meets the policy and a CAPTCHA, and are verified with a six-digit code."),
        (shot("dashboard"), "Dashboard with video guidance and upload",
         "Figure 4.2 shows the dashboard. It summarises videos analysed, violations found, challans issued and cases waiting for review, explains what makes a good video, and accepts an upload with the chosen checks."),
        (shot("analyses"), "Video analyses",
         "Figure 4.3 shows the list of video analyses with the checks run, status, findings, review cases and challans for each upload."),
        (shot("analysis"), "Analysis result with video, overlays and findings",
         "Figure 4.4 shows the result of one analysis. The video can be played with the helmet, triple-riding and lane overlays; violations are marked on the timeline; the results list shows each challan and review case with its confidence."),
        (shot("violations"), "Violations with decision path",
         "Figure 4.5 shows every finding across analyses with its evidence thumbnail, track, confidence and status, including merged duplicates."),
        (shot("review"), "Human review of an uncertain case",
         "Figure 4.6 shows the human review page. The reviewer sees the evidence frames and clip, the plate field, the model's explanation and the reason it was unsure, and decides to issue a challan, reject or ask for more evidence."),
        (shot("challans"), "Issued challans",
         "Figure 4.7 shows the issued challans with their number, offence, plate, source, confidence and a PDF link."),
        (shot("challan"), "Challan detail with evidence",
         "Figure 4.8 shows the detail of one challan with the evidence frames, the challan details and the evidence fingerprint."),
        (pdf_shot(), "First page of a generated e-challan PDF",
         "Figure 4.9 shows the first page of an e-challan PDF generated by the platform."),
        (shot("evidence"), "Evidence gallery",
         "Figure 4.10 shows the evidence gallery, where every finding keeps its frames, clips, boxes, track, confidence and fingerprint."),
        (shot("twin2"), "Digital twin of the scene",
         "Figure 4.11 shows the digital twin: a 3-D reconstruction of the scene from the shared tracks, with every vehicle and violation on a timeline."),
        (shot("lanes"), "Lane detection from vehicle movement",
         "Figure 4.12 shows the lane detection page. Lanes are estimated from how vehicles move, not from paint, and are used as context only."),
        (shot("analytics"), "Analytics",
         "Figure 4.13 shows the analytics page with violation types, the outcome of every finding, the confidence distribution and findings by model."),
        (shot("system"), "Models and system",
         "Figure 4.14 shows the models used by the installation, their status and how decisions are made."),
        (shot("settings"), "Settings",
         "Figure 4.15 shows the settings page for profile, password, appearance, upload defaults and code delivery."),
    ]
    for path, cap, text in shots:
        s += p(text)
        wd = 250 if "pdf" in str(path) else 380
        f, _ = figure(path, cap, width=wd, max_h=400)
        s += f
    # ---------------------------------------------------------- 4.2
    s += h("4.2", "Evaluation Parameters")
    s += p("TraffiSense is evaluated on four groups of parameters: detection quality, decision quality, measurement "
           "quality and system behaviour. The first measures the vehicle detector, the second measures the "
           "violation checks as enforcement decisions, the third measures speed and signal readings, and the "
           "fourth measures the platform as a working application.")
    s += h("4.2.1", "Detection Metrics", 2)
    s += p("A detection is a true positive (TP) when its box overlaps a ground-truth box of the same class with "
           "IoU ≥ 0.5; unmatched detections are false positives (FP) and unmatched ground-truth boxes are false "
           "negatives (FN). Precision and recall are")
    e, _ = eq(r"\mathrm{Precision} = \frac{TP}{TP+FP}, \qquad \mathrm{Recall} = \frac{TP}{TP+FN}")
    s += e
    s += p("The average precision AP of a class is the area under its precision–recall curve, and the mean "
           "average precision over <i>C</i> classes is")
    e, _ = eq(r"\mathrm{mAP} = \frac{1}{C}\sum_{c=1}^{C} \int_0^1 P_c(R)\, dR")
    s += e
    s += p("mAP@0.5 uses the IoU threshold 0.5; mAP@0.5:0.95 averages it over thresholds from 0.5 to 0.95.")
    s += h("4.2.2", "Decision Metrics", 2)
    s += p("Because TraffiSense may answer “not sure”, each check is measured as an enforcement decision. The "
           "most important number is the <b>wrong-challan rate</b>, the share of confident violation calls that "
           "are wrong; the <b>not-sure rate</b> is the share of cases sent to review; and <b>recall of called "
           "violations</b> is the share of true violations that are confirmed automatically:")
    e, _ = eq(r"\mathrm{WrongRate} = \frac{FP_{call}}{TP_{call}+FP_{call}}, \qquad \mathrm{NotSure} = \frac{N_{review}}{N}, \qquad \mathrm{Caught} = \frac{TP_{call}}{N_{viol}}")
    s += e
    s += p("For decisions with an answer, accuracy among called cases is the share of answers that are correct, and "
           "the F1-score combines precision and recall:")
    e, _ = eq(r"F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}")
    s += e
    s += h("4.2.3", "Measurement Metrics", 2)
    s += p("For the speed estimator, the mean absolute percentage error between the middle estimate "
           "<i>ŷ<sub>i</sub></i> and the true speed <i>y<sub>i</sub></i>, and the coverage of the 90 % range, "
           "are reported:")
    e, _ = eq(r"\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{\hat{y}_i - y_i}{y_i}\right|, \qquad \mathrm{Coverage} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\left[y_i \in [v_{lo,i},\, v_{hi,i}]\right]")
    s += e
    s += p("A well-calibrated range has a coverage close to 90 %. For the signal-state model, the per-class share "
           "of correctly classified lamp crops is reported. For the plate reader, the exact-plate accuracy, the "
           "character accuracy and the share of correct plates among READ decisions are reported.")
    s += p("For signal control in simulation, the main measure is the <b>mean wait per person</b>: the time each "
           "vehicle spends stopped before the stop line, weighted by its occupants <i>p<sub>v</sub></i>, so that a "
           "bus with forty passengers counts more than a two-wheeler with one. The wait per vehicle, the worst wait, "
           "the ambulance wait, red-light jumps per hour, throughput and the vehicles left queued at the end are "
           "reported with it:")
    e, _ = eq(r"\bar{W}_{person} = \frac{\sum_{v} p_v \, W_v}{\sum_{v} p_v}")
    s += e
    s += h("4.2.4", "System and Functional Metrics", 2)
    s += p("At the system level the evaluation covers the functional correctness of each workflow, the number of "
           "findings, review cases, challans and merged duplicates per video, and the processing time compared "
           "with the length of the video (the real-time factor).")
    s += h("4.2.5", "Evaluation Protocol", 2)
    s += p("To make the reported numbers honest and reproducible, the following protocol is applied. Data are split "
           "by <b>camera or by video</b>, never by frame, because frames of the same scene are nearly identical "
           "and would leak into the test. Thresholds of the decision rules are chosen on out-of-fold predictions "
           "with the explicit target of zero (or the fewest possible) wrong calls, and the chosen operating point "
           "is then fixed. End-to-end results are additionally checked on complete videos that were not used for "
           "training or tuning, grading every reported rider or vehicle by eye. Because violations are rare, "
           "accuracy alone would look high even for a useless model, so the wrong-call count, the number caught "
           "and the not-sure rate are always reported together. Signal controllers are always compared on identical "
           "simulated arrivals (the same random seed), and differences are called better or worse only when the 95 % "
           "confidence interval of the paired difference excludes zero.")
    # ---------------------------------------------------------- 4.3
    s += h("4.3", "Performance Evaluation")
    s += p("This section reports the evaluation of each component with the parameters defined in Section 4.2, "
           "followed by platform-level results from the analyses stored in the running system.")
    s += h("4.3.1", "Vehicle Detection", 2)
    d = json.load(open("/Users/sachinnair/Desktop/traffic violation system/runs/idd_final/idd_metrics.json"))["test"]
    rows = []
    for c in ["car", "bus", "truck", "autorickshaw", "motorcycle", "rider", "person", "bicycle", "traffic light"]:
        v = d["per_class"][c]
        rows.append([c.capitalize().replace("Autorickshaw", "Auto-rickshaw"), f"{v['P']:.3f}", f"{v['R']:.3f}", f"{v['mAP50']:.3f}", f"{v['mAP50-95']:.3f}"])
    rows.append(["<b>All 15 classes</b>", f"<b>{d['P']:.3f}</b>", f"<b>{d['R']:.3f}</b>", f"<b>{d['mAP50']:.3f}</b>", f"<b>{d['mAP50-95']:.3f}</b>"])
    s += p("Table 4.6 reports the performance of the IDD-trained YOLO11s detector on the 4,197 held-out test "
           "images. The classes that matter for enforcement — cars, buses, trucks, auto-rickshaws, motorcycles "
           "and riders — reach mAP@0.5 between 0.66 and 0.79 with precision above 0.84. The overall mean is "
           "lowered by rare classes such as animals, caravans, trains and the generic vehicle class.")
    t, _ = table("Performance of the vehicle detector on the IDD test split", ["Class", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
                 rows, [0.28, 0.18, 0.18, 0.18, 0.18])
    s += t
    s += p("Figure 4.16 shows the validation curves during training and Figure 4.17 the per-class scores.")
    f, _ = figure(F / "c_train.png", "Validation precision, recall and mAP of the vehicle detector during training", width=420)
    s += f
    f, _ = figure(F / "c_class.png", "Per-class precision, recall and mAP@0.5 on the IDD test split", width=430)
    s += f
    s += p("Recall below 0.7 means that some vehicles are missed in a single frame. Tracking compensates for much "
           "of this, because a vehicle needs to be detected only in some of the frames in which it is visible to "
           "keep its track.")
    s += h("4.3.2", "Helmet Check", 2)
    s += p("The helmet check was evaluated in two ways. Cross-validation on the 24 usable training videos (each "
           "video scored by a model that never saw it) measures the classifier; the seven unseen videos (Varanasi, "
           "Bengaluru, Hyderabad, Kochi, Mysuru, Bhopal and a Hyderabad police clip) measure the complete tool with "
           "its final precision-first rule. Table 4.7 reports both.")
    t, _ = table("Performance of the helmet check", ["Test", "Measure", "Result"], [
        ["Cross-validation, 3,366 persons (3 % not-sure setting)", "Accuracy of called answers", "91.8 %"],
        ["", "No-helmet riders caught", "667 / 873 (76.4 %)"],
        ["", "Helmeted riders called “no helmet”", "50 / 1,228"],
        ["Cross-validation, final rule", "Helmeted riders called “no helmet”", "4 / 1,228"],
        ["7 unseen videos, 367 riders, final rule", "“No helmet” calls (wrong)", "29 (1 wrong)"],
        ["", "“Helmet” calls (wrong)", "175 (3 wrong)"],
        ["", "Wrong among checkable answers", "4 / 167 (2.4 %)"],
        ["", "Not sure", "44 % (71 of 162 unreadable even for a person)"],
        ["", "Clear no-helmet riders caught", "27 / 81 (33 %)"],
    ], [0.42, 0.33, 0.25])
    s += t
    s += p("Figure 4.18 compares the first version of the tool with the final rule on the unseen videos. "
           "Requiring 0.98 probability over at least three agreeing frames reduced wrong no-helmet calls from 20 "
           "to 1, at the cost of more “not sure” answers, which go to review.")
    f, _ = figure(F / "c_helmet.png", "Helmet check on seven unseen videos: first version and final rule", width=400)
    s += f
    s += h("4.3.3", "Triple-Riding Check", 2)
    s += p("Triple riding is rare in real footage (about one bike in 150), so it was tested on held-out cameras "
           "containing 980 two-wheelers with 125 real triples. Table 4.8 compares decision rules with the same "
           "not-sure budget of 8.3 %.")
    t, _ = table("Triple-riding check on held-out cameras (980 bikes, 125 triples)", ["Decision rule", "Triples caught", "Wrong TRIPLE calls", "Not sure"], [
        ["Classifier, ≥ 1 agreeing frame", "20 (16 %)", "0", "8.3 %"],
        ["Classifier, ≥ 3 agreeing frames", "37 (30 %)", "0", "8.3 %"],
        ["<b>Classifier + heads on seat, ≥ 3 frames (chosen)</b>", "<b>50 (40 %)</b>", "<b>0</b>", "<b>8.3 %</b>"],
        ["Chosen rule: triples caught or sent to review", "90 (72 %)", "–", "–"],
    ], [0.46, 0.18, 0.18, 0.18])
    s += t
    s += p("On full street-level videos from Indore, Pune and Jaipur (about 700 motorcycles), the tool made one "
           "wrong triple call (a rider with a side-saddle passenger) and missed the few visible triples, which were "
           "children seated between adults or bikes visible for less than a second. The check is therefore strong "
           "on clear police-CCTV views and weaker on busy hand-held footage.")
    s += h("4.3.4", "Signal State and Red-Light Check", 2)
    s += p("Table 4.9 reports the signal-state classifier on held-out clips and on independent long clips graded "
           "by eye.")
    t, _ = table("Performance of the signal-state classifier", ["Test", "Result"], [
        ["Held-out LISA / TVD crops: red", "2,252 / 2,252 red; TVD red 72 / 73"],
        ["Held-out LISA / TVD crops: yellow, green", "Yellow 72 / 72; green 1,114 / 1,132; TVD green 76 / 78"],
        ["Held-out LISA crops: not a light", "456 / 461"],
        ["Indian crops, fine-tuned (two halves)", "Real reds 25 / 26; non-lights called red 1.1–3.8 %"],
        ["Fixed camera, US junction, 66 s (graded frames)", "56 / 57 correct (98 %)"],
        ["Night camera, Las Vegas, 5 min", "38 / 41 correct when a call is made; green or yellow never read as red"],
        ["Red-light crossings, Indian junction clip", "2 / 2 correct (limited test material)"],
    ], [0.45, 0.55])
    s += t
    s += p("The colour-change safeguard means that a signboard, red car or brick paving read as a red lamp can at "
           "most cause a review, never a challan. Red-light crossings could be tested on only a few clips, because "
           "most available footage either shows no signal facing the camera or is cut into short shots; this is "
           "the main gap in the evaluation.")
    s += h("4.3.5", "Emergency-Vehicle Exemption", 2)
    s += p("Table 4.10 and Figure 4.19 show the trade-off of the emergency probe on 170 held-out emergency tracks "
           "and 2,450 ordinary tracks. At the chosen exemption threshold of 0.956, 98 emergency vehicles were "
           "exempted and no ordinary vehicle was wrongly exempted; with the review band, 143 of the 170 were "
           "exempted or reviewed.")
    ev = json.load(open("/Users/sachinnair/Desktop/traffic violation system/runs/ambulance_cls/eval.json"))["sweeps"]
    rows = []
    for th in ["0.5", "0.7", "0.9", "0.95", "0.98"]:
        a, b = ev["emergency (probe)"][th], ev["emergency (zero-shot)"][th]
        rows.append([th, f"{a['caught']} / 170", f"{a['false']} / 2,450", f"{b['caught']} / 170", f"{b['false']} / 2,450"])
    t, _ = table("Emergency-vehicle recognition at different thresholds", ["Threshold", "Probe: caught", "Probe: wrongly flagged", "Zero-shot: caught", "Zero-shot: wrongly flagged"],
                 rows, [0.16, 0.2, 0.22, 0.2, 0.22])
    s += t
    f, _ = figure(F / "c_emerg.png", "Emergency vehicles caught against ordinary vehicles wrongly flagged", width=400)
    s += f
    s += h("4.3.6", "Number-Plate Reading", 2)
    s += p("Table 4.11 compares the plate reader with general OCR baselines on the 315 held-out test plates, and "
           "reports the certainty rule.")
    t, _ = table("Performance of the number-plate reader", ["Component / setting", "Result"], [
        ["Plate detector recall (held-out plates)", "99.4 %"],
        ["EasyOCR (baseline), exact plate", "13.7 %"],
        ["Pretrained fast-plate-ocr, exact plate", "45.5 %"],
        ["Fine-tuned Indian reader, exact plate", "81.6 %"],
        ["Fine-tuned Indian reader, characters", "95.4 %"],
        ["Certainty rule (character confidence 0.90)", "99.0 % of READ plates correct; 70 % of plates READ"],
        ["Video audit of READ plates", "20 / 20 correct"],
    ], [0.55, 0.45])
    s += t
    s += p("On real CCTV footage most plates are too small to read with certainty, so relatively few vehicles "
           "receive a READ plate; the reviewer types the plate from the crop for the others. This is the intended "
           "behaviour: no guessed plate is ever printed.")
    s += h("4.3.7", "Speed Estimation", 2)
    s += p("Table 4.12 reports the calibrated speed estimator, calibrated on one carriageway and tested on the "
           "other, so that calibration and test use different references.")
    t, _ = table("Calibrated speed estimation on reference scenes", ["Test", "Vehicles", "Mean error", "90 % range contains truth"], [
        ["M6, calibrated left, tested right", "33", "6.3 km/h (8.4 %)", "97 %"],
        ["M6, calibrated right, tested left", "27", "4.3 km/h (11.4 %)", "85 %"],
        ["Autobahn, calibrated left, tested right", "39", "10.7 km/h (8.8 %)", "100 %"],
        ["Autobahn, calibrated right, tested left", "38", "10.3 km/h (7.6 %)", "100 %"],
        ["<b>All</b>", "<b>137</b>", "<b>≈ 9 %</b>", "<b>96 %</b>"],
    ], [0.4, 0.14, 0.22, 0.24])
    s += t
    s += p("Without calibration every vehicle was labelled “speed ?”, so no wrong number was shown, although "
           "the hidden middle estimates were 13.6 % and 53 % off on the two scenes. Because the calibrated error is "
           "about 9 %, the platform confirms over-speeding only when the whole 90 % range is above the limit.")
    s += h("4.3.8", "Platform-Level Results and Functional Testing", 2)
    c = sqlite3.connect(DB)
    nan = c.execute("select count(*) from analyses where status='done'").fetchone()[0]
    nf = c.execute("select count(*) from findings").fetchone()[0]
    nch = c.execute("select count(*) from challans").fetchone()[0]
    nrev = c.execute("select count(*) from findings where status='pending_review'").fetchone()[0]
    ndup = c.execute("select count(*) from findings where status='duplicate'").fetchone()[0]
    s += p(f"The platform was exercised with {nan} uploaded videos of Indian and other traffic, including a "
           f"Bengaluru junction, a Lalbagh signal, red-light jumpers, Mumbai traffic, a Tirupati police-camera clip "
           f"and highway footage. Across these analyses the checks produced {nf} findings: {nch} automatic challans, "
           f"{nrev} cases waiting for review and {ndup} repeat detections merged into existing cases. Figure 4.20 "
           f"shows how the raw findings of each check were routed before merging.")
    f, _ = figure(F / "c_outcome.png", "Routing of raw findings by check across all analyses", width=400)
    s += f
    rows = []
    for fn, cr, fi, m, aid in c.execute("select filename, created, finished, meta, id from analyses where status='done' order by created"):
        m = json.loads(m or "{}")
        o = m.get("original") or m
        res = f"{o.get('width')}×{o.get('height')}"
        n = c.execute("select count(*) from findings where analysis_id=?", (aid,)).fetchone()[0]
        chn = c.execute("select count(*) from challans where analysis_id=?", (aid,)).fetchone()[0]
        name = fn if len(fn) <= 30 else fn[:27] + "..."
        rows.append([name.replace("_", "_​"), res, f"{m.get('duration', 0):.0f}", f"{(fi - cr) / 60:.1f}", str(n), str(chn)])
    t, _ = table("Processing results of the uploaded videos", ["Video", "Resolution", "Length (s)", "Processing (min)", "Findings", "Challans"],
                 rows, [0.36, 0.15, 0.12, 0.14, 0.11, 0.12], small=True)
    s += t
    s += p("Figure 4.21 relates processing time to video length on the Apple M4. Most 720p–1080p videos were "
           "processed in one to two times their length; the 4K video and the long low-resolution police clip took "
           "longer, because the 4K file needs a working copy and the police clip contains many riders, each of "
           "which is classified in every frame. Repeated uploads of the same footage are recognised and their "
           "vehicles are not challaned again.")
    f, _ = figure(F / "c_time.png", "Processing time against video length", width=380)
    s += f
    s += p("Table 4.14 lists the functional test scenarios. The status column records what was observed on the "
           "running platform; scenarios that could not yet be confirmed from stored data are marked to be recorded.")
    t, _ = table("Functional test scenarios and observed behaviour", ["ID", "Scenario", "Expected behaviour", "Status"], [
        ["T1", "Sign up and verify", "CAPTCHA and six-digit code required; account usable only after verification.", "Pass (2 verified accounts)"],
        ["T2", "Sign in and session", "HttpOnly session cookie; pages behind sign-in redirect to /login without it.", "Pass"],
        ["T3", "Upload and background analysis", "Upload returns at once; stage and progress update until done.", f"Pass ({nan} analyses done)"],
        ["T4", "Automatic challan", "Certain findings become challans with number, offences and evidence hash.", f"Pass ({nch} issued)"],
        ["T5", "Challan PDF", "PDF downloads with evidence, QR code and fingerprint.", "Pass (HTTP 200, PDF opened)"],
        ["T6", "Evidence collection", "Frames, crop, clips and SHA-256 stored for every finding.", "Pass"],
        ["T7", "One case per vehicle", "Repeat detections merged; at most one challan per vehicle.", f"Pass ({ndup} merged)"],
        ["T8", "Re-upload of same footage", "Vehicles already handled are not challaned again.", "Pass (Mumbai re-upload: 0 new)"],
        ["T9", "Human review decision", "Challan / reject / more evidence recorded with note and plate.", "To be recorded"],
        ["T10", "Emergency exemption", "Ambulance, police or fire vehicle exempted with reason.", "To be recorded on platform"],
        ["T11", "Delete analysis", "Merged findings elsewhere restored and re-merged.", "To be recorded"],
        ["T12", "Rate limit and code attempts", "Repeated sign-in or code attempts are blocked.", "To be recorded"],
    ], [0.07, 0.24, 0.45, 0.24], small=True)
    s += t
    s += h("4.3.9", "Adaptive Signal Control in Simulation", 2)
    s += p("All results in this section are <b>simulated</b>: they come from the Unity junction simulator, whose "
           "driver-behaviour parameters (patience, dwell times) are estimates, so they show how the controllers "
           "compare rather than predict real delays. Figure 4.22 shows the simulator. Each controller was run for "
           "30 simulated minutes in six scenarios with three random seeds, every controller receiving the same "
           "arrivals in each seed. Table 4.15 and Figure 4.23 report the mean wait per person.")
    f, _ = figure(F / "sim_overview.png", "The Unity junction simulator: an Indian four-arm junction with mixed traffic", width=440)
    s += f
    t, _ = table("Mean wait per person (s) in simulation: 30 min, 3 seeds, identical arrivals", ["Scenario", "Fixed 30 s", "Webster", "Adaptive v1", "PRAVAH", "PRAVAH vs Adaptive v1"], [
        ["City balanced", "38.9", "36.3", "26.9", "<b>15.5</b>", "−42 %"],
        ["Office rush (N–S)", "89.8", "36.3", "23.3", "<b>19.4</b>", "−17 %"],
        ["Market road (cattle, bus stops)", "41.5", "30.7", "22.3", "<b>14.4</b>", "−35 %"],
        ["Peak near capacity", "87.0", "98.1", "75.7", "<b>73.8</b>", "−3 %"],
        ["Night", "31.6", "11.1", "5.5", "<b>2.9</b>", "−47 %"],
        ["Ambulance corridor", "41.0", "29.6", "20.2", "<b>15.9</b>", "−21 %"],
    ], [0.3, 0.13, 0.13, 0.14, 0.13, 0.17])
    s += t
    f, _ = figure(F / "sim_person_wait.png", "Mean wait per person by controller and scenario (lower is better)", width=400)
    s += f
    s += p("PRAVAH gave the lowest wait per person in every scenario, 17–47 % below the queue-pressure adaptive "
           "controller in five of them. Ambulances waited 0.1–0.4 s on average in five scenarios and 7.6 s near "
           "capacity. The gains are not uniform: near capacity PRAVAH was on par with Adaptive v1 (73.8 s against "
           "75.7 s), with a longer worst-case wait (295 s against 268 s) and slightly more red-light jumps, and in "
           "the ambulance corridor its worst wait was 148 s against 101 s.")
    s += p("<b>Robustness.</b> Because the simulator's assumptions are estimates, the comparison was repeated with "
           "each assumption changed: driver patience halved and increased by half, bus dwell halved and doubled, "
           "camera range 50 m and 150 m, demand −20 % and +20 %, a two-wheeler-heavy mix, no red-light jumping and "
           "no free left turn, with ten seeds each (2,220 runs of 30 simulated minutes). Table 4.16 shows that "
           "PRAVAH kept the lowest mean wait per person under every one of the twelve assumption sets. Paired over "
           "all 72 combinations of assumption and scenario, it was significantly better than Adaptive v1 in 60 and "
           "never significantly worse on this measure. The trade-off appears in other measures: against Adaptive v1 "
           "it had significantly more red-light jumps in 11 combinations and a longer worst wait in 4.")
    t, _ = table("Robustness of the mean wait per person to the simulator assumptions", ["Simulator assumption", "PRAVAH", "Best other (Adaptive v1)", "Scenarios where PRAVAH is significantly better"], [
        ["Baseline", "23.1", "27.3", "5 / 6"],
        ["Impatient drivers (patience × 0.5)", "19.3", "24.0", "5 / 6"],
        ["Patient drivers (patience × 1.5)", "24.7", "28.9", "5 / 6"],
        ["Short bus stops (dwell × 0.5)", "18.1", "23.2", "6 / 6"],
        ["Long bus stops (dwell × 2)", "48.9", "56.5", "3 / 6"],
        ["Short camera range (50 m)", "24.0", "28.3", "5 / 6"],
        ["Long camera range (150 m)", "23.1", "28.1", "5 / 6"],
        ["Demand −20 %", "10.9", "15.6", "6 / 6"],
        ["Demand +20 %", "51.0", "55.2", "5 / 6"],
        ["Two-wheeler-heavy mix (58 %)", "15.7", "20.3", "6 / 6"],
        ["No red-light jumping", "24.9", "28.8", "5 / 6"],
        ["No free left turn", "30.6", "34.7", "4 / 6"],
    ], [0.4, 0.14, 0.2, 0.26], small=True)
    s += t
    s += p("<b>Python in the loop.</b> In a shadow test, the C# and Python implementations of PRAVAH received "
           "identical junction states at every decision and disagreed in 0 of 83,702 decisions, so the Python "
           "bridge reproduces the controller exactly. Running the project's earlier rule-based signal controller "
           "through the bridge (Table 4.17) showed that it does well in moderate traffic but breaks down near "
           "capacity: the mean wait rose to 98.7 s, throughput fell to 2,339 vehicles per hour and 486 vehicles were "
           "left queued. The lack of a waiting-time limit and of a spillback check explains this, and motivated both "
           "features in PRAVAH.")
    t, _ = table("The project's rule-based signal controller run through the Python bridge (simulated)", ["Scenario", "Wait per vehicle (s)", "Wait per person (s)", "Worst wait (s)", "Throughput (veh/h)", "Left queued"], [
        ["City balanced", "29.8", "26.5", "220", "2,373", "40"],
        ["Office rush (N–S)", "29.4", "27.3", "309", "2,432", "33"],
        ["Market road", "27.6", "25.1", "181", "2,237", "35"],
        ["Peak near capacity", "98.7", "95.0", "390", "2,339", "486"],
        ["Night", "9.5", "4.6", "133", "654", "3"],
        ["Ambulance corridor", "23.1", "20.9", "169", "2,195", "26"],
    ], [0.26, 0.15, 0.15, 0.13, 0.17, 0.14], small=True)
    s += t
    s += h("4.3.10", "Comparison with the Project-A Proposal", 2)
    s += p("Table 4.18 compares the TraffiSense architecture proposed in Project-A with what has "
           "been implemented in this Project-B platform. The comparison is qualitative.")
    t, _ = table("Project-A proposal compared with the Project-B implementation", ["Aspect", "Project-A proposal", "Project-B implementation"], [
        ["Input", "Four simulated CCTV streams of one junction", "Uploaded recorded videos from any fixed camera"],
        ["Detection", "YOLOv8 fine-tuned on UVH-26 and IDD", "YOLO11s fine-tuned on IDD, run once and shared by all checks"],
        ["Violations", "No helmet, triple riding, red light", "No helmet, triple riding, red light, over-speeding"],
        ["Multi-frame verification", "K of N frames per track", "Weighted votes, minimum frames, agreement and colour-change rules"],
        ["ANPR", "EasyOCR / PaddleOCR", "Fine-tuned Indian reader with READ / NOT SURE / UNREADABLE"],
        ["Emergency vehicles", "Signal pre-emption", "Exemption from challans with CLIP probe"],
        ["Explainability", "Grad-CAM heatmaps and rule reasons", "Plain-language reasons, uncertainty reasons, evidence frames and clips, fingerprint"],
        ["Human review", "Low-confidence queue", "Review page with plate entry and three decisions; one case per vehicle"],
        ["Signal control and simulation", "Adaptive timing validated in a SUMO digital twin", "PRAVAH controller evaluated in a Unity simulator of an Indian junction (simulated results); 3-D twin of the analysed scene"],
        ["Lanes", "Painted lane detection for density", "Lanes from vehicle movement, context only"],
        ["Interface", "Flask / Streamlit dashboard", "React + FastAPI platform with accounts and 18 pages"],
    ], [0.22, 0.34, 0.44], small=True)
    s += t
    s += h("4.3.11", "Discussion", 2)
    s += p("The implemented system shows that an evidence-first enforcement workflow can be assembled from trained "
           "components and standard web technology. Three design decisions deserve comment. First, running the "
           "vehicle detection once and sharing it gives every check, the plate reader and the duplicate merging "
           "the same track numbers, which makes it possible to say with certainty that two findings belong to the "
           "same vehicle. Second, choosing every operating point for zero (or near-zero) wrong calls on unseen "
           "cameras turns the models into tools that can be trusted when they do speak: one wrong no-helmet call "
           "in 29 on new videos, and no wrong triple-riding or emergency exemption calls on held-out cameras. "
           "Third, treating “not sure” as a first-class answer, with its reason shown to the reviewer, "
           "keeps the human in charge of exactly the cases where the machine is weak.")
    s += p("For signal control, counting waiting people instead of vehicles and penalising long red waits gave the "
           "largest gains where traffic is mixed and moderate, and the advantage held when the simulator's "
           "assumptions were changed. Near capacity no controller can create capacity that the junction does not "
           "have, and PRAVAH was only on par with the adaptive baseline there. These remain simulated results; "
           "field validation with measured queues at a real junction is needed before any claim about real "
           "delays.")
    s += p("Overall, the results show that the two parts of TraffiSense complement each other. The enforcement part "
           "makes few wrong decisions and leaves the uncertain cases to people, and the signal-control part shows in "
           "simulation that the traffic measured by the same cameras can be used to reduce the time people wait at "
           "a junction.")
    s += p("The cost of this precision is recall and reviewer time. At the chosen settings the helmet check "
           "confirms only a third of clear no-helmet riders automatically and sends 44 % of riders to review on "
           "phone-camera footage, and the triple-riding check confirms 40 % of triples on police-camera views. "
           "Both numbers are much better on fixed, elevated CCTV views than on hand-held street video, which is why "
           "the dashboard guides operators towards suitable footage.")
    s += p("The main limitations are the following. Red-light crossings were tested on little real material. Some "
           "training and test labels were produced with AI assistance and have not been re-checked by a person. "
           "Plates on real CCTV are often too small to read with certainty. Night footage remains weaker for "
           "signals and emergency vehicles. Finally, TraffiSense is a prototype that analyses recorded video; its "
           "challans are not legal notices, and it must not be used for real enforcement without validation, "
           "certified speed measurement, legal review and institutional controls.")
    return s


def ch5():
    s = chapter(5, "Summary")
    s += p("In this report, the implementation of TraffiSense, an evidence-first traffic enforcement platform for "
           "recorded traffic video, is presented. The report began with the need for reliable and explainable "
           "enforcement of two-wheeler and signal offences on Indian roads and reviewed recent literature on adaptive "
           "signal control, emergency-vehicle priority, real-time object detection, Indian traffic datasets, "
           "two-wheeler violation detection and low-resolution number-plate recognition. The research gaps that emerged were single-frame decisions, the lack of a “not sure” outcome, "
           "guessed plates, duplicate penalties, missing emergency exemptions, weak evidence, and signal control "
           "that counts vehicles rather than people.")
    s += p("The implemented system answers these gaps with a React frontend of 18 pages, a FastAPI backend with an "
           "SQLite database and a model layer of trained tools. A single detection and tracking pass feeds the "
           "helmet, triple-riding, red-light and speed checks and the plate reader. Every decision is taken over "
           "many frames with thresholds chosen on unseen cameras, emergency vehicles are exempted, plates are "
           "printed only when read with certainty, repeat detections are merged so that a vehicle receives at most "
           "one challan, and every finding carries evidence frames, clips, reasons and a SHA-256 fingerprint. "
           "Uncertain cases are decided by a reviewer. For signal control, the PRAVAH controller, which weights the "
           "people waiting and penalises long red waits, gave the lowest mean wait per person of four controllers in "
           "a Unity simulator of an Indian junction, robustly across twelve sets of simulator assumptions, and the "
           "platform can replay the traffic of an analysed video through the simulator. Table 5.1 summarises what the integrated prototype delivers.")
    t, _ = table("Components delivered in the integrated prototype", ["Component", "Status in the prototype"], [
        ["Accounts: sign-up with CAPTCHA, verification codes, sign-in, reset", "Implemented"],
        ["Video upload, background analysis with progress", "Implemented"],
        ["Shared vehicle detection and tracking (IDD YOLO11s + ByteTrack)", "Implemented"],
        ["Helmet check", "Implemented"],
        ["Triple-riding check", "Implemented"],
        ["Red-light check with automatic stop line", "Implemented"],
        ["Speed estimate", "Implemented (needs a calibration and a limit)"],
        ["Number-plate reader (READ / NOT SURE / UNREADABLE)", "Implemented"],
        ["Emergency-vehicle exemption", "Implemented"],
        ["Evidence, fingerprint, e-challan PDF", "Implemented"],
        ["Human review and one case per vehicle", "Implemented"],
        ["Digital twin, lane detection, analytics", "Implemented (lanes as context only)"],
        ["Adaptive signal control (PRAVAH) in a Unity junction simulator", "Implemented (simulated results only)"],
        ["Signal-timing page: video traffic replayed through the simulator", "Implemented (simulated results only)"],
        ["Python bridge for Python signal controllers", "Implemented"],
    ], [0.62, 0.38])
    s += t
    s += p("The evaluation parameters, namely detection precision, recall and mAP, wrong-call rate, not-sure rate, "
           "violations caught, speed error and range coverage, and functional tests, were defined, and Chapter 4 "
           "reports them for every component and for the platform. The platform improves the consistency, "
           "transparency and traceability of enforcement decisions and focuses the reviewer's time on the cases "
           "that need a person. It remains an academic prototype whose outputs must be reviewed by an authorised "
           "human.")
    s += p("<b>Future scope.</b> The following extensions would strengthen the prototype:")
    s += bullets([
        "More labelled street-level and night-time data for the helmet and triple-riding checks, and a head detector that counts the people on the seat.",
        "Retraining the plate detector on enlarged vehicle crops from Indian street video, so that more plates are read with certainty.",
        "Longer fixed-camera junction footage to test red-light crossings, and re-checking the AI-assisted labels by people.",
        "Live camera streams instead of uploaded recordings.",
        "Field validation of the PRAVAH signal controller with measured queue lengths at a real junction.",
        "Coordination of neighbouring junctions along a corridor, and pedestrian phases in the signal plan.",
        "Integration with the official e-challan system, after validation with the traffic authority, certified speed measurement and the applicable legal and privacy controls.",
    ])
    s += p("<b>Conclusion.</b> TraffiSense shows that one set of junction cameras can serve both enforcement and "
           "signal control. For enforcement, multi-frame decisions, an explicit “not sure” answer, certain-only "
           "plates, emergency exemption and fingerprinted evidence make every challan explainable and keep people "
           "in charge of doubtful cases. For signal control, counting the people who wait and penalising long red "
           "waits reduced the simulated waiting time at an Indian junction. Together they form a sound basis for a "
           "responsible, field-validated deployment in the future.")
    return s


def references():
    s = unnumbered("References", "refs")
    for i, k in enumerate(REFS, 1):
        s += [Paragraph(f"[{i}]&nbsp;&nbsp;{REFS[k]}", S["ref"])]
    return s


def publications():
    """Survey Paper B, reproduced page by page (no separate publications page)"""
    s = []
    import pymupdf
    from reportlab.platypus import Image as RLImage
    sp = pymupdf.open(SURVEY_B)
    for i, pg in enumerate(sp):
        out = F / f"survey_b_p{i + 1}.png"
        pg.get_pixmap(dpi=170).save(out)
        hgt = E.PAGE_H - 70 - 66 - 24
        wid = hgt * pg.rect.width / pg.rect.height
        img = RLImage(str(out), width=wid, height=hgt)
        img.hAlign = "CENTER"
        s += [PageBreak(), img]
    return s


def acknowledgement():
    s = unnumbered("Acknowledgement", "ack")
    s += p("The development of TraffiSense: An Explainable, Digital-Twin-Driven Intelligent Traffic Management and Violation Detection System benefited immensely from the valuable guidance and "
           "continuous support provided by our project supervisor, Dr. Siddharth Hariharan. The supervisor's "
           "expertise in computer vision and proactive mentorship were instrumental in shaping the vision and "
           "technical execution of this project, and the insistence on honest evaluation and evidence-based "
           "decisions was vital in ensuring the reliability and responsible design of the system.")
    s += p("We are deeply grateful to Dr. Sharvari Govilkar, Head of the Computer Engineering Department, for "
           "providing our team with the necessary resources and encouragement to successfully conduct this project. "
           "This support played a critical role in enhancing both our educational experience and the development "
           "of explainable, AI-driven traffic enforcement solutions.")
    s += p("Our sincere thanks also go to Dr. Sandeep M. Joshi, Principal of Pillai College of Engineering, for "
           "offering us this opportunity and for fostering an academic environment that encourages advancements in "
           "responsible AI and smart-city technology.")
    s += p("We would also like to express our heartfelt gratitude to our mentors, faculty members and fellow peers "
           "whose valuable inputs, constructive feedback and inspiration helped refine TraffiSense.")
    s += [Spacer(1, 30)]
    from reportlab.lib.styles import ParagraphStyle
    for n in ["Sachin Nair", "Ranjith Iyer", "Aditya Vijay Kumar"]:
        s += [Paragraph(n, ParagraphStyle("n", fontName="TNR", fontSize=12, leading=23.7))]
    return s
