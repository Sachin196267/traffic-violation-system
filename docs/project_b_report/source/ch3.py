"""Chapter 3: Implemented System: TraffiSense."""
from story import *
from engine import FIG

F = FIG


def ch3():
    s = chapter(3, "Implemented System: TraffiSense")
    s += h("3.1", "Overview")
    s += p("This chapter presents TraffiSense, the evidence-first traffic enforcement platform that has been implemented "
           "in this project. TraffiSense is an academic prototype for analysing recorded traffic video. It is not an "
           "autonomous enforcement authority: its challans are produced only when a model is certain under strict, "
           "pre-measured rules, every other case is decided by an authorised reviewer, and every output carries the "
           "evidence needed to check it.")
    s += p("The implemented system integrates the following capabilities. An authentication module lets operators "
           "create an account with an e-mail address or phone number, verify it with a one-time code, sign in and "
           "reset a forgotten password. An analysis module accepts a video upload with a choice of checks, keeps the "
           "original file unchanged, and processes it in the background while the page shows live progress. A model "
           "layer detects and tracks vehicles once and runs the helmet, triple-riding, red-light and speed checks, "
           "the number-plate reader and the emergency-vehicle probe. An evidence module stores frames, crops and "
           "clips for every finding and fingerprints them. A decision module routes each finding, merges repeat "
           "detections of the same vehicle and issues e-challans with a PDF. Review, evidence, digital-twin, lane, "
           "analytics and settings pages complete the platform, and a signal-timing page runs the traffic measured in "
           "a video through a Unity junction simulator to compare signal controllers. All records are stored in an SQLite database "
           ". Table 3.1 summarises the implemented system.")
    t, _ = table("Summary of the implemented TraffiSense system", ["Aspect", "Implementation"], [
        ["System type", "Web-based, evidence-first traffic enforcement platform for recorded video (academic prototype)."],
        ["User role", "Operator (uploads video and reviews cases); every account is verified before first use."],
        ["Frontend", "React 19, TypeScript and Vite; react-router; three.js for the 3-D digital twin; 18 pages."],
        ["Backend", "FastAPI on uvicorn, port 8600, Python 3.14; serves the API and the built web app."],
        ["Model layer", "Ultralytics YOLO11 detectors and classifiers on PyTorch (Apple GPU via MPS), ByteTrack, CLIP probe, ONNX plate reader."],
        ["Checks", "Helmet, triple riding, red-light jumping, over-speeding; number plates; emergency exemption; lanes as context."],
        ["Evidence", "Three frames, crop, original and annotated clips (H.264 via FFmpeg), SHA-256 fingerprint."],
        ["Database", "SQLite in WAL mode: users, sessions, codes, analyses, findings, reviews, challans."],
        ["Decision principle", "Precision first; uncertain cases to human review; at most one challan per vehicle."],
        ["Signal control (simulated)", "Unity 6 junction simulator (C#) with four controllers including PRAVAH; Python TCP bridge for Python controllers; results always labelled SIMULATED."],
    ], [0.24, 0.76])
    s += t
    s += p("The functional requirements derived from the project definition are: account creation, verification, "
           "sign-in and password reset; video upload with a selection of checks; background processing with "
           "progress; detection of the four offence types; number-plate reading; emergency-vehicle exemption; "
           "evidence collection; routing to automatic challan, review or exemption; human review with plate entry; "
           "e-challan PDF download; and evidence, twin, lane and analytics views. The main non-functional "
           "requirements are precision before recall, explainability of every decision, integrity of evidence, "
           "secure handling of passwords and sessions, recovery after a server restart, and a modular model layer "
           "in which a detector can be added or replaced without changing the rest of the platform.")
    s += h("3.1.1", "Existing System Architecture", 2)
    s += p("In the existing process, footage from CCTV cameras or phones is watched by an officer, either at the "
           "junction or on a control-room screen. When the officer notices an offence, it is noted from what was "
           "seen at that moment, the plate is read or written down by hand, and a challan is issued, usually with "
           "little recorded evidence and no explanation. Figure 3.1 shows this flow together with the limitation "
           "that arises at each stage.")
    f, _ = figure(F / "f3_1.png", "Existing enforcement flow and its limitations", width=450)
    s += f
    s += p("Single-purpose enforcement cameras automate parts of the third and fourth stages, but they still decide "
           "on single frames, they cover one offence each, and they give the reviewer no account of why a vehicle "
           "was flagged. Table 3.2 lists the main limitations and the response taken in TraffiSense.")
    t, _ = table("Limitations of the existing system and the response in TraffiSense", ["Limitation of existing systems", "Response in TraffiSense"], [
        ["Manual watching; most footage never reviewed", "Every uploaded video is processed completely in the background."],
        ["Decision from one glance or one frame", "Tracked, multi-frame decisions with minimum frame counts and agreement checks."],
        ["Forced decisions on unclear cases", "Explicit “not sure” band; uncertain cases go to the human review queue."],
        ["Misread or guessed plates", "Plate printed only when READ with certainty; otherwise the reviewer confirms it."],
        ["Emergency vehicles caught by rules", "Ambulance, police and fire vehicles exempted with a recorded reason."],
        ["Duplicate penalties", "Duplicate merging and a unique index: one challan per vehicle."],
        ["No evidence trail or reasons", "Frames, clips, reasons, confidence and SHA-256 fingerprint on every challan."],
    ], [0.4, 0.6])
    s += t
    s += h("3.1.2", "Proposed System Architecture", 2)
    s += p("The proposed architecture of TraffiSense has three tiers and a model layer. Operators work through a "
           "<b>React frontend</b>, built with Vite and served by the same server. The frontend talks to the "
           "<b>FastAPI backend</b> on port 8600 through a JSON API; the backend provides authentication, the "
           "analyses API and file serving, findings, review and challans, and the statistics behind the dashboard, "
           "analytics and digital twin. Video analysis is delegated to a background <b>analysis pipeline</b>, which "
           "calls the trained tools of the <b>model layer</b> as separate processes. Persistent data live in "
           "<b>SQLite</b> and in one folder per analysis. Figure 3.2 shows the architecture.")
    f, _ = figure(F / "f3_2.png", "Proposed architecture of TraffiSense", width=450)
    s += f
    s += p("Each trained tool runs as a subprocess with its own models and decision rules, and the pipeline only "
           "reads its JSON output. This keeps the web server responsive while the GPU is busy, isolates a failing "
           "tool, and lets a model be retrained or replaced by writing one <i>Detector</i> class. The deployment "
           "view in Figure 3.3 shows the processes that run when the platform is started with `./start.sh`.")
    f, _ = figure(F / "f3_3.png", "Runtime and deployment view of TraffiSense", width=440)
    s += f
    s += p("The responsibilities of each component are summarised in Table 3.3.")
    t, _ = table("Components of the proposed architecture and their responsibilities", ["Component", "Runtime", "Responsibility"], [
        ["React frontend", "Browser; built to web/dist", "Pages for upload, results, review, challans, evidence, twin, lanes, analytics, settings."],
        ["FastAPI backend", "uvicorn, port 8600", "Authentication, sessions, CAPTCHA, verification codes, rate limits, API, PDF generation."],
        ["Analysis pipeline", "Thread pool, one analysis at a time", "Working copy, shared detection, checks, plates, evidence, merging, challans, lanes."],
        ["Model layer", "Python subprocesses (MPS)", "Detection and tracking, helmet, triple riding, red light, speed, plates, emergency probe, lanes."],
        ["Signal simulator", "Unity 6, headless batch mode", "Indian four-arm junction; runs every controller on identical arrivals; 3-D views for demonstration."],
        ["Signal bridge", "Python TCP server, port 5055", "Lets controllers written in Python run the simulated signal; PRAVAH and the project controller."],
        ["SQLite + files", "data/platform.db, data/analyses/&lt;id&gt;/", "Users, sessions, analyses, findings, reviews, challans; original video and evidence."],
    ], [0.2, 0.27, 0.53])
    s += t
    s += p("The frontend has 18 pages, grouped by purpose in Figure 3.4: four public authentication pages and "
           "fourteen pages behind sign-in, which share one layout with a sidebar and a live count of running "
           "analyses and cases waiting for review. Table 3.4 lists each page and its purpose.")
    f, _ = figure(F / "f3_4.png", "Map of the 18 frontend pages grouped by purpose", width=440)
    s += f
    t, _ = table("Frontend pages of TraffiSense", ["Route", "Group", "Purpose"], [
        ["/login", "Public", "Sign in with e-mail or phone and password"],
        ["/signup", "Public", "Create an account (name, contact, password policy, CAPTCHA)"],
        ["/verify", "Public", "Enter the six-digit verification code"],
        ["/forgot", "Public", "Request a reset code and set a new password"],
        ["/", "Analysis", "Dashboard: totals, video guidance, upload, review queue preview"],
        ["/analyses", "Analysis", "All video analyses with status, findings, reviews and challans"],
        ["/analyses/:id", "Analysis", "Video player with overlays, results list, lane structure, notes"],
        ["/twin/:id", "Analysis", "3-D digital twin of the scene with violations on a timeline"],
        ["/lanes/:id", "Analysis", "Lanes estimated from vehicle movement (context only)"],
        ["/signal/:id", "Analysis", "Signal timing (SIMULATED): measured traffic replayed through the junction simulator"],
        ["/violations", "Enforcement", "Every finding with its decision path and filters"],
        ["/review/:id", "Enforcement", "Human review: evidence, reasons, plate, decision"],
        ["/challans", "Enforcement", "Issued challans with source, confidence and PDF"],
        ["/challans/:id", "Enforcement", "Challan detail with evidence and download"],
        ["/evidence", "Platform", "Gallery of evidence frames for all findings"],
        ["/analytics", "Platform", "Violation types, outcomes, confidence distribution"],
        ["/system", "Platform", "Models and their status, how decisions are made"],
        ["/settings", "Platform", "Profile, password, theme, upload defaults, code delivery"],
    ], [0.22, 0.17, 0.61])
    s += t
    # ---------------------------------------------------------- 3.2
    s += h("3.2", "Implementation Details")
    s += p("This section describes how the platform is implemented: the methodology and techniques behind each "
           "check, the diagrams that describe the behaviour of the system, and the hardware and software used to "
           "build and run it.")
    s += h("3.2.1", "Algorithm / Methodology / Techniques", 2)
    s += p("An analysis follows the pipeline in Figure 3.5. The steps are described in the paragraphs below.")
    f, _ = figure(F / "f3_5.png", "Video analysis pipeline of TraffiSense", width=400, max_h=420)
    s += f
    s += h3("A. Video intake and working copy")
    s += p("The uploaded file (up to 2,048 MB) is stored unchanged in `original/` and is never modified, so the "
           "evidence can always be traced back to it. The video is probed for resolution, frame rate and length. "
           "When it is larger than 1080p or faster than 30 frames per second, a 1080p working copy is made with "
           "FFmpeg and the models run on that copy, while a browser-playable H.264 copy is prepared in parallel. "
           "Frames are sampled at about 5 per second, so the sampling step for a video with frame rate <i>f</i> is")
    e, _ = eq(r"k = \max\left(1,\ \mathrm{round}\left(f\,/\,5\right)\right)")
    s += e
    s += h3("B. Shared vehicle detection and tracking")
    s += p("A YOLO11s detector fine-tuned on the Indian Driving Dataset (input 960 px, confidence 0.3) "
           "finds cars, buses, trucks, auto-rickshaws, motorcycles, riders, persons and other classes in every "
           "sampled frame, and ByteTrack links the boxes into tracks. This pass runs <b>once</b> per "
           "video; the helmet, triple-riding and plate tools replay its cached detections through an import hook, "
           "so all checks see identical boxes and track numbers. Boxes are compared with the intersection over "
           "union,")
    e, _ = eq(r"\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}")
    s += e
    s += p("which is also used to link a finding to the tracked vehicle behind it (IoU ≥ 0.3) and to find "
           "neighbouring vehicles whose plates must be read together.")
    s += h3("C. Helmet check")
    s += p("Only ridden or moving motorcycles that a second, COCO-trained detector also confirms are checked "
           "; bicycles, parked bikes and look-alikes are skipped. A head counts only if it sits over the "
           "bike, which removes posters, murals and bystanders, and a rider re-found by the tracker is stitched "
           "back into one person. A YOLO11s classifier at 128 px labels each head crop as helmet, no helmet or "
           "not a head. The frame probabilities <i>p<sub>t</sub></i> of a rider are combined with weights "
           "<i>w<sub>t</sub></i> that grow with the head size:")
    e, _ = eq(r"\bar{p}_c = \frac{\sum_t w_t\, p_{t,c}}{\sum_t w_t}, \qquad c \in \{\mathrm{helmet},\ \mathrm{no\ helmet}\}")
    s += e
    s += p("The rider is called <b>helmet</b> if the combined helmet probability is at least 0.90 and <b>no "
           "helmet</b> only if the combined no-helmet probability is at least 0.98 over at least three frames. If "
           "fewer than 85 % of the confident frames agree, or if the clearest view says the opposite, the answer "
           "is <b>not sure</b>. A confirmed no-helmet rider in the second or later seat is still sent to review, "
           "because a person standing next to the bike can be mistaken for a pillion rider.")
    e, _ = eq([r"d = \mathrm{no\ helmet} \quad \mathrm{if}\ \bar{p}_{nh} \geq 0.98,\ n \geq 3,\ \mathrm{agreement} \geq 0.85",
               r"d = \mathrm{helmet} \quad \mathrm{if}\ \bar{p}_{h} \geq 0.90",
               r"d = \mathrm{not\ sure} \quad \mathrm{otherwise}"])
    s += e
    s += h3("D. Triple-riding check")
    s += p("For every tracked motorcycle the crop of the bike with its riders is classified by a YOLO11s classifier "
           "(256 px) into one, two, three-or-more riders or not a bike, and the number of heads sitting on the seat "
           "is counted. Four features are combined by a logistic model fitted on held-out cameras: the logits of "
           "the three-or-more and two-rider probabilities, the share of frames with at least three heads on the "
           "seat, and the maximum head count divided by four. With standardised features "
           "<i>z<sub>j</sub></i> = (<i>x<sub>j</sub></i> − μ<sub>j</sub>) / σ<sub>j</sub>,")
    e, _ = eq(r"s = \sigma\left(b + \sum_{j=1}^{4} w_j z_j\right), \qquad \sigma(u) = \frac{1}{1+e^{-u}}")
    s += e
    s += p("A motorcycle is called <b>triple riding</b> only if <i>s</i> ≥ 0.897 and at least three frames agree; "
           "between 0.839 and 0.897 it is <b>not sure</b> and goes to review. The thresholds were chosen for zero "
           "wrong triple calls on held-out cameras with at most about one bike in twelve marked not sure.")
    s += h3("E. Red-light check")
    s += p("Signal heads are found by the detectors and by a colour finder that looks for red and green lamps next "
           "to each other that are never lit together. A YOLO11s classifier at 64 px labels each lamp crop as red, "
           "yellow, green or not a light. Because a lamp keeps its colour for seconds and changes in a fixed order, "
           "the per-frame probabilities are smoothed over time in the manner of a hidden Markov model: with "
           "transition matrix <i>A</i> and frame likelihoods <i>b<sub>t</sub></i>, the filtered state is")
    e, _ = eq(r"\alpha_t(j) \propto b_t(j) \sum_i \alpha_{t-1}(i)\, A_{ij}")
    s += e
    s += p("The controlled direction and the stop line are found automatically: the stop line is placed where the "
           "front of the queue of vehicles that came this way stands during red. A vehicle commits a violation when "
           "its ground point crosses the stop line in the controlled direction while the smoothed signal has been "
           "red for at least one second, both at the last sighting before the line and at the first after it. A "
           "challan additionally requires that the head was <b>seen changing colour</b> in the shot, so a red "
           "signboard or a red vehicle can at most cause a review. Just-turned-red cases, weak signal readings and "
           "short tracks go to review.")
    s += h3("F. Emergency-vehicle exemption")
    s += p("Each flagged vehicle crop is embedded with a CLIP ViT-B/16 image encoder, and a logistic "
           "probe trained with scikit-learn gives the probabilities of ambulance, police, fire and other. "
           "A vehicle is <b>exempt</b> if the emergency probability is at least 0.956 and is sent to review if it "
           "is at least 0.541. Exempt findings never become challans, and the reason is stored with the finding.")
    s += h3("G. Speed estimate")
    s += p("Speed is checked only when a one-time calibration (a known road length or standard lane dashes) and a "
           "speed limit are supplied. Box bottoms are mapped to the road plane using the camera geometry, and a "
           "velocity <i>v</i> = Δ<i>d</i> / Δ<i>t</i> is fitted per vehicle. The uncertain inputs (lens, "
           "vehicle size, tracking noise, model error) are varied in about 300 Monte-Carlo repetitions, in the "
           "spirit of the bootstrap, and the 90 % range is taken from the 5th and 95th percentiles:")
    e, _ = eq(r"[\,v_{lo},\ v_{hi}\,] = [\,Q_{0.05}(v^{(1..M)}),\ Q_{0.95}(v^{(1..M)})\,]")
    s += e
    s += p("Over-speeding is confirmed only if <i>v<sub>lo</sub></i> is above the limit and the confidence is "
           "high or medium; if the range includes legal speeds, the case is reviewed.")
    s += h3("H. Number-plate reading")
    s += p("For the vehicles flagged by a check, and the vehicles overlapping them, a YOLO plate detector finds the "
           "plate in every frame, and a fast-plate-ocr reader fine-tuned on Indian plates and exported to ONNX "
           "reads it. Characters are voted across frames, weighting each reading by its character "
           "confidence, and the result is checked against the Indian registration format (for example "
           "MH12AB1234). The decision is <b>READ</b> when two or more sharp frames agree without a sharp "
           "disagreement, <b>NOT SURE</b> when a reading exists but is not certain, and <b>UNREADABLE</b> "
           "otherwise. If the same plate is read on two vehicles, it is kept only on the vehicle with at least "
           "twice as many agreeing reads; otherwise all become NOT SURE. Only READ plates are printed; a NOT SURE "
           "reading is shown to the reviewer to confirm or correct.")
    s += h3("I. Routing, evidence and challan issue")
    s += p("Each finding is routed as shown in Figure 3.6. For every finding the pipeline saves the frames at "
           "<i>t</i> − 0.6 s, <i>t</i> and <i>t</i> + 0.6 s (raw and with the box drawn), a close-up crop, a "
           "clip of the original video and a clip of the tool's annotated video, and the plate crop when "
           "available. The evidence fingerprint is the SHA-256 hash over the names and contents of all "
           "evidence files <i>e</i><sub>1</sub>, …, <i>e<sub>n</sub></i> in sorted order:")
    e, _ = eq(r"H = \mathrm{SHA256}\left(\mathrm{name}_1 \,\|\, \mathrm{bytes}_1 \,\|\, \cdots \,\|\, \mathrm{name}_n \,\|\, \mathrm{bytes}_n\right)")
    s += e
    f, _ = figure(F / "f3_6.png", "Routing of a finding to challan, review or exemption", width=430)
    s += f
    s += p("A confirmed finding becomes an automatic challan numbered `TVP-YYYYMMDD-XXXXXX`, which stores the "
           "offences charged, the plate (a READ plate or the plate typed by the reviewer) and the evidence hash. A "
           "reviewer can issue a challan, reject the case or mark that more evidence is needed. The challan PDF is "
           "generated with ReportLab and contains the vehicle and offence details, the explanation, "
           "the evidence frames and crops, a QR code with the challan number and hash, and the fingerprint in the "
           "footer.")
    s += h3("J. One case per vehicle")
    s += p("The same vehicle can be flagged several times: by two checks, for two riders, after the tracker loses "
           "and re-finds it, or in a second upload of the same footage. TraffiSense groups such detections with a "
           "union–find structure (Figure 3.7). Two findings are joined when they belong to the same shared "
           "track or to tracks that the tracker re-found within one second at the same place and size, when the "
           "same plate was READ, when the same violation occurs within 8 s with near-identical close-ups (CLIP "
           "cosine similarity ≥ 0.92), or when another upload of the same footage shows the same vehicle at "
           "the same moment.")
    f, _ = figure(F / "f3_7.png", "One case per vehicle: merging repeat detections", width=430)
    s += f
    s += p("Re-uploads are recognised from a motion signature: each video is reduced to 64 × 36 grey frames at 5 "
           "per second, frame differences <i>d<sub>t</sub></i> are normalised, and the best alignment <i>k</i> "
           "between two videos maximises the average correlation of their motion,")
    e, _ = eq(r"k^{*} = \arg\max_k \frac{1}{|O_k|} \sum_{t \in O_k} \hat{d}^{A}_{t} \cdot \hat{d}^{B}_{t+k}, \qquad \mathrm{accepted\ if\ score} \geq 0.5")
    s += e
    s += p("Motion is compared instead of the picture itself, because the same fixed camera on another day shows "
           "the same road but never the same moving traffic. In each group one case is kept, in the order: a "
           "reviewer's challan, an automatic challan, a case waiting for more evidence, then the most confident "
           "case waiting for review. The others become <i>duplicate</i> records that remain linked for audit, and "
           "a different offence of the same vehicle is added to the kept case and, when certain, to its challan. "
           "A unique database index guarantees at most one challan per case. When an analysis is deleted, the "
           "findings merged into its cases are restored and re-merged.")
    s += h3("K. Lanes and digital twin")
    s += p("LaneTrace estimates lanes from how vehicles actually move during the first 30 seconds, ignoring painted "
           "markings, which are often faded in India; camera movement and scene cuts are detected. Lanes are "
           "<b>context only</b>: they support the evidence view and the digital twin and never create a challan. "
           "The digital twin rebuilds the scene in 3-D with three.js from the shared tracks: vehicles "
           "are placed on the road plane using the camera horizon, coloured by class, and violations are marked on "
           "a timeline.")
    s += h3("L. Adaptive signal control in simulation (PRAVAH)")
    s += p("Signal timing cannot be tested on live roads, so it is developed and evaluated in a microscopic simulator "
           "of an Indian four-arm junction built in Unity 6. Its core is a headless model with a 0.1 s "
           "step: vehicles follow the Intelligent Driver Model, drive on the left and arrive with an Indian "
           "mix (38 % two-wheelers, 28 % cars, 16 % autorickshaws, 8 % trucks, 6 % buses, 4 % bicycles) weighted in "
           "passenger-car units (PCU). Two-wheelers take lateral positions across the lane and creep past the stop "
           "line, each driver has a patience after which they jump the red, left turns are free, buses and autos "
           "stop just past the junction, a vehicle stuck in the junction box blocks cross traffic, and stray cattle "
           "occasionally block the kerb lane. A 3-D view with day, night and monsoon lighting and eight camera modes "
           "is used for demonstration.")
    s += p("PRAVAH (Person-weighted, Red-patience-Aware, Vehicle-mix Adaptive Headway control) is the signal "
           "controller developed in this project. It serves one approach at a time and uses only what a junction "
           "camera sees within 100 m of the stop line. A person who has waited <i>w</i> seconds contributes an "
           "urgency")
    e, _ = eq(r"U(w) = 1 + \left(\frac{w}{\tau}\right)^{2}, \qquad \tau = 75\ \mathrm{s}")
    s += e
    s += p("so that long red waits, which lead to red-light jumping, become expensive quickly. Each waiting vehicle "
           "counts its typical occupants (about 40 for a bus, 1.4 for a two-wheeler). The time needed to clear "
           "approach <i>a</i> and the gain of serving it next are")
    e, _ = eq(r"C_a = 2 + \frac{Q_a}{\hat{s}_a}, \qquad G_a = \frac{\sum_{i \in a} p_i \, U(w_i) \, \varphi(o_a)}{C_a + L}")
    s += e
    s += p("where <i>Q<sub>a</sub></i> is the queued load in PCU, <i>ŝ<sub>a</sub></i> the discharge rate, "
           "initialised from the Indian Roads Congress saturation flow of 525 PCU/h per metre of approach width "
           " and learned online from the vehicles that actually cross during green, <i>p<sub>i</sub></i> the "
           "occupants, <i>L</i> = 5 s the amber and all-red loss, and φ(<i>o</i>) = clip(1 − (<i>o</i> − 0.5)/0.4, 0, 1) "
           "a factor that discounts an approach whose exit is more than half full, so that the controller does not "
           "feed a blocked junction box. After its planned queue has cleared, a green continues only while it "
           "relieves more urgency per second than the best other approach would; any approach that has waited 90 s "
           "is served next. For an ambulance, the switch is timed so that the queue ahead of it has cleared when it "
           "arrives.")
    s += p("Four controllers are compared on identical arrivals: a fixed 30 s rotation, a fixed-time plan designed "
           "with Webster's method from the true flows, a queue-pressure adaptive controller (Adaptive v1) "
           "and PRAVAH. A TCP bridge on port 5055 lets controllers written in Python, including a Python port of "
           "PRAVAH and the project's own rule-based signal controller, run the simulated signal. On the "
           "<i>/signal/:id</i> page, the vehicles counted and classified in an analysed video, split by direction of "
           "travel, become the arrivals of one approach, the operator supplies assumptions for the other three, and "
           "the simulator runs every controller headless on a private copy of the project. Every number on this "
           "page is labelled SIMULATED.")
    s += h3("M. Authentication and security")
    s += p("Passwords must have at least 10 characters with upper- and lower-case letters, a digit and a symbol, and "
           "are stored as PBKDF2-HMAC-SHA256 hashes with a 16-byte random salt and 310,000 iterations. "
           "Sign-up is protected by a self-hosted CAPTCHA whose answer is bound to a signed token that expires "
           "after 5 minutes and can be used once. Accounts are activated by a six-digit code that is stored only as "
           "a keyed hash, expires after 10 minutes and allows at most 5 attempts. Sessions use an opaque random "
           "token in an HttpOnly cookie; only its HMAC is stored, and sessions expire after 7 days. Sensitive "
           "endpoints are rate-limited with a sliding window, and every API call checks the session on the server.")
    s += h("3.2.2", "Use Case Diagram / Activity Diagram / DFD", 2)
    s += p("<b>Use case diagram.</b> Figure 3.8 shows the use cases of TraffiSense. The operator signs up, verifies "
           "the account and signs in, uploads a video with a choice of checks, follows its progress, views the "
           "results, opens the evidence, twin and lane views and downloads challan PDFs. In the reviewer role the "
           "same user works through the review queue, confirms or types a plate and issues a challan, rejects the "
           "case or asks for more evidence. The model layer is a system actor that performs the analysis. Table 3.5 "
           "maps actors to use cases.")
    f, _ = figure(F / "f3_8.png", "Use case diagram of the TraffiSense platform", width=380, max_h=360)
    s += f
    t, _ = table("Mapping of actors to use cases", ["Actor", "Use cases"], [
        ["Operator", "Sign up, verify, sign in, reset password; upload a video and choose checks; follow progress; view results, evidence, digital twin and lanes; download challan PDFs; view analytics and settings."],
        ["Reviewer (operator in review role)", "Open the review queue; inspect frames, clips, reasons and the plate crop; confirm or type the plate; issue a challan, reject, or mark that more evidence is needed."],
        ["Model layer (system)", "Detect and track vehicles; run the selected checks; read plates; recognise emergency vehicles; estimate lanes."],
    ], [0.3, 0.7])
    s += t
    s += p("<b>Data flow diagrams.</b> The level-0 DFD in Figure 3.9 treats TraffiSense as a single process that "
           "exchanges data with the operator and the reviewer, uses the trained models and stores data in SQLite "
           "and the evidence folders. The level-1 DFD in Figure 3.10 decomposes the platform into six processes and "
           "four data stores, as listed in Table 3.6.")
    f, _ = figure(F / "f3_9.png", "Level-0 data flow diagram", width=420)
    s += f
    f, _ = figure(F / "f3_10.png", "Level-1 data flow diagram", width=430)
    s += f
    t, _ = table("Processes and data stores of the level-1 DFD", ["ID", "Name", "Description"], [
        ["1.0", "Authentication", "Sign-up with CAPTCHA, verification codes, sign-in, sessions, password reset."],
        ["2.0", "Upload and analysis queue", "Stores the original video, records options, queues the analysis."],
        ["3.0", "Run detectors", "Shared detection and tracking; helmet, triple-riding, red-light and speed checks."],
        ["4.0", "Evidence and plate linking", "Evidence frames and clips, plate reading, fingerprint, duplicate merging."],
        ["5.0", "Human review", "Presents uncertain cases; records the reviewer's decision, note and plate."],
        ["6.0", "Challan and PDF", "Issues one challan per vehicle and generates the e-challan PDF."],
        ["D1", "Users / sessions", "Accounts, hashed passwords, session hashes, verification codes."],
        ["D2", "Analyses", "Uploads, options, status, progress, metadata and artefacts."],
        ["D3", "Findings", "Findings with route, status, reasons, evidence and links to merged duplicates."],
        ["D4", "Challans", "Challan number, source, reviewer, plate, offences and evidence hash."],
    ], [0.08, 0.3, 0.62])
    s += t
    s += p("<b>Activity diagram.</b> The activity diagram in Figure 3.11 follows a video from upload to the final "
           "outcome across the operator, platform and reviewer lanes. After evidence is collected and duplicates "
           "are merged, the flow splits: a certain finding becomes an automatic challan, and an uncertain one goes "
           "to the reviewer, whose decision is recorded.")
    f, _ = figure(F / "f3_11.png", "Activity diagram of the video analysis workflow", width=360, max_h=410)
    s += f
    s += p("<b>Sequence of an analysis.</b> Figure 3.12 shows the message sequence of an analysis. The browser posts "
           "the video to `POST /api/analyses`; the server stores it, inserts the analysis row and submits it to the "
           "worker, then returns its id. While the browser polls the analysis, the worker runs the shared detection "
           "and the checks, reads plates for flagged vehicles, saves evidence and findings, merges duplicates and "
           "issues challans. Finally the browser can download a challan PDF. Table 3.7 lists the main API endpoints.")
    f, _ = figure(F / "f3_12.png", "Sequence diagram of a video analysis", width=430)
    s += f
    t, _ = table("Main API endpoints of TraffiSense", ["Endpoint", "Purpose"], [
        ["GET /api/auth/captcha; POST /api/auth/signup", "New CAPTCHA; create an account and send a verification code"],
        ["POST /api/auth/verify, /resend, /login, /logout", "Verify the code; resend it; sign in (session cookie); sign out"],
        ["POST /api/auth/forgot, /reset", "Send a reset code; set a new password"],
        ["POST /api/analyses; GET /api/analyses[/{id}]", "Upload a video with options; list analyses or read one with its findings"],
        ["DELETE /api/analyses/{id}", "Delete an analysis; merged findings elsewhere are restored"],
        ["GET /api/analyses/{id}/files/{path}; /twin", "Serve evidence and videos; digital-twin scene and tracks"],
        ["GET /api/findings[/{id}]; POST /api/findings/{id}/review", "List or open findings; record a review decision"],
        ["GET /api/challans[/{id}]; /api/challans/{id}/pdf", "List and open challans; download the e-challan PDF"],
        ["GET /api/stats, /api/analytics, /api/system, /api/plates", "Dashboard counts, analytics, model status, plate results"],
        ["GET, POST /api/analyses/{id}/signal-sim", "Traffic measured in the video and the last simulation; start a simulated signal-timing comparison"],
    ], [0.5, 0.5])
    s += t
    s += p("<b>Step-by-step workflows.</b> The operator workflow is:")
    s += numbered([
        "Create an account, solve the CAPTCHA and enter the six-digit code; then sign in.",
        "On the dashboard, read the guidance for a good video and drop a video file; choose the checks (no helmet, triple riding, red light; speed with a calibration).",
        "Follow the live stage and progress bar while the analysis runs in the background.",
        "Open the results: play the video with overlays, read each finding's reasons and confidence, and open its evidence.",
        "Download the PDF of any automatic challan, and open the digital twin and lane views for context.",
    ])
    s += p("The review workflow is:")
    s += numbered([
        "Open the human review queue, ordered with the most confident cases first.",
        "Inspect the evidence frames, clips and the model's reasons, including why the case was uncertain.",
        "Confirm or type the number plate from the plate crop if it was not read with certainty.",
        "Decide: issue a challan (choosing the offences to charge), reject the case, or mark that more evidence is needed. The decision and note are recorded.",
    ])
    s += h("3.2.3", "Hardware and Software Specifications", 2)
    s += p("TraffiSense is developed and run on an Apple Silicon Mac. The models run on the Apple GPU through PyTorch's "
           "MPS backend; heavier training jobs were also run on Kaggle. Table 3.8 gives the hardware and Table 3.9 "
           "lists the software.")
    t, _ = table("Hardware specification", ["Item", "Specification"], [
        ["Processor / GPU", "Apple M4 (CPU, integrated GPU via Metal / MPS, media engine for H.264)"],
        ["Memory", "16 GB unified memory"],
        ["Storage", "Space for the virtual environment, model weights, datasets and per-analysis evidence folders"],
        ["Training (additional)", "Kaggle notebooks (CPU and T4 GPU) for longer training runs"],
        ["Network", "Needed only for package installation and for e-mail delivery of verification codes"],
    ], [0.28, 0.72])
    s += t
    t, _ = table("Software specification", ["Category", "Software"], [
        ["Operating system", "macOS 26"],
        ["Languages / runtimes", "Python 3.14 (backend and tools), Node.js 24 and npm (frontend build)"],
        ["Frontend", "React 19, TypeScript 6, Vite 8, react-router 7, three.js, lucide-react"],
        ["Backend", "FastAPI 0.141, uvicorn, Pydantic, SQLite (stdlib sqlite3), Pillow, ReportLab 5"],
        ["Deep learning and vision", "PyTorch 2.14 (MPS), Ultralytics 8.4 (YOLO11, ByteTrack), OpenCV 5, open_clip, scikit-learn"],
        ["Plate reader", "fast-plate-ocr (fine-tuned), ONNX Runtime"],
        ["Media", "FFmpeg (H.264 via VideoToolbox, software fallback)"],
        ["Traffic simulation", "Unity 6 (6000.6.4f1) with the Universal Render Pipeline, C#; CC0 sky and road textures"],
        ["Version control", "Git"],
    ], [0.28, 0.72])
    s += t
    s += p("The platform is configured through environment variables; no secret is stored in the code. Table 3.10 "
           "lists the configuration parameters.")
    t, _ = table("Configuration parameters", ["Parameter", "Default", "Purpose"], [
        ["PORT", "8600", "Port of the web server"],
        ["TVP_DATA", "app/data", "Folder of the database and analyses"],
        ["TVP_PYTHON", ".venv/bin/python", "Python used to run the tools"],
        ["TVP_SMTP_HOST / USER / PASS / FROM", "unset", "E-mail delivery of codes; unset = developer mode"],
        ["SESSION_DAYS", "7", "Session lifetime"],
        ["MAX_UPLOAD_MB", "2048", "Maximum upload size"],
        ["TVP_DEVICE", "mps (if available)", "Device for the models"],
        ["TVP_UNITY", "newest Unity in Unity Hub", "Unity editor used to run the junction simulator headless"],
    ], [0.38, 0.22, 0.4])
    s += t
    s += p("The platform is started with one command. The script builds the web app if needed and starts the "
           "server:")
    s += code(["# one-time setup (project root)", "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
               "cd app/web && npm install && npm run build", "", "# start the platform -> http://127.0.0.1:8600",
               "./app/start.sh", "", "# frontend development (proxies /api to :8600)", "cd app/web && npm run dev"])
    s += p("Without SMTP settings, verification codes are shown in a clearly marked developer banner and written to "
           "`data/dev_outbox.log`, which allows the platform to be demonstrated offline.")
    s += p("The junction simulator runs headless from the command line, which is also how the platform starts it. "
           "Controllers written in Python are served through the bridge:")
    s += code(["# compare all controllers -> signal_sim/Results/compare.md",
               "Unity -batchmode -nographics -projectPath signal_sim \\",
               "      -executeMethod SignalSim.Experiment.Run", "",
               "# Python controller through the TCP bridge (port 5055)",
               "python3 signal_bridge/server.py --controller pravah", "",
               "# robustness study -> signal_sim/Results/robustness.csv",
               "Unity -batchmode -nographics -projectPath signal_sim \\",
               "      -executeMethod SignalSim.Robustness.Run -seeds 10",
               "python3 signal_sim/Tools/robustness_report.py"])
    return s
