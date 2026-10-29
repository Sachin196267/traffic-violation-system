"""Abstract, Chapter 1 (Introduction) and Chapter 2 (Literature Survey)."""
from story import *
from engine import FIG

F = FIG


def abstract():
    s = []
    s += [Mark("mode", mode="roman")]
    s += unnumbered("Abstract", "abstract")[1:]
    s += p("This project presents TraffiSense: An Explainable, Digital-Twin-Driven Intelligent Traffic Management and Violation Detection System, an evidence-first traffic enforcement platform that turns recorded "
           "traffic video into explainable violation decisions for Indian roads. The core problem addressed is that "
           "two-wheeler offences such as riding without a helmet and triple riding, and signal offences such as "
           "red-light jumping, are still enforced mostly by officers watching junctions or control-room screens, "
           "which is slow, tiring and prone to single-glance mistakes. TraffiSense integrates several deep-learning "
           "components into one web application: a YOLO11 vehicle detector fine-tuned on the Indian Driving Dataset "
           "with ByteTrack multi-object tracking; a rider-level helmet classifier; a rider-count classifier for triple "
           "riding; a signal-state model with temporal smoothing and automatic stop-line estimation for red-light "
           "jumping; a calibrated speed estimator that reports a 90 % speed range; an Indian number-plate reader that "
           "prints a plate only when it is read with certainty; and a CLIP-based probe that exempts ambulances, police "
           "vehicles and fire engines. Every decision is taken over many frames of the same tracked vehicle, carries a "
           "plain-language explanation, a confidence value and fingerprinted evidence frames and clips, and is routed "
           "to an automatic e-challan only when the model is certain. Uncertain cases go to a human review queue, and "
           "repeat detections of the same vehicle are merged so that a vehicle never receives two challans. The "
           "platform is built with a React and TypeScript frontend, a FastAPI backend and an SQLite database, and "
           "includes secure sign-up with verification, a digital-twin view, lane estimation and analytics. For signal "
           "control, a Unity simulator of an Indian four-arm junction is used to evaluate PRAVAH, a person-weighted, "
           "red-patience-aware adaptive signal controller developed in this project; in simulation it gave the lowest "
           "mean wait per person of the four controllers compared under every one of twelve tested sets of simulator "
           "assumptions, and the platform can replay the traffic measured in an uploaded video through it. TraffiSense "
           "is a decision-support prototype: it favours “not sure” over a wrong challan.")
    return s


def lists(entries):
    """List of Figures / List of Tables pages"""
    s = []
    s += unnumbered("List of Figures", "lof")[1:]
    for key, text in lists.LOF:
        s.append(DotLine(text, entries.get(key, "0"), 0, height=18.2))
    s += unnumbered("List of Tables", "lot")
    for key, text in lists.LOT:
        s.append(DotLine(text, entries.get(key, "0"), 0, height=18.2))
    return s


def ch1():
    s = chapter(1, "Introduction")
    s += h("1.1", "Fundamentals")
    s += p("Two-wheelers form the largest share of road traffic in Indian cities, and they are involved in a large "
           "share of road deaths. The Motor Vehicles (Amendment) Act, 2019 makes riding without a protective "
           "headgear, carrying more than one pillion rider and jumping a red signal punishable offences. In "
           "practice these rules are enforced mainly by traffic personnel who watch a junction, or by operators who "
           "watch CCTV screens in a control room. The approach does not scale to the number of cameras that cities "
           "now install, it depends on the attention of individual officers, and a decision taken from one glance at "
           "one frame is easily wrong when the view is blurred, partly hidden or badly lit.")
    s += p("Deep learning makes it possible for a computer to watch the same footage. Real-time object detectors of "
           "the YOLO family locate every vehicle, rider and person in a frame in a single pass [@yolov12], and "
           "multi-object trackers such as ByteTrack link these detections over time so that each road user keeps a "
           "stable identity. Once a motorcycle is tracked, its riders, their heads and its number plate "
           "can be examined in every frame in which they are visible, and evidence from many frames can be combined "
           "before any decision is taken. Earlier Indian work on two-wheeler enforcement has shown that such "
           "pipelines can reach high detection accuracy, but also that low light, occlusion and camera placement "
           "still cause errors when a decision rests on a single frame [@deshpande].")
    s += p("Accuracy alone is not enough for enforcement. A challan affects a citizen, so the reason for it must be "
           "visible, the evidence must be preserved, and a person must be able to check a doubtful case. A wrong "
           "challan on an unclear rider or a guessed plate is worse than no challan, and plates on wide camera views "
           "are often too small to read at all [@laroca]. A practical enforcement system therefore needs three things at once: reliable "
           "multi-frame perception, an explanation and evidence trail for every decision, and a clear rule for when "
           "a human must decide.")
    s += p("<b>TraffiSense</b> was proposed in the preceding Project-A as an architecture for explainable, "
           "multi-frame violation detection with human review, a digital twin and adaptive signal control. In this "
           "Project-B it has been implemented as a working web platform together with a junction simulator for "
           "signal control. An operator uploads recorded traffic "
           "video; the platform detects and tracks every vehicle, runs the selected violation checks, reads number "
           "plates, collects evidence and decides for every finding whether it becomes an automatic e-challan, goes "
           "to a human reviewer, or is exempt because the vehicle is an ambulance, a police vehicle or a fire engine.")
    s += p("TraffiSense is deliberately positioned as an <b>evidence-first decision-support</b> system. Its operating "
           "rule is precision first: when a model is not sure, the case is sent to a person instead of producing a "
           "challan, and a number plate is printed only when it has been read with certainty across frames. Every "
           "output is information intended for an authorised human, who remains responsible for enforcement.")
    s += h3("Key concepts used in this report")
    s += p("Table 1.1 summarises the technical terms that recur throughout the report. The distinction between a "
           "<i>finding</i>, a <i>case</i> and a <i>challan</i> is particularly important, because a single vehicle "
           "can produce several findings but must never receive more than one challan.")
    t, _ = table("Key concepts used in TraffiSense", ["Concept", "Meaning in the context of TraffiSense"], [
        ["Analysis", "One uploaded video together with the checks selected for it, its processing status and results."],
        ["Finding", "One detection of a possible offence by one detector for one tracked vehicle: time, confidence, model decision and reasons."],
        ["Route", "Where a finding goes: <i>confirmed</i> (automatic challan), <i>review</i> (human queue) or <i>exempt</i> (emergency vehicle)."],
        ["Multi-frame decision", "A decision combined over all frames in which the same tracked vehicle or rider is visible, never from one frame."],
        ["Operating point", "The thresholds of a model's decision rule, chosen on held-out cameras so that wrong challans are avoided."],
        ["Not sure", "The model's answer when the evidence lies between its thresholds; such cases are reviewed by a person."],
        ["Plate decision", "READ (certain, may be printed), NOT SURE (shown to the reviewer to confirm) or UNREADABLE."],
        ["Evidence fingerprint", "SHA-256 hash over every evidence file of a finding, printed on the challan."],
        ["Duplicate merging", "Grouping all findings of one vehicle into one case, so that one vehicle receives at most one challan."],
        ["Context only", "Output such as lane structure that supports understanding but never creates a challan."],
        ["Human-in-the-loop", "Design principle that an authorised person decides every case the model is not certain about."],
    ], [0.26, 0.74])
    s += t
    s += p("Figure 1.1 gives a high-level view of how these elements fit together. The operator works through the "
           "React web application. The FastAPI server stores the upload and hands it to a background analysis "
           "worker, which runs the trained detectors and turns their output into findings with evidence. Each "
           "finding is then routed to an automatic challan, to human review, or to exemption.")
    f, _ = figure(F / "f1_1.png", "High-level view of the TraffiSense enforcement workflow", width=460)
    s += f
    s += p("A further design goal is to make the result easy to trust. For every finding the operator can see the "
           "evidence frames before, at and after the moment of the offence, a short clip of the original video, the "
           "model's confidence, the reasons it gave in plain language and, where available, the number-plate crop. "
           "The same information is printed on the e-challan PDF together with the evidence fingerprint.")
    s += h("1.2", "Objectives")
    s += p("The objective of this work is to design, implement and demonstrate an explainable, evidence-first "
           "traffic enforcement platform for recorded Indian traffic video. The specific objectives are as follows:")
    s += numbered([
        "To detect and track every vehicle in an uploaded video with a detector trained on Indian road scenes and a multi-object tracker, running the detection pass once and sharing it with all checks.",
        "To detect riding without a helmet, triple riding, red-light jumping and over-speeding with multi-frame decision rules whose thresholds favour “not sure” over a wrong challan.",
        "To read Indian number plates across frames and print a plate only when it is read with certainty.",
        "To exempt emergency vehicles (ambulance, police, fire) from challans and record the reason.",
        "To attach evidence frames, clips, plain-language reasons and a SHA-256 fingerprint to every finding, and to generate a downloadable e-challan PDF.",
        "To route every uncertain finding to a human review queue and guarantee at most one challan per vehicle.",
        "To deliver the platform as a secure multi-page web application with verified accounts, analytics, a digital-twin view and lane estimation.",
        "To design an adaptive signal controller for Indian mixed traffic and evaluate it against fixed-time and adaptive baselines in a simulator of an Indian junction, clearly labelling every such result as simulated.",
    ])
    s += h("1.3", "Scope")
    s += p("The scope of the project covers an integrated academic prototype consisting of a React and TypeScript "
           "frontend of 18 pages, a FastAPI backend with an SQLite database, and a model layer of trained tools for "
           "vehicle detection and tracking, helmet use, triple riding, signal state and red-light crossing, speed "
           "estimation, number-plate reading, emergency-vehicle recognition and lane estimation, together with a Unity "
           "junction simulator and the PRAVAH signal controller. The prototype "
           "supports account creation with verification codes, sign-in and password reset, video upload with a "
           "choice of checks, live progress, results with evidence, a human review queue, automatic and reviewed "
           "e-challans with PDF download, an evidence gallery, a digital-twin scene, lane detection, analytics, "
           "settings, and a simulated signal-timing comparison for the traffic of an analysed video.")
    s += p("The signal-control part of the scope covers a microscopic simulator of an Indian four-arm junction "
           "with mixed traffic, red-light jumping, free left turns and bus stops; the PRAVAH controller and three "
           "baseline controllers (fixed 30 s, Webster and queue-pressure adaptive); a Python bridge through which "
           "controllers written in Python can run the simulated signal; and a page of the platform that replays the "
           "traffic measured in an analysed video through the simulator. All signal-timing results are simulated "
           "and are labelled as such wherever they appear.")
    s += p("The following boundaries apply. They are part of the project definition:")
    s += bullets([
        "TraffiSense analyses <b>recorded</b> video uploaded by an operator. It does not connect to live camera streams and does not control real traffic signals; adaptive signal control is implemented and evaluated only in simulation, in a Unity junction simulator that takes the place of the SUMO digital twin proposed in Project-A.",
        "TraffiSense is a decision-support prototype. Its challans are records produced by the platform for review and demonstration; they are not legally issued notices and are not connected to any government e-challan system.",
        "Speed is checked only when a one-time camera calibration and a speed limit are supplied, and lane position never creates a challan.",
        "Verification codes are delivered by e-mail when SMTP settings are provided; without them the platform runs in a clearly marked developer mode.",
    ])
    s += h("1.4", "Organization of the Project Report")
    s += p("The report is organized as follows:")
    s += bullets([
        "<b>Chapter 1</b> introduces the fundamentals of camera-based traffic enforcement and adaptive signal control, and states the objectives and the scope of the project.",
        "<b>Chapter 2</b> reviews ten recent works on adaptive signal control, emergency-vehicle priority, real-time object detection, Indian traffic datasets, two-wheeler violation detection and low-resolution number-plate recognition, and summarises the research gaps.",
        "<b>Chapter 3</b> presents the implemented system: the overview, the existing and proposed architectures, the algorithms and techniques including the PRAVAH signal controller and the junction simulator, the use case, data flow, activity and sequence diagrams, and the hardware and software specifications.",
        "<b>Chapter 4</b> describes the inputs, datasets, database and outputs, shows the screens of the platform, defines the evaluation parameters and reports the performance of every component, including signal control in simulation.",
        "<b>Chapter 5</b> summarises the work and the components delivered, and outlines the future scope.",
        "The references, the list of publications with the survey paper, and the acknowledgement follow.",
    ])
    return s


def ch2():
    s = chapter(2, "Literature Survey")
    s += h("2.1", "Introduction")
    s += p("In this chapter the relevant literature is reviewed. It describes the techniques used in camera-based "
           "traffic enforcement and adaptive signal control, and the limitations that motivate the design of "
           "TraffiSense. Ten recent works (2025–2026) are reviewed, and their summary is presented at the end of "
           "the chapter.")
    s += p("Traffic management has moved from officers at the junction and fixed signal plans towards camera-based "
           "systems driven by deep learning. Signal timing can now follow measured demand [@tolani] and even be "
           "decided by large language models [@llmlight], and emergency vehicles can be detected and given priority "
           "[@alruwaili]. Real-time detectors understand traffic scenes from ordinary cameras [@yolov12], datasets "
           "recorded on Indian roads teach them local vehicle types [@uvh], and dedicated systems judge helmet use "
           "[@dhelmet], count riders [@deshpande,saravanan], generate e-tickets [@dashcop] and read number plates at "
           "low resolution [@laroca].")
    s += p("Figure 2.1 groups the reviewed works into six families: adaptive signal control, emergency-vehicle "
           "priority, real-time object detection, Indian traffic datasets, two-wheeler violation detection, and "
           "number-plate recognition. The following sections review each work.")
    f, _ = figure(F / "f2_1.png", "Families of AI techniques reviewed in the literature", width=440)
    s += f
    s += h("2.2", "Literature Review")
    s += p("Each subsection describes the reviewed work, its method and results, and its main advantages and "
           "disadvantages for an Indian enforcement and signal-control system.")
    s += h("2.2.1", "Adaptive Traffic Signal Control", 2)
    s += p("Tolani et al. proposed a machine-learning based system for adaptive traffic prediction and control. "
           "Proximity sensors placed upstream of the signal measure the speed and density of approaching vehicles, "
           "and a model built on the Edge Impulse platform predicts vehicle density and arrival time, so that the "
           "signal timing is adjusted in real time instead of following preset timers [@tolani]. Lai et al. "
           "introduced LLMLight, in which a large language model acts as the decision agent for signal control: it "
           "receives a prompt describing the current traffic and reasons about which phase to serve. A specialised "
           "backbone, LightGPT, was trained for the task, and experiments on ten real and synthetic datasets, "
           "together with an evaluation by fifteen human experts, showed better generalisation and "
           "interpretability than reinforcement-learning controllers [@llmlight].")
    s += bullets(["<b>Advantages:</b> green times follow real demand; the language-model agent can explain its choice of phase.",
                  "<b>Disadvantages:</b> demand is counted in vehicles, not people; long red waits that provoke red-light jumping are not treated as a risk; both assume lane-based queues."])
    s += h("2.2.2", "Emergency-Vehicle Priority", 2)
    s += p("Alruwaili et al. addressed emergency response times with a system that detects ambulances from both "
           "sound and video. Sirens are recognised by an LSTM network on MFCC features and ambulances in images by "
           "a transfer-learned ResNet18 network, and the result drives Raspberry Pi-based signal prioritisation so "
           "that the ambulance's approach receives green [@alruwaili].")
    s += bullets(["<b>Advantages:</b> two independent cues (sound and image) for detection; low-cost hardware.",
                  "<b>Disadvantages:</b> priority is granted as soon as the vehicle is detected, not timed to the queue in front of it; emergency vehicles are not exempted from violation checks."])
    s += h("2.2.3", "Real-Time Object Detection", 2)
    s += p("Tian et al. presented YOLOv12, an attention-centric real-time object detector that brings the accuracy "
           "of attention mechanisms into the YOLO family while keeping the speed of CNN-based detectors. Its "
           "smallest model reaches 40.6 % mAP on COCO with a latency of 1.64 ms on a T4 GPU, outperforming earlier "
           "real-time detectors [@yolov12].")
    s += bullets(["<b>Advantages:</b> real-time speed with one model for all road users; strong accuracy for its size.",
                  "<b>Disadvantages:</b> trained on generic data that lacks Indian vehicle types; a single frame gives no identity over time."])
    s += h("2.2.4", "Indian Traffic Datasets", 2)
    s += p("Sharma et al. released the Urban Vision Hackathon dataset, UVH-26, containing 26,646 full-HD images "
           "from about 2,800 Bengaluru Safe-City CCTV cameras. The images were annotated by 565 students with 1.8 "
           "million bounding boxes over 14 Indian vehicle classes, such as two-wheeler, autorickshaw and "
           "tempo-traveller, and detection models trained on the data were released with it [@uvh].")
    s += bullets(["<b>Advantages:</b> elevated CCTV viewpoint like that of junction cameras; local vehicle classes.",
                  "<b>Disadvantages:</b> a single city; a very large download."])
    s += h("2.2.5", "Helmet Violation Detection", 2)
    s += p("Deshpande et al. developed a two-stage helmet violation detection system for Indian smart-city "
           "scenarios, in which a DetectNet_v2 model from the NVIDIA TAO toolkit localises riders and YOLOv8 "
           "detects helmet violations and number plates. On 1,715 images from Indian roads it reported 98.56 % "
           "helmet detection and 97.6 % number-plate detection accuracy [@dhelmet].")
    s += bullets(["<b>Advantages:</b> high accuracy on curated images; links the violation directly to the plate.",
                  "<b>Disadvantages:</b> decisions on single images; no explicit “not sure” outcome for unclear riders."])
    s += h("2.2.6", "Triple-Riding Detection", 2)
    s += p("Deshpande et al. also proposed a triple-riding detection system for surveillance cameras mounted at "
           "different heights and angles. A ResNet18-based DetectNet_v2 handles occlusion and distant vehicles, and "
           "YOLOv8 counts riders and reads plates, giving 91.42 % triple-rider, 98 % two-wheeler and 81 % "
           "number-plate detection accuracy [@deshpande]. Saravanan and Rajini classified double and triple riders "
           "using a modified Mask R-CNN framework with DenseNet-121 and ResNet-50 networks, reporting 97.93 % "
           "triple-rider detection, 97.41 % classification and 98.22 % segmentation accuracy across different "
           "weather conditions [@saravanan].")
    s += bullets(["<b>Advantages:</b> high accuracy; robustness to camera angle and weather.",
                  "<b>Disadvantages:</b> riders of neighbouring bikes can be counted as passengers; plate reading is much weaker than rider counting; single-frame decisions."])
    s += h("2.2.7", "Automated E-Ticket Generation", 2)
    s += p("Rawat et al. built DashCop, an end-to-end system that processes dashcam video to detect triple riding "
           "and helmet non-compliance and generates e-tickets automatically. A segmentation and cross-association "
           "module links riders to their motorcycles, and the annotated RideSafe-400 video dataset was released for "
           "evaluation [@dashcop].")
    s += bullets(["<b>Advantages:</b> works on video end to end, from detection to the e-ticket; rider–motorcycle association.",
                  "<b>Disadvantages:</b> moving dashcam viewpoint; tickets are issued without a human-review step or an evidence fingerprint."])
    s += h("2.2.8", "Low-Resolution Number-Plate Recognition", 2)
    s += p("Laroca et al. organised the ICPR 2026 competition on low-resolution license plate recognition using the "
           "LRLPR-26 dataset, which contains 20,000 training and 3,000 test tracks, each pairing low- and "
           "high-resolution images of the same plate captured by real surveillance cameras. It shows that distance "
           "and compression often make plates unreadable, which is the usual situation for plates in wide "
           "junction-camera views [@laroca].")
    s += bullets(["<b>Advantages:</b> realistic surveillance data; multi-frame tracks of the same plate.",
                  "<b>Disadvantages:</b> non-Indian plates; even the best methods leave many plates unreadable, so a reader must be able to abstain."])
    s += p("Table 2.1 compares the technique families on the criteria that matter most for an enforcement and "
           "signal-control platform.")
    t, _ = table("Comparison of the technique families", ["Technique family", "Accuracy", "Transparency", "Relevance to TraffiSense"], [
        ["Adaptive signal control [@tolani,llmlight]", "Gains over fixed timing", "Medium (rules) to low (LLM)", "PRAVAH controller in a Unity junction simulator"],
        ["Emergency priority [@alruwaili]", "High with two cues", "High", "Emergency exemption and timed ambulance priority"],
        ["Object detection [@yolov12]", "High in real time", "Low per frame", "Shared vehicle detection pass (IDD-trained YOLO11s)"],
        ["Indian datasets [@uvh]", "Large gains from local data", "Not applicable", "Indian vehicle classes in training"],
        ["Two-wheeler violations [@dhelmet,deshpande,saravanan,dashcop]", "High on curated images", "Medium", "Helmet and rider-count checks with a “not sure” band"],
        ["Low-resolution plates [@laroca]", "Low on distant plates", "High with per-character votes", "READ / NOT SURE / UNREADABLE plate decisions"],
    ], [0.3, 0.2, 0.2, 0.3])
    s += t
    s += h("2.3", "Summary of Literature Survey")
    s += p("The literature shows a clear shift from manual observation and fixed signal plans to deep-learning "
           "perception and adaptive control, but most systems still decide on single frames, solve one problem in "
           "isolation, and do not say when they are unsure. Table 2.2 summarises the surveyed works.")
    t, _ = table("Literature survey summary", ["SN", "Technique / work", "Author and year", "Advantages and disadvantages"], [
        ["1", "Machine-learning based adaptive traffic prediction and control using Edge Impulse", "Tolani et al., 2025 [@tolani]",
         "<b>Advantages:</b> green times follow predicted density and arrival time. <b>Disadvantages:</b> needs upstream sensors; demand counted per vehicle."],
        ["2", "LLMLight: large language models as traffic signal control agents", "Lai et al., 2025 [@llmlight]",
         "<b>Advantages:</b> better generalisation and interpretability than RL on ten datasets. <b>Disadvantages:</b> heavy model; lane-based benchmark networks."],
        ["3", "LSTM and ResNet18 for ambulance routing and signal control", "Alruwaili et al., 2025 [@alruwaili]",
         "<b>Advantages:</b> sirens and images both detect ambulances; Raspberry Pi signal priority. <b>Disadvantages:</b> priority on detection, regardless of the queue ahead."],
        ["4", "YOLOv12: attention-centric real-time object detectors", "Tian et al., 2025 [@yolov12]",
         "<b>Advantages:</b> 40.6 % mAP at 1.64 ms on a T4 GPU. <b>Disadvantages:</b> generic training data; no Indian vehicle classes."],
        ["5", "UVH-26 Indian traffic dataset and models", "Sharma et al., 2025 [@uvh]",
         "<b>Advantages:</b> 26,646 CCTV images, 1.8 million boxes, 14 Indian classes. <b>Disadvantages:</b> single city; very large download."],
        ["6", "Rider helmet violation detection with NVIDIA TAO and YOLOv8", "Deshpande et al., 2025 [@dhelmet]",
         "<b>Advantages:</b> 98.56 % helmet and 97.6 % plate detection on 1,715 images. <b>Disadvantages:</b> single-image decisions; no “not sure” outcome."],
        ["7", "Two-wheeler rider identification and triple-riding detection", "Deshpande et al., 2025 [@deshpande]",
         "<b>Advantages:</b> 91.42 % triple-rider and 98 % two-wheeler detection. <b>Disadvantages:</b> 81 % plate detection; single-frame decisions."],
        ["8", "Triple-rider violations with an optimised Mask R-CNN", "Saravanan and Rajini, 2025 [@saravanan]",
         "<b>Advantages:</b> 97.93 % triple-rider detection across weather conditions. <b>Disadvantages:</b> heavy two-stage model; no plate reading."],
        ["9", "DashCop: e-ticket generation from dashcam video", "Rawat et al., 2025 [@dashcop]",
         "<b>Advantages:</b> end-to-end e-tickets; rider–motorcycle association; RideSafe-400 dataset. <b>Disadvantages:</b> moving viewpoint; no human review."],
        ["10", "ICPR 2026 competition on low-resolution plate recognition", "Laroca et al., 2026 [@laroca]",
         "<b>Advantages:</b> 20,000 training and 3,000 test tracks of real surveillance plates. <b>Disadvantages:</b> non-Indian plates; many plates remain unreadable."],
    ], [0.06, 0.32, 0.2, 0.42], small=True)
    s += t
    s += p("<b>Research gaps.</b> The survey exposes several gaps that motivate the TraffiSense design. First, most "
           "violation detectors decide on single frames, so glare, occlusion or a neighbouring rider can produce a "
           "wrong record. Second, systems rarely state when they are <i>not sure</i>; a forced decision on an unclear "
           "rider becomes a wrong challan. Third, number-plate readers return their best guess, although a guessed "
           "plate charges the wrong citizen. Fourth, the same vehicle is often detected many times, by different "
           "checks or in a re-uploaded video, and no mechanism prevents several challans for one vehicle. Fifth, "
           "emergency vehicles are seldom exempted automatically. Sixth, explanation and evidence are usually "
           "limited to a confidence score. Seventh, adaptive signal controllers count vehicles rather than the "
           "people waiting, which suits Indian mixed traffic poorly. Table 2.3 maps these gaps to the corresponding TraffiSense design "
           "decisions.")
    t, _ = table("Research gaps and the corresponding TraffiSense design decisions", ["Research gap", "TraffiSense design decision"], [
        ["Single-frame decisions", "Shared detection and ByteTrack; every check votes over all frames of a tracked vehicle or rider."],
        ["No “not sure” outcome", "Operating points chosen on held-out cameras for zero wrong calls; between thresholds a case goes to human review."],
        ["Guessed number plates", "Per-character voting across frames and an Indian format check; only READ plates are printed."],
        ["Several challans for one vehicle", "Union-find merging by track, plate, appearance and re-upload matching; unique index of one challan per case."],
        ["Emergency vehicles challaned", "CLIP probe exempts ambulance, police and fire vehicles with a recorded reason."],
        ["Weak evidence and explanation", "Three evidence frames, crop, clips, plain-language reasons and a SHA-256 fingerprint on every finding and challan."],
        ["Signal control counts vehicles, not people", "PRAVAH weights waiting people and long red waits; evaluated in a simulator of an Indian junction."],
    ], [0.32, 0.68])
    s += t
    return s
