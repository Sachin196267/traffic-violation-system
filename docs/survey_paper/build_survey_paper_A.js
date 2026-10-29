// IEEE-style two-column survey paper for the traffic-violation + adaptive-signal project.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, SectionType, LevelFormat, VerticalAlign,
} = require("docx");

const OUT = "/Users/sachinnair/Desktop/traffic violation system/docs/survey_paper";
const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN_LR = 812, COL_GAP = 360;
const TEXT_W = PAGE_W - 2 * MARGIN_LR;            // 10282 DXA
const COL_W = (TEXT_W - COL_GAP) / 2;             // 4961 DXA
const pxFromDxa = (d) => Math.round(d / 1440 * 96);

// ---------- text helpers ----------
const r = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, superScript: o.sup, subScript: o.sub, color: o.color });
// rich: array of strings or [text, opts]
const runs = (parts, size) => parts.map((p) => (typeof p === "string" ? r(p, { size }) : r(p[0], { size, ...p[1] })));
const body = (parts, o = {}) => new Paragraph({
  children: runs(Array.isArray(parts) ? parts : [parts], o.size),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 240 }, indent: o.noIndent ? undefined : { firstLine: 240 },
});
const heading = (text) => new Paragraph({
  children: [r(text.toUpperCase(), { size: 20 })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
});
const sub = (text) => new Paragraph({ children: [r(text, { italics: true })], spacing: { before: 120, after: 60 }, keepNext: true });
const bullet = (parts) => new Paragraph({
  children: runs(Array.isArray(parts) ? parts : [parts]), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 240 },
});
const numbered = (parts) => new Paragraph({
  children: runs(Array.isArray(parts) ? parts : [parts]), numbering: { reference: "contrib", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 240 },
});
const equation = (parts, n) => new Paragraph({
  children: [...runs(parts), r(`\t(${n})`)], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  tabStops: [{ type: "right", position: COL_W - 20 }],
});
const caption = (text) => new Paragraph({ children: [r(text, { size: 16 })], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 } });
const tableTitle = (num, text) => [
  new Paragraph({ children: [r(`TABLE ${num}`, { size: 16 })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 }, keepNext: true }),
  new Paragraph({ children: [r(text.toUpperCase(), { size: 16 })], alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: true }),
];
const figure = (file, widthDxa) => {
  const buf = fs.readFileSync(path.join(OUT, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const pw = pxFromDxa(widthDxa);
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: pw, height: Math.round(pw * h / w) },
      altText: { title: file, description: file, name: file } })],
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
  });
};

// ---------- table helper ----------
const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
function table(widths, header, rows, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, isHead, align) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: line, bottom: line, left: line, right: line },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "E7ECF3" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: runs(Array.isArray(text) ? text : [isHead ? [text, { bold: true }] : text], o.size || 16),
      alignment: align || (isHead ? AlignmentType.CENTER : AlignmentType.LEFT),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((row) => new TableRow({ cantSplit: true, children: row.map((c, i) => cell(c, widths[i], false, o.align && o.align[i])) })),
    ],
  });
}

// ---------- page / section setup ----------
const pageProps = (cols) => ({
  page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1300, left: MARGIN_LR, right: MARGIN_LR } },
  column: cols === 2 ? { count: 2, space: COL_GAP, equalWidth: true } : { count: 1 },
});
const section = (cols, children, first = false) => ({
  properties: { ...pageProps(cols), ...(first ? {} : { type: SectionType.CONTINUOUS }) }, children,
});

// ---------- title block ----------
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const author = (name, lines, email) => new TableCell({
  width: { size: Math.floor(TEXT_W / 3), type: WidthType.DXA },
  borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, children: [r(name, { size: 20 })] }),
    ...lines.map((l) => new Paragraph({ alignment: AlignmentType.CENTER, children: [r(l, { size: 18, italics: l.startsWith("Dept") })] })),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [r(email, { size: 18, color: "1F3B63" })] }),
  ],
});
const affil = ["Dept. of Computer Engineering", "Pillai College of Engineering", "Navi Mumbai, India"];
const W3 = Math.floor(TEXT_W / 3);
const blank = () => new TableCell({ width: { size: W3, type: WidthType.DXA }, borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder }, children: [new Paragraph("")] });
const authorTable = new Table({
  width: { size: W3 * 3, type: WidthType.DXA }, columnWidths: [W3, W3, W3], alignment: AlignmentType.CENTER,
  borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
  rows: [
    new TableRow({ children: [author("Sachin Nair", affil, "nsachin23comp@student.mes.ac.in"), author("Aditya Vijay Kumar", affil, "[email]"), author("Ranjit Iyer", affil, "[email]")] }),
    new TableRow({ children: [blank(), author("Dr. Siddharth Hariharan", affil, "[email]"), blank()] }),
  ],
});
const TITLE = "TraffiSense: An Explainable, Digital-Twin-Driven Intelligent Traffic Management and Violation Detection System";
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [r(TITLE, { size: 44 })] }),
  authorTable,
];

const abstract = [
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [
    r("Abstract", { bold: true, italics: true, size: 18 }), r("—", { bold: true, size: 18 }),
    r("Indian cities face rapid growth in vehicles, and their junctions carry heterogeneous, lane-less traffic in which two-wheelers, autorickshaws, cars, buses and trucks share the same road. Violations such as riding without a helmet, triple riding and jumping a red light are common, yet they are still largely detected by officers watching camera feeds, and many junctions run fixed signal plans that ignore the traffic actually waiting. Recent advances in deep learning have made it possible to detect vehicles, riders, helmets and number plates in camera video in real time, and to adjust signal timing from measured demand. This paper surveys ten recent works (2025–2026) on adaptive and intelligent signal control, emergency-vehicle priority, real-time object detection, Indian traffic datasets, two-wheeler violation detection and low-resolution number-plate recognition. Based on the gaps identified, it proposes an integrated system in which one shared video-analysis layer serves both automated enforcement, through precision-first violation modules and an e-challan platform, and person-weighted adaptive signal control evaluated in a simulator of an Indian junction.", { bold: true, size: 18 }),
  ] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
    r("Keywords", { bold: true, italics: true, size: 18 }), r("—", { bold: true, size: 18 }),
    r("Intelligent Transportation Systems, Traffic Violation Detection, Adaptive Traffic Signal Control, Number Plate Recognition, Helmet Detection, Triple Riding, Emergency Vehicle Priority, Deep Learning, YOLO", { bold: true, italics: true, size: 18 }),
  ] }),
];

const intro = [
  heading("I. Introduction"),
  body("Road traffic in Indian cities is unlike the lane-disciplined traffic for which most enforcement and signal-control systems were designed. Two-wheelers make up a large share of vehicles and move between lanes, autorickshaws, buses and trucks share the same carriageway, and left turns are often free. Violations such as riding without a helmet, carrying three people on a two-wheeler and jumping a red light are frequent and are a major cause of road injuries, but manual enforcement cannot watch every junction at every hour."),
  body("At the same time, many junctions still run fixed-time signal plans. When the plan does not match the traffic, vehicles wait at red while the green approach is empty, queues spill back into neighbouring junctions, and long waits encourage drivers to jump the signal. Emergency vehicles such as ambulances are delayed in the same queues, which can cost lives."),
  body("Deep learning has changed what a camera can do. Real-time detectors such as YOLO [4] find vehicles and people in each frame, and datasets recorded on Indian roads [5] allow these detectors to learn local vehicle types such as autorickshaws and tempo-travellers. Researchers have used them to detect helmet violations [6], triple riding [7], [8] and to generate e-tickets automatically [9], while low-resolution number-plate recognition has become a research problem of its own [10]. In signal control, machine learning [1] and large language models [2] are being used to choose signal timing, and emergency vehicles can be detected from their sirens and appearance and given priority [3]."),
  body("The objective of this survey is to study these recent methods, compare their approaches and results, identify the gaps that matter for Indian junctions and, based on them, propose an integrated system that combines automated violation detection with adaptive signal control using the same junction cameras."),
];

const lit = [
  heading("II. Literature Survey"),
  body("Tolani et al. [1] proposed a machine-learning based system for adaptive traffic prediction and control. Proximity sensors placed upstream of the signal measure the speed and density of approaching vehicles, and a model built on the Edge Impulse platform predicts vehicle density and arrival time. The signal timing is then adjusted in real time instead of following preset timers, which reduces congestion and waiting time."),
  body("Lai et al. [2] introduced LLMLight, in which a large language model acts as the decision agent for traffic signal control. The model receives a prompt describing the current traffic conditions and reasons about which phase to serve. A specialised backbone, LightGPT, was trained for the task, and experiments on ten real and synthetic datasets, together with an evaluation by fifteen human experts, showed better generalisation and interpretability than reinforcement-learning controllers."),
  body("Alruwaili et al. [3] addressed emergency response times with a system that detects ambulances from both sound and video. Sirens are recognised by an LSTM network on MFCC features and ambulances in images by a transfer-learned ResNet18 network, and the result drives Raspberry Pi-based signal prioritisation so that the ambulance's approach receives green."),
  body("Tian et al. [4] presented YOLOv12, an attention-centric real-time object detector that brings the accuracy of attention mechanisms into the YOLO family while keeping the speed of CNN-based detectors. Its smallest model reaches 40.6 % mAP with a latency of 1.64 ms on a T4 GPU."),
  body("Sharma et al. [5] released the Urban Vision Hackathon dataset, UVH-26, containing 26,646 full-HD images from about 2,800 Bengaluru Safe-City CCTV cameras. The images were annotated by 565 students with 1.8 million bounding boxes over 14 Indian vehicle classes, and detection models trained on the data were also released."),
  body("Deshpande et al. [6] developed a two-stage helmet violation detection system for Indian smart-city scenarios, in which a DetectNet_v2 model from the NVIDIA TAO toolkit localises riders and YOLOv8 detects helmet violations and number plates. On 1,715 images from Indian roads it reported 98.56 % helmet detection and 97.6 % number-plate detection accuracy."),
  body("Deshpande et al. [7] also proposed a triple-riding detection system for surveillance cameras mounted at different heights and angles. A ResNet18-based DetectNet_v2 handles occlusion and distant vehicles and YOLOv8 counts riders and reads plates, giving 91.42 % triple-rider, 98 % two-wheeler and 81 % number-plate detection accuracy."),
  body("Saravanan and Rajini [8] classified double and triple riders using a modified Mask R-CNN framework with DenseNet-121 and ResNet-50 networks, reporting 97.93 % triple-rider detection, 97.41 % classification and 98.22 % segmentation accuracy across different weather conditions."),
  body("Rawat et al. [9] built DashCop, an end-to-end system that processes dashcam video to detect triple riding and helmet non-compliance and generates e-tickets automatically. A segmentation and cross-association module links riders to their motorcycles, and the annotated RideSafe-400 video dataset was released for evaluation."),
  body("Laroca et al. [10] organised the ICPR 2026 competition on low-resolution license plate recognition using the LRLPR-26 dataset, which contains 20,000 training and 3,000 test tracks, each pairing low- and high-resolution images of the same plate from real surveillance cameras. It shows that distance and compression often make plates unreadable, which is the usual situation for plates in wide junction-camera views."),
  body("From the survey, four gaps are identified for Indian junctions. The signal controllers [1], [2] measure demand in vehicles, although a bus carrying forty people and a two-wheeler carrying one count very differently, and they do not treat long red waits as a cause of red-light jumping. Emergency priority [3] is granted as soon as the vehicle is detected, not timed to the queue ahead of it. The violation detectors [6]–[9] are evaluated by accuracy, while an enforcement system must also recognise when it is not sure, since a wrong challan affects a citizen, and plates are often too small to read [10]. Finally, enforcement and signal control are studied separately even though they use the same cameras. Table I summarises the surveyed papers."),
];

const litRows = [
  ["1", "M. Tolani, G. E. Saathwik, A. Roy, L. A. Ameeth, D. B. Rao, A. Bajpai and A. Balodi, \u201cMachine Learning Based Adaptive Traffic Prediction and Control Using Edge Impulse Platform,\u201d Scientific Reports, 2025 [1]",
   "Proposes an autonomous signal-timing system driven by the density of vehicles approaching the signal. Proximity sensors placed at a fixed distance upstream measure vehicle speed and density, and an Edge-Impulse machine-learning model predicts density and arrival time. Signal timings are adjusted in real time instead of using preset timers, reducing congestion and delay. [1]"],
  ["2", "S. Lai, Z. Xu, W. Zhang, H. Liu and H. Xiong, \u201cLLMLight: Large Language Models as Traffic Signal Control Agents,\u201d ACM SIGKDD (KDD), 2025 [2]",
   "Uses a large language model as the decision agent for traffic signal control, prompted with real-time traffic conditions. A specialised backbone, LightGPT, is trained for signal control to reduce cost. Experiments on ten real and synthetic datasets and evaluation by fifteen human experts show better generalisation and interpretability than reinforcement-learning methods. [2]"],
  ["3", "M. Alruwaili, A. Ali, M. Almutairi, A. Alsahyan and M. Mohamed, \u201cLSTM and ResNet18 for Optimized Ambulance Routing and Traffic Signal Control in Emergency Situations,\u201d Scientific Reports, 2025 [3]",
   "Presents a real-time traffic management system to reduce emergency medical response times. Sirens are detected with an LSTM on MFCC spectrograms and ambulances in images with a transfer-learned ResNet18; the outputs drive Raspberry Pi-based signal prioritisation. Priority is granted on detection. [3]"],
  ["4", "Y. Tian, Q. Ye and D. Doermann, \u201cYOLOv12: Attention-Centric Real-Time Object Detectors,\u201d arXiv, 2025 [4]",
   "Introduces an attention-centric YOLO framework that keeps the speed of CNN-based detectors while gaining the accuracy of attention. YOLOv12-N reaches 40.6 % mAP with 1.64 ms latency on a T4 GPU, outperforming earlier real-time detectors. [4]"],
  ["5", "A. Sharma et al., \u201cThe Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic,\u201d arXiv, 2025 [5]",
   "Releases UVH-26: 26,646 1080p images from about 2,800 Bengaluru Safe-City CCTV cameras, annotated by 565 students with 1.8 million boxes over 14 Indian vehicle classes such as two-wheeler, auto-rickshaw and tempo-traveller, along with detection models trained on the data. [5]"],
  ["6", "U. U. Deshpande et al., \u201cComputer-Vision Based Automatic Rider Helmet Violation Detection and Vehicle Identification in Indian Smart City Scenarios Using NVIDIA TAO Toolkit and YOLOv8,\u201d Frontiers in Artificial Intelligence, 2025 [6]",
   "A two-stage system: DetectNet_v2 localises riders and YOLOv8 detects helmet violations and extracts number plates. On a custom dataset of 1,715 motorcycle images from Indian roads it reports 98.56 % helmet detection and 97.6 % number-plate detection accuracy. [6]"],
  ["7", "U. U. Deshpande et al., \u201cAutomatic Two-Wheeler Rider Identification and Triple-Riding Detection in Surveillance Systems Using Deep-Learning Models,\u201d Discover Artificial Intelligence, 2025 [7]",
   "Detects triple riding from surveillance cameras mounted at different heights and angles. A ResNet18-based DetectNet_v2 handles occlusion and distant vehicles, and YOLOv8 counts riders and reads plates, reporting 91.42 % triple-rider, 98 % two-wheeler and 81 % number-plate detection accuracy. [7]"],
  ["8", "M. Saravanan and G. K. Rajini, \u201cAutomatic Detection of Triples Rider Violations Using an Optimized Mask R-CNN Framework,\u201d Systems Science & Control Engineering, 2025 [8]",
   "Classifies double and triple riders, with and without motorcycles, using a modified Mask R-CNN with DenseNet-121 and ResNet-50 networks. Reports 97.93 % triple-rider detection, 97.41 % double/triple classification and 98.22 % segmentation accuracy across weather conditions. [8]"],
  ["9", "D. Rawat, K. Gupta, A. B. Roy and R. K. Sarvadevabhatla, \u201cDashCop: Automated E-ticket Generation for Two-Wheeler Traffic Violations Using Dashcam Videos,\u201d arXiv, 2025 [9]",
   "An end-to-end system that processes vehicle-mounted dashcam video to detect triple riding and helmet non-compliance and generate e-tickets. A segmentation and cross-association module links riders to their motorcycles, and the RideSafe-400 annotated video dataset is released for evaluation. [9]"],
  ["10", "R. Laroca et al., \u201cICPR 2026 Competition on Low-Resolution License Plate Recognition,\u201d ICPR, 2026 [10]",
   "Reports the first competition dedicated to low-resolution plate recognition on real surveillance data. The LRLPR-26 dataset has 20,000 training and 3,000 test tracks of paired low- and high-resolution plate images, showing how distance and compression make plates hard to read. [10]"],
];
// Paper column in the sample: bold title (year) [n]
const paperTitle = (p) => { const m = p.match(/“(.+?),?”.*?(20\d\d) \[(\d+)\]/); return `${m[1]} (${m[2]}) [${m[3]}]`; };

const proposed = [
  heading("III. Proposed System"),
  body("The proposed system (Fig. 1) uses one shared video-analysis layer to serve two purposes: automated enforcement, which produces evidence and e-challans, and adaptive signal control, which uses the measured traffic to decide signal timing. Every module follows one rule: an automatic decision is made only when the evidence is certain, and uncertain cases are sent to a human reviewer instead of being guessed."),
  body("Video from junction cameras is sampled at a few frames per second. A YOLO detector trained on Indian driving data finds vehicles, riders and pedestrians, and a multi-object tracker keeps one identity for each vehicle across frames, so that every module examines the same tracked vehicle and each vehicle is decided once."),
  body("The violation modules work on these tracks. The helmet module classifies each rider's head on enlarged crops and votes over several frames before reporting a violation. The triple-riding module counts the riders on each two-wheeler and keeps a “not sure” band between its thresholds. The red-light module reads the signal state and checks whether a vehicle crosses the stop line during red, and the speed module reports a calibrated speed range with a confidence level rather than a single value."),
  body("For identification, the number-plate module searches for the plate inside each tracked vehicle, reads it with a recogniser fine-tuned on Indian plates, combines the readings across frames and checks the Indian registration format. Each plate is marked as READ, NOT SURE or UNREADABLE, and only READ plates are printed on a challan. An emergency-vehicle classifier identifies ambulances, police and fire vehicles so that they are exempted from violations and can be given priority at the signal."),
  body("Confirmed violations are stored on a web platform built with FastAPI and React, where reviewers see the video with each finding on a timeline, confirm or reject uncertain cases, and issue PDF e-challans that carry the evidence frames and a plain-language explanation."),
  body("For signal control, the system proposes PRAVAH (Person-weighted, Red-patience-Aware, Vehicle-mix Adaptive Headway control). Instead of counting vehicles, PRAVAH weights each waiting vehicle by its typical number of occupants, and the urgency of a wait grows with the square of its duration so that long red waits, which lead to red-light jumping, are served before they become dangerous. The time needed to clear each approach is estimated from its queue and a discharge rate learned online, the controller avoids sending traffic into a blocked exit, no approach waits longer than a fixed limit, and an ambulance's green is timed so that the queue ahead of it has cleared when it arrives. Because signal timing cannot be tested on live roads, the controller is evaluated in a Unity-based simulator of an Indian four-arm junction with mixed traffic, red-light jumping, free left turns and bus stops, which can also replay traffic measured from analysed videos."),
];

const applications = [
  heading("IV. Applications"),
  body([["Automated traffic enforcement: ", { bold: true }], "Helmet, triple-riding, red-light and speeding violations can be detected from existing junction CCTV cameras, reducing the manual effort of traffic police."]),
  body([["E-challan generation: ", { bold: true }], "Confirmed violations with readable number plates can be converted into e-challans carrying evidence frames, making penalties transparent and verifiable."]),
  body([["Adaptive signal control: ", { bold: true }], "Signal timing can follow the people actually waiting at a junction, reducing delay, congestion and fuel consumption."]),
  body([["Emergency-vehicle priority: ", { bold: true }], "Ambulances, fire engines and police vehicles can be detected, exempted from violations and given a timely green signal."]),
  body([["Smart-city traffic monitoring: ", { bold: true }], "Vehicle counts, mix, speeds and queue lengths collected by the system can help authorities plan roads and signal timings."]),
  body([["Road-safety analysis: ", { bold: true }], "Records of frequent violations by location and time can guide awareness campaigns and the placement of enforcement resources."]),
];

const future = [
  heading("V. Future Scope"),
  body("The system can be improved by training the detection modules on more street-level and low-resolution footage, so that small or distant riders and number plates are recognised more often. More violation types, such as wrong-side driving, mobile-phone use while riding and illegal parking, can be added on the same shared tracks. The adaptive signal controller can be validated at a real junction with measured queue lengths, extended to coordinate neighbouring junctions along a corridor and to include pedestrian phases, and deployed on edge devices beside the camera. Integration with the national vehicle registration database would allow e-challans to be delivered directly to vehicle owners."),
];

const refs = [
  "M. Tolani, G. E. Saathwik, A. Roy, L. A. Ameeth, D. B. Rao, A. Bajpai and A. Balodi, \u201cMachine learning based adaptive traffic prediction and control using edge impulse platform,\u201d Scientific Reports, vol. 15, art. 17161, 2025, doi: 10.1038/s41598-025-00762-4.",
  "S. Lai, Z. Xu, W. Zhang, H. Liu and H. Xiong, \u201cLLMLight: Large language models as traffic signal control agents,\u201d in Proc. 31st ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD), vol. 1, 2025, pp. 2335\u20132346, doi: 10.1145/3690624.3709379.",
  "M. Alruwaili, A. Ali, M. Almutairi, A. Alsahyan and M. Mohamed, \u201cLSTM and ResNet18 for optimized ambulance routing and traffic signal control in emergency situations,\u201d Scientific Reports, vol. 15, art. 6011, 2025, doi: 10.1038/s41598-025-89651-4.",
  "Y. Tian, Q. Ye and D. Doermann, \u201cYOLOv12: Attention-centric real-time object detectors,\u201d arXiv:2502.12524, 2025.",
  "A. Sharma, C. Mhatre, S. Gawali, R. Bokkasam, B. Kishore, V. Pattanaik, T. Rambha, A. R. Pinjari, V. Kovvali, A. Chakraborty, P. Rathore, R. Krishnapuram and Y. Simmhan, \u201cThe Urban Vision Hackathon dataset and models: Towards image annotations and accurate vision models for Indian traffic,\u201d arXiv:2511.02563, 2025.",
  "U. U. Deshpande, G. K. O. Michael, S. D. C. S. Araujo, V. Deshpande, R. Patil, R. A. A. Chate, V. R. Tandur, S. S. Goudar, S. Ingale and V. Charantimath, \u201cComputer-vision based automatic rider helmet violation detection and vehicle identification in Indian smart city scenarios using NVIDIA TAO toolkit and YOLOv8,\u201d Frontiers in Artificial Intelligence, vol. 8, art. 1582257, 2025, doi: 10.3389/frai.2025.1582257.",
  "U. U. Deshpande, S. Shanbhag, R. Patil, R. A. A. Chate, S. D. C. S. Araujo, K. Pinto, J. Arjun, M. Patil and B. Nirwani, \u201cAutomatic two-wheeler rider identification and triple-riding detection in surveillance systems using deep-learning models,\u201d Discover Artificial Intelligence, vol. 5, art. 104, 2025, doi: 10.1007/s44163-025-00263-3.",
  "M. Saravanan and G. K. Rajini, \u201cAutomatic detection of triples rider violations using an optimized mask R-CNN framework,\u201d Systems Science & Control Engineering, vol. 13, no. 1, art. 2546829, 2025, doi: 10.1080/21642583.2025.2546829.",
  "D. Rawat, K. Gupta, A. B. Roy and R. K. Sarvadevabhatla, \u201cDashCop: Automated e-ticket generation for two-wheeler traffic violations using dashcam videos,\u201d arXiv:2503.00428, 2025.",
  "R. Laroca et al., \u201cICPR 2026 competition on low-resolution license plate recognition,\u201d in Proc. Int. Conf. Pattern Recognition (ICPR), 2026, arXiv:2604.22506.",
];
const references = [
  heading("References"),
  ...refs.map((t, i) => new Paragraph({
    children: [r(`[${i + 1}]\t${t}`, { size: 16 })], alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 50 }, indent: { left: 400, hanging: 400 }, tabStops: [{ type: "left", position: 400 }],
  })),
];

const doc = new Document({
  creator: "Sachin Nair", title: TITLE,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    section(1, titleBlock, true),
    section(2, [...abstract, ...intro, ...lit]),
    section(1, [...tableTitle("I", "Literature Survey"),
      table([900, 3300, 6082], ["SR. No.", "Paper", "Paper Summary"], litRows.map(([n, paper, summary]) => [n, [[paperTitle(paper), { bold: true }]], summary]),
        { align: [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.JUSTIFIED], size: 18 }),
      new Paragraph({ spacing: { after: 120 }, children: [] })]),
    section(2, proposed),
    section(1, [figure("fig1_architecture.png", TEXT_W - 3700), caption("Fig. 1: Block diagram of Proposed System.")]),
    section(2, [...applications, ...future, ...references]),
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const f = path.join(OUT, "Survey_Paper_A.docx");
  fs.writeFileSync(f, buf);
  console.log("wrote", f);
});
