"""Diagrams for the TraffiSense report, drawn in the pastel box style of the reference report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Times New Roman"

GREEN, BLUE, PURPLE, YELLOW, ORANGE, GREY, RED, WHITE = (
    "#DDEFDD", "#DCE8F7", "#E6DDF3", "#FBF1C7", "#FBE3CF", "#EEEEEE", "#F6D5D5", "#FFFFFF")
EDGE = "#4a4a4a"


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=BLUE, fs=9, bold=False, dashed=False, ec=EDGE, round_=True, align="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2" if round_ else "square,pad=0",
                       fc=fc, ec=ec, lw=0.9, ls="--" if dashed else "-")
    ax.add_patch(p)
    if text:
        ax.text(x + (w / 2 if align == "center" else 1.5), y + h / 2, text, ha=align, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True, linespacing=1.15)
    return (x, y, w, h)


def label(ax, x, y, text, fs=9, bold=False, ha="left", va="top", color="black", style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight="bold" if bold else "normal", color=color, fontstyle=style)


def arrow(ax, p, q, text=None, fs=7.5, both=False, ls="-", rad=0.0, tpos=0.5, toff=(0, 1.2)):
    a = FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=9, lw=0.9, color="#333",
                        ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        mx, my = p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fs, fontstyle="italic", color="#333")


def line(ax, pts, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#333", lw=0.9, ls=ls)


def mid(b, side):
    x, y, w, h = b
    return {"t": (x + w / 2, y + h), "b": (x + w / 2, y), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1.1 high-level workflow
def fig_1_1():
    fig, ax = canvas(8.6, 3.4)
    a = box(ax, 2, 22, 11, 7, "Operator\n(uploads video)", GREEN, bold=True)
    b = box(ax, 17, 22, 11, 7, "React web app\n(TraffiSense UI)", BLUE)
    c = box(ax, 32, 22, 12, 7, "FastAPI server\n(:8600)", BLUE)
    d = box(ax, 48, 22, 12, 7, "Analysis worker\n(background)", PURPLE)
    e = box(ax, 62.5, 22, 9, 7, "Findings\n+ evidence", ORANGE)
    arrow(ax, mid(a, "r"), mid(b, "l")); arrow(ax, mid(b, "r"), mid(c, "l")); arrow(ax, mid(c, "r"), mid(d, "l"))
    arrow(ax, mid(d, "r"), mid(e, "l"))
    m = box(ax, 44, 6, 20, 9, "Detectors: helmet, triple riding,\nred light, speed, plate reader,\nemergency check (lanes = context)", PURPLE, fs=8)
    arrow(ax, mid(d, "b"), (54, 15), both=True)
    s = box(ax, 24, 6, 14, 8, "SQLite database +\nevidence folders", GREY)
    arrow(ax, mid(c, "b"), (34, 14), both=True)
    cc = box(ax, 74, 25.5, 9, 6, "Automatic\nchallan", RED, fs=8.5)
    rv = box(ax, 74, 16, 9, 6, "Human\nreview", YELLOW, fs=8.5)
    ex = box(ax, 74, 6.5, 9, 6, "Exempt\n(emergency)", GREEN, fs=8.5)
    arrow(ax, (71.5, 27), mid(cc, "l"))
    arrow(ax, (71.5, 24), mid(rv, "l"))
    arrow(ax, (70, 22), (74, 9.5))
    label(ax, 3, 15, "Every decision is multi-frame,\nexplained and fingerprinted", fs=8, style="italic", color="#8b1a1a")
    save(fig, "f1_1")


# ---------------------------------------------------------------- Fig 2.1 technique families
def fig_2_1():
    fig, ax = canvas(7.2, 3.2)
    box(ax, 22, 23, 28, 6.5, "AI techniques for traffic management\nand violation detection", GREY, bold=True, fs=9.5)
    fams = [("Adaptive signal\ncontrol\n[1], [2]", GREEN), ("Emergency-vehicle\npriority\n[3]", GREEN),
            ("Real-time object\ndetection\n[4]", BLUE), ("Indian traffic\ndatasets\n[5]", BLUE),
            ("Two-wheeler\nviolations\n[6]-[9]", PURPLE), ("Low-resolution\nnumber plates\n[10]", PURPLE)]
    w, gap = 10.6, 1.2
    x0 = (72 - (6 * w + 5 * gap)) / 2
    line(ax, [(36, 23), (36, 19.5)])
    line(ax, [(x0 + w / 2, 19.5), (x0 + 5 * (w + gap) + w / 2, 19.5)])
    for i, (txt, c) in enumerate(fams):
        x = x0 + i * (w + gap)
        box(ax, x, 6, w, 10, txt, c, fs=7.8)
        arrow(ax, (x + w / 2, 19.5), (x + w / 2, 16))
    label(ax, 36, 1.0, "Green: signal control   Blue: perception   Purple: enforcement", fs=7.5, ha="center", va="bottom", style="italic")
    save(fig, "f2_1")


# ---------------------------------------------------------------- Fig 3.1 existing flow
def fig_3_1():
    fig, ax = canvas(7.2, 3.3)
    steps = ["CCTV / phone\nfootage", "Officer watches\nscreen or junction", "Offence noted\nfrom one view",
             "Plate read or\nwritten by hand", "Challan issued\n(few reasons)"]
    lims = ["Footage reviewed\nonly partly", "Tiring, slow,\nnot round-the-clock", "Single-frame:\nglare, occlusion",
            "Misread plates;\nno confidence", "No evidence trail\nor explanation"]
    for i, (s, l) in enumerate(zip(steps, lims)):
        x = 1 + i * 14.4
        b = box(ax, x, 18, 12.5, 9, s, BLUE, fs=8.5)
        r = box(ax, x, 3, 12.5, 8, l, RED, fs=8)
        arrow(ax, mid(b, "b"), mid(r, "t"), ls="--")
        if i < 4:
            arrow(ax, mid(b, "r"), (x + 14.4, 22.5))
    label(ax, 1, 31, "Existing process", bold=True, fs=9)
    label(ax, 1, 14.2, "Limitation at each stage", bold=True, fs=9, color="#8b1a1a")
    save(fig, "f3_1")


# ---------------------------------------------------------------- Fig 3.2 architecture
def fig_3_2():
    fig, ax = canvas(7.2, 5.6)
    op = box(ax, 4, 49, 14, 5, "Operator / Reviewer", GREEN, bold=True)
    box(ax, 2, 36, 68, 10, "", BLUE, dashed=True)
    label(ax, 3, 45.3, "Presentation tier: React + TypeScript (Vite build), served by the same server", bold=True, fs=8.5)
    for i, t in enumerate(["Auth pages\n(sign in, sign up,\nverify, reset)", "Analyses, results,\nevidence, twin, lanes",
                           "Review queue,\nchallans, PDF", "Analytics, models,\nsettings"]):
        box(ax, 4 + i * 16.5, 37.5, 15, 6, t, WHITE, fs=7.8)
    arrow(ax, mid(op, "b"), (11, 46))
    box(ax, 2, 15, 42, 18, "", GREEN, dashed=True)
    label(ax, 3, 32.3, "FastAPI server (Python, uvicorn, :8600)", bold=True, fs=8.5)
    for i, t in enumerate(["Auth + sessions + CAPTCHA + codes", "Analyses API + file serving",
                           "Findings, review, challans, PDF", "Stats, analytics, digital-twin scene"]):
        box(ax, 4, 26.5 - i * 3.0, 24, 2.5, t, WHITE, fs=7.5)
    pw = box(ax, 30, 17, 12, 11, "Analysis\npipeline\n(background\nworker)", PURPLE, fs=8)
    arrow(ax, (36, 36), (23, 33), "REST / JSON", both=True, tpos=0.5, toff=(4, -1))
    box(ax, 48, 6, 22, 27, "", PURPLE, dashed=True)
    label(ax, 49, 32.3, "Model layer (tool subprocesses)", bold=True, fs=8.5)
    for i, t in enumerate(["Shared detection + tracking", "Helmet check", "Triple-riding check", "Red-light check",
                           "Speed estimate", "Number-plate reader", "Emergency probe (CLIP)", "Lanes (context only)"]):
        box(ax, 50, 28.2 - i * 2.85, 18, 2.4, t, WHITE, fs=7.5)
    arrow(ax, mid(pw, "r"), (48, 22.5), "process\ncalls", both=True, toff=(0, 0.5))
    db = box(ax, 2, 2, 20, 8, "SQLite (WAL)\nusers, sessions, analyses,\nfindings, reviews, challans", GREY, fs=7.8)
    fs_ = box(ax, 24, 2, 20, 8, "data/analyses/<id>/\noriginal video, evidence,\nclips, SHA-256", GREY, fs=7.8)
    arrow(ax, (12, 15), mid(db, "t"), both=True); arrow(ax, (34, 15), mid(fs_, "t"), both=True)
    save(fig, "f3_2")


# ---------------------------------------------------------------- Fig 3.3 runtime view
def fig_3_3():
    fig, ax = canvas(7.2, 3.6)
    br = box(ax, 2, 24, 15, 8, "Browser\n(React SPA)", GREEN, fs=8.5)
    up = box(ax, 24, 22, 20, 12, "uvicorn process\nFastAPI app + static web/dist\nport 8600", BLUE, fs=8.5)
    arrow(ax, mid(br, "r"), mid(up, "l"), "HTTP, cookie session", both=True)
    pool = box(ax, 24, 6, 20, 11, "ThreadPoolExecutor\n1 analysis at a time\n(+ lane pool, side pool)", PURPLE, fs=8.5)
    arrow(ax, mid(up, "b"), mid(pool, "t"), "submit(id)")
    tl = box(ax, 52, 20, 20, 14, "Tool subprocesses\n(.venv Python, Ultralytics,\nPyTorch on Apple GPU via\nMPS; plate reader ONNX)", PURPLE, fs=8)
    ff = box(ax, 52, 4, 20, 10, "ffmpeg\n(H.264 copies, clips,\nmotion fingerprint)", ORANGE, fs=8.5)
    arrow(ax, mid(pool, "r"), (52, 27), both=True)
    arrow(ax, (44, 9), mid(ff, "l"))
    d = box(ax, 2, 4, 15, 12, "app/data/\nplatform.db\nanalyses/<id>/", GREY, fs=8.5)
    arrow(ax, mid(pool, "l"), mid(d, "r"), both=True)
    save(fig, "f3_3")


# ---------------------------------------------------------------- Fig 3.4 page map
def fig_3_4():
    fig, ax = canvas(7.2, 4.0)
    ax.set_ylim(-2.5, 40)
    groups = [("Public (4)", GREEN, ["/login  Sign in", "/signup  Create account", "/verify  Verify code", "/forgot  Reset password"]),
              ("Analysis (6)", BLUE, ["/  Dashboard + upload", "/analyses  Video analyses", "/analyses/:id  Results", "/twin/:id  Digital twin", "/lanes/:id  Lane detection", "/signal/:id  Signal timing (sim)"]),
              ("Enforcement (4)", RED, ["/violations  Violations", "/review/:id  Human review", "/challans  Challans", "/challans/:id  Challan detail"]),
              ("Platform (4)", PURPLE, ["/evidence  Evidence", "/analytics  Analytics", "/system  Models & system", "/settings  Settings"])]
    for i, (g, c, items) in enumerate(groups):
        x = 1 + i * 17.8
        box(ax, x, 31, 16.5, 5, g, c, bold=True, fs=9)
        for j, t in enumerate(items):
            b = box(ax, x, 24.4 - j * 4.5, 16.5, 3.8, t, WHITE, fs=7.6)
        line(ax, [(x + 8.25, 31), (x + 8.25, 28.2)])
    label(ax, 36, -0.2, "Every page after sign-in sits inside the shared layout (sidebar + session check)", fs=7.8,
          ha="center", va="bottom", style="italic")
    save(fig, "f3_4")


# ---------------------------------------------------------------- Fig 3.5 pipeline
def fig_3_5():
    fig, ax = canvas(7.2, 7.4)
    a = box(ax, 20, 66, 32, 5, "Upload: original video kept unchanged", GREEN, fs=8.5)
    b = box(ax, 20, 58, 32, 5, "Probe; 1080p working copy if 4K or > 30 fps", BLUE, fs=8.5)
    c = box(ax, 20, 50, 32, 5, "Shared detection + ByteTrack (5 fps, once)", BLUE, fs=8.5)
    arrow(ax, mid(a, "b"), mid(b, "t")); arrow(ax, mid(b, "b"), mid(c, "t"))
    dets = [("Helmet\ncheck", PURPLE), ("Triple\nriding", PURPLE), ("Red light\n+ signal", PURPLE), ("Speed\nrange", PURPLE)]
    xs = [3, 20.5, 38, 55.5]
    for (t, col), x in zip(dets, xs):
        d = box(ax, x, 38, 14, 7, t, col, fs=8.5)
        arrow(ax, (36, 50), (x + 7, 45))
    e = box(ax, 3, 28, 66.5, 5.5, "Findings: track, time, confidence, model decision, reasons, route", ORANGE, fs=8.5)
    for x in xs:
        arrow(ax, (x + 7, 38), (x + 7, 33.5))
    p = box(ax, 3, 19, 20, 5.5, "Plate reader on\nflagged vehicles", YELLOW, fs=8)
    v = box(ax, 26, 19, 20, 5.5, "Evidence: 3 frames, crop,\nclips, SHA-256", YELLOW, fs=8)
    m = box(ax, 49, 19, 20.5, 5.5, "Merge duplicates:\none case per vehicle", YELLOW, fs=8)
    for q in (p, v, m):
        arrow(ax, (q[0] + q[2] / 2, 28), mid(q, "t"))
    r1 = box(ax, 3, 6, 20, 7, "Confirmed\n-> automatic challan", RED, fs=8)
    r2 = box(ax, 26, 6, 20, 7, "Uncertain\n-> human review", YELLOW, fs=8)
    r3 = box(ax, 49, 6, 20.5, 7, "Emergency vehicle\n-> exempt", GREEN, fs=8)
    for q in (r1, r2, r3):
        arrow(ax, (59, 19) if q is r3 else (36, 19) if q is r2 else (13, 19), mid(q, "t"))
    l = box(ax, 3, 0.5, 66.5, 3.5, "Lanes (LaneTrace) run afterwards - context and digital twin only, never a challan", GREY, fs=7.8)
    save(fig, "f3_5")


# ---------------------------------------------------------------- Fig 3.6 routing
def fig_3_6():
    fig, ax = canvas(7.2, 3.8)
    f = box(ax, 2, 15, 15, 9, "Finding from\na detector", ORANGE, bold=True, fs=9)
    q1 = box(ax, 22, 15, 16, 9, "Emergency vehicle?\n(CLIP probe\np >= 0.956)", YELLOW, fs=8)
    q2 = box(ax, 43, 15, 15, 9, "Model's own rule\nsure? enough\nframes? 1 rider?", YELLOW, fs=8)
    arrow(ax, mid(f, "r"), mid(q1, "l")); arrow(ax, mid(q1, "r"), mid(q2, "l"), "no")
    ex = box(ax, 22, 2, 16, 7, "EXEMPT\nreason recorded", GREEN, fs=8.5)
    arrow(ax, mid(q1, "b"), mid(ex, "t"), "yes", toff=(2, 0))
    ok = box(ax, 62, 25, 10, 8, "CONFIRMED\nautomatic\nchallan", RED, fs=8)
    rv = box(ax, 62, 6, 10, 8, "REVIEW\nhuman\nqueue", BLUE, fs=8)
    arrow(ax, (58, 22), mid(ok, "l"), "yes"); arrow(ax, (58, 17), mid(rv, "l"), "no / unsure", toff=(-6, -2.5))
    label(ax, 43, 36, "Reviewer decision: challan / reject / needs evidence", fs=8, style="italic")
    arrow(ax, mid(rv, "t"), (67, 25), ls="--")
    save(fig, "f3_6")


# ---------------------------------------------------------------- Fig 3.7 merge
def fig_3_7():
    fig, ax = canvas(7.2, 3.6)
    srcs = ["(a) same shared track /\ntracker re-found it", "(b) same plate read\nwith certainty",
            "(c) same violation <= 8 s,\nCLIP similarity >= 0.92", "(d) same footage uploaded\nagain (motion match)"]
    for i, t in enumerate(srcs):
        b = box(ax, 1, 28 - i * 7.5, 22, 6, t, BLUE, fs=7.8)
        arrow(ax, mid(b, "r"), (30, 17))
    u = box(ax, 30, 11, 14, 12, "Union-find\ngroups =\none vehicle", PURPLE, fs=8.5)
    k = box(ax, 50, 19, 21, 10, "Keep one case: reviewer's\nchallan > automatic challan >\nneeds evidence > most\nconfident review", GREEN, fs=7.8)
    d = box(ax, 50, 4, 21, 10, "Others -> 'duplicate' (kept\nfor audit, linked); other\noffences listed on the kept\ncase / its challan", GREY, fs=7.8)
    arrow(ax, mid(u, "r"), mid(k, "l")); arrow(ax, mid(u, "r"), mid(d, "l"))
    save(fig, "f3_7")


# ---------------------------------------------------------------- Fig 3.8 use case
def actor(ax, x, y, name):
    ax.add_patch(Ellipse((x, y + 6), 2.4, 2.4, fc="white", ec="#333", lw=0.9))
    line(ax, [(x, y + 4.8), (x, y + 1.5)]); line(ax, [(x - 2, y + 3.8), (x + 2, y + 3.8)])
    line(ax, [(x, y + 1.5), (x - 1.6, y - 1)]); line(ax, [(x, y + 1.5), (x + 1.6, y - 1)])
    ax.text(x, y - 2, name, ha="center", va="top", fontsize=8.5, fontweight="bold")


def uc(ax, x, y, t, w=19, h=4.2):
    ax.add_patch(Ellipse((x, y), w, h, fc=BLUE, ec="#333", lw=0.8))
    ax.text(x, y, t, ha="center", va="center", fontsize=7.6)
    return (x, y, w)


def fig_3_8():
    fig, ax = canvas(7.2, 6.4)
    ax.add_patch(Rectangle((13, 1), 47, 62, fc="none", ec="#333", lw=0.9))
    ax.text(36.5, 61.5, "TraffiSense platform", ha="center", va="top", fontsize=9, fontweight="bold")
    actor(ax, 5, 44, "Operator"); actor(ax, 5, 14, "Reviewer"); actor(ax, 68, 30, "Model layer\n(system)")
    left = [("Sign up / verify / sign in", 56), ("Upload video, choose checks", 50.5), ("Watch progress, view results", 45),
            ("Open evidence, twin, lanes", 39.5), ("Download challan PDF", 34)]
    right = [("Review uncertain case", 26), ("Type / confirm plate", 20.5), ("Issue challan / reject /\nneeds evidence", 15),
             ("View analytics, models,\nsettings", 9), ("Reset password", 4)]
    for t, y in left:
        c = uc(ax, 36.5, y, t, w=26)
        line(ax, [(7.5, 47), (36.5 - 13, y)])
    for t, y in right:
        c = uc(ax, 36.5, y, t, w=26)
        line(ax, [(7.5, 17), (36.5 - 13, y)])
    for y in (50.5, 45, 26):
        line(ax, [(65.5, 33), (36.5 + 13, y)], ls="--")
    save(fig, "f3_8")


# ---------------------------------------------------------------- DFDs
def proc(ax, x, y, r, t, fc=BLUE):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * 0.8, fc=fc, ec="#333", lw=0.9))
    ax.text(x, y, t, ha="center", va="center", fontsize=8)


def store(ax, x, y, w, t):
    line(ax, [(x, y + 3), (x + w, y + 3)]); line(ax, [(x, y), (x + w, y)]); line(ax, [(x, y), (x, y + 3)])
    ax.text(x + 1, y + 1.5, t, ha="left", va="center", fontsize=7.8)


def fig_3_9():
    fig, ax = canvas(7.2, 3.6)
    o = box(ax, 2, 22, 13, 7, "Operator", GREEN, bold=True)
    r = box(ax, 2, 5, 13, 7, "Reviewer", GREEN, bold=True)
    proc(ax, 37, 18, 11, "0\nTraffiSense\nplatform", PURPLE)
    m = box(ax, 59, 22, 13, 7, "Trained\nmodels", YELLOW)
    s = box(ax, 59, 5, 13, 7, "SQLite +\nevidence files", GREY)
    arrow(ax, (15, 27), (27, 22), "video, checks, login", fs=7)
    arrow(ax, (27, 19), (15, 24), "results, PDF", fs=7, toff=(0, -3))
    arrow(ax, (15, 9), (27, 14), "decision, plate", fs=7, toff=(-2, 0.5))
    arrow(ax, (27, 12), (15, 7), "review queue", fs=7, toff=(0, -3.5))
    arrow(ax, (48, 21), (59, 25), "frames", fs=7); arrow(ax, (59, 23), (48, 19), "detections", fs=7, toff=(0, -3))
    arrow(ax, (47, 14), (59, 9), both=True)
    save(fig, "f3_9")


def fig_3_10():
    fig, ax = canvas(7.2, 5.2)
    op = box(ax, 1, 42, 11, 6, "Operator", GREEN, bold=True, fs=8.5)
    rv = box(ax, 1, 6, 11, 6, "Reviewer", GREEN, bold=True, fs=8.5)
    P = {"1": (25, 44, "1.0\nAuthentication"), "2": (25, 27, "2.0\nUpload +\nanalysis queue"),
         "3": (46, 40, "3.0\nRun detectors"), "4": (46, 22, "4.0\nEvidence +\nplate linking"),
         "5": (25, 9, "5.0\nHuman\nreview"), "6": (62, 9, "6.0\nChallan +\nPDF")}
    for k, (x, y, t) in P.items():
        proc(ax, x, y, 6.5, t)
    store(ax, 55, 44, 16, "D1  Users / sessions")
    store(ax, 59, 30, 13, "D2  Analyses")
    store(ax, 58, 17.5, 14, "D3  Findings")
    store(ax, 40, 1, 14, "D4  Challans")
    arrow(ax, (12, 46), (18.5, 45)); arrow(ax, (12, 43), (19, 29))
    arrow(ax, (31.5, 45), (55, 45.5)); arrow(ax, (31, 29), (39.5, 38))
    arrow(ax, (52, 38), (59, 32)); arrow(ax, (46, 34.8), (46, 27.2))
    arrow(ax, (52.5, 21), (58, 19)); arrow(ax, (40, 19), (31, 11.5))
    arrow(ax, (12, 9), (18.5, 9), both=True); arrow(ax, (31.5, 9), (55.5, 9))
    arrow(ax, (59, 4.2), (54, 2.5)); arrow(ax, (65, 17.5), (63, 14.2))
    arrow(ax, (31, 26), (39.5, 23.5))
    save(fig, "f3_10")


# ---------------------------------------------------------------- Fig 3.11 activity
def fig_3_11():
    fig, ax = canvas(7.2, 8.2)
    lanes = ["Operator", "Platform", "Reviewer"]
    for i, n in enumerate(lanes):
        ax.add_patch(Rectangle((1 + i * 23.5, 1), 23.5, 79, fc=[GREEN, BLUE, YELLOW][i], alpha=0.25, ec="#333", lw=0.8))
        ax.text(12.75 + i * 23.5, 78.5, n, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.add_patch(Ellipse((12.75, 74), 2.2, 2.2, fc="black"))
    def act(lane, y, t, w=19):
        x = 12.75 + lane * 23.5 - w / 2
        return box(ax, x, y, w, 4.6, t, WHITE, fs=7.8)
    a1 = act(0, 66, "Sign in (verified\naccount)")
    a2 = act(0, 58, "Upload video,\nchoose checks")
    b1 = act(1, 58, "Store original, queue\nanalysis")
    b2 = act(1, 50, "Detect + track,\nrun checks")
    b3 = act(1, 42, "Evidence, plates,\nmerge duplicates")
    dm = (36.25, 35)
    ax.add_patch(plt.Polygon([(dm[0], dm[1] + 3), (dm[0] + 4, dm[1]), (dm[0], dm[1] - 3), (dm[0] - 4, dm[1])], fc=WHITE, ec="#333"))
    ax.text(dm[0] + 5, dm[1] + 1.5, "certain?", fontsize=7.8, style="italic")
    b4 = act(1, 24, "Issue automatic\nchallan + PDF")
    c1 = act(2, 30, "Inspect frames, clips,\nreasons, plate")
    c2 = act(2, 21, "Challan / reject /\nneeds evidence")
    a3 = act(0, 14, "View results,\ndownload PDF")
    arrow(ax, (12.75, 72.9), mid(a1, "t")); arrow(ax, mid(a1, "b"), mid(a2, "t"))
    arrow(ax, mid(a2, "r"), mid(b1, "l")); arrow(ax, mid(b1, "b"), mid(b2, "t")); arrow(ax, mid(b2, "b"), mid(b3, "t"))
    arrow(ax, mid(b3, "b"), (dm[0], dm[1] + 3))
    arrow(ax, (dm[0], dm[1] - 3), mid(b4, "t"), "yes", toff=(2, 0))
    arrow(ax, (dm[0] + 4, dm[1]), mid(c1, "l"), "no")
    arrow(ax, mid(c1, "b"), mid(c2, "t"))
    arrow(ax, mid(c2, "l"), (36.25 + 9.5, 26.3), "challan", fs=7, toff=(0, 0.3))
    arrow(ax, mid(b4, "l"), mid(a3, "r"))
    ax.add_patch(Ellipse((12.75, 6), 2.6, 2.6, fc="white", ec="black", lw=1.2))
    ax.add_patch(Ellipse((12.75, 6), 1.6, 1.6, fc="black"))
    arrow(ax, mid(a3, "b"), (12.75, 7.3))
    save(fig, "f3_11")


# ---------------------------------------------------------------- Fig 3.12 sequence
def fig_3_12():
    fig, ax = canvas(7.2, 5.6)
    parts = ["Browser", "FastAPI", "Worker", "Tools", "SQLite / files"]
    xs = [7, 22, 37, 52, 66]
    for x, p in zip(xs, parts):
        box(ax, x - 6, 50, 12, 4.5, p, BLUE, bold=True, fs=8.5)
        line(ax, [(x, 50), (x, 2)], ls="--")
    msgs = [(0, 1, 46, "POST /api/analyses (video, checks)"), (1, 4, 42.5, "save original, insert row"),
            (1, 2, 39, "submit(analysis id)"), (1, 0, 35.5, "201 analysis id"),
            (0, 1, 32, "GET /api/analyses/{id} (poll)"), (2, 3, 28.5, "shared detection, then checks"),
            (3, 2, 25, "JSON decisions"), (2, 3, 21.5, "plate reader on flagged tracks"),
            (2, 4, 18, "evidence + SHA-256, findings"), (2, 4, 14.5, "merge duplicates, challans"),
            (1, 0, 11, "status = done, results"), (0, 1, 7.5, "GET /api/challans/{id}/pdf"), (1, 0, 4, "PDF e-challan")]
    for a, b, y, t in msgs:
        arrow(ax, (xs[a], y), (xs[b], y), t, fs=7.2, ls="--" if a > b else "-", toff=(0, 0.25))
    save(fig, "f3_12")


if __name__ == "__main__":
    for f in [fig_1_1, fig_2_1, fig_3_1, fig_3_2, fig_3_3, fig_3_4, fig_3_5, fig_3_6, fig_3_7, fig_3_8, fig_3_9,
              fig_3_10, fig_3_11, fig_3_12]:
        f()
    print("diagrams done")
