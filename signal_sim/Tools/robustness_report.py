"""
Statistics for Results/robustness.csv (Editor/Robustness.cs): is PRAVAH's advantage real, and does it survive when the
simulator's assumptions change?

Paired design: in each (condition, scenario, seed) every controller gets identical arrivals, so PRAVAH is compared
seed-by-seed (difference = other - PRAVAH; for waits, jumps and queues positive = PRAVAH better; for throughput
negative = PRAVAH better). 95 % confidence interval of the mean difference (t distribution). A cell counts as
"better" / "worse" only when the interval excludes zero; otherwise "no clear difference".
Usage: python3 signal_sim/Tools/robustness_report.py [Results/robustness.csv]  -> Results/robustness.md
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "Results"
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.05}
METRICS = [("person_wait", "wait per person", "s", True), ("avg_wait", "wait per vehicle", "s", True), ("max_wait", "worst wait", "s", True),
           ("red_jumps_h", "red-light jumps", "/h", True), ("queued_end", "left queued", "veh", True), ("throughput_h", "throughput", "veh/h", False)]


def t95(n):
    return T95.get(n) or T95[min(T95, key=lambda k: abs(k - n))]


def ci(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, t95(len(xs)) * st.stdev(xs) / len(xs) ** 0.5


def main(path):
    rows = list(csv.DictReader(open(path)))
    data = defaultdict(dict)   # (cond, scen, seed) -> controller -> row
    for r in rows:
        data[(r["condition"], r["scenario"], int(r["seed"]))][r["controller"]] = r
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    scens = list(dict.fromkeys(r["scenario"] for r in rows))
    seeds = sorted({k[2] for k in data})
    out = [f"# PRAVAH robustness - {len(rows)} simulation runs\n",
           f"{len(conds)} simulator assumptions x {len(scens)} scenarios x {len(seeds)} seeds x controllers, 30 simulated minutes each. "
           "Paired comparison (identical arrivals per seed), 95 % confidence intervals; \"better\"/\"worse\" only when the interval excludes 0.\n"]

    # 1. baseline means with CI
    out.append("## 1. Baseline (the simulator's default assumptions)\n")
    out.append("| Scenario | Controller | Wait per person (s) | Wait per vehicle (s) | Worst wait (s) | Red-light jumps /h | Throughput (veh/h) |")
    out.append("|---|---|---|---|---|---|---|")
    ctls = list(dict.fromkeys(r["controller"] for r in rows if r["condition"] == "baseline"))
    for sc in scens:
        for c in ctls:
            vals = [data[("baseline", sc, s)][c] for s in seeds if c in data[("baseline", sc, s)]]
            cells = []
            for k in ("person_wait", "avg_wait", "max_wait", "red_jumps_h", "throughput_h"):
                m, h = ci([float(v[k]) for v in vals])
                cells.append(f"{m:.1f} ± {h:.1f}" if k != "throughput_h" else f"{m:.0f} ± {h:.0f}")
            out.append(f"| {sc} | {c} | " + " | ".join(cells) + " |")

    # 2. PRAVAH vs each opponent across all assumptions
    out.append("\n## 2. PRAVAH against each controller, over every assumption and scenario\n")
    out.append("Cells = (assumption, scenario) pairs. Better / no clear difference / worse for PRAVAH.\n")
    opps = [c for c in ctls if c != "PRAVAH"]
    out.append("| Against | " + " | ".join(m[1] for m in METRICS) + " |")
    out.append("|---|" + "---|" * len(METRICS))
    detail = defaultdict(list)
    for o in opps:
        cnt = {m[0]: [0, 0, 0] for m in METRICS}
        for cd in conds:
            for sc in scens:
                pairs = [(data[(cd, sc, s)].get(o), data[(cd, sc, s)].get("PRAVAH")) for s in seeds]
                pairs = [(a, b) for a, b in pairs if a and b]
                if len(pairs) < 2:
                    continue
                for k, label, unit, lower_better in METRICS:
                    d = [float(a[k]) - float(b[k]) for a, b in pairs]
                    if not lower_better:
                        d = [-x for x in d]
                    m, h = ci(d)
                    i = 0 if m - h > 0 else 2 if m + h < 0 else 1
                    cnt[k][i] += 1
                    detail[(o, k)].append((cd, sc, m, h, i))
        out.append(f"| {o} | " + " | ".join(f"{cnt[k][0]} / {cnt[k][1]} / {cnt[k][2]}" for k, *_ in METRICS) + " |")

    # 3. where PRAVAH is worse (the honest part)
    out.append("\n## 3. Every case where PRAVAH is significantly worse\n")
    worse = [(o, k, cd, sc, m, h) for (o, k), L in detail.items() for cd, sc, m, h, i in L if i == 2]
    if not worse:
        out.append("None.\n")
    else:
        out.append("| Against | Metric | Assumption | Scenario | PRAVAH worse by (95 % CI) |")
        out.append("|---|---|---|---|---|")
        unit = {k: u for k, _, u, _ in METRICS}; name = {k: n for k, n, _, _ in METRICS}
        for o, k, cd, sc, m, h in sorted(worse, key=lambda x: (x[1], x[0], x[2])):
            out.append(f"| {o} | {name[k]} | {cd} | {sc} | {-m:.1f} ± {h:.1f} {unit[k]} |")

    # 4. per-assumption headline: PRAVAH vs the best other controller, person wait, averaged over scenarios
    out.append("\n## 4. Does the main result survive each assumption change? (wait per person, mean over scenarios)\n")
    out.append("| Assumption | PRAVAH (s) | Best other (s) | Best other | PRAVAH better in scenarios |")
    out.append("|---|---|---|---|---|")
    for cd in conds:
        means = {}
        for c in ctls:
            v = [float(data[(cd, sc, s)][c]["person_wait"]) for sc in scens for s in seeds if c in data[(cd, sc, s)]]
            if v:
                means[c] = st.mean(v)
        others = {c: v for c, v in means.items() if c != "PRAVAH"}
        if "PRAVAH" not in means or not others:
            continue
        bo = min(others, key=others.get)
        wins = sum(1 for cd2, sc, m, h, i in detail[(bo, "person_wait")] if cd2 == cd and i == 0)
        out.append(f"| {cd} | {means['PRAVAH']:.1f} | {others[bo]:.1f} | {bo} | {wins} / {len(scens)} |")
    (RES / "robustness.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "robustness.csv")
