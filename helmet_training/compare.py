#!/usr/bin/env python3
"""Tabulate evaluation reports side by side -> helmet_training/reports/comparison.md (+ .json)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "helmet_training/reports/eval"


def load(tag):
    r = json.loads((EV / tag / "report.json").read_text())
    ph = EV / tag / "ph_rider_level.json"
    r["ph"] = json.loads(ph.read_text())["overall"] if ph.exists() else None
    return r


def row(tag, r):
    t = r["test_operating_point"]; u = r["ultralytics_test"]; bg = r["background_false_positives"]
    ex = r["extreme_conditions(test, synthetic)"]
    hard = [k for k in ex if k != "clean"]
    mean_hard = sum(ex[k]["macro_F1"] for k in hard) / len(hard)
    worst = min(hard, key=lambda k: ex[k]["macro_F1"])
    sz = r["size_vs_recall(pooled over scale sweep)"]
    ph = r["ph"]
    return {
        "tag": tag, "imgsz": r["imgsz"], "conf": r["operating_conf(chosen_on_val)"],
        "val_mAP50": r["ultralytics_val"]["mAP50"], "test_mAP50": u["mAP50"], "test_mAP50-95": u["mAP50-95"],
        "helmet_P": t["helmet"]["P"], "helmet_R": t["helmet"]["R"], "helmet_F1": t["helmet"]["F1"],
        "no_helmet_P": t["no_helmet"]["P"], "no_helmet_R": t["no_helmet"]["R"], "no_helmet_F1": t["no_helmet"]["F1"],
        "macro_F1": t["macro_F1"],
        "false_NO_helmet(n/%)": f'{t["false_no_helmet"]["count"]} / {100 * t["false_no_helmet"]["rate_of_helmet_gt"]:.1f}%',
        "false_helmet(n/%)": f'{t["false_helmet"]["count"]} / {100 * t["false_helmet"]["rate_of_no_helmet_gt"]:.1f}%',
        "missed(h/nh)": f'{t["missed"]["helmet"]}/{t["missed"]["no_helmet"]}',
        "spurious(h/nh)": f'{t["spurious"]["as_helmet"]}/{t["spurious"]["as_no_helmet"]}',
        "bg_FP/img": bg["fp_per_image"],
        "hard_mean_macroF1": round(mean_hard, 3), "worst_condition": f"{worst} ({ex[worst]['macro_F1']:.2f})",
        "recall_xs<16": sz.get("xs(<16px)", {}).get("correct_class_recall"), "recall_s16-32": sz.get("s(16-32px)", {}).get("correct_class_recall"),
        "recall_m32-64": sz.get("m(32-64px)", {}).get("correct_class_recall"), "recall_l>=64": sz.get("l(>=64px)", {}).get("correct_class_recall"),
        "RIDERS_ONLY helmet P/R": f'{r["riders_only(test, clean)"]["helmet"]["P"]}/{r["riders_only(test, clean)"]["helmet"]["R"]} (n={r["riders_only(test, clean)"]["helmet"]["n_gt"]})',
        "RIDERS_ONLY no_helmet P/R": f'{r["riders_only(test, clean)"]["no_helmet"]["P"]}/{r["riders_only(test, clean)"]["no_helmet"]["R"]} (n={r["riders_only(test, clean)"]["no_helmet"]["n_gt"]})',
        "RIDERS_ONLY macro_F1": r["riders_only(test, clean)"]["macro_F1"],
        "RIDERS_ONLY false_NO_helmet / false_helmet": f'{r["riders_only(test, clean)"]["false_no_helmet"]["count"]} / {r["riders_only(test, clean)"]["false_helmet"]["count"]}',
        "PH_rider_recall(h/nh)": f'{ph["helmet"]["rider_recall(correct/n)"]}/{ph["no_helmet"]["rider_recall(correct/n)"]}' if ph else "-",
    }


def main(tags):
    rows = [row(t, load(t)) for t in tags]
    keys = list(rows[0])
    md = ["| metric | " + " | ".join(r["tag"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in keys[1:]:
        md.append(f"| {k} | " + " | ".join(str(r[k]) for r in rows) + " |")
    # condition table
    md += ["", "### Extreme conditions (test set, synthetic) — macro-F1", "",
           "| condition | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    reps = [load(t) for t in tags]
    for c in reps[0]["extreme_conditions(test, synthetic)"]:
        md.append(f"| {c} | " + " | ".join(f'{r["extreme_conditions(test, synthetic)"][c]["macro_F1"]:.3f}' for r in reps) + " |")
    md += ["", "### Real-condition subsets (clean test) — macro-F1", "", "| subset | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for c in reps[0]["real_condition_subsets(test, clean)"]:
        md.append(f"| {c} (n_h={reps[0]['real_condition_subsets(test, clean)'][c]['helmet']['n_gt']}, n_nh={reps[0]['real_condition_subsets(test, clean)'][c]['no_helmet']['n_gt']}) | "
                  + " | ".join(f'{r["real_condition_subsets(test, clean)"].get(c, {}).get("macro_F1", "-")}' for r in reps) + " |")
    out = "\n".join(md)
    (ROOT / "helmet_training/reports/comparison.md").write_text(out + "\n")
    (ROOT / "helmet_training/reports/comparison.json").write_text(json.dumps(rows, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
