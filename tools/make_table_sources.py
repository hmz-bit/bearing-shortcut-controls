#!/usr/bin/env python3
"""Write results/TABLE_SOURCES.csv: for every number printed in results/CROSS_DATASET_SUMMARY (.txt/.md)
the file and the key or column it comes from, with the value read from that file and a check that the printed
number equals the stored value rounded to the printed precision.  Standard library only.

Usage: python tools/make_table_sources.py        (run from anywhere; paths are relative to the repository root)
"""
import csv
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
_cache = {}


def jload(rel):
    if rel not in _cache:
        with open(os.path.join(RES, rel), encoding="utf8") as fh:
            _cache[rel] = json.load(fh)
    return _cache[rel]


def jget(rel, path):
    o = jload(rel)
    for k in path:
        o = o[k]
    return o


def csv_rows(rel):
    with open(os.path.join(RES, rel), encoding="utf8", newline="") as fh:
        return list(csv.DictReader(fh))


def decimals(s):
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


OUT = []
FAILS = []


def add(location, printed, rel, key, value, derived=""):
    """printed: the number as printed (string, sign included); value: float read from the source."""
    p = float(printed)
    ok = abs(round(value, decimals(printed)) - p) < 1e-9
    if not ok:
        FAILS.append((location, printed, rel, key, value))
    OUT.append([location, printed, rel, key, repr(value), "yes" if ok else "NO", derived])


def J(location, printed, rel, path, derived=""):
    add(location, printed, rel, "/".join(str(k) for k in path), jget(rel, path), derived)


def diff(location, point_s, lo_s, hi_s, rel, key_path_prefix, kind="ci_list", names=("point", "ci_low", "ci_high"), cikey="ci"):
    """Difference with 95% interval: three numbers from one source object."""
    if kind == "ci_list":                   # {point, ci:[lo, hi]}
        J(location + " (point)", point_s, rel, key_path_prefix + ["point"])
        J(location + " (CI low)", lo_s, rel, key_path_prefix + [cikey, 0])
        J(location + " (CI high)", hi_s, rel, key_path_prefix + [cikey, 1])
    else:                                   # {macro:{point, ci_low, ci_high}} or {point, ci_low, ci_high}
        J(location + " (point)", point_s, rel, key_path_prefix + ["point"])
        J(location + " (CI low)", lo_s, rel, key_path_prefix + ["ci_low"])
        J(location + " (CI high)", hi_s, rel, key_path_prefix + ["ci_high"])


T = "table row "
# ---------------------------------------------------------------- PU
J(T + "PU, column base (LF-PNCN)", "85.63", "tier1/metrics.json", ["metrics", "LF-PNCN", "macro_mean"],
  "same value: tier1/metrics_table.csv column macro_mean row LF-PNCN; phase2/eval_metrics.json tier1/metrics/LF-PNCN/macro")
J(T + "PU, column rowz_drop", "88.85", "phase2/eval_metrics.json", ["tier1", "metrics", "rowz_drop", "macro"])
J(T + "PU, column B1", "79.58", "tier1/metrics.json", ["metrics", "B1 (WDCNN envelope)", "macro_mean"],
  "B1 on PU = third-party saved logits scored by scripts/evaltools (see docs/DATA.md)")
diff(T + "PU, column rowz_drop - base", "3.22", "-0.88", "8.72", "phase2/eval_metrics.json", ["tier1", "paired", "rowz_drop - LF-PNCN", "macro"], "flat")
diff(T + "PU, column rowz_drop - B1", "9.27", "4.66", "15.68", "phase2/eval_metrics.json", ["tier1", "paired", "rowz_drop - B1", "macro"], "flat")
diff("Notes, base minus B1 on PU", "6.05", "0.31", "11.93", "tier1/metrics.json", ["paired", "LF-PNCN - B1 (WDCNN envelope)"], "flat")
# ---------------------------------------------------------------- CWRU
f = "cwru/cwru_results_b1.json"
J(T + "CWRU, column base", "88.40", f, ["base", "macro"], "also cwru/cwru_results.json")
J(T + "CWRU, column rowz_drop", "95.29", f, ["rowz_drop", "macro"], "also cwru/cwru_results.json")
J(T + "CWRU, column B1", "71.75", f, ["b1", "macro"])
diff(T + "CWRU, column rowz_drop - base", "6.89", "1.47", "11.36", f, ["rowz_drop_minus_base_macro"])
diff(T + "CWRU, column rowz_drop - B1", "23.54", "15.50", "31.17", f, ["rowz_drop_minus_b1_macro"])
diff("Notes, base minus B1 on CWRU", "16.66", "10.42", "24.20", f, ["base_minus_b1_macro"])
# ---------------------------------------------------------------- UORED
f = "uored/uored_results_b1.json"
J(T + "UORED-VAFCLS, column base", "80.92", f, ["base", "macro"], "also uored/uored_results.json")
J(T + "UORED-VAFCLS, column rowz_drop", "91.55", f, ["rowz_drop", "macro"], "also uored/uored_results.json")
J(T + "UORED-VAFCLS, column B1", "94.21", f, ["b1", "macro"])
diff(T + "UORED-VAFCLS, column rowz_drop - base", "10.64", "5.20", "17.35", f, ["rowz_drop_minus_base_macro"])
diff(T + "UORED-VAFCLS, column rowz_drop - B1", "-2.66", "-6.21", "0.29", f, ["rowz_drop_minus_b1_macro"])
diff("Notes, base minus B1 on UORED", "-13.30", "-21.54", "-6.68", f, ["base_minus_b1_macro"])
# ---------------------------------------------------------------- XJTU-SY
f = "xjtu/xjtu_results.json"
J(T + "XJTU-SY, column base", "74.07", f, ["primary", "base", "macro"])
J(T + "XJTU-SY, column rowz_drop", "76.04", f, ["primary", "rowz_drop", "macro"])
J(T + "XJTU-SY, column B1", "69.84", f, ["primary", "b1", "macro"])
diff(T + "XJTU-SY, column rowz_drop - base", "1.97", "-4.0", "7.4", f, ["primary", "rowz_drop_minus_base_macro"])
diff(T + "XJTU-SY, column rowz_drop - B1", "6.20", "-6.3", "13.4", f, ["primary", "rowz_drop_minus_b1_macro"])
diff("Notes, base minus B1 on XJTU", "4.23", "-8.8", "10.5", f, ["primary", "base_minus_b1_macro"])
# ---------------------------------------------------------------- Paderborn run-to-failure
f = "rtf/rtf_results.json"
J(T + "Paderborn run-to-failure, column base", "76.42", f, ["primary", "base", "macro"])
J(T + "Paderborn run-to-failure, column rowz_drop", "78.85", f, ["primary", "rowz_drop", "macro"])
J(T + "Paderborn run-to-failure, column B1", "72.51", f, ["primary", "b1", "macro"])
diff(T + "Paderborn run-to-failure, column rowz_drop - base", "2.43", "-1.54", "6.59", f, ["primary", "rowz_drop_minus_base_macro"])
diff(T + "Paderborn run-to-failure, column rowz_drop - B1", "6.35", "2.09", "12.15", f, ["primary", "rowz_drop_minus_b1_macro"])
diff("Notes, base minus B1 on Paderborn run-to-failure", "3.92", "-1.43", "9.88", f, ["primary", "base_minus_b1_macro"])

# ---------------------------------------------------------------- notes: life-stage controls
J("Notes, life-stage-only detector on XJTU-SY", "84.9", "xjtu/null_life_stage.json", ["oracle_real_labels_macro_mean_100_splits"],
  "score = 1[faulty stage], true labels, mean over 100 splits; script scripts/xjtu/null_life_stage.py")
J("Notes, 56 threshold exceeded: XJTU-SY base permutation control", "63.1", "xjtu/xjtu_results.json", ["permutation_control", "base", "held_macro_mean"],
  "threshold 56 is declared in protocols/PREDECLARED_XJTU.txt item 14 (and PREDECLARED_RTF.txt item 12); printed value rounded here to one decimal")
J("Notes, 56 threshold exceeded: XJTU-SY rowz_drop permutation control", "58.2", "xjtu/xjtu_results.json", ["permutation_control", "rowz_drop", "held_macro_mean"], "")
J("Notes, 56 threshold: XJTU-SY B1 permutation control (below 56)", "55.8", "xjtu/xjtu_results.json", ["permutation_control", "b1", "held_macro_mean"], "only B1 is below 56 on XJTU-SY")
J("Notes, 56 threshold exceeded: Paderborn RTF base label-pair permutation", "65.2", "rtf/rtf_results.json", ["permutation_control", "perm_base", "held_macro_mean"], "")
J("Notes, 56 threshold exceeded: Paderborn RTF rowz_drop label-pair permutation", "64.2", "rtf/rtf_results.json", ["permutation_control", "perm_rowz_drop", "held_macro_mean"], "")
J("Notes, 56 threshold exceeded: Paderborn RTF B1 label-pair permutation", "63.6", "rtf/rtf_results.json", ["permutation_control", "perm_b1", "held_macro_mean"], "")

# own B1 versus published value on PU: mean of column 'diff' (6 validation splits, 3 seeds)
rows = csv_rows("b1/val_3seed_table.csv")
m = sum(float(r["diff"]) for r in rows) / len(rows)
add("Notes, own B1 about +3 above the published value on PU", "3", "b1/val_3seed_table.csv", "mean of column diff over 6 rows", m,
    "6 validation splits, 3-seed mean of own B1 minus published per-split Macro-AUROC; stated as 'about +3' (value 2.95)")
J("Notes, own B1 minus published value on CWRU: own value", "71.75", "cwru/cwru_results_b1.json", ["b1", "macro"],
  "published value 74.51 comes from the cited paper (Vieira et al., arXiv v5) and is NOT reproduced in this repository; the printed -2.8 is 71.75 - 74.51 = -2.76")

# ---------------------------------------------------------------- "What the data support"
pts = {"PU": 3.22, "CWRU": 6.89, "UORED": 10.64, "XJTU": 1.97, "RTF": 2.43}
for ds, rel, path in [("PU", "phase2/eval_metrics.json", ["tier1", "paired", "rowz_drop - LF-PNCN", "macro", "point"]),
                      ("CWRU", "cwru/cwru_results_b1.json", ["rowz_drop_minus_base_macro", "point"]),
                      ("UORED", "uored/uored_results_b1.json", ["rowz_drop_minus_base_macro", "point"]),
                      ("XJTU", "xjtu/xjtu_results.json", ["primary", "rowz_drop_minus_base_macro", "point"]),
                      ("RTF", "rtf/rtf_results.json", ["primary", "rowz_drop_minus_base_macro", "point"])]:
    lo = {"PU": ["tier1", "paired", "rowz_drop - LF-PNCN", "macro", "ci_low"]}.get(ds, path[:-1] + ["ci", 0])
    v = jget(rel, path); l = jget(rel, lo)
    add(f"'What the data support': rowz_drop minus base point estimate, {ds}", f"{pts[ds]:.2f}", rel, "/".join(path), v,
        f"range '+2.0 to +10.6' is the min and max over the five datasets; CI lower bound {l:.2f}: " + ("above 0 (significant)" if l > 0 else "not above 0"))
lows = [jget("phase2/eval_metrics.json", ["tier1", "paired", "rowz_drop - LF-PNCN", "macro", "ci_low"]),
        jget("cwru/cwru_results_b1.json", ["rowz_drop_minus_base_macro", "ci", 0]),
        jget("uored/uored_results_b1.json", ["rowz_drop_minus_base_macro", "ci", 0]),
        jget("xjtu/xjtu_results.json", ["primary", "rowz_drop_minus_base_macro", "ci", 0]),
        jget("rtf/rtf_results.json", ["primary", "rowz_drop_minus_base_macro", "ci", 0])]
add("Corrections: 'lower confidence bounds are above -4.0 points'", "-4.0", "five files above (CI low of rowz_drop minus base)", "min over datasets", min(lows),
    f"minimum {min(lows):.4f} (XJTU-SY); this is slightly below -4.0 when not rounded")
# above B1 in four of five; significant on PU, CWRU, RTF
for ds, rel, path in [("PU", "phase2/eval_metrics.json", ["tier1", "paired", "rowz_drop - B1", "macro"]),
                      ("CWRU", "cwru/cwru_results_b1.json", ["rowz_drop_minus_b1_macro"]),
                      ("UORED", "uored/uored_results_b1.json", ["rowz_drop_minus_b1_macro"]),
                      ("XJTU", "xjtu/xjtu_results.json", ["primary", "rowz_drop_minus_b1_macro"]),
                      ("RTF", "rtf/rtf_results.json", ["primary", "rowz_drop_minus_b1_macro"])]:
    o = jget(rel, path)
    pt = o["point"]; lo = o["ci_low"] if "ci_low" in o else o["ci"][0]
    OUT.append([f"'What the data support': rowz_drop minus B1 sign and significance, {ds}", "", rel, "/".join(path),
                f"point={pt!r}; ci_low={lo!r}", "n/a", "above B1" if pt > 0 else "below B1"])
    OUT[-1][-1] += "; significant (CI low > 0)" if lo > 0 else "; not significant"

# ---------------------------------------------------------------- weak outer-race bearings
def per_bearing(location, rel, filt, key):
    rows = [r for r in csv_rows(rel) if filt(r)]
    val = "; ".join(f"{r['model']}={float(r['auroc']):.1f}" for r in rows)
    OUT.append([location, "", rel, key, val, "n/a", "weak outer-race bearing named in the Notes (AUROC x100)"])

for b in ("8",):
    per_bearing("Notes, weak outer-race bearing UORED O8", "uored/per_bearing_auroc.csv",
                lambda r: r["label"] == "outer" and r["bearing"] == b and r["state"] == "all", "rows label=outer, bearing=8, state=all, column auroc")
per_bearing("Notes, weak outer-race bearing CWRU DE_OR_14", "cwru/per_bearing_auroc_b1.csv",
            lambda r: r["label"] == "outer" and r["group"] == "DE_OR_14", "rows label=outer, group=DE_OR_14, column auroc")
for b in ("B01", "B11", "B16"):
    per_bearing(f"Notes, weak outer-race bearing Paderborn run-to-failure {b}", "rtf/per_bearing_auroc.csv",
                lambda r, b=b: r["label"] == "outer" and r["bearing"] == b and r["negatives"] == "all_negatives", f"rows label=outer, bearing={b}, negatives=all_negatives, column auroc")
for b in ("B3_1", "B3_2"):
    per_bearing(f"Notes, weak outer-race bearing XJTU-SY {b[1:]}", "xjtu/per_bearing_auroc.csv",
                lambda r, b=b: r["label"] == "outer" and r["bearing"] == b and r["negatives"] == "all_negatives", f"rows label=outer, bearing={b}, negatives=all_negatives, column auroc")
OUT.append(["Notes, weak outer-race bearings PU KA15/KA22/KA30", "", "phase2/post_hoc_eval_log.txt", "table 'outer bearings: mean pairwise AUROC' rows KA15, KA22, KA30",
            "see file (columns LF-PNCN, rowz_drop, B1)", "n/a", "also phase2/diag1_log.txt and diag5_log.txt (no outer-race line in the envelope spectrum)"])

# ---------------------------------------------------------------- Corrections and additions
def logval(rel, pattern):
    with open(os.path.join(RES, rel), encoding="utf8") as fh:
        t = fh.read()
    m = re.search(pattern, t)
    return float(m.group(1)), m.group(0)
v, line = logval("audit/stage_only_reference_log.txt", r"UORED stage-only reference: .*?macro ([0-9.]+)")
add("Corrections: UORED stage-only reference", "70.0", "audit/stage_only_reference_log.txt", "line 'UORED stage-only reference ... macro'", v, "script scripts/audit/stage_only_reference.py")
v, line = logval("audit/stage_only_reference_log.txt", r"Paderborn run-to-failure stage-only reference: .*?macro ([0-9.]+)")
add("Corrections: Paderborn run-to-failure stage-only reference", "84.1", "audit/stage_only_reference_log.txt", "line 'Paderborn run-to-failure stage-only reference ... macro'", v, "script scripts/audit/stage_only_reference.py")
J("Corrections: XJTU-SY stage-only reference", "84.9", "xjtu/null_life_stage.json", ["oracle_real_labels_macro_mean_100_splits"])
for (ds, rel, nm, printed) in [("XJTU-SY", "xjtu/xjtu_results.json", "rowz_drop", "81.3"), ("XJTU-SY", "xjtu/xjtu_results.json", "b1", "67.5"),
                               ("Paderborn run-to-failure", "rtf/rtf_results.json", "rowz_drop", "72.96"), ("Paderborn run-to-failure", "rtf/rtf_results.json", "b1", "66.10"),
                               ("Paderborn run-to-failure", "rtf/rtf_results.json", "base", "68.52")]:
    J(f"Corrections: stage-matched control, {ds} {nm}", printed, rel, ["late_life_negatives", nm, "macro"], "pre-declared descriptive analysis (negatives restricted to late-life windows of bearings without the defect)")
J("Corrections: whole-bearing permutation, Paderborn run-to-failure (about 49), base", "48.8", "rtf/rtf_results.json", ["permutation_control", "permw_base", "held_macro_mean"], "post hoc; addendum protocols/PREDECLARED_RTF_ADDENDUM1.txt")
J("Corrections: whole-bearing permutation, Paderborn run-to-failure (about 49), rowz_drop", "48.9", "rtf/rtf_results.json", ["permutation_control", "permw_rowz_drop", "held_macro_mean"], "")
J("Corrections: whole-bearing permutation, Paderborn run-to-failure (about 49), B1", "49.1", "rtf/rtf_results.json", ["permutation_control", "permw_b1", "held_macro_mean"], "")
# leave-one-negative-bearing-out ranges (point estimates) from the log
with open(os.path.join(RES, "audit", "stage_matched_robustness_log.txt"), encoding="utf8") as fh:
    log = fh.read()
parts = re.findall(r"rowz_drop - B1 over leave-one-out: min ([0-9.-]+) max ([0-9.-]+); rowz_drop - base: min ([0-9.-]+) max ([0-9.-]+)", log)
blocks = log.split("== ")[1:]
for (ds, blk, (a, b, c, d), pr) in zip(("XJTU-SY", "Paderborn run-to-failure"), blocks, parts, (("8.5", "17.2", "1.3", "8.7"), ("5.6", "9.7", "2.3", "6.1"))):
    nrows = len([ln for ln in blk.splitlines() if re.match(r"\s+[A-Z0-9_]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.-]+\s+[0-9.-]+$", ln)])
    for lab, val, p in (("rowz_drop minus B1, min", a, pr[0]), ("rowz_drop minus B1, max", b, pr[1]), ("rowz_drop minus base, min", c, pr[2]), ("rowz_drop minus base, max", d, pr[3])):
        add(f"Corrections: leave-one-negative-bearing-out, {ds}, {lab}", p, "audit/stage_matched_robustness_log.txt", "summary line of the dataset block", float(val), "script scripts/audit/stage_matched_robustness.py; point estimates")
    add(f"Corrections: number of dropped bearings, {ds}", "13" if ds == "XJTU-SY" else "14", "audit/stage_matched_robustness_log.txt", "number of table rows of the dataset block", float(nrows), "")

# numbers that repeat or describe other numbers
OUT.append(["Corrections: model scores in parentheses (UORED 80.9 / 91.6 / 94.2; XJTU-SY 74.1 / 76.0 / 69.8; Paderborn run-to-failure 76.4 / 78.9 / 72.5)", "",
            "see the table rows above", "columns base, rowz_drop, B1", "same values rounded to one decimal", "n/a", "no new number"])
OUT.append(["Corrections: 'significant in two' and 'all five datasets'", "", "see the rows 'rowz_drop minus base' above", "CI low > 0 on CWRU and UORED only",
            "see value_in_source of the 'What the data support' rows", "n/a", "count derived from the five intervals"])
OUT.append(["Corrections: life-stage definition of UORED (first 5 files, files 25-30, files 45-50)", "", "docs/DATA.md", "section UORED-VAFCLS",
            "dataset authors' definition (Sehri et al. 2023)", "n/a", "external source, quoted in docs/DATA.md"])
OUT.append(["Corrections: first and last 10% rule (XJTU-SY, Paderborn run-to-failure) and declaration dates 16:43 and 16:06", "", "protocols/PREDECLARED_XJTU.txt; protocols/PREDECLARED_RTF.txt; protocols/TIMESTAMPS.csv",
            "item 2 of each file; column protocol_mtime", "see files", "n/a", "protocol text and file modification times"])

with open(os.path.join(RES, "TABLE_SOURCES.csv"), "w", encoding="utf8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["location_in_CROSS_DATASET_SUMMARY", "printed_value", "source_file_under_results", "key_path_or_column", "value_in_source", "printed_equals_rounded_source", "comment"])
    w.writerows([r for r in OUT if r])
print(f"{len(OUT)} rows written; mismatches: {len(FAILS)}")
for x in FAILS:
    print("MISMATCH", x)
