"""Robustness of the stage-matched analysis (positives = faulty-stage windows with the label, negatives = faulty-stage windows without it) to
mislabelled negatives: leave-one-negative-bearing-out. For each bearing that serves as a negative for a label, drop all its windows and recompute
the mean per-split Macro-AUROC of base, rowz_drop and B1; report the range of the differences over all dropped bearings.
Point estimates only (no bootstrap). Usage: python -I stage_matched_robustness.py
"""
import glob, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORK = pathlib.Path(__file__).resolve().parents[2]


def load(dataset, model):
    pats = {"xjtu": f"results/xjtu/pred_{model}_r*.parquet", "rtf": f"results/rtf/pred_{model}_r*.parquet"}[dataset]
    files = [f for f in sorted(glob.glob(str(WORK / pats))) if "perm" not in f and "sens" not in f]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stage_matched(d, drop=None):
    """mean over splits of Macro-AUROC on faulty-stage windows only (inner and outer), optionally without one bearing."""
    out = []
    for _, g in d.groupby("split"):
        g = g[g["stage"] == "faulty"]
        if drop is not None:
            g = g[g["bearing"] != drop]
        vals = []
        for lab in ("inner", "outer"):
            y = g[f"y_{lab}"].to_numpy()
            if y.min() == y.max():
                vals = None
                break
            vals.append(roc_auc_score(y, g[f"s_{lab}"].to_numpy()))
        if vals:
            out.append(np.mean(vals))
    return 100 * np.mean(out), len(out)


for ds in ("xjtu", "rtf"):
    P = {m: load(ds, m) for m in ("base", "rowz_drop", "b1")}
    ref = {m: stage_matched(P[m]) for m in P}
    print(f"== {ds}: stage-matched macro, all negatives:", {m: round(v[0], 2) for m, v in ref.items()}, "valid splits", {m: v[1] for m, v in ref.items()})
    neg_bearings = sorted(P["rowz_drop"][(P["rowz_drop"]["stage"] == "faulty") & ((P["rowz_drop"]["y_inner"] == 0) | (P["rowz_drop"]["y_outer"] == 0))]["bearing"].unique())
    rows = []
    for b in neg_bearings:
        r = {m: stage_matched(P[m], drop=b)[0] for m in P}
        rows.append((b, r["base"], r["rowz_drop"], r["b1"], r["rowz_drop"] - r["b1"], r["rowz_drop"] - r["base"]))
    t = pd.DataFrame(rows, columns=["dropped", "base", "rowz_drop", "b1", "rowz_minus_b1", "rowz_minus_base"]).round(2)
    print(t.to_string(index=False))
    print(f"   rowz_drop - B1 over leave-one-out: min {t.rowz_minus_b1.min():.2f} max {t.rowz_minus_b1.max():.2f}; rowz_drop - base: min {t.rowz_minus_base.min():.2f} max {t.rowz_minus_base.max():.2f}")
