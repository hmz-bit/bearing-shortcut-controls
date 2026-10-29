"""Stage-only reference detectors: a score that knows only whether a window comes from a faulty life stage (no fault-type information).
UORED: score = state > 0 (developing or faulty file). Paderborn run-to-failure: score = stage == 'faulty'. Same splits and labels as the model predictions.
Prints the mean per-split inner, outer and macro AUROC (x100). The XJTU-SY counterpart is in results/xjtu/null_life_stage.json.
Usage: python -I stage_only_reference.py
"""
import glob, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORK = pathlib.Path(__file__).resolve().parents[2]


def per_split_macro(d, score_fn):
    out = []
    for _, g in d.groupby("split"):
        sc = score_fn(g)
        if sc.min() == sc.max():
            continue
        ai, ao = roc_auc_score(g.y_inner, sc), roc_auc_score(g.y_outer, sc)
        out.append((ai, ao, (ai + ao) / 2))
    return 100 * np.array(out).mean(0), len(out)


uored = pd.read_parquet(WORK / "results" / "uored" / "pred_base.parquet")
m, n = per_split_macro(uored, lambda g: (g["state"] > 0).astype(float).to_numpy())
print("UORED stage-only reference: inner %.1f outer %.1f macro %.1f (%d splits)" % (*m, n))

files = [f for f in sorted(glob.glob(str(WORK / "results" / "rtf" / "pred_base_r*.parquet"))) if "perm" not in f]
rtf = pd.concat([pd.read_parquet(f) for f in files])
m, n = per_split_macro(rtf, lambda g: (g["stage"] == "faulty").astype(float).to_numpy())
print("Paderborn run-to-failure stage-only reference: inner %.1f outer %.1f macro %.1f (%d splits)" % (*m, n))
