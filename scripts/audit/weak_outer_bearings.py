"""Per-bearing outer-race AUROC from the saved predictions (post hoc, descriptive; no training, no tuning).

For every dataset with outer-race labels and for the base model and rowz_drop: in each evaluation split, each outer-race
positive unit (bearing, or fault configuration for CWRU) that is in the test set is scored against all outer-race negative
windows of the same test set (AUROC over windows); the value of a unit is the mean over the splits in which it is a test unit.
Original labels of each dataset (all windows).  Output: results/audit/weak_outer_bearings.csv.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "audit"


def load(pattern_list):
    fs = []
    for p in pattern_list:
        fs += sorted(glob.glob(str(R / p)))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


SOURCES = {
    "PU": ({"base": ["tier1/lfpncn_predictions_long.parquet"],
            "rowz_drop": ["phase2/eval/rowz_drop_r5-104_predictions_long.parquet"]}, "bearing"),
    "UORED": ({"base": ["uored/pred_base.parquet"], "rowz_drop": ["uored/pred_rowz_drop.parquet"]}, "bearing"),
    "CWRU": ({"base": ["cwru/pred_base_r*.parquet"], "rowz_drop": ["cwru/pred_rowz_drop_r*.parquet"]}, "unit"),
    "XJTU-SY": ({"base": ["xjtu/pred_base_r*.parquet"], "rowz_drop": ["xjtu/pred_rowz_drop_r*.parquet"]}, "bearing"),
    "RTF": ({"base": ["rtf/pred_base_r*.parquet"], "rowz_drop": ["rtf/pred_rowz_drop_r*.parquet"]}, "bearing"),
}


def per_unit(d, ucol):
    keys = [c for c in ("run", "scenario", "split") if c in d.columns]
    rows = []
    for _, g in d.groupby(keys):
        neg = g[g.y_outer == 0]
        if len(neg) == 0:
            continue
        for u, gu in g[g.y_outer == 1].groupby(ucol):
            y = np.r_[np.ones(len(gu)), np.zeros(len(neg))]
            s = np.r_[gu.s_outer.to_numpy(), neg.s_outer.to_numpy()]
            rows.append((u, roc_auc_score(y, s)))
    return pd.DataFrame(rows, columns=["unit", "auc"]).groupby("unit").auc.agg(["mean", "count"])


res = []
for ds, (src, ucol) in SOURCES.items():
    cols = {}
    for m, pats in src.items():
        try:
            d = load(pats)
        except ValueError:
            print(ds, m, "no files")
            continue
        r = per_unit(d, ucol)
        cols[m] = r["mean"] * 100
        cols["n_splits_" + m] = r["count"]
    t = pd.DataFrame(cols)
    t.insert(0, "dataset", ds)
    res.append(t.reset_index())
out = pd.concat(res, ignore_index=True).rename(columns={"index": "unit", "unit": "unit"})
OUT.mkdir(exist_ok=True)
out.round(2).to_csv(OUT / "weak_outer_bearings.csv", index=False)
print(out.round(1).to_string())
