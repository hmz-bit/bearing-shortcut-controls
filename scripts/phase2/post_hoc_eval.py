"""Post-hoc (after the evaluation, NOT used for any selection): per-bearing outer-race behaviour on the 100 Tier 1 evaluation splits.
For each outer bearing: mean over the splits in which it is a test bearing of its pairwise AUROC against the negatives of that split (outer head);
for each negative bearing: share of its windows above the split's 90%-TPR threshold of the outer head."""
import sys, pathlib, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p2common as P
W = P.WORK
frames = {"LF-PNCN": pd.read_parquet(W / "results/tier1/lfpncn_predictions_long.parquet"),
          "rowz_drop": pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P.OUT / "eval/rowz_drop_r*_predictions_long.parquet")))]),
          "B1": pd.read_parquet(W / "scripts/evaltools/results/b1_predictions_long.parquet")}
pd.set_option("display.width", 200)
res = {}
for k, f in frames.items():
    auc, fa = {}, {}
    for s, g in f.groupby("split"):
        thr = np.quantile(g.loc[g.y_outer == 1, "s_outer"], 0.10)
        neg = g[g.y_outer == 0]
        for b, gb in neg.groupby("bearing"):
            fa.setdefault(b, []).append((gb.s_outer >= thr).mean())
        for b in g[g.y_outer == 1].bearing.unique():
            gg = g[(g.bearing == b) | (g.y_outer == 0)]
            auc.setdefault(b, []).append(100 * roc_auc_score((gg.bearing == b).astype(int), gg.s_outer))
    res[k] = (auc, fa)
print("outer bearings: mean pairwise AUROC (outer bearing vs negatives), mean over splits where tested")
print(pd.DataFrame({k: {b: np.mean(v) for b, v in a.items()} for k, (a, _) in res.items()}).round(1).to_string())
print("\nnegative bearings: share of windows above the 90%-TPR threshold of the outer head (false-alarm rate), mean over splits")
print(pd.DataFrame({k: {b: 100 * np.mean(v) for b, v in a.items()} for k, (_, a) in res.items()}).round(1).to_string())
