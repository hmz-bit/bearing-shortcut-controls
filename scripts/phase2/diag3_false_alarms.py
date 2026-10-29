"""Phase A (3): which negative windows trigger the outer-race head in the TUNING splits (validation predictions of the baseline LF-PNCN,
results/phase2/tuning/base_s0.parquet written by run_tuning.py). Threshold per split = score giving 90 % TPR on the outer label."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p2common as P

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
name = sys.argv[1] if len(sys.argv) > 1 else "base_s0"
d = pd.read_parquet(P.OUT / "tuning" / f"{name}.parquet")
rows = []
for label, other in (("outer", "s_outer"), ("inner", "s_inner")):
    d[f"alarm_{label}"] = np.nan
    for s, g in d.groupby("split"):
        y = g[f"y_{label}"].to_numpy(); sc = g[other].to_numpy()
        thr = np.quantile(sc[y == 1], 0.10)             # 90 % of positives at or above thr
        d.loc[g.index, f"alarm_{label}"] = (sc >= thr).astype(float)
print(f"=== {name}: share of windows above the 90%-TPR threshold of the OUTER head, by bearing (all 5 tuning splits pooled) ===")
t = d.groupby(["bearing"]).agg(n=("alarm_outer", "size"), alarm_rate=("alarm_outer", "mean"), mean_logit=("s_outer", "mean"),
                              n_splits=("split", "nunique"))
t["cls"] = ["healthy" if b.startswith("K0") else "outer" if b.startswith("KA") else "inner" for b in t.index]
print(t.sort_values(["cls", "alarm_rate"], ascending=[True, False]).round(3).to_string())
print("\n=== OUTER head alarm rate (at 90% TPR) by class, per split ===")
print(d.assign(cls=d.bearing.map(lambda b: "healthy" if b.startswith("K0") else "outer" if b.startswith("KA") else "inner"))
      .groupby(["split", "cls"]).alarm_outer.mean().unstack().round(2).to_string())
neg = d[d.y_outer == 0]
print("\n=== negatives (healthy+inner) only: false-alarm rate of the outer head by operating condition and class ===")
neg = neg.assign(cls=neg.bearing.map(lambda b: "healthy" if b.startswith("K0") else "inner"))
print(neg.groupby(["condition", "cls"]).alarm_outer.mean().unstack().round(3).to_string())
print("\n=== positives (outer): detection rate by bearing and condition (TPR at the split threshold) ===")
pos = d[d.y_outer == 1]
print(pos.groupby(["bearing", "condition"]).alarm_outer.mean().unstack().round(2).to_string())
print("\n=== within-split AUROC of the outer head, outer bearing vs negatives, per outer bearing (pairwise) ===")
rows = []
for s, g in d.groupby("split"):
    for bear in g[g.y_outer == 1].bearing.unique():
        gg = g[(g.bearing == bear) | (g.y_outer == 0)]
        rows.append(dict(split=s, bearing=bear, auc=100 * roc_auc_score((gg.bearing == bear).astype(int), gg.s_outer)))
print(pd.DataFrame(rows).pivot(index="bearing", columns="split", values="auc").round(1).to_string())
print("\n=== the 10 healthy/inner bearing-condition cells with the highest outer alarm rate ===")
cell = neg.groupby(["bearing", "condition"]).alarm_outer.agg(["mean", "size"]).sort_values("mean", ascending=False).head(10)
print(cell.round(3).to_string())
print("\n=== inner head, for comparison: alarm rate (90% TPR) on negatives by class ===")
n2 = d[d.y_inner == 0].assign(cls=lambda x: x.bearing.map(lambda b: "healthy" if b.startswith("K0") else "outer"))
print(n2.groupby("cls").alarm_inner.mean().round(3).to_string())
