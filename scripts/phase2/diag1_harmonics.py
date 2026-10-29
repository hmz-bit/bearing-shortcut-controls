"""Phase A (1): strength of the fault-order peaks in the model input (t_mad) per bearing and operating condition.
Descriptive, no model. Uses the 17 Vieira bearings (all of them belong to the tuning splits' train or val sets).
Outputs: results/phase2/diag1_peak_table.csv (bearing, cond, mode, band, h: median/MAD of centre value, contrast, single-feature AUROC vs healthy)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p2common as P

meta = P.load_meta()
X = P.load_t_mad()                       # (N,3,15,64)
b, c = meta["bearing"], meta["condition"]
CEN = slice(30, 34)                      # +-2 grid steps around xi=0 (grid has no exact 0: idx 31/32 straddle)
centre = X[:, :, :, 31:33].mean(-1)      # (N,3,15)
cmax = X[:, :, :, CEN].max(-1)
rowmed = np.median(X, axis=-1)           # off-peak level of the row (median over xi)
contrast = cmax - rowmed
rows = []
for cond in P.CONDS + ["ALL"]:
    mc = np.ones(len(b), bool) if cond == "ALL" else (c == cond)
    hmask = mc & np.isin(b, P.HEALTHY)
    for bear in P.BEARINGS:
        m = mc & (b == bear)
        for mode, mname, same in ((0, "OR", P.OUTER), (1, "IR", P.INNER)):
            for band in range(3):
                for h in range(1, 6):
                    r = band * 5 + h - 1
                    v = centre[m, mode, r]
                    vh = centre[hmask, mode, r]
                    med = np.median(v)
                    mad = 1.4826 * np.median(np.abs(v - med))
                    if bear in P.HEALTHY:
                        auc = np.nan
                    elif (bear in same):
                        auc = roc_auc_score(np.r_[np.ones(len(v)), np.zeros(len(vh))], np.r_[v, vh])
                    else:
                        auc = np.nan
                    rows.append(dict(cond=cond, bearing=bear, cls=("healthy" if bear in P.HEALTHY else "outer" if bear in P.OUTER else "inner"),
                                     mode=mname, band=band + 1, h=h, n=int(m.sum()), centre_med=med, centre_mad=mad,
                                     contrast_med=float(np.median(contrast[m, mode, r])), auc_vs_healthy=auc))
df = pd.DataFrame(rows)
df.to_csv(P.OUT / "diag1_peak_table.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)

# Summary 1: OR mode, centre value, ALL conditions, per bearing: rows band x h
for mode in ("OR", "IR"):
    print(f"\n=== {mode} mode, median over windows of tensor value at xi=0 (log10(1+SNR)); condition ALL ===")
    sub = df[(df["mode"] == mode) & (df.cond == "ALL")]
    piv = sub.pivot_table(index="bearing", columns=["band", "h"], values="centre_med").round(2)
    print(piv.to_string())
    print(f"--- {mode} single-feature AUROC (bearing windows vs healthy windows), ALL ---")
    piv = sub[sub.cls != "healthy"].pivot_table(index="bearing", columns=["band", "h"], values="auc_vs_healthy").round(2) * 100
    print(piv.round(0).to_string())

print("\n=== OR mode per condition, outer bearings: mean over h=1..5 of AUROC vs healthy, by band ===")
sub = df[(df["mode"] == "OR") & (df.cls == "outer") & (df.cond != "ALL")]
print(sub.groupby(["bearing", "cond", "band"]).auc_vs_healthy.mean().unstack("band").mul(100).round(1).to_string())
print("\n=== OR mode per condition, outer bearings: mean over band of centre_med for h=1,2 ===")
sub2 = df[(df["mode"] == "OR") & (df.cond != "ALL") & (df.h <= 2)]
print(sub2.groupby(["cond", "cls", "h"]).centre_med.mean().unstack("h").round(2).to_string())
print("\n=== OR mode per condition per bearing class: median(centre_med) per band (h pooled 1..5) ===")
sub3 = df[(df["mode"] == "OR") & (df.cond != "ALL")]
print(sub3.groupby(["cond", "cls", "band"]).centre_med.median().unstack("band").round(2).to_string())

# baseline noise-floor of the tensor: per-row median over xi, healthy vs outer
print("\n=== median row level (off-peak) of OR tensor, per band, by class ===")
for cls, names in (("healthy", P.HEALTHY), ("outer", P.OUTER), ("inner", P.INNER)):
    m = np.isin(b, names)
    print(cls, np.round([float(np.median(rowmed[m, 0, k * 5:(k + 1) * 5])) for k in range(3)], 3))
