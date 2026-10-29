"""Phase A (2): resolution at 900 rpm and effect of longer windows on the order tensor. Needs results/phase2/cache/feat_L2.npy, feat_L4.npy.
Descriptive (no model, no labels used except bearing class for grouping)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p2common as P
import geometry as G
import variants as V
from pu_loader import load_vibration

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
ff25 = G.fault_frequencies(25.0, G.PU_6203)
ff15 = G.fault_frequencies(15.0, G.PU_6203)
print("f_o, f_i, f_b, f_c at 1500 rpm:", {k: round(v, 2) for k, v in ff25.items()})
print("f_o, f_i, f_b, f_c at  900 rpm:", {k: round(v, 2) for k, v in ff15.items()})
print("\n=== width of the harmonic window (Hz) and number of FFT bins in it (OR mode: +-3 % of h*f_o) ===")
rows = []
for rpm, ff, fr in ((1500, ff25, 25.0), (900, ff15, 15.0)):
    for h in range(1, 6):
        for mode, fm, fmod in (("OR", ff["f_o"], 0.0), ("IR", ff["f_i"], fr)):
            delta = max(0.03 * h * fm, 1.2 * fmod)
            rows.append(dict(rpm=rpm, mode=mode, h=h, centre_Hz=round(h * fm, 1), width_Hz=round(2 * delta, 2),
                             bins_1s=round(2 * delta / 1.0, 1), bins_2s=round(2 * delta / 0.5, 1), bins_4s=round(2 * delta / 0.25, 1),
                             hz_per_grid_pt=round(2 * delta / 63, 3)))
print(pd.DataFrame(rows).to_string(index=False))

# measured shaft speed vs nominal (sample)
meta = P.load_meta()
print("\n=== measured speed (mean |speed channel|) vs nominal, sample of files ===")
rng = np.random.default_rng(0)
recs = []
for bear in P.BEARINGS:
    for cond in P.CONDS:
        fns = np.unique(meta["fname"][(meta["bearing"] == bear) & (meta["condition"] == cond)])
        for f in rng.choice(fns, 2, replace=False):
            m, _ = load_vibration(str(P.WORK / "data" / "pu" / bear / f))
            recs.append(dict(bearing=bear, cond=cond, rpm=m["rpm_measured"]))
sp = pd.DataFrame(recs)
print(sp.groupby("cond").rpm.agg(["mean", "std", "min", "max"]).round(1).to_string())
nom = {"N15": 1500.0, "N09": 900.0}
sp["dev_pct"] = [100 * (r - nom[c[:3]]) / nom[c[:3]] for r, c in zip(sp.rpm, sp.cond)]
print("deviation from nominal in percent (cond: mean, min, max):")
print(sp.groupby("cond").dev_pct.agg(["mean", "min", "max"]).round(2).to_string())

# tensor differences between window lengths
X = {"L1": V.load_feat("L1"), "L2": V.load_feat("L2"), "L4": V.load_feat("L4")}
b, c = meta["bearing"], meta["condition"]
CEN = slice(30, 34)


def feat(Xl, mode=0):
    cm = Xl[:, mode][:, :, CEN].max(-1)                  # (N,15)
    rm = np.median(Xl[:, mode], axis=-1)
    return cm, rm


print("\n=== OR mode: centre-peak (max over +-2 grid pts) and row level, mean over bands, h=1 and h=2 only; by condition and class ===")
rows = []
for L in ("L1", "L2", "L4"):
    cm, rm = feat(X[L])
    for cond in P.CONDS:
        for cls, names in (("healthy", P.HEALTHY), ("outer", P.OUTER)):
            m = (c == cond) & np.isin(b, names)
            for h in (1, 2):
                cols = [band * 5 + h - 1 for band in range(3)]
                rows.append(dict(L=L, cond=cond, cls=cls, h=h, peak=cm[m][:, cols].mean(), rowlevel=rm[m][:, cols].mean()))
d = pd.DataFrame(rows)
d["contrast"] = d.peak - d.rowlevel
print(d.pivot_table(index=["cond", "cls", "h"], columns="L", values=["peak", "contrast"]).round(2).to_string())

print("\n=== OR single-feature AUROC (pooled outer windows vs pooled healthy windows of the same condition), peak value, mean over bands, by L ===")
rows = []
for L in ("L1", "L2", "L4"):
    cm, rm = feat(X[L])
    for cond in P.CONDS:
        for h in range(1, 6):
            cols = [band * 5 + h - 1 for band in range(3)]
            for bear in P.OUTER:
                mo = (c == cond) & (b == bear)
                mh = (c == cond) & np.isin(b, P.HEALTHY)
                a = np.mean([roc_auc_score(np.r_[np.ones(mo.sum()), np.zeros(mh.sum())], np.r_[cm[mo][:, k], cm[mh][:, k]]) for k in cols])
                rows.append(dict(L=L, cond=cond, h=h, bearing=bear, auc=100 * a))
d = pd.DataFrame(rows)
print("-- by condition, h=1..2 pooled over outer bearings --")
print(d[d.h <= 2].groupby(["cond", "L"]).auc.mean().unstack("L").round(1).to_string())
print("-- by bearing at 900 rpm, mean over h=1..5 --")
print(d[d.cond == "N09_M07_F10"].groupby(["bearing", "L"]).auc.mean().unstack("L").round(1).to_string())
print("-- by bearing at 1500 rpm (N15_M07_F10), mean over h=1..5 --")
print(d[d.cond == "N15_M07_F10"].groupby(["bearing", "L"]).auc.mean().unstack("L").round(1).to_string())
d.to_csv(P.OUT / "diag2_auc_by_window_length.csv", index=False)
mad = {k: float(np.mean(np.abs(X[k] - X["L1"]))) for k in ("L2", "L4")}
print("\nmean |tensor(Lx) - tensor(L1)| (all entries):", mad, " mean |L1| =", float(np.mean(np.abs(X["L1"]))))
