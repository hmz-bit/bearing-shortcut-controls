import sys, json, pathlib
W = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "scripts" / "evaltools"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import bootstrap as bs
d = pd.read_parquet(W / "results" / "audit" / "a3_predictions_long.parquet")
cols = ["split", "bearing", "y_inner", "y_outer", "s_inner", "s_outer"]
print(d.columns.tolist(), d.groupby("variant").pair.nunique().to_dict())
A = d[d.variant == "rowz_drop"][cols]; B = d[d.variant == "base"][cols]
for nm, f in (("base", B), ("rowz_drop", A)):
    pm = [ (roc_auc_score(g.y_inner, g.s_inner), roc_auc_score(g.y_outer, g.s_outer)) for _, g in f.groupby("split")]
    pm = np.array(pm) * 100
    print(nm, "inner %.2f outer %.2f macro %.2f" % (pm[:, 0].mean(), pm[:, 1].mean(), pm.mean()))
r = bs.paired_bootstrap_multi_split(A, B, B=2000, seed=0)
print(json.dumps({k: r[k] for k in r if k != "draws"}, indent=1, default=float))
# per-bearing outer-race detail: mean score rank view, AUROC of each artificial outer bearing vs healthy+inner
pb = {}
for nm, f in (("base", B), ("rowz_drop", A)):
    out = {}
    for b in sorted(set(f.bearing)):
        if not b.startswith("KA"): continue
        v = []
        for _, g in f.groupby("split"):
            pos = g[g.bearing == b].s_outer.values; neg = g[(~g.bearing.str.startswith("KA"))].s_outer.values
            v.append(np.mean(pos[:, None] > neg[None, :]) if len(pos) and len(neg) else np.nan)
        out[b] = 100 * np.nanmean(v)
    pb[nm] = out
print(pd.DataFrame(pb).round(1))
