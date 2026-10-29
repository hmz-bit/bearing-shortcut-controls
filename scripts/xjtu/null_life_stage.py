"""POST-HOC (not pre-declared) reference for the bearing-label permutation control of results/xjtu/PREDECLARED_XJTU.txt item 14.
Under that permutation the healthy-stage windows keep label 0, so a detector that only knows "faulty stage versus healthy stage" has a held-out
Macro-AUROC above 50. This script reproduces the same 40 (split, permutation) draws (same seeds, same train/test bearings) and computes the
held-out Macro-AUROC of the score = 1[stage == faulty] (an ideal life-stage detector), and of the same score on the real labels, for comparison
with the models' permutation results in per_split_perm_*.csv. Output: results/xjtu/null_life_stage.json and prints a summary."""
import sys, pathlib, glob, json
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import xjtu_common as C

w = C.load_windows()
Y, exp, stage = w["Y"], w["exp"], w["stage"]
score = (stage == "faulty").astype(float)
splits = C.load_splits()
rows = []
for run in range(5, 25):
    tr_ids, te_ids = splits[run]
    te = np.isin(exp, te_ids)
    for k in (0, 1):
        Yp = C.perm_Y(Y, exp, stage, run, k, tr_ids, te_ids)
        if Yp is None:
            continue
        a = [roc_auc_score(Yp[te][:, c], score[te]) for c in (0, 1)]
        rows.append(dict(split=run, perm=k, oracle_perm_macro=100 * np.mean(a)))
d = pd.DataFrame(rows)
real = []
for run in range(5, 105):
    tr_ids, te_ids = splits[run]
    te = np.isin(exp, te_ids)
    try:
        real.append(100 * np.mean([roc_auc_score(Y[te][:, c], score[te]) for c in (0, 1)]))
    except ValueError:
        pass
out = dict(oracle_perm_macro_mean=float(d.oracle_perm_macro.mean()), oracle_perm_macro_sd=float(d.oracle_perm_macro.std()), n_draws=len(d),
           oracle_real_labels_macro_mean_100_splits=float(np.mean(real)))
mod = {}
for n in ("base", "rowz_drop", "b1"):
    q = pd.concat([pd.read_csv(f) for f in glob.glob(str(C.RES / f"per_split_perm_{n}_r*.csv"))]); q = q[q.skipped == 0]
    m = q.merge(d, on=["split", "perm"])
    mod[n] = dict(model_perm_macro=100 * float(m.macro.mean()), oracle_perm_macro=float(m.oracle_perm_macro.mean()), model_minus_oracle=float((100 * m.macro - m.oracle_perm_macro).mean()), n=len(m))
out["models_vs_oracle_on_same_draws"] = mod
json.dump(out, open(C.RES / "null_life_stage.json", "w"), indent=1)
print(json.dumps(out, indent=1))
