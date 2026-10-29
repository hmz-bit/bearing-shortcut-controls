"""Post hoc diagnostic declared in results/rf/PREDECLARED_RF_ADDENDUM1.txt: primary random forest on the time-statistics block (columns 0-4) and on the
spectral block (columns 5-19) only. Usage: python -I run_rf_ablation.py cwru|uored|xjtu|rtf  ->  results/rf/ablation_<dataset>.json"""
import sys, pathlib, json, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as R
import run_rf as RR

SUBSETS = {"time5": slice(0, 5), "spectral15": slice(5, 20)}


def main():
    ds = sys.argv[1]
    if not (R.RES / "PREDECLARED_RF_ADDENDUM1.txt").exists():
        raise SystemExit("addendum missing")
    z = dict(np.load(R.DATA / f"features_{ds}.npz", allow_pickle=False))
    X, Y = z["feat"], z["Y"]
    out = {}
    for name, cols in SUBSETS.items():
        m_all, m_sm, t0 = [], [], time.time()
        for sid, tr, te, meta in RR.GEN[ds](z):
            S = R.fit_scores(X[tr][:, cols], Y[tr], X[te][:, cols], R.PARAMS_PRIMARY)
            ai, ao, am = RR.auroc2(Y[te], S)
            m_all.append((ai, ao, am))
            if ds in ("xjtu", "rtf"):
                keep = meta["stage"].to_numpy() == "faulty"
                Yt = Y[te][keep]; St = S[keep]
                if Yt[:, 0].min() != Yt[:, 0].max() and Yt[:, 1].min() != Yt[:, 1].max():
                    m_sm.append(np.mean([roc_auc_score(Yt[:, 0], St[:, 0]), roc_auc_score(Yt[:, 1], St[:, 1])]))
        a = 100 * np.array(m_all)
        out[name] = dict(inner=float(a[:, 0].mean()), outer=float(a[:, 1].mean()), macro=float(a[:, 2].mean()), sd_over_splits=float(a[:, 2].std()), n_splits=len(a))
        if m_sm:
            out[name].update(stage_matched_macro=float(100 * np.mean(m_sm)), stage_matched_valid_splits=len(m_sm))
        print(ds, name, json.dumps(out[name]), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(R.RES / f"ablation_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
