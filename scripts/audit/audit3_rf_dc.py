"""Part H of results/audit3/PREDECLARED_AUDIT3.txt: random forest on UORED with the window mean removed (rf_dm, rfu_dm).
Usage: PYTHONUTF8=1 python -I audit3_rf_dc.py     Output: results/audit3/pred_uored_{rf_dm,rfu_dm}.parquet, per_split_*.csv, h_check.json"""
import sys, pathlib, os, glob, json, time
from multiprocessing import Pool
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "rf")); sys.path.insert(0, str(WORK / "scripts" / "lfpncn")); sys.path.insert(0, str(WORK / "scripts" / "xjtu")); sys.path.insert(0, str(WORK / "scripts" / "rtf"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as R
import build_features as BF
import run_rf as RR
import features as F

OUT = WORK / "results" / "audit3"


def proc(path):
    base = os.path.basename(path)
    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), encoding="utf-8-sig")
    assert len(x) == 420000
    starts = list(range(0, len(x) - F.WIN + 1, BF.UO_HOP))
    feats = np.stack([R.window_features(x[s:s + F.WIN] - x[s:s + F.WIN].mean(), BF.UO_FR, F.FS_TARGET, BF.UO_GEOM) for s in starts])
    return base, feats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z = dict(np.load(R.DATA / "features_uored.npz", allow_pickle=False))
    files = sorted(glob.glob(str(WORK / "data" / "uored" / "csv" / "*.csv")))
    assert len(files) == 60
    with Pool(4) as p:
        res = p.map(proc, files, chunksize=2)
    fname = np.concatenate([[r[0]] * len(r[1]) for r in res])
    assert (fname == z["fname"]).all(), "row order differs from the stored features"
    feat = np.concatenate([r[1] for r in res])
    assert feat.shape == z["feat"].shape and np.isfinite(feat).all()
    spec_diff = float(np.abs(feat[:, 5:] - z["feat"][:, 5:]).max())
    ts_diff = np.abs(feat[:, :5] - z["feat"][:, :5]).max(0).tolist()
    print(f"features rebuilt in {time.time() - t0:.0f}s; max abs diff of the 15 spectral columns {spec_diff:.3e}; of the 5 time statistics {ts_diff}", flush=True)
    json.dump(dict(spectral_max_abs_diff=spec_diff, time_stat_max_abs_diff=ts_diff), open(OUT / "h_check.json", "w"))
    Y = z["Y"]
    z["feat"] = feat
    for name, params in (("rf_dm", R.PARAMS_PRIMARY), ("rfu_dm", R.PARAMS_U)):
        preds, rows = [], []
        for sid, tr, te, meta in RR.GEN["uored"](z):
            S = R.fit_scores(feat[tr], Y[tr], feat[te], params)
            ai, ao, am = RR.auroc2(Y[te], S)
            p = meta.copy()
            p["y_inner"], p["y_outer"] = Y[te, 0].astype(int), Y[te, 1].astype(int)
            p["s_inner"], p["s_outer"] = S[:, 0], S[:, 1]
            preds.append(p)
            rows.append(dict(split=sid, auroc_inner=ai, auroc_outer=ao, macro=am))
        pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_uored_{name}.parquet")
        pd.DataFrame(rows).to_csv(OUT / f"per_split_uored_{name}.csv", index=False)
        print(f"DONE {name}: macro {100 * np.mean([r['macro'] for r in rows]):.2f} inner {100 * np.mean([r['auroc_inner'] for r in rows]):.2f} outer {100 * np.mean([r['auroc_outer'] for r in rows]):.2f} t={time.time() - t0:.0f}s", flush=True)
