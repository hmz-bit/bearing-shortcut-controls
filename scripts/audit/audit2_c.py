"""Part C of results/audit2/PREDECLARED_AUDIT2.txt: metadata-only control (random forest on non-fault metadata, bearing-wise splits).
Usage: PYTHONUTF8=1 python -I audit2_c.py   Output: results/audit2/c_results.json, c_log.txt"""
import sys, pathlib, json, re, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "rf")); sys.path.insert(0, str(WORK / "scripts" / "evaltools"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as R
import run_rf as RR
import pu_data as pu

OUT = WORK / "results" / "audit2"
THRESH = 56.0


def macro(Y, S):
    try:
        return 100 * (roc_auc_score(Y[:, 0], S[:, 0]) + roc_auc_score(Y[:, 1], S[:, 1])) / 2
    except ValueError:
        return None


def features(ds, z):
    if ds == "xjtu":
        w = np.load(WORK / "data/xjtu/xjtu_windows.npz", allow_pickle=False)
        return np.stack([w["cond"], w["raw_sum"] / w["raw_n"], w["raw_n"]], 1).astype(float), "condition id, DC offset, record length"
    if ds == "rtf":
        w = np.load(WORK / "data/rtf/rtf_windows.npz", allow_pickle=False)
        return np.stack([w["f_r"], w["raw_sum"] / w["raw_n"], w["raw_n"]], 1).astype(float), "set speed, DC offset, record length"
    if ds == "uored":
        st = pd.read_csv(WORK / "data/uored/b1_file_stats.csv").set_index("fname")
        fn = np.array([f if f.endswith(".csv") else f + ".csv" for f in z["fname"]])
        miss = set(fn) - set(st.index)
        assert not miss, list(miss)[:5]
        s = st.loc[fn]
        return np.stack([s.raw_sum / s.n_raw, s.n_raw], 1).astype(float), "DC offset, record length"
    if ds == "cwru":
        return np.stack([z["load"], (z["sig_loc"] == "FE")], 1).astype(float), "load, sensor location"
    raise KeyError(ds)


def run_ds(ds):
    z = dict(np.load(R.DATA / f"features_{ds}.npz", allow_pickle=False))
    X, desc = features(ds, z)
    assert X.shape[0] == len(z["Y"]) and np.isfinite(X).all()
    Y = z["Y"]
    vals = []
    for sid, tr, te, meta in RR.GEN[ds](z):
        S = R.fit_scores(X[tr], Y[tr], X[te], R.PARAMS_U)
        m = macro(Y[te], S)
        if m is not None:
            vals.append(m)
    return dict(dataset=ds, features=desc, mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), n_splits=len(vals), flagged=bool(np.mean(vals) > THRESH))


def run_pu():
    z = np.load(WORK / "data/cache/pu_windows.npz", allow_pickle=False)
    keep = np.isin(z["bearing"], ["K001", "K002", "K003", "K004", "K005", "K006", "KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"])
    b, cond = z["bearing"][keep], z["condition"][keep]
    num = np.array([[int(x) for x in re.findall(r"\d+", c)] for c in cond], float)   # N, M, F
    Y = np.array([pu.bearing_labels(x) for x in b], float)
    vals = []
    for run in range(5, 105):
        tr_ids, te_ids = pu.load_eval_split(run)
        tr, te = np.isin(b, tr_ids), np.isin(b, te_ids)
        S = R.fit_scores(num[tr], Y[tr], num[te], R.PARAMS_U)
        m = macro(Y[te], S)
        if m is not None:
            vals.append(m)
    return dict(dataset="pu", features="speed, load torque, radial force (from condition code)", mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), n_splits=len(vals), flagged=bool(np.mean(vals) > THRESH))


if __name__ == "__main__":
    out = []
    t0 = time.time()
    for ds in ("uored", "cwru", "xjtu", "rtf"):
        r = run_ds(ds); out.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    r = run_pu(); out.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    (OUT / "c_results.json").write_text(json.dumps(out, indent=1))
