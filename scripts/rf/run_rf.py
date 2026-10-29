"""Random-forest baseline (results/rf/PREDECLARED_RF.txt) on the splits of an existing experiment; no tuning.
Usage: python -I run_rf.py cwru|uored|xjtu|rtf [rf rfu]      (default: both variants)
Output: results/rf/pred_<dataset>_<variant>.parquet (same columns and row order as the existing pred_*.parquet of the dataset) and per_split_<dataset>_<variant>.csv."""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "xjtu")); sys.path.insert(0, str(WORK / "scripts" / "rtf"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as R


def auroc2(Y, S):
    a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    return a, b, (a + b) / 2


def splits_cwru(z):
    spl = WORK / "bearing-data-leakage" / "data" / "splits" / "cwru" / "eval"
    wid, grp, sig, load, Y = z["wid"], z["group"], z["sig_loc"], z["load"], z["Y"]
    faulty_loc = np.array([g[:2] for g in grp])
    healthy = sig != faulty_loc
    unit = np.where(healthy, np.char.add("H_", grp), grp)
    for run in range(3, 53):
        for scen in ("DE", "FE"):
            tr_w = set(pd.read_parquet(spl / f"train_{scen}_run_{run}.parquet")["waveform_id"])
            te_w = set(pd.read_parquet(spl / f"test_{scen}_run_{run}.parquet")["waveform_id"])
            tr, te = np.isin(wid, list(tr_w)), np.isin(wid, list(te_w))
            assert not (tr_w & te_w) and not np.any(tr & te)
            assert not (set(grp[tr]) & set(grp[te])), "configuration shared between train and test"
            assert not (set(sig[tr & healthy]) & set(sig[te & healthy])), "healthy bearing shared between train and test"
            meta = pd.DataFrame({"run": run, "scenario": scen, "split": f"{run}_{scen}", "unit": unit[te], "group": grp[te], "healthy": healthy[te],
                                 "sig_loc": sig[te], "load": load[te].astype(int)})
            yield f"{run}_{scen}", tr, te, meta


def splits_uored(z):
    spl = WORK / "bearing-data-leakage" / "data" / "splits" / "uored-vafcls" / "eval"
    bearing = z["bearing"]
    for run in range(5, 105):
        sp = pd.read_parquet(spl / f"run_{run}.parquet").iloc[0]
        tr_ids, te_ids = [int(i) for i in sp["train_ids"]], [int(i) for i in sp["test_ids"]]
        assert not set(tr_ids) & set(te_ids)
        tr, te = np.isin(bearing, tr_ids), np.isin(bearing, te_ids)
        assert set(bearing[te].tolist()) == set(te_ids) and set(bearing[tr].tolist()) == set(tr_ids)
        meta = pd.DataFrame({"split": run, "bearing": bearing[te].astype(int), "cls": z["cls"][te], "state": z["state"][te].astype(int)})
        yield run, tr, te, meta


def splits_xjtu(z):
    import xjtu_common as C
    sp = C.load_splits(); exp = z["exp"]
    for run in range(5, 105):
        tr_ids, te_ids = sp[run]
        assert not set(tr_ids) & set(te_ids)
        tr, te = np.isin(exp, tr_ids), np.isin(exp, te_ids)
        assert set(exp[te]) == set(te_ids) and set(exp[tr]) == set(tr_ids)
        yield run, tr, te, pd.DataFrame({"split": run, "bearing": exp[te], "stage": z["stage"][te], "fileno": z["fileno"][te]})


def splits_rtf(z):
    import rtf_common as C
    sp = C.load_splits(); exp = z["exp"]
    for run in range(5, 105):
        tr_ids, te_ids = sp[run]
        assert not set(tr_ids) & set(te_ids)
        tr, te = np.isin(exp, tr_ids), np.isin(exp, te_ids)
        assert set(exp[te]) == set(te_ids) and set(exp[tr]) == set(tr_ids)
        yield run, tr, te, pd.DataFrame({"split": run, "bearing": exp[te], "stage": z["stage"][te], "M": z["M"][te]})


GEN = {"cwru": splits_cwru, "uored": splits_uored, "xjtu": splits_xjtu, "rtf": splits_rtf}


def main():
    ds = sys.argv[1]
    variants = sys.argv[2:] or ["rf", "rfu"]
    if not (R.RES / "PREDECLARED_RF.txt").exists():
        raise SystemExit("pre-declaration missing")
    z = dict(np.load(R.DATA / f"features_{ds}.npz", allow_pickle=False))
    X, Y = z["feat"], z["Y"]
    assert X.shape[1] == 20 and np.isfinite(X).all()
    for v in variants:
        preds, rows, t0 = [], [], time.time()
        for sid, tr, te, meta in GEN[ds](z):
            S = R.fit_scores(X[tr], Y[tr], X[te], R.VARIANTS[v])
            ai, ao, am = auroc2(Y[te], S)
            p = meta.copy()
            p["y_inner"], p["y_outer"] = Y[te, 0].astype(int), Y[te, 1].astype(int)
            p["s_inner"], p["s_outer"] = S[:, 0], S[:, 1]
            preds.append(p)
            rows.append(dict(split=sid, auroc_inner=ai, auroc_outer=ao, macro=am, n_train=int(tr.sum()), n_test=int(te.sum())))
            print(f"{ds} {v} split {sid}: macro {100 * am:.1f} (in {100 * ai:.1f} out {100 * ao:.1f}) t={time.time() - t0:.0f}s", flush=True)
        pd.concat(preds, ignore_index=True).to_parquet(R.RES / f"pred_{ds}_{v}.parquet")
        pd.DataFrame(rows).to_csv(R.RES / f"per_split_{ds}_{v}.csv", index=False)
        print(f"DONE {ds} {v}: macro {100 * np.mean([r['macro'] for r in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
