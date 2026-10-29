"""Frozen external validation on UORED-VAFCLS (see results/uored/PREDECLARED_UORED.txt). No tuning.
Usage: python -I run_uored.py base rowz_drop   (runs Vieira's 100 evaluation splits run_5..104 once per variant)
Locked until PREDECLARED_UORED.txt exists. Output: results/uored/pred_<variant>.parquet (long) and per_split_<variant>.csv.
"""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "phase2"))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V
import common as C

OUT = WORK / "results" / "uored"
SPLITS = WORK / "bearing-data-leakage" / "data" / "splits" / "uored-vafcls" / "eval"


def main():
    if not (OUT / "PREDECLARED_UORED.txt").exists():
        raise SystemExit("pre-declaration missing")
    C.set_determinism()
    dev = P.device()
    z = np.load(WORK / "data" / "uored" / "uored_windows.npz", allow_pickle=False)
    X, Y, bearing, cls, state = z["t_mad"], z["Y"], z["bearing"], z["cls"], z["state"]
    cfg = dict(P.BASE_CFG)
    for name in sys.argv[1:]:
        opt = dict(V.VARIANTS[name]["opt"])
        assert V.VARIANTS[name]["feat"] == "L1"
        preds, rows, t0 = [], [], time.time()
        for run in range(5, 105):
            sp = pd.read_parquet(SPLITS / f"run_{run}.parquet").iloc[0]
            tr_ids, te_ids = [int(i) for i in sp["train_ids"]], [int(i) for i in sp["test_ids"]]
            assert not set(tr_ids) & set(te_ids)
            r = T.train_eval(X, Y, bearing, tr_ids, te_ids, cfg, seed=run, opt=opt, dev=dev)
            he = r["held_mask"]
            assert set(bearing[he].tolist()) == set(te_ids) and set(bearing[r["train_mask"]].tolist()) == set(tr_ids)
            S = r["scores"]
            preds.append(pd.DataFrame({"split": run, "bearing": bearing[he].astype(int), "cls": cls[he], "state": state[he].astype(int),
                                       "y_inner": Y[he, 0].astype(int), "y_outer": Y[he, 1].astype(int),
                                       "s_inner": S[:, 0], "s_outer": S[:, 1]}))
            rows.append(dict(split=run, auroc_inner=r["inner"], auroc_outer=r["outer"], macro=r["macro"], seconds=r["seconds"]))
            print(f"{name} split {run}: macro {100*r['macro']:.1f} (in {100*r['inner']:.1f} out {100*r['outer']:.1f}) t={time.time()-t0:.0f}s", flush=True)
        pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_{name}.parquet")
        pd.DataFrame(rows).to_csv(OUT / f"per_split_{name}.csv", index=False)
        print(f"DONE {name}: macro {100*np.mean([x['macro'] for x in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
