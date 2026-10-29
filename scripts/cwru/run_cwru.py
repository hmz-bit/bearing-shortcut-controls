"""Frozen external validation on CWRU (see results/cwru/PREDECLARED_CWRU.txt). No tuning.
Usage: python -I run_cwru.py --runs 3-12 base rowz_drop      (each run has two scenarios DE and FE; each (run, scenario) once per variant)
Locked until PREDECLARED_CWRU.txt exists. CPU: set P2_DEVICE=cpu P2_THREADS=2.
Output: results/cwru/pred_<variant>_r<lo>-<hi>.parquet (long) and per_split_<variant>_r<lo>-<hi>.csv
"""
import argparse, sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "phase2"))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V
import common as C

OUT = WORK / "results" / "cwru"
SPL = WORK / "bearing-data-leakage" / "data" / "splits" / "cwru" / "eval"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("variants", nargs="+"); a = ap.parse_args()
    if not (OUT / "PREDECLARED_CWRU.txt").exists():
        raise SystemExit("pre-declaration missing")
    lo, hi = map(int, a.runs.split("-"))
    C.set_determinism()
    dev = P.device()
    z = np.load(WORK / "data" / "cwru" / "cwru_windows.npz", allow_pickle=False)
    X, Y, wid, grp, sig, load = z["t_mad"], z["Y"], z["wid"], z["group"], z["sig_loc"], z["load"]
    faulty_loc = np.array([g[:2] for g in grp])
    healthy = sig != faulty_loc                                  # signal of the non-faulty bearing of the pair
    unit = np.where(healthy, np.char.add("H_", grp), grp)
    cfg = dict(P.BASE_CFG)
    for name in a.variants:
        opt = dict(V.VARIANTS[name]["opt"]); assert V.VARIANTS[name]["feat"] == "L1"
        preds, rows, t0 = [], [], time.time()
        for run in range(lo, hi + 1):
            for scen in ("DE", "FE"):
                tr_w = set(pd.read_parquet(SPL / f"train_{scen}_run_{run}.parquet")["waveform_id"])
                te_w = set(pd.read_parquet(SPL / f"test_{scen}_run_{run}.parquet")["waveform_id"])
                tr, te = np.isin(wid, list(tr_w)), np.isin(wid, list(te_w))
                assert not (tr_w & te_w) and not np.any(tr & te)
                assert not (set(grp[tr]) & set(grp[te])), "configuration shared between train and test"
                assert not (set(sig[tr & healthy]) & set(sig[te & healthy])), "healthy bearing shared between train and test"
                seed = run if scen == "DE" else run + 1000
                r = T.train_eval(X, Y, wid, None, None, cfg, seed=seed, opt=opt, dev=dev, train_mask=tr, held_mask=te)
                S = r["scores"]
                preds.append(pd.DataFrame({"run": run, "scenario": scen, "split": f"{run}_{scen}", "unit": unit[te], "group": grp[te], "healthy": healthy[te],
                                           "sig_loc": sig[te], "load": load[te].astype(int), "y_inner": Y[te, 0].astype(int), "y_outer": Y[te, 1].astype(int),
                                           "s_inner": S[:, 0], "s_outer": S[:, 1]}))
                rows.append(dict(run=run, scenario=scen, auroc_inner=r["inner"], auroc_outer=r["outer"], macro=r["macro"], n_train=r["n_train"], seconds=r["seconds"]))
                print(f"{name} run {run} {scen}: macro {100*r['macro']:.1f} (in {100*r['inner']:.1f} out {100*r['outer']:.1f}) {r['seconds']:.0f}s t={time.time()-t0:.0f}s", flush=True)
        tag = f"{name}_r{lo}-{hi}{a.tag}"
        pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_{tag}.parquet")
        pd.DataFrame(rows).to_csv(OUT / f"per_split_{tag}.csv", index=False)
        print(f"DONE {tag}: macro {100*np.mean([x['macro'] for x in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
