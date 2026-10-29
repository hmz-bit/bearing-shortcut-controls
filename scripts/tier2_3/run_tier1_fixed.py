"""Tier 1 completeness for the fixed-Hz ablation F: 100 evaluation splits (runs 5..104), Tier 1 protocol and config
(chosen_config_t_mad.json), standardiser from training windows only, seed of split r = r.
Writes results/tier2_3/f_tier1_<lo>-<hi>_predictions_long.parquet (+ per_split csv).
Usage: python -I run_tier1_fixed.py --runs 5-104
"""
import argparse, time, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common23 as K
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="5-104")
    ap.add_argument("--variant", default="t_fix", help="t_fix (F) or t_mad (reproduction check of Tier 1)")
    ap.add_argument("--tag", default="f_tier1")
    a = ap.parse_args()
    lo, hi = K.parse_range(a.runs)
    C.set_determinism()
    d = K.load_all((a.variant,))
    X, Y, bearing = d["X"][a.variant], d["Y"], d["bearing"]
    keep = np.isin(bearing, C.VIEIRA_BEARINGS)
    X, Y, bearing = X[keep], Y[keep], bearing[keep]
    dev = K.device()
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = C.load_split("eval", run)
        assert not set(tr_ids) & set(te_ids)
        tr, he = np.isin(bearing, tr_ids), np.isin(bearing, te_ids)
        assert set(bearing[tr]) == set(tr_ids) and set(bearing[he]) == set(te_ids) and not np.any(tr & he)
        r = K.train_generic(X, Y, tr, {"test": he}, seed=run, dev=dev)
        S = r["scores"]["test"]
        m, ai, ao = C.macro_auroc_np(Y[he], S)
        preds.append(K.long_frame(run, bearing[he], Y[he], S))
        rows.append(dict(split=run, auroc_inner=ai, auroc_outer=ao, macro=m, train_seconds=r["seconds"], n_train=r["n_train"], n_test=int(he.sum())))
        print(f"{a.tag} split {run}: macro {100*m:.2f} (in {100*ai:.1f} out {100*ao:.1f}) {r['seconds']:.1f}s elapsed {time.time()-t0:.0f}s", flush=True)
    K.OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(K.OUT / f"{a.tag}_{lo}-{hi}_predictions_long.parquet")
    pd.DataFrame(rows).to_csv(K.OUT / f"{a.tag}_{lo}-{hi}_per_split.csv", index=False)
    print("DONE", f"{100*np.mean([r['macro'] for r in rows]):.2f}", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
