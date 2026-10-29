"""Tier 3 (H3, cross-speed). For each of the 100 Tier 1 evaluation splits: train ONLY on windows of condition N15_M07_F10
(1500 rpm) of the training bearings; evaluate the same trained model on the test bearings' windows at N15_M07_F10 (same speed)
and N09_M07_F10 (900 rpm). Methods: lfpncn (t_mad), b4 (t_max), f (t_fix), b3 (RF on b3).
Config: tier1 chosen_config_t_mad.json for every neural method; seed of split r = r. Standardiser: training windows only.
Output: results/tier2_3/t3_<method>_<lo>-<hi>_predictions_long.parquet with column test_cond.
Usage: python -I run_tier3.py --methods lfpncn b4 f b3 --runs 5-104
"""
import argparse, time, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common23 as K
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lfpncn", "b4", "f", "b3"])
    ap.add_argument("--runs", default="5-104")
    a = ap.parse_args()
    lo, hi = K.parse_range(a.runs)
    C.set_determinism()
    variants = sorted({K.NEURAL[m] for m in a.methods if m in K.NEURAL})
    d = K.load_all(tuple(variants), need_b3="b3" in a.methods)
    bearing, cond, Y = d["bearing"], d["condition"], d["Y"]
    dev = K.device()
    out = {m: [] for m in a.methods}
    rows, t0 = [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = C.load_split("eval", run)
        assert not set(tr_ids) & set(te_ids)
        tr = np.isin(bearing, tr_ids) & (cond == K.COND_SAME)
        masks = {K.COND_SAME: np.isin(bearing, te_ids) & (cond == K.COND_SAME),
                 K.COND_SLOW: np.isin(bearing, te_ids) & (cond == K.COND_SLOW)}
        assert set(bearing[tr]) == set(tr_ids) and not (set(bearing[tr]) & set(te_ids))
        for tc, m in masks.items():
            assert set(bearing[m]) == set(te_ids) and not np.any(m & tr)
        for meth in a.methods:
            if meth == "b3":
                t1 = time.time()
                scores = {tc: K.rf_scores(d["b3"], Y, tr, m) for tc, m in masks.items()}
                secs = time.time() - t1
            else:
                r = K.train_generic(d["X"][K.NEURAL[meth]], Y, tr, masks, seed=run, dev=dev)
                secs, scores = r["seconds"], r["scores"]
            msg = []
            for tc, m in masks.items():
                S = scores[tc]
                mac, ai, ao = C.macro_auroc_np(Y[m], S)
                out[meth].append(K.long_frame(run, bearing[m], Y[m], S, test_cond=tc))
                rows.append(dict(method=meth, split=run, test_cond=tc, auroc_inner=ai, auroc_outer=ao, macro=mac,
                                 train_seconds=secs, n_train=int(tr.sum()), n_test=int(m.sum())))
                msg.append(f"{tc[:3]} {100*mac:.1f}")
            print(f"T3 {meth:7s} split {run}: {' | '.join(msg)} {secs:.1f}s elapsed {time.time()-t0:.0f}s", flush=True)
    K.OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{lo}-{hi}"
    for meth, fr in out.items():
        pd.concat(fr, ignore_index=True).to_parquet(K.OUT / f"t3_{meth}_{tag}_predictions_long.parquet")
    pd.DataFrame(rows).to_csv(K.OUT / f"t3_{'-'.join(a.methods)}_{tag}_per_split.csv", index=False)
    print("DONE", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
