"""Tier 2 (artificial to real, second half of H2). 15 pairs of held-out healthy bearings (C(6,2) of K001..K006).
Train: 12 artificial bearings (KA01 KA03 KA05 KA06 KA07 KA08 KA09 | KI01 KI03 KI05 KI07 KI08) + the 4 remaining healthy bearings, all four
operating conditions. Test: 11 real faulty bearings (KA04 KA15 KA16 KA22 KA30 | KI04 KI14 KI16 KI17 KI18 KI21) + the 2 held-out healthy bearings.
Methods lfpncn (t_mad), b4 (t_max), f (t_fix): 2 seeds per pair (seed = pair index p and p + 1000); b3: RF (seed 42).
Config: tier1 chosen_config_t_mad.json for every neural method; standardiser from training windows only.
Output: results/tier2_3/t2_<method>_<lo>-<hi>_predictions_long.parquet: split = pair index*10 + seed index
(0: seed p, 1: seed p+1000), plus pair and seed columns.
Usage: python -I run_tier2.py --methods lfpncn b4 f b3 --pairs 0-14
"""
import argparse, time, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common23 as K
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lfpncn", "b4", "f", "b3"])
    ap.add_argument("--pairs", default="0-14")
    a = ap.parse_args()
    lo, hi = K.parse_range(a.pairs)
    C.set_determinism()
    variants = sorted({K.NEURAL[m] for m in a.methods if m in K.NEURAL})
    d = K.load_all(tuple(variants), need_b3="b3" in a.methods)
    bearing, Y = d["bearing"], d["Y"]
    dev = K.device()
    out = {m: [] for m in a.methods}
    rows, t0 = [], time.time()
    for p in range(lo, hi + 1):
        held = list(K.PAIRS[p])
        train_ids = K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held]
        test_ids = K.REAL_FAULTY + held
        assert not set(train_ids) & set(test_ids) and len(train_ids) == 16 and len(test_ids) == 13
        tr, te = np.isin(bearing, train_ids), np.isin(bearing, test_ids)
        assert set(bearing[tr]) == set(train_ids) and set(bearing[te]) == set(test_ids) and not np.any(tr & te)
        for meth in a.methods:
            runs = [(0, p)] if meth == "b3" else [(0, p), (1, p + 1000)]
            for si, seed in runs:
                if meth == "b3":
                    t1 = time.time()
                    S = K.rf_scores(d["b3"], Y, tr, te)
                    secs = time.time() - t1
                else:
                    r = K.train_generic(d["X"][K.NEURAL[meth]], Y, tr, {"test": te}, seed=seed, dev=dev)
                    S, secs = r["scores"]["test"], r["seconds"]
                mac, ai, ao = C.macro_auroc_np(Y[te], S)
                out[meth].append(K.long_frame(p * 10 + si, bearing[te], Y[te], S, pair=p, seed=seed))
                rows.append(dict(method=meth, pair=p, held_healthy="+".join(held), seed=seed, auroc_inner=ai, auroc_outer=ao,
                                 macro=mac, train_seconds=secs, n_train=int(tr.sum()), n_test=int(te.sum())))
                print(f"T2 {meth:7s} pair {p} ({'+'.join(held)}) seed {seed}: macro {100*mac:.1f} "
                      f"(in {100*ai:.1f} out {100*ao:.1f}) {secs:.1f}s elapsed {time.time()-t0:.0f}s", flush=True)
    K.OUT.mkdir(parents=True, exist_ok=True)
    for meth, fr in out.items():
        pd.concat(fr, ignore_index=True).to_parquet(K.OUT / f"t2_{meth}_{lo}-{hi}_predictions_long.parquet")
    pd.DataFrame(rows).to_csv(K.OUT / f"t2_{'-'.join(a.methods)}_{lo}-{hi}_per_pair.csv", index=False)
    print("DONE", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
