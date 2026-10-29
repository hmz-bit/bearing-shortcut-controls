"""Repeated shuffled-label runs (training labels permuted across windows, LF-PNCN t_mad). A single 6-bearing test set is a very noisy
draw for a null model (windows of one bearing get nearly the same score), so the null AUROC is averaged over many splits/pairs.
Tier 3: splits lo..hi, both test speeds. Tier 2: pairs lo..hi (seed = pair index).
Writes results/tier2_3/shuffle_<tier>_<lo>-<hi>.json.   Usage: python -I sanity_shuffle.py --tier t3 --range 5-24
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import common23 as K
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--range", required=True)
    a = ap.parse_args()
    lo, hi = K.parse_range(a.range)
    C.set_determinism()
    d = K.load_all(("t_mad",))
    X, Y, bearing, cond = d["X"]["t_mad"], d["Y"], d["bearing"], d["condition"]
    dev = K.device()
    rows = []
    for i in range(lo, hi + 1):
        if a.tier == "t3":
            tr_ids, te_ids = C.load_split("eval", i)
            tr = np.isin(bearing, tr_ids) & (cond == K.COND_SAME)
            masks = {"same": np.isin(bearing, te_ids) & (cond == K.COND_SAME), "slow": np.isin(bearing, te_ids) & (cond == K.COND_SLOW)}
            r = K.train_generic(X, Y, tr, masks, seed=i, dev=dev, shuffle_window=True)
            row = dict(idx=i, **{k: 100 * C.macro_auroc_np(Y[masks[k]], S)[0] for k, S in r["scores"].items()})
        else:
            held = list(K.PAIRS[i])
            tr = np.isin(bearing, K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held])
            te = np.isin(bearing, K.REAL_FAULTY + held)
            r = K.train_generic(X, Y, tr, {"test": te}, seed=i, dev=dev, shuffle_window=True)
            row = dict(idx=i, test=100 * C.macro_auroc_np(Y[te], r["scores"]["test"])[0])
        rows.append({k: float(v) for k, v in row.items()})
        print(rows[-1], flush=True)
    (K.OUT / f"shuffle_{a.tier}_{lo}-{hi}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
