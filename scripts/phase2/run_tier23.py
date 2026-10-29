"""Phase C, Tier 3 (cross-speed) and Tier 2 (artificial to real) for a pre-declared variant whose input is the Tier 1 cache (feat == 'L1').
Same protocol as scripts/tier2_3/run_tier3.py and run_tier2.py (config = Tier 1 chosen config, seed = run for Tier 3; pair p with seeds p and p+1000 for Tier 2).
LOCKED until PREDECLARED.txt exists and P2_ALLOW_EVAL=1.
Usage: P2_ALLOW_EVAL=1 python -I run_tier23.py <variant> --tier 3 --runs 5-104   |   --tier 2 --pairs 0-14
Output: results/phase2/eval/t3_<variant>_<lo>-<hi>_predictions_long.parquet (column test_cond) / t2_<variant>_<lo>-<hi>_predictions_long.parquet
"""
import argparse, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V
import common23 as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--runs", default="5-104")
    ap.add_argument("--pairs", default="0-14")
    a = ap.parse_args()
    if not P.PRED.exists() or a.name not in P.PRED.read_text():
        raise SystemExit("variant not listed in PREDECLARED.txt")
    v = V.VARIANTS[a.name]
    assert v["feat"] == "L1", "run_tier23 supports only variants on the Tier 1 cache input"
    import common as C
    C.set_determinism()
    d = K.load_all(("t_mad",))
    X, Y, bearing, cond = d["X"]["t_mad"], d["Y"], d["bearing"], d["condition"]
    opt = dict(v["opt"])
    assert opt.get("extra") is None
    cfg = dict(P.BASE_CFG); cfg.update(v.get("cfg", {}))
    dev = P.device()
    out_dir = P.OUT / "eval"; out_dir.mkdir(exist_ok=True)
    frames, t0 = [], time.time()
    if a.tier == 3:
        lo, hi = K.parse_range(a.runs)
        for run in range(lo, hi + 1):
            tr_ids, te_ids = P.load_split("eval", run)
            tr = np.isin(bearing, tr_ids) & (cond == K.COND_SAME)
            m15 = np.isin(bearing, te_ids) & (cond == K.COND_SAME)
            m09 = np.isin(bearing, te_ids) & (cond == K.COND_SLOW)
            r = T.train_eval(X, Y, bearing, None, None, cfg, seed=run, opt=opt, dev=dev, train_mask=tr, held_mask=m15, extra_held={"n09": m09})
            for tc, m, S in ((K.COND_SAME, m15, r["scores"]), (K.COND_SLOW, m09, r["extra"]["n09"]["scores"])):
                frames.append(K.long_frame(run, bearing[m], Y[m], S, test_cond=tc))
            print(f"T3 {a.name} split {run}: 1500 {100*r['macro']:.1f} | 900 {100*r['extra']['n09']['macro']:.1f} {r['seconds']:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"t3_{a.name}_{lo}-{hi}_predictions_long.parquet")
    else:
        lo, hi = K.parse_range(a.pairs)
        for p in range(lo, hi + 1):
            held = list(K.PAIRS[p])
            train_ids = K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held]
            test_ids = K.REAL_FAULTY + held
            tr, te = np.isin(bearing, train_ids), np.isin(bearing, test_ids)
            for si, seed in ((0, p), (1, p + 1000)):
                r = T.train_eval(X, Y, bearing, None, None, cfg, seed=seed, opt=opt, dev=dev, train_mask=tr, held_mask=te)
                frames.append(K.long_frame(p * 10 + si, bearing[te], Y[te], r["scores"], pair=p, seed=seed))
                print(f"T2 {a.name} pair {p} seed {seed}: macro {100*r['macro']:.1f} (in {100*r['inner']:.1f} out {100*r['outer']:.1f}) elapsed {time.time()-t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"t2_{a.name}_{lo}-{hi}_predictions_long.parquet")
    print("DONE", round(time.time() - t0))


if __name__ == "__main__":
    main()
