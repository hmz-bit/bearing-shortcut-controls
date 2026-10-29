"""Phase C, Tier 1: evaluate ONE pre-declared variant on the 100 evaluation splits (run 5..104) with the Tier 1 protocol
(config = Tier 1 chosen config, seed = run, standardiser from training windows only).
LOCKED until results/phase2/PREDECLARED.txt exists AND the environment variable P2_ALLOW_EVAL=1 is set; the variant name must be listed there.
Usage: P2_ALLOW_EVAL=1 python -I run_eval.py <variant> [--runs 5-104] [--seed-offset 0]
Output: results/phase2/eval/<variant>_r<lo>-<hi>[_so<offset>]_predictions_long.parquet (split, bearing, y_inner, y_outer, s_inner, s_outer) + _per_split.csv
"""
import argparse, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--runs", default="5-104")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    if not P.PRED.exists() or a.name not in P.PRED.read_text():
        raise SystemExit("variant not listed in PREDECLARED.txt")
    import common as C
    C.set_determinism()
    lo, hi = map(int, a.runs.split("-"))
    dev = P.device()
    meta = P.load_meta()
    v = V.VARIANTS[a.name]
    X = V.load_feat(v["feat"])
    opt = dict(v["opt"])
    if opt.get("extra") == "tstats":
        opt["extra"] = V.load_extra("tstats", meta)
    cfg = dict(P.BASE_CFG); cfg.update(v.get("cfg", {}))
    out_dir = P.OUT / "eval"; out_dir.mkdir(exist_ok=True)
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = P.load_split("eval", run)
        assert not set(tr_ids) & set(te_ids)
        r = T.train_eval(X, meta["Y"], meta["bearing"], tr_ids, te_ids, cfg, seed=run + a.seed_offset, opt=opt, dev=dev)
        he, tr = r["held_mask"], r["train_mask"]
        assert set(meta["bearing"][he]) == set(te_ids) and set(meta["bearing"][tr]) == set(tr_ids)
        S = r["scores"]
        preds.append(pd.DataFrame({"split": run, "bearing": meta["bearing"][he], "y_inner": meta["Y"][he, 0].astype(int),
                                   "y_outer": meta["Y"][he, 1].astype(int), "s_inner": S[:, 0], "s_outer": S[:, 1]}))
        rows.append(dict(split=run, auroc_inner=r["inner"], auroc_outer=r["outer"], macro=r["macro"], train_seconds=r["seconds"]))
        print(f"{a.name} split {run}: macro {100*r['macro']:.2f} (in {100*r['inner']:.1f} out {100*r['outer']:.1f}) {r['seconds']:.1f}s elapsed {time.time()-t0:.0f}s", flush=True)
    tag = f"{a.name}_r{lo}-{hi}" + ("" if a.seed_offset == 0 else f"_so{a.seed_offset}")
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / f"{tag}_predictions_long.parquet")
    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_per_split.csv", index=False)
    print(f"DONE {tag}: macro {100*np.mean([x['macro'] for x in rows]):.2f}")


if __name__ == "__main__":
    main()
