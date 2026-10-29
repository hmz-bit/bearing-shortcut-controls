"""Frozen LF-PNCN variants on the XJTU-SY windows (results/xjtu/PREDECLARED_XJTU.txt).
Usage: python -I run_lfpncn.py <variant> <first> <last> [perm|sens5]    variant in {base, rowz_drop}; splits first..last (5..104)
 'perm': bearing-label permutation control (2 permutations per split); 'sens5': 5% sensitivity rule (training and test windows restricted to sens5).
Output: results/xjtu/pred_[sens5_]<variant>_r<first>-<last>.parquet (long) + per_split_*.csv with held-out and TRAIN macro AUROC."""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "phase2"))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V
import common as CC
import xjtu_common as C


def main():
    name, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else ""
    perm, sens = mode == "perm", mode == "sens5"
    if not (C.RES / "PREDECLARED_XJTU.txt").exists():
        raise SystemExit("pre-declaration missing")
    CC.set_determinism()
    dev = P.device()
    w = C.load_windows()
    keep = w["sens5"] if sens else np.ones(len(w["Y"]), bool)
    X, Y, exp, stage, fileno = w["t_mad"][keep], w["Y"][keep], w["exp"][keep], w["stage"][keep], w["fileno"][keep]
    splits = C.load_splits()
    cfg = dict(P.BASE_CFG)
    opt = dict(V.VARIANTS[name]["opt"])
    assert V.VARIANTS[name]["feat"] == "L1"
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = splits[run]
        assert not set(tr_ids) & set(te_ids)
        for k in ((0, 1) if perm else (-1,)):
            Yr = Y
            if perm:
                Yr = C.perm_Y(Y, exp, stage, run, k, tr_ids, te_ids)
                if Yr is None:
                    rows.append(dict(split=run, perm=k, skipped=1)); continue
            r = T.train_eval(X, Yr, exp, tr_ids, te_ids, cfg, seed=run + (1000 * (k + 1) if perm else 0), opt=opt, dev=dev, score_train=True)
            he, tr = r["held_mask"], r["train_mask"]
            assert set(exp[he]) == set(te_ids) and set(exp[tr]) == set(tr_ids)
            tm, ti, to = CC.macro_auroc_np(Yr[tr], r["train_scores"])
            S = r["scores"]
            if not perm:
                preds.append(pd.DataFrame({"split": run, "bearing": exp[he], "stage": stage[he], "fileno": fileno[he], "y_inner": Yr[he, 0].astype(int),
                                           "y_outer": Yr[he, 1].astype(int), "s_inner": S[:, 0], "s_outer": S[:, 1]}))
            rows.append(dict(split=run, perm=k, skipped=0, auroc_inner=r["inner"], auroc_outer=r["outer"], macro=r["macro"], train_macro=tm, seconds=r["seconds"]))
            print(f"{name}{' perm'+str(k) if perm else ''}{' sens5' if sens else ''} split {run}: held {100*r['macro']:.1f} (IR {100*r['inner']:.1f} OR {100*r['outer']:.1f}) train {100*tm:.1f} t={time.time()-t0:.0f}s", flush=True)
    tag = f"{'perm_' if perm else ''}{'sens5_' if sens else ''}{name}_r{lo}-{hi}"
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(C.RES / f"pred_{tag}.parquet")
    pd.DataFrame(rows).to_csv(C.RES / f"per_split_{tag}.csv", index=False)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
