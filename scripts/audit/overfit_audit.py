"""Overfitting / memorisation audit (A1, A2, A3 of results/audit/PREDECLARED_AUDIT.txt). Works with `python -I`.
Usage: python -I overfit_audit.py a12 | a3
Never loads evaluation splits (A1/A2 use the 5 tuning splits; A3 uses the fixed Tier 2 healthy pairs on the 12 artificial bearings).
"""
import sys, pathlib, time, itertools
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "phase2"))
sys.path.insert(0, str(WORK / "scripts" / "tier2_3"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p2common as P
import trainer as T
import variants as V
import common as C
import common23 as K

OUT = WORK / "results" / "audit"
OUT.mkdir(exist_ok=True)
NAMES = ["base", "rowz_drop"]


def macro(Y, S):
    return C.macro_auroc_np(Y, S)


def a12():
    C.set_determinism()
    dev = P.device()
    meta = P.load_meta()
    X = V.load_feat("L1")
    bearing, Y = meta["bearing"], meta["Y"]
    cfg = dict(P.BASE_CFG)
    rng = np.random.RandomState(7000)
    rows, t0 = [], time.time()
    for run in P.TUNING_RUNS:
        tr_ids, va_ids = P.load_split("tuning", run)
        for name in NAMES:
            opt = dict(V.VARIANTS[name]["opt"])
            r = T.train_eval(X, Y, bearing, tr_ids, va_ids, cfg, seed=1000 + run, opt=opt, dev=dev, score_train=True)
            tr = r["train_mask"]
            m_tr, i_tr, o_tr = macro(Y[tr], r["train_scores"])
            rows.append(dict(test="A1", split=run, variant=name, perm=-1, train_macro=m_tr, held_macro=r["macro"], held_inner=r["inner"], held_outer=r["outer"]))
            print(f"A1 split {run} {name}: train {100*m_tr:.1f} held {100*r['macro']:.1f} gap {100*(m_tr-r['macro']):.1f} t={time.time()-t0:.0f}s", flush=True)
        # permutation draws are shared by both variants (same permuted labels)
        done = 0
        while done < 5:
            bl = sorted(set(tr_ids) | set(va_ids))
            lab = {b: Y[bearing == b][0].copy() for b in bl}
            perm = rng.permutation(len(bl))
            Yp = Y.copy()
            for b, j in zip(bl, perm):
                Yp[bearing == b] = lab[bl[j]]
            hm = np.isin(bearing, va_ids)
            if any(len(np.unique(Yp[hm][:, k])) < 2 for k in (0, 1)):
                continue
            for name in NAMES:
                opt = dict(V.VARIANTS[name]["opt"])
                r = T.train_eval(X, Yp, bearing, tr_ids, va_ids, cfg, seed=9000 + 10 * run + done, opt=opt, dev=dev, score_train=True)
                m_tr = macro(Yp[r["train_mask"]], r["train_scores"])[0]
                rows.append(dict(test="A2", split=run, variant=name, perm=done, train_macro=m_tr, held_macro=r["macro"], held_inner=r["inner"], held_outer=r["outer"]))
                print(f"A2 split {run} perm {done} {name}: train {100*m_tr:.1f} held {100*r['macro']:.1f} t={time.time()-t0:.0f}s", flush=True)
            done += 1
        pd.DataFrame(rows).to_csv(OUT / "a12_results.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "a12_results.csv", index=False)
    print("DONE a12", round(time.time() - t0))


def a3():
    C.set_determinism()
    dev = P.device()
    d = K.load_all(("t_mad",))
    X, Y, bearing = d["X"]["t_mad"], d["Y"], d["bearing"]
    cfg = dict(P.BASE_CFG)
    frames, t0 = [], time.time()
    for p, held in enumerate(K.PAIRS):
        held = list(held)
        train_ids = K.REAL_FAULTY + [h for h in K.HEALTHY if h not in held]
        test_ids = K.ARTIFICIAL + held
        tr, te = np.isin(bearing, train_ids), np.isin(bearing, test_ids)
        assert not np.any(tr & te)
        for name in NAMES:
            opt = dict(V.VARIANTS[name]["opt"])
            r = T.train_eval(X, Y, bearing, None, None, cfg, seed=p, opt=opt, dev=dev, train_mask=tr, held_mask=te)
            f = K.long_frame(p, bearing[te], Y[te], r["scores"], pair=p, variant=name)
            frames.append(f)
            print(f"A3 pair {p} {name}: macro {100*r['macro']:.1f} (in {100*r['inner']:.1f} out {100*r['outer']:.1f}) t={time.time()-t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(OUT / "a3_predictions_long.parquet")
    print("DONE a3", round(time.time() - t0))


if __name__ == "__main__":
    {"a12": a12, "a3": a3}[sys.argv[1]]()
