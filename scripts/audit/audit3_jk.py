"""Parts J and K of results/audit3/PREDECLARED_AUDIT3.txt.
J: XJTU-SY leave-one-condition-out.   Usage: python -I audit3_jk.py J
K: faulty-stage-only fault-type task.  Usage: python -I audit3_jk.py K <xjtu|rtf> <base|rowz_drop|rf|rfu|meta>
Analysis of K: python -I audit3_jk.py Kan"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (WORK / "scripts" / "phase2", WORK / "scripts" / "xjtu", WORK / "scripts" / "rtf", WORK / "scripts" / "rf", HERE):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as RC

OUT = WORK / "results" / "audit3"
OUT.mkdir(exist_ok=True)
RF_PARAMS = {"rf": RC.PARAMS_PRIMARY, "rfu": RC.PARAMS_U, "meta": RC.PARAMS_U}


def auc_pair(Y, S, mask=None):
    if mask is not None:
        Y, S = Y[mask], S[mask]
    try:
        a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    except ValueError:
        return None
    return 100 * a, 100 * b, 100 * (a + b) / 2


def run_lf(name, X, Y, exp, tr, he, seed, dev):
    import p2common as P, trainer as T, variants as V
    opt = dict(V.VARIANTS[name]["opt"])
    r = T.train_eval(X, Y, exp, None, None, dict(P.BASE_CFG), seed=seed, opt=opt, dev=dev, train_mask=tr, held_mask=he)
    return r["scores"]


def part_j():
    import xjtu_common as C, p2common as P, common as CC
    CC.set_determinism(); dev = P.device()
    w = C.load_windows()
    X, Y, exp, stage, cond = w["t_mad"], w["Y"], w["exp"], w["stage"], w["cond"]
    z = dict(np.load(RC.DATA / "features_xjtu.npz", allow_pickle=False)); assert (z["exp"] == exp).all()
    F = z["feat"]
    res = {}
    for c in (1, 2, 3):
        te = cond == c; tr = ~te
        fm = stage == "faulty"
        row = {}
        for name in ("base", "rowz_drop"):
            S = run_lf(name, X, Y, exp, tr, te, 7000 + c, dev)
            Sf = np.full((len(Y), 2), np.nan); Sf[te] = S
            row[name] = dict(all=auc_pair(Y[te], S), stage_matched=auc_pair(Y[te & fm], Sf[te & fm]))
        for name in ("rf", "rfu"):
            S = RC.fit_scores(F[tr], Y[tr], F[te], RF_PARAMS[name])
            Sf = np.full((len(Y), 2), np.nan); Sf[te] = S
            row[name] = dict(all=auc_pair(Y[te], S), stage_matched=auc_pair(Y[te & fm], Sf[te & fm]))
        res[f"hold_out_condition_{c}"] = row
        print("J", c, {k: {m: (None if v is None else [round(t, 1) for t in v]) for m, v in r.items()} for k, r in row.items()}, flush=True)
    (OUT / "j_results.json").write_text(json.dumps(res, indent=1))


def part_k(ds, model):
    import p2common as P, common as CC
    CC.set_determinism(); dev = P.device()
    if ds == "xjtu":
        import xjtu_common as C
        w = C.load_windows(); sp = C.load_splits()
        meta = np.stack([w["cond"]], 1).astype(float)
    else:
        import rtf_common as C
        w = C.load_windows(); sp = C.load_splits()
        sr = np.array([1.0 if int(e[1:]) <= 9 else 0.0 for e in w["exp"]])
        meta = np.stack([w["f_r"], sr], 1).astype(float)
    X, Y, exp, stage = w["t_mad"], w["Y"], w["exp"], w["stage"]
    fm = stage == "faulty"
    if model in RF_PARAMS and model != "meta":
        z = dict(np.load(RC.DATA / f"features_{ds}.npz", allow_pickle=False)); assert (z["exp"] == exp).all(); F = z["feat"]
    preds, rows, skipped, t0 = [], [], 0, time.time()
    for run in range(5, 105):
        tr_ids, te_ids = sp[run]
        tr, te = np.isin(exp, tr_ids) & fm, np.isin(exp, te_ids) & fm
        if any(len(np.unique(Y[te][:, c])) < 2 for c in (0, 1)) or any(len(np.unique(Y[tr][:, c])) < 2 for c in (0, 1)):
            skipped += 1; continue
        if model not in RF_PARAMS:
            S = run_lf(model, X, Y, exp, tr, te, run, dev)
        else:
            Xf = meta if model == "meta" else F
            S = RC.fit_scores(Xf[tr], Y[tr], Xf[te], RF_PARAMS[model])
        a = auc_pair(Y[te], S)
        preds.append(pd.DataFrame({"split": run, "bearing": exp[te], "stage": stage[te], "y_inner": Y[te, 0].astype(int), "y_outer": Y[te, 1].astype(int),
                                   "s_inner": S[:, 0], "s_outer": S[:, 1]}))
        rows.append(dict(split=run, inner=a[0], outer=a[1], macro=a[2]))
        if run % 10 == 5:
            print(f"K {ds} {model} split {run}: macro {a[2]:.1f} t={time.time() - t0:.0f}s", flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_k_{ds}_{model}.parquet")
    pd.DataFrame(rows).to_csv(OUT / f"per_split_k_{ds}_{model}.csv", index=False)
    print(f"DONE K {ds} {model}: macro {np.mean([r['macro'] for r in rows]):.2f} inner {np.mean([r['inner'] for r in rows]):.2f} outer {np.mean([r['outer'] for r in rows]):.2f} defined {len(rows)} skipped {skipped}", flush=True)


def part_kan():
    import audit2_ab as A, audit3_e_i as E
    res = {}
    for ds in ("xjtu", "rtf"):
        fr = {m: A.std(pd.read_parquet(OUT / f"pred_k_{ds}_{m}.parquet"), ds) for m in ("base", "rowz_drop", "rf", "rfu", "meta", "b1")}
        # clusters present
        clusters = sorted(set().union(*[set(f.cluster) for f in fr.values()]))
        types = A.cluster_types(fr["base"], sorted(set(fr["base"].cluster)))
        # use the cluster list of base for bootstrap; all models share the same splits and bearings
        clusters = sorted(set(fr["base"].cluster))
        M = A.draws(clusters, types, np.random.default_rng(A.SEED)); ones = np.ones((1, len(clusters)))
        tabs = {k: A.pair_table(f, clusters) for k, f in fr.items()}
        pairs = (("rowz_drop", "base"), ("rowz_drop", "rf"), ("rowz_drop", "meta"), ("rowz_drop", "b1"), ("base", "rf"), ("base", "b1"), ("rf", "b1"), ("rf", "meta"))
        res[ds] = E.contrast(tabs, M, ones, ["macro", 0, 1], pairs)
        print("Kan", ds, {l: {k: round(v, 1) for k, v in r["points"].items()} for l, r in res[ds].items()}, flush=True)
    (OUT / "k_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "J": part_j()
    elif a[0] == "K": part_k(a[1], a[2])
    elif a[0] == "Kan": part_kan()
