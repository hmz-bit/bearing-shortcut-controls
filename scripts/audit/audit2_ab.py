"""Parts A and B of results/audit2/PREDECLARED_AUDIT2.txt (evaluation only, existing predictions, no training).
A: healthy-only negatives (window level) for PU, UORED, CWRU.  B: unit-level evaluation for all five datasets.
Intervals: paired bearing-level cluster bootstrap (B = 2000, seed 20261029), stratified by cluster label type.
Usage: PYTHONUTF8=1 python -I audit2_ab.py   (paths relative to this file)"""
import glob, json, sys, pathlib, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
R = WORK / "results"
OUT = R / "audit2"
B, SEED = 2000, 20261029
KNOWN = {  # window-level Macro-AUROC means from results/CROSS_DATASET_SUMMARY.md and results/rf/SUMMARY.md
    ("pu", "base"): 85.63, ("pu", "rowz_drop"): 88.85, ("pu", "b1"): 79.58,
    ("uored", "base"): 80.92, ("uored", "rowz_drop"): 91.55, ("uored", "b1"): 94.21, ("uored", "rf"): 94.43,
    ("cwru", "base"): 88.40, ("cwru", "rowz_drop"): 95.29, ("cwru", "b1"): 71.75, ("cwru", "rf"): 80.35,
    ("xjtu", "base"): 74.07, ("xjtu", "rowz_drop"): 76.04, ("xjtu", "b1"): 69.84, ("xjtu", "rf"): 79.77,
    ("rtf", "base"): 76.42, ("rtf", "rowz_drop"): 78.85, ("rtf", "b1"): 72.51, ("rtf", "rf"): 85.70,
}


def cat(pattern, exclude=("perm", "sens5", "smoke", "partial")):
    fs = [f for f in sorted(glob.glob(str(pattern))) if not any(x in pathlib.Path(f).name for x in exclude)]
    assert fs, pattern
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def std(df, ds):
    """Common columns: split, cluster, unit, healthy, state2 (UORED faulty stage), y_inner, y_outer, s_inner, s_outer."""
    d = df.copy()
    d["split"] = d["split"].astype(str)
    if ds == "pu":
        d = d[d.split.astype(int).between(5, 104)]
        d["cluster"] = d["bearing"].astype(str); d["unit"] = d["cluster"]; d["healthy"] = d["cluster"].str.startswith("K0")
    elif ds == "uored":
        d["cluster"] = "U" + d["bearing"].astype(int).astype(str); d["healthy"] = d["cls"] == "H"; d["unit"] = d["cluster"] + "|" + d["cls"].astype(str)
    elif ds == "cwru":
        d["cluster"] = d["unit"].astype(str); d["healthy"] = d["healthy"].astype(bool)
    else:
        d["cluster"] = d["bearing"].astype(str); d["unit"] = d["cluster"] + "|" + d["stage"].astype(str); d["healthy"] = d["stage"] == "healthy"
    d["state2"] = (d["state"] == 2) if "state" in d else False
    return d[["split", "cluster", "unit", "healthy", "state2", "y_inner", "y_outer", "s_inner", "s_outer"]].reset_index(drop=True)


def load_models(ds):
    if ds == "pu":
        m = {"base": pd.read_parquet(R / "tier1/lfpncn_predictions_long.parquet"),
             "rowz_drop": pd.read_parquet(R / "phase2/eval/rowz_drop_r5-104_predictions_long.parquet"),
             "b1": pd.read_parquet(WORK / "scripts/evaltools/results/b1_predictions_long.parquet")}
    elif ds == "uored":
        m = {k: pd.read_parquet(R / f"uored/pred_{k}.parquet") for k in ("base", "rowz_drop", "b1")}
        m["rf"] = pd.read_parquet(R / "rf/pred_uored_rf.parquet")
    elif ds == "cwru":
        m = {"base": cat(R / "cwru/pred_base_r*.parquet"), "rowz_drop": cat(R / "cwru/pred_rowz_drop_r*.parquet"),
             "b1": pd.read_parquet(R / "cwru/pred_b1_r3-52.parquet"), "rf": pd.read_parquet(R / "rf/pred_cwru_rf.parquet")}
    else:
        m = {"base": cat(R / f"{ds}/pred_base_r*.parquet"), "rowz_drop": cat(R / f"{ds}/pred_rowz_drop_r*.parquet"),
             "b1": pd.read_parquet(R / f"{ds}/pred_b1_r5-104.parquet"), "rf": pd.read_parquet(R / f"rf/pred_{ds}_rf.parquet")}
    return {k: std(v, ds) for k, v in m.items()}


def window_macro(d):
    out = []
    for _, g in d.groupby("split"):
        try:
            out.append((roc_auc_score(g.y_inner, g.s_inner) + roc_auc_score(g.y_outer, g.s_outer)) / 2)
        except ValueError:
            pass
    return 100 * np.mean(out), len(out)


def unit_frame(d):
    g = d.groupby(["split", "unit"], sort=False)
    u = g.agg(cluster=("cluster", "first"), healthy=("healthy", "first"), state2=("state2", "first"),
              y_inner=("y_inner", "first"), y_outer=("y_outer", "first"), s_inner=("s_inner", "mean"), s_outer=("s_outer", "mean")).reset_index()
    return u


def pair_table(d, clusters, keep=None):
    """C (S,2,Bn,Bn), Npos, Nneg (S,2,Bn); splits list. keep: dict label -> boolean mask over rows (default all)."""
    cidx = {c: i for i, c in enumerate(clusters)}
    splits = sorted(d.split.unique(), key=lambda x: (len(x), x))
    Bn = len(clusters)
    C = np.zeros((len(splits), 2, Bn, Bn)); Np = np.zeros((len(splits), 2, Bn)); Nn = np.zeros((len(splits), 2, Bn))
    gi = {s: ix for s, ix in d.groupby("split").indices.items()}
    for si, s in enumerate(splits):
        g = d.iloc[gi[s]]
        for li, (yc, sc) in enumerate((("y_inner", "s_inner"), ("y_outer", "s_outer"))):
            gg = g[keep[li].iloc[gi[s]].to_numpy()] if keep is not None else g
            y = gg[yc].to_numpy() == 1
            if y.sum() == 0 or (~y).sum() == 0:
                continue
            ci = gg.cluster.map(cidx).to_numpy()
            sp, sn = gg[sc].to_numpy()[y], gg[sc].to_numpy()[~y]
            cmp = (sp[:, None] > sn[None, :]).astype(np.float32) + 0.5 * (sp[:, None] == sn[None, :])
            Pm = np.zeros((y.sum(), Bn), np.float32); Pm[np.arange(y.sum()), ci[y]] = 1
            Nm = np.zeros((int((~y).sum()), Bn), np.float32); Nm[np.arange((~y).sum()), ci[~y]] = 1
            C[si, li] = Pm.T @ cmp @ Nm
            Np[si, li] = Pm.sum(0); Nn[si, li] = Nm.sum(0)
    return dict(C=C, Np=Np, Nn=Nn, splits=splits)


def macro(T, M):
    """M (D,Bn) -> (D,) mean over defined splits of macro AUROC; also returns number of defined splits per draw."""
    D = M.shape[0]
    per = np.full((D, len(T["splits"])), np.nan)
    ok = np.ones((D, len(T["splits"])), bool)
    res = np.zeros((D, len(T["splits"]), 2))
    for l in range(2):
        num = np.einsum("di,sij,dj->ds", M, T["C"][:, l], M, optimize=True)
        dp, dn = M @ T["Np"][:, l].T, M @ T["Nn"][:, l].T
        good = (dp > 0) & (dn > 0)
        res[:, :, l] = np.where(good, num / np.where(good, dp * dn, 1), np.nan)
    per = res.mean(2)
    return np.nanmean(per, axis=1) * 100, np.sum(~np.isnan(per), axis=1)


def draws(clusters, types, rng):
    Bn = len(clusters)
    M = np.zeros((B, Bn))
    groups = {}
    for i, t in enumerate(types):
        groups.setdefault(t, []).append(i)
    for t, idx in groups.items():
        idx = np.array(idx)
        pick = rng.integers(0, len(idx), size=(B, len(idx)))
        for b in range(B):
            np.add.at(M[b], idx[pick[b]], 1)
    return M


def cluster_types(d, clusters):
    t = d.groupby("cluster").apply(lambda g: tuple(sorted({(int(a), int(b)) for a, b in zip(g.y_inner, g.y_outer)})), include_groups=False)
    return [str(t[c]) for c in clusters]


def run(ds, name_filter, level, models, keep_fn=None):
    """Returns dict model -> (point, ci?) and paired differences."""
    frames = {k: (unit_frame(v) if level == "unit" else v) for k, v in models.items()}
    clusters = sorted(set(frames["base"].cluster))
    types = cluster_types(frames["base"], clusters)
    rng = np.random.default_rng(SEED)
    M = draws(clusters, types, rng)
    ones = np.ones((1, len(clusters)))
    tabs, pts, boots, nsplit = {}, {}, {}, {}
    for k, f in frames.items():
        keep = keep_fn(f) if keep_fn else None
        tabs[k] = pair_table(f, clusters, keep)
        p, n = macro(tabs[k], ones)
        pts[k], nsplit[k] = float(p[0]), int(n[0])
        boots[k] = macro(tabs[k], M)[0]
    res = {"points": pts, "n_splits_defined": nsplit, "n_clusters": len(clusters), "diff": {}}
    for a, b in (("rowz_drop", "base"), ("rowz_drop", "b1"), ("rowz_drop", "rf"), ("base", "b1")):
        if a in pts and b in pts:
            dd = boots[a] - boots[b]
            res["diff"][f"{a}-{b}"] = [pts[a] - pts[b], float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allres = {}
    t0 = time.time()
    for ds in ("pu", "uored", "cwru", "xjtu", "rtf"):
        models = load_models(ds)
        san = {}
        for k, v in models.items():
            m, n = window_macro(v)
            san[k] = (round(m, 2), n, KNOWN.get((ds, k)))
            flag = "" if KNOWN.get((ds, k)) is None or abs(m - KNOWN[(ds, k)]) <= 0.05 else "  <-- DEVIATION > 0.05"
            print(f"SANITY {ds} {k}: recomputed {m:.2f} over {n} splits, known {KNOWN.get((ds, k))}{flag}", flush=True)
        dev = [k for k, (m, n, kn) in san.items() if kn is not None and abs(m - kn) > 0.05]
        if dev:
            allres[ds] = {"sanity": san, "STOPPED": f"deviation for {dev}"}
            continue
        r = {"sanity": san}
        if ds in ("pu", "uored", "cwru"):
            def keepA(f):
                return [(f.y_inner == 1) | f.healthy, (f.y_outer == 1) | f.healthy]
            r["A1_healthy_only_window"] = run(ds, "A1", "window", models, keepA)
            if ds == "uored":
                def keepA2(f):
                    return [((f.y_inner == 1) & f.state2) | f.healthy, ((f.y_outer == 1) & f.state2) | f.healthy]
                r["A2_uored_faulty_stage_positives_healthy_negatives"] = run(ds, "A2", "window", models, keepA2)
        r["B_unit_level"] = run(ds, "B", "unit", models)
        allres[ds] = r
        print(ds, "done", round(time.time() - t0), "s", flush=True)
        (OUT / "ab_results.json").write_text(json.dumps(allres, indent=1))
    (OUT / "ab_results.json").write_text(json.dumps(allres, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
