"""Analysis pre-declared in results/uored/PREDECLARED_UORED.txt: paired, stratified bearing-level cluster bootstrap (B=2000) of Macro-AUROC
(base vs rowz_drop) on UORED, with a weighted-AUROC fast path verified against sklearn. Also descriptive per-bearing outer/inner AUROC.
"""
import sys, pathlib, json
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = WORK / "results" / "uored"
NAMES = ["base", "rowz_drop"]
GROUPS = [list(range(1, 6)), list(range(6, 11)), list(range(11, 16)), list(range(16, 21))]   # inner, outer, ball, cage bearings
B = 2000


def prep(df):
    """per (split, label): sorted y, bearing, plus unweighted AUROC."""
    out = {}
    for s, g in df.groupby("split", sort=True):
        for l in ("inner", "outer"):
            sc = g[f"s_{l}"].to_numpy(); y = g[f"y_{l}"].to_numpy(); b = g["bearing"].to_numpy()
            o = np.argsort(sc, kind="stable")
            out[(s, l)] = (y[o].astype(float), b[o], sc[o])
    return out


def weighted_auroc(M, y, b):
    """M (B, 21) bearing multiplicities; y, b sorted by score. Returns (B,) AUROC (NaN if undefined). Ties ignored (continuous logits)."""
    W = M[:, b]
    pos, neg = W * y, W * (1 - y)
    cum_neg = np.cumsum(neg, axis=1)
    num = (pos * cum_neg).sum(1)
    dp, dn = pos.sum(1), neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((dp > 0) & (dn > 0), num / (dp * dn), np.nan)


def macro_per_split(tab, splits, M):
    ms = []
    for s in splits:
        a = [weighted_auroc(M, *tab[(s, l)][:2]) for l in ("inner", "outer")]
        ms.append(np.stack(a, 0))          # (2, B)
    return np.stack(ms, 0)                 # (S, 2, B)


def main():
    P = {n: pd.read_parquet(OUT / f"pred_{n}.parquet") for n in NAMES}
    a, bb = P["base"], P["rowz_drop"]
    assert (a[["split", "bearing", "y_inner", "y_outer"]].to_numpy() == bb[["split", "bearing", "y_inner", "y_outer"]].to_numpy()).all()
    tabs = {n: prep(P[n]) for n in NAMES}
    splits = sorted(a["split"].unique())
    # verification of the weighted fast path against sklearn on three random multiplicity vectors
    rng = np.random.default_rng(1)
    for _ in range(3):
        m = np.zeros(21)
        for grp in GROUPS:
            for k in rng.integers(0, 5, 5): m[grp[k]] += 1
        s = splits[int(rng.integers(0, len(splits)))]
        for l in ("inner", "outer"):
            y, bi, sc = tabs["base"][(s, l)]
            w = m[bi]
            if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                ref = roc_auc_score(y, sc, sample_weight=w)
                got = weighted_auroc(m[None, :], y, bi)[0]
                assert abs(ref - got) < 1e-9, (ref, got)
    print("weighted AUROC fast path matches sklearn")
    # draws
    rng = np.random.default_rng(0)
    M = np.zeros((B, 21))
    for grp in GROUPS:
        d = rng.integers(0, len(grp), size=(B, len(grp)))
        for k in range(len(grp)):
            np.add.at(M, (np.arange(B), np.array(grp)[d[:, k]]), 1)
    ones = np.ones((1, 21))
    res, boot_macro, boot_lab = {}, {}, {}
    for n in NAMES:
        point = macro_per_split(tabs[n], splits, ones)[:, :, 0]      # (S, 2)  unweighted == plain per-split AUROC
        bm = macro_per_split(tabs[n], splits, M)                     # (S, 2, B)
        boot_lab[n] = bm
        boot_macro[n] = np.nanmean(bm.mean(1), axis=0)               # mean over splits of macro, (B,)
        res[n] = dict(inner=100 * np.nanmean(point[:, 0]), outer=100 * np.nanmean(point[:, 1]), macro=100 * np.nanmean(point.mean(1)))
        lo, hi = np.percentile(100 * boot_macro[n][~np.isnan(boot_macro[n])], [2.5, 97.5])
        res[n].update(macro_ci=[lo, hi], sd_over_splits=100 * np.std(point.mean(1)))
    for l, li in (("inner", 0), ("outer", 1)):
        d = 100 * (np.nanmean(boot_lab["rowz_drop"][:, li], axis=0) - np.nanmean(boot_lab["base"][:, li], axis=0))
        res[f"diff_{l}"] = dict(point=res["rowz_drop"][l] - res["base"][l], ci=list(np.percentile(d[~np.isnan(d)], [2.5, 97.5])))
    d = 100 * (boot_macro["rowz_drop"] - boot_macro["base"])
    res["diff_macro"] = dict(point=res["rowz_drop"]["macro"] - res["base"]["macro"], ci=list(np.percentile(d[~np.isnan(d)], [2.5, 97.5])), p_le_0=float((d <= 0).mean()))
    print(json.dumps(res, indent=1, default=float))
    # descriptive per-bearing AUROC (own windows vs all negatives of the same split), averaged over the splits where the bearing is in test
    rows = []
    for n in NAMES:
        df = P[n]
        for l in ("inner", "outer"):
            grp = range(1, 6) if l == "inner" else range(6, 11)
            for bearing in grp:
                for st in ("all", 1, 2):
                    v = []
                    for s, g in df.groupby("split"):
                        gb = g[(g.bearing == bearing) & (g[f"y_{l}"] == 1)]      # faulty recordings of this bearing only
                        if st != "all":
                            gb = gb[gb.state == st]
                        if len(gb) == 0:
                            continue
                        neg = g[g[f"y_{l}"] == 0][f"s_{l}"].to_numpy()
                        pos = gb[f"s_{l}"].to_numpy()
                        v.append(np.mean(pos[:, None] > neg[None, :]))
                    rows.append(dict(model=n, label=l, bearing=bearing, state=st, auroc=100 * np.mean(v) if v else np.nan, n_splits=len(v)))
    pb = pd.DataFrame(rows)
    pb.to_csv(OUT / "per_bearing_auroc.csv", index=False)
    pv = pb.pivot_table(index=["label", "bearing", "state"], columns="model", values="auroc").round(1)
    print(pv.to_string())
    json.dump(res, open(OUT / "uored_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
