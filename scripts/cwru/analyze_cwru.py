"""Analysis pre-declared in results/cwru/PREDECLARED_CWRU.txt: paired cluster bootstrap (B=2000, seed 0) of Macro-AUROC, base vs rowz_drop,
clusters = faulty bearings (group) and healthy configurations ("H_"+group); strata = IR groups, OR groups, B groups, healthy clusters.
Weighted-AUROC fast path verified against sklearn. Descriptive: per-faulty-bearing AUROC and results by scenario.
"""
import glob, json, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

WORK = pathlib.Path(__file__).resolve().parents[2]
OUT = WORK / "results" / "cwru"
NAMES = ["base", "rowz_drop"]
B = 2000


def load(name):
    fs = [f for f in sorted(glob.glob(str(OUT / f"pred_{name}_r*.parquet"))) if "smoke" not in f]
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.sort_values(["run", "scenario"], kind="stable").reset_index(drop=True)


def prep(df, uid):
    out = {}
    for s, g in df.groupby("split", sort=False):
        for l in ("inner", "outer"):
            sc = g[f"s_{l}"].to_numpy(); y = g[f"y_{l}"].to_numpy(); u = g["unit"].map(uid).to_numpy()
            o = np.argsort(sc, kind="stable")
            out[(s, l)] = (y[o].astype(float), u[o], sc[o])
    return out


def weighted_auroc(M, y, u):
    W = M[:, u]
    pos, neg = W * y, W * (1 - y)
    num = (pos * np.cumsum(neg, axis=1)).sum(1)
    dp, dn = pos.sum(1), neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((dp > 0) & (dn > 0), num / (dp * dn), np.nan)


def per_split(tab, splits, M):
    return np.stack([np.stack([weighted_auroc(M, *tab[(s, l)][:2]) for l in ("inner", "outer")], 0) for s in splits], 0)   # (S, 2, B)


def main():
    P = {n: load(n) for n in NAMES}
    a, b = P["base"], P["rowz_drop"]
    assert a["split"].nunique() == 100 and b["split"].nunique() == 100, (a["split"].nunique(), b["split"].nunique())
    cols = ["split", "unit", "y_inner", "y_outer"]
    assert (a[cols].to_numpy() == b[cols].to_numpy()).all()
    units = sorted(a["unit"].unique()); uid = {u: i for i, u in enumerate(units)}
    strata = {}
    for u in units:
        k = "healthy" if u.startswith("H_") else u.split("_")[1]
        strata.setdefault(k, []).append(uid[u])
    print({k: len(v) for k, v in strata.items()})
    splits = list(dict.fromkeys(a["split"]))
    tabs = {n: prep(P[n], uid) for n in NAMES}
    rng = np.random.default_rng(1)
    for _ in range(4):               # fast path vs sklearn
        m = np.zeros(len(units))
        for idx in strata.values():
            for k in rng.integers(0, len(idx), len(idx)): m[idx[k]] += 1
        s = splits[int(rng.integers(0, len(splits)))]
        for l in ("inner", "outer"):
            y, u, sc = tabs["base"][(s, l)]; w = m[u]
            if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                assert abs(roc_auc_score(y, sc, sample_weight=w) - weighted_auroc(m[None], y, u)[0]) < 1e-9
    print("weighted AUROC fast path matches sklearn")
    rng = np.random.default_rng(0)
    M = np.zeros((B, len(units)))
    for idx in strata.values():
        idx = np.array(idx); d = rng.integers(0, len(idx), size=(B, len(idx)))
        for k in range(len(idx)):
            np.add.at(M, (np.arange(B), idx[d[:, k]]), 1)
    ones = np.ones((1, len(units)))
    res, bm, bl = {}, {}, {}
    for n in NAMES:
        pt = per_split(tabs[n], splits, ones)[:, :, 0]
        bl[n] = per_split(tabs[n], splits, M)
        bm[n] = np.nanmean(bl[n].mean(1), axis=0)
        lo, hi = np.percentile(100 * bm[n][~np.isnan(bm[n])], [2.5, 97.5])
        sc = P[n].drop_duplicates("split").set_index("split")["scenario"].reindex(splits).to_numpy()
        res[n] = dict(inner=100 * pt[:, 0].mean(), outer=100 * pt[:, 1].mean(), macro=100 * pt.mean(), macro_ci=[lo, hi], sd_over_splits=100 * pt.mean(1).std(),
                      macro_DE=100 * pt[sc == "DE"].mean(), macro_FE=100 * pt[sc == "FE"].mean(),
                      outer_DE=100 * pt[sc == "DE", 1].mean(), outer_FE=100 * pt[sc == "FE", 1].mean(),
                      min_split_macro=100 * pt.mean(1).min())
    for l, li in (("inner", 0), ("outer", 1)):
        d = 100 * (np.nanmean(bl["rowz_drop"][:, li], axis=0) - np.nanmean(bl["base"][:, li], axis=0))
        res[f"diff_{l}"] = dict(point=res["rowz_drop"][l] - res["base"][l], ci=list(np.percentile(d[~np.isnan(d)], [2.5, 97.5])))
    d = 100 * (bm["rowz_drop"] - bm["base"])
    res["diff_macro"] = dict(point=res["rowz_drop"]["macro"] - res["base"]["macro"], ci=list(np.percentile(d[~np.isnan(d)], [2.5, 97.5])), p_le_0=float((d <= 0).mean()))
    res["n_splits"] = len(splits)
    print(json.dumps(res, indent=1, default=float))
    rows = []
    for n in NAMES:
        df = P[n]
        for l, typ in (("inner", "IR"), ("outer", "OR")):
            for grp in sorted(g for g in df["group"].unique() if g.split("_")[1] == typ):
                v = []
                for s, g in df.groupby("split", sort=False):
                    pos = g[(g["unit"] == grp) & (g[f"y_{l}"] == 1)][f"s_{l}"].to_numpy()
                    if len(pos) == 0: continue
                    neg = g[g[f"y_{l}"] == 0][f"s_{l}"].to_numpy()
                    v.append(np.mean(pos[:, None] > neg[None, :]))
                rows.append(dict(model=n, label=l, group=grp, auroc=100 * np.mean(v), n_splits=len(v)))
    pb = pd.DataFrame(rows); pb.to_csv(OUT / "per_bearing_auroc.csv", index=False)
    print(pb.pivot_table(index=["label", "group"], columns="model", values="auroc").round(1).to_string())
    json.dump(res, open(OUT / "cwru_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
