"""Analysis pre-declared in results/cwru/PREDECLARED_B1_CWRU.txt: paired cluster bootstrap (B=2000, seed 0, the same draws as analyze_cwru.py) of
Macro-AUROC for base, rowz_drop and B1 on CWRU; rowz_drop-B1 and base-B1 differences with the pre-stated decision rule; per-bearing AUROC for B1.
Does not modify any file of the LF-PNCN analysis (writes cwru_results_b1.json, per_bearing_auroc_b1.csv)."""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import analyze_cwru as A

OUT = A.OUT
NAMES = ["base", "rowz_drop", "b1"]


def verdict(pt, ci):
    return "SUPPORTED" if (pt > 0 and ci[0] > 0) else ("INCONCLUSIVE" if pt >= 0 else "NOT SUPPORTED")


def main():
    P = {n: A.load(n) for n in NAMES}
    for n in NAMES:
        assert P[n]["split"].nunique() == 100, (n, P[n]["split"].nunique())
    cols = ["split", "unit", "y_inner", "y_outer"]
    for n in NAMES[1:]:
        assert (P["base"][cols].to_numpy() == P[n][cols].to_numpy()).all(), n
    units = sorted(P["base"]["unit"].unique()); uid = {u: i for i, u in enumerate(units)}
    strata = {}
    for u in units:
        k = "healthy" if u.startswith("H_") else u.split("_")[1]
        strata.setdefault(k, []).append(uid[u])
    print({k: len(v) for k, v in strata.items()})
    splits = list(dict.fromkeys(P["base"]["split"]))
    tabs = {n: A.prep(P[n], uid) for n in NAMES}
    rng = np.random.default_rng(1)
    for _ in range(4):               # fast path vs sklearn, for every model
        m = np.zeros(len(units))
        for idx in strata.values():
            for k in rng.integers(0, len(idx), len(idx)): m[idx[k]] += 1
        s = splits[int(rng.integers(0, len(splits)))]
        for n in NAMES:
            for l in ("inner", "outer"):
                y, u, sc = tabs[n][(s, l)]; w = m[u]
                if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                    assert abs(roc_auc_score(y, sc, sample_weight=w) - A.weighted_auroc(m[None], y, u)[0]) < 1e-9
    print("weighted AUROC fast path matches sklearn (all three models)")
    rng = np.random.default_rng(0)                     # identical draws to analyze_cwru.py
    M = np.zeros((A.B, len(units)))
    for idx in strata.values():
        idx = np.array(idx); d = rng.integers(0, len(idx), size=(A.B, len(idx)))
        for k in range(len(idx)):
            np.add.at(M, (np.arange(A.B), idx[d[:, k]]), 1)
    ones = np.ones((1, len(units)))
    res, bm, bl = {}, {}, {}
    for n in NAMES:
        pt = A.per_split(tabs[n], splits, ones)[:, :, 0]
        bl[n] = A.per_split(tabs[n], splits, M)
        bm[n] = np.nanmean(bl[n].mean(1), axis=0)
        lo, hi = np.percentile(100 * bm[n][~np.isnan(bm[n])], [2.5, 97.5])
        sc = P[n].drop_duplicates("split").set_index("split")["scenario"].reindex(splits).to_numpy()
        res[n] = dict(inner=100 * pt[:, 0].mean(), outer=100 * pt[:, 1].mean(), macro=100 * pt.mean(), macro_ci=[lo, hi], sd_over_splits=100 * pt.mean(1).std(),
                      macro_DE=100 * pt[sc == "DE"].mean(), macro_FE=100 * pt[sc == "FE"].mean(),
                      outer_DE=100 * pt[sc == "DE", 1].mean(), outer_FE=100 * pt[sc == "FE", 1].mean(), min_split_macro=100 * pt.mean(1).min())
    for a, b in (("rowz_drop", "b1"), ("base", "b1"), ("rowz_drop", "base")):
        for lab, li in (("macro", None), ("inner", 0), ("outer", 1)):
            if li is None:
                d = 100 * (bm[a] - bm[b]); pt = res[a]["macro"] - res[b]["macro"]
            else:
                d = 100 * (np.nanmean(bl[a][:, li], axis=0) - np.nanmean(bl[b][:, li], axis=0)); pt = res[a][lab] - res[b][lab]
            d = d[~np.isnan(d)]
            ci = list(np.percentile(d, [2.5, 97.5]))
            res[f"{a}_minus_{b}_{lab}"] = dict(point=pt, ci=ci, p_le_0=float((d <= 0).mean()))
            if lab == "macro" and b == "b1":
                res[f"{a}_minus_{b}_{lab}"]["rule"] = verdict(pt, ci)
                res[f"{a}_minus_{b}_{lab}"]["H2_ge_plus6_met_on_point_estimate"] = bool(pt >= 6)
    res["n_splits"] = len(splits)
    print(json.dumps(res, indent=1, default=float))
    rows = []
    df = P["b1"]
    for l, typ in (("inner", "IR"), ("outer", "OR")):
        for grp in sorted(g for g in df["group"].unique() if g.split("_")[1] == typ):
            v = []
            for s, g in df.groupby("split", sort=False):
                pos = g[(g["unit"] == grp) & (g[f"y_{l}"] == 1)][f"s_{l}"].to_numpy()
                if len(pos) == 0: continue
                neg = g[g[f"y_{l}"] == 0][f"s_{l}"].to_numpy()
                v.append(np.mean(pos[:, None] > neg[None, :]))
            rows.append(dict(model="b1", label=l, group=grp, auroc=100 * np.mean(v), n_splits=len(v)))
    old = pd.read_csv(OUT / "per_bearing_auroc.csv")
    allp = pd.concat([old[old.model != "b1"], pd.DataFrame(rows)])
    allp.to_csv(OUT / "per_bearing_auroc_b1.csv", index=False)
    print(allp.pivot_table(index=["label", "group"], columns="model", values="auroc").round(1).to_string())
    json.dump(res, open(OUT / "cwru_results_b1.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
