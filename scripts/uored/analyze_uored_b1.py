"""Pre-declared comparison (results/uored/PREDECLARED_B1_UORED.txt): paired stratified bearing-level cluster bootstrap (B=2000, same draws as
analyze_uored.py) of Macro-AUROC for base, rowz_drop and B1 on UORED; rowz_drop-B1 and base-B1 differences; per-bearing descriptive AUROC for B1."""
import sys, pathlib, json
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import analyze_uored as A

OUT = A.OUT
NAMES = ["base", "rowz_drop", "b1"]


def main():
    P = {n: pd.read_parquet(OUT / f"pred_{n}.parquet") for n in NAMES}
    key = ["split", "bearing", "y_inner", "y_outer"]
    for n in NAMES[1:]:
        assert (P["base"][key].to_numpy() == P[n][key].to_numpy()).all(), n
    tabs = {n: A.prep(P[n]) for n in NAMES}
    splits = sorted(P["base"]["split"].unique())
    rng = np.random.default_rng(0)
    M = np.zeros((A.B, 21))
    for grp in A.GROUPS:
        d = rng.integers(0, len(grp), size=(A.B, len(grp)))
        for k in range(len(grp)):
            np.add.at(M, (np.arange(A.B), np.array(grp)[d[:, k]]), 1)
    ones = np.ones((1, 21))
    res, bl, bmac = {}, {}, {}
    for n in NAMES:
        point = A.macro_per_split(tabs[n], splits, ones)[:, :, 0]
        bl[n] = A.macro_per_split(tabs[n], splits, M)
        bmac[n] = np.nanmean(bl[n].mean(1), axis=0)
        lo, hi = np.percentile(100 * bmac[n][~np.isnan(bmac[n])], [2.5, 97.5])
        res[n] = dict(inner=100 * np.nanmean(point[:, 0]), outer=100 * np.nanmean(point[:, 1]), macro=100 * np.nanmean(point.mean(1)), macro_ci=[lo, hi],
                      sd_over_splits=100 * np.std(point.mean(1)))
    for a, b in (("rowz_drop", "b1"), ("base", "b1"), ("rowz_drop", "base")):
        for lab, li in (("macro", None), ("inner", 0), ("outer", 1)):
            if li is None:
                d = 100 * (bmac[a] - bmac[b]); pt = res[a]["macro"] - res[b]["macro"]
            else:
                d = 100 * (np.nanmean(bl[a][:, li], axis=0) - np.nanmean(bl[b][:, li], axis=0)); pt = res[a][lab] - res[b][lab]
            d = d[~np.isnan(d)]
            res[f"{a}_minus_{b}_{lab}"] = dict(point=pt, ci=list(np.percentile(d, [2.5, 97.5])), p_le_0=float((d <= 0).mean()))
    print(json.dumps(res, indent=1, default=float))
    rows = []
    df = P["b1"]
    for l in ("inner", "outer"):
        for bearing in (range(1, 6) if l == "inner" else range(6, 11)):
            for st in ("all", 1, 2):
                v = []
                for s, g in df.groupby("split"):
                    gb = g[(g.bearing == bearing) & (g[f"y_{l}"] == 1)]
                    if st != "all":
                        gb = gb[gb.state == st]
                    if len(gb) == 0:
                        continue
                    neg = g[g[f"y_{l}"] == 0][f"s_{l}"].to_numpy()
                    v.append(np.mean(gb[f"s_{l}"].to_numpy()[:, None] > neg[None, :]))
                rows.append(dict(model="b1", label=l, bearing=bearing, state=st, auroc=100 * np.mean(v) if v else np.nan))
    pb = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "per_bearing_auroc.csv")
    allp = pd.concat([old[old.model != "b1"], pb])
    allp.to_csv(OUT / "per_bearing_auroc.csv", index=False)
    print(allp[allp.state == "all"].pivot_table(index=["label", "bearing"], columns="model", values="auroc").round(1).to_string())
    json.dump(res, open(OUT / "uored_results_b1.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
