"""Merge shards and compute metrics for B1 (own WDCNN-envelope implementation).
usage: python -I analyze_b1.py val|t1shuf|t3|t2 [tag]
Writes results/b1/<tier>[_tag]_predictions_long.parquet, _per_split.csv, _summary.json (and val_vs_published.csv for val).
Macro-AUROC in points (x100). FPR90 = horizontal-average convention (evaltools.metrics.mean_fpr_at_tpr).
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORK, "scripts", "evaltools"))
import metrics as mt

OUT = os.path.join(WORK, "results", "b1")


def load_shards(tag):
    fs = sorted(glob.glob(os.path.join(OUT, "shards", tag, "split_*.parquet")), key=lambda p: int(os.path.basename(p)[6:-8]))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    info = [json.load(open(f[:-8] + ".json")) for f in fs]
    return df, info


def per_split(df, by=("split",)):
    rows = []
    for key, g in df.groupby(list(by), sort=True):
        m, per = mt.macro_auroc(g[["y_inner", "y_outer"]].to_numpy(), g[["s_inner", "s_outer"]].to_numpy(), True)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), inner=100 * per[0], outer=100 * per[1], macro=100 * m, n_windows=len(g)))
    return pd.DataFrame(rows)


def fpr90(df):
    r = {}
    for lab in ("inner", "outer"):
        ys, ss = [], []
        for s, g in df.groupby("split", sort=True):
            ys.append(g[f"y_{lab}"].to_numpy()); ss.append(g[f"s_{lab}"].to_numpy())
        r[lab] = 100 * mt.mean_fpr_at_tpr(ys, ss, 0.9, "horizontal")
    return r


def ms(x):
    return f"{np.mean(x):.2f} +- {np.std(x, ddof=0):.2f}"


def main():
    kind = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    name = kind + (("_" + tag) if tag else "")
    shard_tag = {"val": "t1val"}.get(kind, kind) + (("_" + tag) if tag else "")
    df, info = load_shards(shard_tag)
    df.to_parquet(os.path.join(OUT, f"{name}_predictions_long.parquet"), index=False)
    summ = dict(n_splits=int(df.split.nunique()),
                total_train_seconds=float(sum(i["train_seconds"] for i in info)),
                mean_seconds_per_split=float(np.mean([i["total_seconds"] for i in info])),
                overlap_total=int(sum(i["overlap"] for i in info)))
    if kind == "val":
        pub = pd.read_parquet(os.path.join(WORK, "scripts", "evaltools", "results", "b1_predictions_long.parquet"))
        pp = per_split(pub[pub.split.isin(df.split.unique())])
        mine = per_split(df)
        t = pp.merge(mine, on="split", suffixes=("_pub", "_own"))
        t["diff_macro"] = t.macro_own - t.macro_pub
        t["abs_diff"] = t.diff_macro.abs()
        t.to_csv(os.path.join(OUT, f"{name}_vs_published.csv"), index=False)
        print(t.round(2).to_string(index=False))
        rk = pd.Series(t.macro_own).rank().values, pd.Series(t.macro_pub).rank().values
        summ.update(mean_abs_diff=float(t.abs_diff.mean()), mean_own=float(t.macro_own.mean()), mean_pub=float(t.macro_pub.mean()),
                    mean_signed_diff=float(t.diff_macro.mean()),
                    spearman=float(np.corrcoef(*rk)[0, 1]), pearson=float(np.corrcoef(t.macro_own, t.macro_pub)[0, 1]))
    elif kind == "t1shuf":
        ps = per_split(df)
        ps.to_csv(os.path.join(OUT, f"{name}_per_split.csv"), index=False)
        print(ps.round(2).to_string(index=False))
        summ.update(macro_mean=float(ps.macro.mean()), inner=float(ps.inner.mean()), outer=float(ps.outer.mean()))
    elif kind == "t3":
        res = {}
        for c, g in df.groupby("test_cond"):
            if c not in ("N15_M07_F10", "N09_M07_F10"):
                continue
            ps = per_split(g)
            ps.to_csv(os.path.join(OUT, f"{name}_per_split_{c}.csv"), index=False)
            res[c] = ps.set_index("split")
        a, b = res["N15_M07_F10"], res["N09_M07_F10"]
        loss = (a.macro - b.macro)
        pd.DataFrame(dict(same=a.macro, rpm900=b.macro, loss=loss, same_inner=a.inner, same_outer=a.outer,
                          r900_inner=b.inner, r900_outer=b.outer)).to_csv(os.path.join(OUT, f"{name}_per_split.csv"))
        summ.update(same_speed=dict(macro_mean=float(a.macro.mean()), macro_sd=float(a.macro.std(ddof=0)),
                                    inner=float(a.inner.mean()), outer=float(a.outer.mean()),
                                    fpr90=fpr90(df[df.test_cond == "N15_M07_F10"])),
                    rpm900=dict(macro_mean=float(b.macro.mean()), macro_sd=float(b.macro.std(ddof=0)),
                                inner=float(b.inner.mean()), outer=float(b.outer.mean()),
                                fpr90=fpr90(df[df.test_cond == "N09_M07_F10"])),
                    loss=dict(mean=float(loss.mean()), sd=float(loss.std(ddof=0))))
    elif kind == "t2":
        ps = per_split(df)
        ps.to_csv(os.path.join(OUT, f"{name}_per_split.csv"), index=False)
        print(ps.round(2).to_string(index=False))
        summ.update(macro_mean=float(ps.macro.mean()), macro_sd=float(ps.macro.std(ddof=0)),
                    inner=float(ps.inner.mean()), outer=float(ps.outer.mean()), inner_sd=float(ps.inner.std(ddof=0)),
                    outer_sd=float(ps.outer.std(ddof=0)), fpr90=fpr90(df))
        # per-bearing mean score diagnostics (pooled over the 15 splits)
        pb = df.groupby("bearing")[["s_inner", "s_outer"]].mean()
        pb.to_csv(os.path.join(OUT, f"{name}_bearing_mean_logits.csv"))
    json.dump(summ, open(os.path.join(OUT, f"{name}_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
