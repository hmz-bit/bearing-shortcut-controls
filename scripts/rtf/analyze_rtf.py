"""Pre-declared analysis for the Paderborn run-to-failure experiments (results/rtf/PREDECLARED_RTF.txt, items 9-12 and decision rules).
Reads pred_base_r*.parquet, pred_rowz_drop_r*.parquet, pred_b1_r*.parquet (+ per_split_*.csv, per_split_perm_*.csv); writes rtf_results.json, per_bearing_auroc.csv."""
import sys, pathlib, glob, json
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rtf_common as C

NAMES = ["base", "rowz_drop", "b1"]
B = 2000
NB = 18                                   # bearing index 1..17
ORIDX = [int(b[1:]) for b in sorted(C.ORPOS)]
OTHERIDX = [i for i in range(1, 18) if i not in ORIDX]


def load(name):
    fs = sorted(f for f in glob.glob(str(C.RES / f"pred_{name}_r*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.sort_values(["split"], kind="stable").reset_index(drop=True)


def prep(df, lab, late_neg_only=False):
    """per split: (y, bearing idx, score) sorted by score for label lab; optionally only late-life negatives (control)."""
    out = {}
    for s, g in df.groupby("split", sort=True):
        y = g[f"y_{lab}"].to_numpy()
        keep = np.ones(len(g), bool) if not late_neg_only else ((y == 1) | (g["stage"].to_numpy() == "faulty"))
        sc = g[f"s_{lab}"].to_numpy()[keep]; yy = y[keep].astype(float); bi = g["bearing"].str[1:].astype(int).to_numpy()[keep]
        o = np.argsort(sc, kind="stable")
        out[s] = (yy[o], bi[o], sc[o])
    return out


def wauc(Mm, y, b):
    W = Mm[:, b]
    pos, neg = W * y, W * (1 - y)
    num = (pos * np.cumsum(neg, axis=1)).sum(1)
    dp, dn = pos.sum(1), neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((dp > 0) & (dn > 0), num / (dp * dn), np.nan)


def per_split(tabs, splits, Mm):
    """(S, 2, D) AUROC per split and label for multiplicities Mm (D, NB)."""
    return np.stack([np.stack([wauc(Mm, *tabs[l][s][:2]) for l in ("inner", "outer")], 0) for s in splits], 0)


def summarise(x, point, a=2.5):
    x = x[~np.isnan(x)]
    lo, hi = np.percentile(x, [a, 100 - a])
    return dict(point=float(point), ci=[float(lo), float(hi)], p_le_0=float((x <= 0).mean()))


def main():
    P = {n: load(n) for n in NAMES}
    key = ["split", "bearing", "stage", "M", "y_inner", "y_outer"]
    for n in NAMES[1:]:
        assert len(P[n]) == len(P["base"]) and (P["base"][key].to_numpy() == P[n][key].to_numpy()).all(), n
    splits = sorted(P["base"]["split"].unique()); assert len(splits) == 100
    rng = np.random.default_rng(0)
    Mm = np.zeros((B, NB))
    for grp in (ORIDX, OTHERIDX):
        d = rng.integers(0, len(grp), size=(B, len(grp)))
        for k in range(len(grp)):
            np.add.at(Mm, (np.arange(B), np.array(grp)[d[:, k]]), 1)
    ones = np.ones((1, NB))
    # fast-path check against sklearn (primary tables, random multiplicities)
    tabs0 = {n: {l: prep(P[n], l) for l in ("inner", "outer")} for n in NAMES}
    r2 = np.random.default_rng(1)
    for _ in range(4):
        m = np.zeros(NB)
        for grp in (ORIDX, OTHERIDX):
            for k in r2.integers(0, len(grp), len(grp)): m[grp[k]] += 1
        s = splits[int(r2.integers(0, len(splits)))]
        for l in ("inner", "outer"):
            y, bi, sc = tabs0["base"][l][s]; w = m[bi]
            if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                assert abs(roc_auc_score(y, sc, sample_weight=w) - wauc(m[None, :], y, bi)[0]) < 1e-9
    print("weighted AUROC fast path matches sklearn", flush=True)

    res = {}
    def run(tag, late):
        tabs = {n: {l: prep(P[n], l, late) for l in ("inner", "outer")} for n in NAMES}
        pt, bt = {}, {}
        for n in NAMES:
            pt[n] = per_split(tabs[n], splits, ones)[:, :, 0]          # (S,2) point
            bt[n] = per_split(tabs[n], splits, Mm)                     # (S,2,B)
        out = {}
        for n in NAMES:
            mac = np.nanmean(bt[n].mean(1), axis=0)
            out[n] = dict(inner=100 * float(np.nanmean(pt[n][:, 0])), outer=100 * float(np.nanmean(pt[n][:, 1])), macro=100 * float(np.nanmean(pt[n].mean(1))),
                          macro_ci=[float(x) for x in np.nanpercentile(100 * mac, [2.5, 97.5])], sd_over_splits=100 * float(np.nanstd(pt[n].mean(1))),
                          n_valid_splits=int((~np.isnan(pt[n].mean(1))).sum()))
        for a, b in (("rowz_drop", "base"), ("rowz_drop", "b1"), ("base", "b1")):
            for lab, li in (("macro", None), ("inner", 0), ("outer", 1)):
                if li is None:
                    d = 100 * (np.nanmean(bt[a].mean(1), axis=0) - np.nanmean(bt[b].mean(1), axis=0)); p = out[a]["macro"] - out[b]["macro"]
                else:
                    d = 100 * (np.nanmean(bt[a][:, li], axis=0) - np.nanmean(bt[b][:, li], axis=0)); p = out[a][lab] - out[b][lab]
                out[f"{a}_minus_{b}_{lab}"] = summarise(d, p)
        res[tag] = out
    run("primary", False)
    run("late_life_negatives", True)

    # decisions
    pr = res["primary"]
    dec = {n: ("U1 met" if pr[n]["macro_ci"][0] > 50 else "U1 NOT met") for n in NAMES}
    def rule(x):
        return "supported/confirmed" if x["point"] > 0 and x["ci"][0] > 0 else ("inconclusive" if x["point"] >= 0 else "not supported/not reproduced")
    dec["U2_rowz_drop_minus_base"] = rule(pr["rowz_drop_minus_base_macro"])
    dec["H2_rowz_drop_vs_B1"] = rule(pr["rowz_drop_minus_b1_macro"]) + ("; +6 met" if pr["rowz_drop_minus_b1_macro"]["point"] >= 6 else "; +6 not met")
    dec["H2_base_vs_B1"] = rule(pr["base_minus_b1_macro"]) + ("; +6 met" if pr["base_minus_b1_macro"]["point"] >= 6 else "; +6 not met")
    res["decisions"] = dec

    # train-vs-held gap and permutation controls
    gap, perm = {}, {}
    for n in NAMES:
        d = pd.concat([pd.read_csv(f) for f in glob.glob(str(C.RES / f"per_split_{n}_r*.csv"))])
        gap[n] = dict(train_macro=100 * float(d.train_macro.mean()), held_macro=100 * float(d.macro.mean()), gap=100 * float((d.train_macro - d.macro).mean()), n=len(d))
        for kind in ("perm", "permw"):          # perm = declared (faulty-end pair only), permw = addendum 1 (whole bearing)
            files = glob.glob(str(C.RES / f"per_split_{kind}_{n}_r*.csv"))
            if files:
                q = pd.concat([pd.read_csv(f) for f in files]); q = q[q.skipped == 0]
                perm[f"{kind}_{n}"] = dict(held_macro_mean=100 * float(q.macro.mean()), sd=100 * float(q.macro.std()), n=len(q), train_macro_mean=100 * float(q.train_macro.mean()),
                                           held_inner=100 * float(q.auroc_inner.mean()), held_outer=100 * float(q.auroc_outer.mean()))
    res["train_gap"], res["permutation_control"] = gap, perm

    # per-bearing descriptive AUROC (faulty windows of that bearing vs all negatives / vs late-life negatives of the same split)
    rows = []
    for n in NAMES:
        df = P[n]
        for lab, pos in (("inner", C.IRPOS), ("outer", C.ORPOS)):
            for b in sorted(pos):
                for kind in ("all_negatives", "late_life_negatives"):
                    v = []
                    for s, g in df.groupby("split"):
                        gb = g[(g.bearing == b) & (g[f"y_{lab}"] == 1)]
                        if len(gb) == 0: continue
                        ng = g[g[f"y_{lab}"] == 0]
                        if kind == "late_life_negatives": ng = ng[ng.stage == "faulty"]
                        if len(ng) == 0: continue
                        v.append(np.mean(gb[f"s_{lab}"].to_numpy()[:, None] > ng[f"s_{lab}"].to_numpy()[None, :]))
                    rows.append(dict(model=n, label=lab, bearing=b, negatives=kind, auroc=100 * float(np.mean(v)) if v else np.nan, n_splits=len(v)))
    pb = pd.DataFrame(rows); pb.to_csv(C.RES / "per_bearing_auroc.csv", index=False)
    json.dump(res, open(C.RES / "rtf_results.json", "w"), indent=1, default=float)
    for tag in ("primary", "late_life_negatives"):
        print("==", tag)
        for n in NAMES:
            print(f"  {n:10s} macro {res[tag][n]['macro']:.2f} CI {[round(x,2) for x in res[tag][n]['macro_ci']]} inner {res[tag][n]['inner']:.2f} outer {res[tag][n]['outer']:.2f} sd {res[tag][n]['sd_over_splits']:.2f} valid {res[tag][n]['n_valid_splits']}")
        for k, v in res[tag].items():
            if "_minus_" in k: print(f"  {k:32s} {v['point']:+.2f} {[round(x,2) for x in v['ci']]}")
    print("decisions", json.dumps(dec, indent=1)); print("train gap", json.dumps(gap, indent=1)); print("perm", json.dumps(perm, indent=1))
    print(pb.pivot_table(index=["label", "bearing", "negatives"], columns="model", values="auroc").round(1).to_string())


if __name__ == "__main__":
    main()
