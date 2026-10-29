"""Analysis pre-declared in results/rf/PREDECLARED_RF.txt: paired bearing-level cluster bootstrap (B=2000, seed 0; the same strata and the same
multiplicity draws as the dataset's own analyze script) of Macro-AUROC for base, rowz_drop, B1, RF and RF-U; differences (neural method minus RF);
decision rules R1-R3; for the two run-to-failure sets also the stage-only reference, the stage-matched (late-life-negatives) analysis and
leave-one-negative-bearing-out robustness. AUROC is tie-aware (forest probabilities contain ties) and checked against sklearn.
Usage: python -I analyze_rf.py cwru|uored|xjtu|rtf  ->  results/rf/rf_results_<dataset>.json, per_bearing_auroc_rf_<dataset>.csv"""
import sys, pathlib, glob, json
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import rfcommon as R

B = R.B
EXISTING = ["base", "rowz_drop", "b1"]
RFS = ["rf", "rfu"]
NAMES = EXISTING + RFS


def load_existing(ds, n):
    d = WORK / "results" / {"cwru": "cwru", "uored": "uored", "xjtu": "xjtu", "rtf": "rtf"}[ds]
    if ds == "uored":
        return pd.read_parquet(d / f"pred_{n}.parquet")
    fs = [f for f in sorted(glob.glob(str(d / f"pred_{n}_r*.parquet"))) if "smoke" not in f]
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return x.sort_values(["run", "scenario"] if ds == "cwru" else ["split"], kind="stable").reset_index(drop=True)


CFG = {
    "cwru": dict(unit="unit", key=["split", "unit", "y_inner", "y_outer"], late=False),
    "uored": dict(unit="bearing", key=["split", "bearing", "y_inner", "y_outer"], late=False),
    "xjtu": dict(unit="bearing", key=["split", "bearing", "stage", "fileno", "y_inner", "y_outer"], late=True),
    "rtf": dict(unit="bearing", key=["split", "bearing", "stage", "M", "y_inner", "y_outer"], late=True),
}


def unit_setup(ds, P):
    """returns uid (unit -> index), strata (list of index lists in the order used by the dataset's own analyze script)."""
    if ds == "cwru":
        units = sorted(P["base"]["unit"].unique()); uid = {u: i for i, u in enumerate(units)}
        st = {}
        for u in units:
            st.setdefault("healthy" if u.startswith("H_") else u.split("_")[1], []).append(uid[u])
        return uid, list(st.values())
    if ds == "uored":
        return {b: b for b in range(1, 21)}, [list(range(1, 6)), list(range(6, 11)), list(range(11, 16)), list(range(16, 21))]
    if ds == "xjtu":
        sys.path.insert(0, str(WORK / "scripts" / "xjtu"))
        import xjtu_common as C
        idx = {b: i for i, b in enumerate(C.ALL)}
        return idx, [[idx[b] for b in sorted(C.IRPOS)], [idx[b] for b in sorted(C.ORPOS - C.IRPOS)], [idx[b] for b in sorted(C.CAGE_ONLY)]]
    sys.path.insert(0, str(WORK / "scripts" / "rtf"))
    import rtf_common as C
    orx = [int(b[1:]) for b in sorted(C.ORPOS)]
    return {f"B{i:02d}": i for i in range(1, 18)}, [orx, [i for i in range(1, 18) if i not in orx]]


def make_tabs(df, ucol, uid, late):
    out = {}
    u_all = df[ucol].map(uid).to_numpy()
    stage = df["stage"].to_numpy() if "stage" in df else None
    for s, idx in df.groupby("split", sort=False).indices.items():
        for l in ("inner", "outer"):
            y = df[f"y_{l}"].to_numpy()[idx]; sc = df[f"s_{l}"].to_numpy()[idx]; u = u_all[idx]
            if late:
                keep = (y == 1) | (stage[idx] == "faulty")
                y, sc, u = y[keep], sc[keep], u[keep]
            out[(s, l)] = R.prep_sorted(y, u, sc)
    return out


def per_split(tab, splits, M):
    return np.stack([np.stack([R.wauc(M, *tab[(s, l)][:2], tab[(s, l)][3]) for l in ("inner", "outer")], 0) for s in splits], 0)   # (S,2,D)


def summ(x, pt):
    x = x[~np.isnan(x)]
    return dict(point=float(pt), ci=[float(v) for v in np.percentile(x, [2.5, 97.5])], p_le_0=float((x <= 0).mean()))


def rule(x):
    r = "SUPPORTED" if (x["point"] > 0 and x["ci"][0] > 0) else ("INCONCLUSIVE" if x["point"] >= 0 else "NOT SUPPORTED")
    if x["ci"][1] < 0:
        r += "; RF significantly ahead"
    return r


def main():
    ds = sys.argv[1]
    cfg = CFG[ds]
    P = {n: load_existing(ds, n) for n in EXISTING}
    for n in RFS:
        P[n] = pd.read_parquet(R.RES / f"pred_{ds}_{n}.parquet")
    nspl = P["base"]["split"].nunique(); assert nspl == 100, nspl
    for n in NAMES[1:]:
        assert len(P[n]) == len(P["base"]) and (P["base"][cfg["key"]].to_numpy() == P[n][cfg["key"]].to_numpy()).all(), f"row mismatch {n}"
    print(f"{ds}: all five prediction sets have identical rows ({len(P['base'])}) and keys {cfg['key']}", flush=True)
    if cfg["late"]:   # stage-only pseudo-model: score = window from the faulty life stage (no fault-type information)
        P["stage_only"] = P["base"].copy()
        for l in ("inner", "outer"):
            P["stage_only"][f"s_{l}"] = (P["base"]["stage"] == "faulty").astype(float)
    uid, strata = unit_setup(ds, P)
    nu = max(uid.values()) + 1
    splits = list(dict.fromkeys(P["base"]["split"]))
    rng = np.random.default_rng(0)
    M = np.zeros((B, nu))
    for grp in strata:
        grp = np.array(grp); d = rng.integers(0, len(grp), size=(B, len(grp)))
        for k in range(len(grp)):
            np.add.at(M, (np.arange(B), grp[d[:, k]]), 1)
    ones = np.ones((1, nu))
    res = {"dataset": ds, "n_splits": len(splits)}

    # fast path vs sklearn (weighted, with ties) and point AUROC vs sklearn
    tabs0 = {n: make_tabs(P[n], cfg["unit"], uid, False) for n in P}
    r2 = np.random.default_rng(1); worst = 0.0; nchk = 0
    for _ in range(6):
        m = np.zeros(nu)
        for grp in strata:
            for k in r2.integers(0, len(grp), len(grp)): m[grp[k]] += 1
        s = splits[int(r2.integers(0, len(splits)))]
        for n in ("rf", "rfu", "base"):
            for l in ("inner", "outer"):
                y, u, sc, st = tabs0[n][(s, l)]; w = m[u]
                if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
                    worst = max(worst, abs(roc_auc_score(y, sc, sample_weight=w) - R.wauc(m[None], y, u, st)[0])); nchk += 1
    for n in ("rf", "rfu", "base", "rowz_drop", "b1"):
        s = splits[7]
        for l in ("inner", "outer"):
            y, u, sc, st = tabs0[n][(s, l)]
            worst = max(worst, abs(roc_auc_score(y, sc) - R.wauc(ones, y, u, st)[0])); nchk += 1
    print(f"tie-aware weighted AUROC fast path vs sklearn: {nchk} checks, max abs difference {worst:.2e}", flush=True)
    assert worst < 1e-9
    res["fastpath_check"] = dict(n=nchk, max_abs_diff=worst)

    def run(tag, late):
        names = [n for n in P if not (late and n == "stage_only")]
        tabs = tabs0 if not late else {n: make_tabs(P[n], cfg["unit"], uid, True) for n in names}
        pt, bt = {}, {}
        for n in names:
            pt[n] = per_split(tabs[n], splits, ones)[:, :, 0]
            bt[n] = per_split(tabs[n], splits, M)
        out = {}
        for n in names:
            mac = np.nanmean(bt[n].mean(1), axis=0)
            out[n] = dict(inner=100 * float(np.nanmean(pt[n][:, 0])), outer=100 * float(np.nanmean(pt[n][:, 1])), macro=100 * float(np.nanmean(pt[n].mean(1))),
                          macro_ci=[float(v) for v in np.nanpercentile(100 * mac, [2.5, 97.5])], sd_over_splits=100 * float(np.nanstd(pt[n].mean(1))),
                          n_valid_splits=int((~np.isnan(pt[n].mean(1))).sum()))
        if ds == "cwru" and not late:
            sc_ = P["base"].drop_duplicates("split").set_index("split")["scenario"].reindex(splits).to_numpy()
            for n in names:
                out[n].update(macro_DE=100 * float(pt[n][sc_ == "DE"].mean()), macro_FE=100 * float(pt[n][sc_ == "FE"].mean()),
                              outer_DE=100 * float(pt[n][sc_ == "DE", 1].mean()), outer_FE=100 * float(pt[n][sc_ == "FE", 1].mean()))
        pairs = [("rowz_drop", "rf"), ("base", "rf"), ("b1", "rf"), ("rowz_drop", "rfu"), ("base", "rfu"), ("b1", "rfu"), ("rf", "rfu"),
                 ("rowz_drop", "base"), ("rowz_drop", "b1"), ("base", "b1")]
        if "stage_only" in names:
            pairs += [("rf", "stage_only"), ("rfu", "stage_only"), ("rowz_drop", "stage_only"), ("base", "stage_only"), ("b1", "stage_only")]
        for a, b in pairs:
            for lab, li in (("macro", None), ("inner", 0), ("outer", 1)):
                if li is None:
                    d = 100 * (np.nanmean(bt[a].mean(1), axis=0) - np.nanmean(bt[b].mean(1), axis=0)); p = out[a]["macro"] - out[b]["macro"]
                else:
                    d = 100 * (np.nanmean(bt[a][:, li], axis=0) - np.nanmean(bt[b][:, li], axis=0)); p = out[a][lab] - out[b][lab]
                out[f"{a}_minus_{b}_{lab}"] = summ(d, p)
        res[tag] = out

    run("primary", False)
    if cfg["late"]:
        run("stage_matched", True)
    pr = res["primary"]
    dec = {}
    for n in ("rf", "rfu"):
        dec[f"R1_{n}"] = "met" if pr[n]["macro_ci"][0] > 50 else "NOT met"
    for a in ("rowz_drop", "base", "b1"):
        for b in ("rf", "rfu"):
            x = pr[f"{a}_minus_{b}_macro"]
            dec[f"R2_{a}_minus_{b}"] = rule(x).replace("RF", b.upper()) + ("; +6 met" if (x["point"] >= 6 and a == "rowz_drop") else ("; +6 not met" if a == "rowz_drop" else ""))
    for b in ("rf", "rfu"):
        dec[f"R3_rowz_drop_not_losing_to_{b}"] = "holds" if pr[f"rowz_drop_minus_{b}_macro"]["point"] >= 0 else "FAILS (point estimate below RF)"
    res["decisions"] = dec

    # per-bearing descriptive AUROC (faulty windows of that bearing vs all negatives / late-life negatives of the splits where it is in the test set)
    rows = []
    ucol = cfg["unit"]
    for n in [n for n in P]:
        df = P[n]
        for lab in ("inner", "outer"):
            pos_units = sorted(df.loc[df[f"y_{lab}"] == 1, ucol].unique(), key=lambda v: str(v))
            for b in pos_units:
                for kind in (("all_negatives", "late_life_negatives") if cfg["late"] else ("all_negatives",)):
                    v = []
                    for s, g in df.groupby("split"):
                        gb = g[(g[ucol] == b) & (g[f"y_{lab}"] == 1)]
                        if len(gb) == 0: continue
                        ng = g[g[f"y_{lab}"] == 0]
                        if kind == "late_life_negatives": ng = ng[ng.stage == "faulty"]
                        if len(ng) == 0: continue
                        v.append(roc_auc_score(np.r_[np.ones(len(gb)), np.zeros(len(ng))], np.r_[gb[f"s_{lab}"].to_numpy(), ng[f"s_{lab}"].to_numpy()]))
                    rows.append(dict(model=n, label=lab, bearing=b, negatives=kind, auroc=100 * float(np.mean(v)) if v else np.nan, n_splits=len(v)))
    pb = pd.DataFrame(rows); pb.to_csv(R.RES / f"per_bearing_auroc_rf_{ds}.csv", index=False)

    # leave-one-negative-bearing-out robustness of the stage-matched point estimates (as scripts/audit/stage_matched_robustness.py)
    if cfg["late"]:
        def sm(d, drop=None):
            o = []
            for _, g in d.groupby("split"):
                g = g[g["stage"] == "faulty"]
                if drop is not None: g = g[g["bearing"] != drop]
                vals = []
                for lab in ("inner", "outer"):
                    y = g[f"y_{lab}"].to_numpy()
                    if y.min() == y.max(): vals = None; break
                    vals.append(roc_auc_score(y, g[f"s_{lab}"].to_numpy()))
                if vals: o.append(np.mean(vals))
            return 100 * float(np.mean(o)), len(o)
        ms = EXISTING + RFS
        ref = {m: sm(P[m]) for m in ms}
        negb = sorted(P["rowz_drop"][(P["rowz_drop"]["stage"] == "faulty") & ((P["rowz_drop"]["y_inner"] == 0) | (P["rowz_drop"]["y_outer"] == 0))]["bearing"].unique())
        loo = []
        for b in negb:
            r = {m: sm(P[m], drop=b)[0] for m in ms}
            loo.append(dict(dropped=b, **r))
        lo = pd.DataFrame(loo)
        for m in ("rowz_drop", "base", "b1"):
            lo[f"{m}_minus_rf"] = lo[m] - lo["rf"]
        res["stage_matched_point"] = {m: dict(macro=ref[m][0], n_valid_splits=ref[m][1]) for m in ms}
        res["leave_one_negative_out"] = dict(table=lo.round(2).to_dict(orient="records"),
                                              rowz_drop_minus_rf=[float(lo["rowz_drop_minus_rf"].min()), float(lo["rowz_drop_minus_rf"].max())],
                                              base_minus_rf=[float(lo["base_minus_rf"].min()), float(lo["base_minus_rf"].max())],
                                              b1_minus_rf=[float(lo["b1_minus_rf"].min()), float(lo["b1_minus_rf"].max())])
        print(lo.round(2).to_string(index=False))
    json.dump(res, open(R.RES / f"rf_results_{ds}.json", "w"), indent=1, default=float)

    for tag in [t for t in ("primary", "stage_matched") if t in res]:
        print("==", ds, tag)
        for n in res[tag]:
            if "_minus_" in n: continue
            v = res[tag][n]
            print(f"  {n:10s} macro {v['macro']:.2f} CI [{v['macro_ci'][0]:.2f}, {v['macro_ci'][1]:.2f}] inner {v['inner']:.2f} outer {v['outer']:.2f} sd {v['sd_over_splits']:.2f} valid {v['n_valid_splits']}")
        for k, v in res[tag].items():
            if "_minus_" in k and k.endswith("_macro"): print(f"  {k:34s} {v['point']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
    print("decisions", json.dumps(dec, indent=1))
    print(pb.pivot_table(index=["label", "bearing", "negatives"], columns="model", values="auroc").round(1).to_string())


if __name__ == "__main__":
    main()
