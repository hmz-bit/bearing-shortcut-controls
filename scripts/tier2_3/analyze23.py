"""Metrics and paired bearing-level cluster bootstraps for the fixed-Hz ablation (Tier 1 F), Tier 3 and Tier 2.
Reads results/tier2_3/*_predictions_long.parquet (chunks are concatenated) and the Tier 1 prediction files (read-only).
Writes results/tier2_3/analysis_tier1_F.json, analysis_tier3.json, analysis_tier2.json and csv tables.
Usage: python -I analyze23.py [--B 2000] [--which t1 t3 t2]
"""
import argparse, glob, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common23 as K
import bootstrap as bs
import metrics as mt

T1 = K.WORK / "results" / "tier1"


def cat(pattern):
    fs = sorted(glob.glob(str(K.OUT / pattern)))
    assert fs, pattern
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def per_split(frame):
    rows = []
    for s, g in frame.groupby("split", sort=True):
        m, per = mt.macro_auroc(g[["y_inner", "y_outer"]].to_numpy(), g[["s_inner", "s_outer"]].to_numpy(), True)
        rows.append(dict(split=s, inner=per[0], outer=per[1], macro=m))
    return pd.DataFrame(rows)


def fpr90(frame, label):
    ys = [g[f"y_{label}"].to_numpy() for _, g in frame.groupby("split", sort=True)]
    ss = [g[f"s_{label}"].to_numpy() for _, g in frame.groupby("split", sort=True)]
    return 100 * mt.mean_fpr_at_tpr(ys, ss, 0.9, "horizontal")


def ci(a, alpha=0.05):
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))]


def summ(point, draws):
    lo, hi = ci(draws)
    d = draws[~np.isnan(draws)]
    return dict(point=float(point), ci_low=lo, ci_high=hi, se=float(d.std(ddof=1)), p_le_0=float((d <= 0).mean()))


# ------------------------------------------------------------------------------------------------ Tier 1, F
def tier1(B):
    fr = {"LF-PNCN": pd.read_parquet(T1 / "lfpncn_predictions_long.parquet"),
          "B4": pd.read_parquet(T1 / "b4_tmax_predictions_long.parquet"),
          "F": cat("f_tier1_*_predictions_long.parquet")}
    out = {"metrics": {}, "paired": {}}
    for k, f in fr.items():
        ps = per_split(f)
        m = 100 * ps.macro
        out["metrics"][k] = dict(macro_mean=m.mean(), macro_sd_ddof0=m.std(ddof=0), inner=100 * ps.inner.mean(), outer=100 * ps.outer.mean(),
                                 fpr90_inner=fpr90(f, "inner"), fpr90_outer=fpr90(f, "outer"), n_splits=len(ps))
        print(k, {a: round(b, 2) for a, b in out["metrics"][k].items()})
    for a, b in (("LF-PNCN", "F"), ("LF-PNCN", "B4"), ("F", "B4")):
        r = bs.paired_bootstrap_multi_split(fr[a], fr[b], B=B, seed=0, stratify=True)
        out["paired"][f"{a} - {b}"] = r["diff"]
        d = r["diff"]
        print(f"Tier1 {a} - {b}: {d['point']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]")
    (K.OUT / "analysis_tier1_F.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame([dict(method=k, **v) for k, v in out["metrics"].items()]).to_csv(K.OUT / "tier1_F_metrics_table.csv", index=False)


# ------------------------------------------------------------------------------------------------ Tier 3
def tier3(B):
    meths = {"LF-PNCN": "lfpncn", "B4": "b4", "F": "f", "B3": "b3"}
    fr = {k: cat(f"t3_{v}_*_predictions_long.parquet") for k, v in meths.items()}
    conds = {"same": K.COND_SAME, "slow": K.COND_SLOW}
    splits_used = sorted(fr["LF-PNCN"].split.unique())
    for k, f in fr.items():
        assert sorted(f.split.unique()) == splits_used, k
    tabs, pt = {}, {}
    bearings = sorted(fr["LF-PNCN"].bearing.unique(), key=lambda b: (K.pu.bearing_labels(b), b))
    for k, f in fr.items():
        for c, cn in conds.items():
            g = f[f.test_cond == cn]
            tabs[(k, c)] = bs.build_pair_table(g, bearings)
            pt[(k, c)] = tabs[(k, c)].macro_per_split(np.ones((1, len(bearings))))[0] * 100
    rng = np.random.default_rng(0)
    M = bs.draw_multiplicities(bearings, B, rng, True)
    mac = {key: t.macro_per_split(M) * 100 for key, t in tabs.items()}              # (B, S)
    valid = np.all([~np.isnan(v) for v in mac.values()], axis=0)
    def meanv(a):
        return np.nanmean(np.where(valid, a, np.nan), axis=1)
    loss_d = {k: meanv(mac[(k, "same")] - mac[(k, "slow")]) for k in meths}
    same_d = {k: meanv(mac[(k, "same")]) for k in meths}
    slow_d = {k: meanv(mac[(k, "slow")]) for k in meths}
    out = {"n_splits": len(splits_used), "splits": [int(s) for s in splits_used], "B": B, "methods": {}, "paired": {}}
    for k in meths:
        same, slow = pt[(k, "same")], pt[(k, "slow")]
        loss = same - slow
        out["methods"][k] = dict(
            same_mean=same.mean(), same_sd=same.std(ddof=0), slow_mean=slow.mean(), slow_sd=slow.std(ddof=0),
            loss_mean=loss.mean(), loss_sd=loss.std(ddof=0),
            same_ci=ci(same_d[k]), slow_ci=ci(slow_d[k]), loss_ci=ci(loss_d[k]),
            slow_inner=100 * np.mean([tabs[(k, "slow")].auroc(np.ones((1, len(bearings))))[0, :, 0]]),
            slow_outer=100 * np.mean([tabs[(k, "slow")].auroc(np.ones((1, len(bearings))))[0, :, 1]]),
            same_inner=100 * np.mean([tabs[(k, "same")].auroc(np.ones((1, len(bearings))))[0, :, 0]]),
            same_outer=100 * np.mean([tabs[(k, "same")].auroc(np.ones((1, len(bearings))))[0, :, 1]]))
        print(k, {a: (round(b, 2) if not isinstance(b, list) else [round(x, 2) for x in b]) for a, b in out["methods"][k].items()})
    for cmp in ("B4", "F", "B3"):
        d_loss = summ(out["methods"][cmp]["loss_mean"] - out["methods"]["LF-PNCN"]["loss_mean"], loss_d[cmp] - loss_d["LF-PNCN"])
        d_slow = summ(out["methods"]["LF-PNCN"]["slow_mean"] - out["methods"][cmp]["slow_mean"], slow_d["LF-PNCN"] - slow_d[cmp])
        d_same = summ(out["methods"]["LF-PNCN"]["same_mean"] - out["methods"][cmp]["same_mean"], same_d["LF-PNCN"] - same_d[cmp])
        lf = out["methods"]["LF-PNCN"]["loss_mean"]
        falsified = d_loss["ci_high"] <= 0
        out["paired"][cmp] = dict(loss_diff_cmp_minus_lf=d_loss, slow_auroc_diff_lf_minus_cmp=d_slow, same_auroc_diff_lf_minus_cmp=d_same,
                                  falsified_vs_this=bool(falsified), lf_loss_le_5=bool(lf <= 5.0), lf_loss_smaller=bool(lf < out["methods"][cmp]["loss_mean"]))
        print(f"Tier3 loss({cmp}) - loss(LF): {d_loss['point']:+.2f} [{d_loss['ci_low']:+.2f}, {d_loss['ci_high']:+.2f}] ; "
              f"slow AUROC LF - {cmp}: {d_slow['point']:+.2f} [{d_slow['ci_low']:+.2f}, {d_slow['ci_high']:+.2f}]")
    rows = [dict(method=k, **{a: (str(b) if isinstance(b, list) else b) for a, b in v.items()}) for k, v in out["methods"].items()]
    pd.DataFrame(rows).to_csv(K.OUT / "tier3_table.csv", index=False)
    per = pd.DataFrame({f"{k}_{c}": pt[(k, c)] for k in meths for c in conds}); per.insert(0, "split", splits_used)
    per.to_csv(K.OUT / "tier3_per_split_macro.csv", index=False)
    (K.OUT / "analysis_tier3.json").write_text(json.dumps(out, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ Tier 2
def tier2(B):
    meths = {"LF-PNCN": "lfpncn", "B4": "b4", "F": "f", "B3": "b3"}
    fr = {k: cat(f"t2_{v}_*_predictions_long.parquet") for k, v in meths.items()}
    bearings = sorted(fr["LF-PNCN"].bearing.unique(), key=lambda b: (K.pu.bearing_labels(b), b))
    assert len(bearings) == 17
    one = np.ones((1, len(bearings)))
    tabs, point_pair, rows = {}, {}, []
    for k, f in fr.items():
        tabs[k] = bs.build_pair_table(f, bearings)
        a = tabs[k].auroc(one)[0] * 100                      # (S, 2) per (pair, seed)
        spl = np.array(tabs[k].splits)
        pair = spl // 10
        df = pd.DataFrame(dict(split=spl, pair=pair, inner=a[:, 0], outer=a[:, 1]))
        df["macro"] = df[["inner", "outer"]].mean(axis=1)
        df["seed_idx"] = spl % 10
        df.insert(0, "method", k)
        rows.append(df)
        point_pair[k] = df.groupby("pair")[["inner", "outer", "macro"]].mean()
    per_seed = pd.concat(rows, ignore_index=True)
    per_seed.to_csv(K.OUT / "tier2_per_pair_seed_auroc.csv", index=False)
    pd.concat([v.assign(method=k) for k, v in point_pair.items()]).to_csv(K.OUT / "tier2_per_pair_auroc_seedavg.csv")
    out = {"metrics": {}, "paired": {}, "B": B}
    for k in meths:
        pp = point_pair[k]
        seeds = per_seed[per_seed.method == k]
        out["metrics"][k] = dict(
            macro_mean=pp.macro.mean(), macro_sd_pairs=pp.macro.std(ddof=0), inner=pp.inner.mean(), inner_sd=pp.inner.std(ddof=0),
            outer=pp.outer.mean(), outer_sd=pp.outer.std(ddof=0),
            macro_by_seed={int(s): float(g.macro.mean()) for s, g in seeds.groupby("seed_idx")},
            fpr90_inner=fpr90(fr[k], "inner"), fpr90_outer=fpr90(fr[k], "outer"), min_pair=pp.macro.min(), max_pair=pp.macro.max())
        print(k, {a: (round(b, 2) if not isinstance(b, dict) else b) for a, b in out["metrics"][k].items()})
    # bootstrap: multiplicities over the 17 test bearings (stratified: 6 healthy, 5 outer, 6 inner)
    rng = np.random.default_rng(0)
    M = bs.draw_multiplicities(bearings, B, rng, True)
    healthy_cols = [i for i, b in enumerate(bearings) if K.pu.bearing_class(b) == "healthy"]
    M_faulty = M.copy()
    M_faulty[:, healthy_cols] = 1.0              # resample only the faulty bearings, healthy ones fixed
    def pair_matrix(k, MM):
        a = tabs[k].macro_per_split(MM) * 100    # (B, S)
        spl = np.array(tabs[k].splits)
        cols = []
        for p in range(15):
            idx = np.where(spl // 10 == p)[0]
            cols.append(a[:, idx].mean(axis=1))   # average over seeds (NaN if undefined)
        return np.stack(cols, axis=1)             # (B, 15)
    for tag, MM in (("all17", M), ("faulty_only", M_faulty)):
        pm = {k: pair_matrix(k, MM) for k in meths}
        for cmp in ("B4", "F", "B3"):
            valid = ~np.isnan(pm["LF-PNCN"]) & ~np.isnan(pm[cmp])
            dd = np.nanmean(np.where(valid, pm["LF-PNCN"] - pm[cmp], np.nan), axis=1)
            point = point_pair["LF-PNCN"].macro.mean() - point_pair[cmp].macro.mean()
            # label-wise differences
            res = summ(point, dd)
            res["mean_valid_pairs_per_draw"] = float(valid.sum(axis=1).mean())
            out["paired"].setdefault(tag, {})[cmp] = res
            print(f"Tier2 [{tag}] LF-PNCN - {cmp}: {point:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}] valid pairs/draw {res['mean_valid_pairs_per_draw']:.1f}")
        if tag == "all17":
            for k in meths:
                v = np.nanmean(pm[k], axis=1)
                out["metrics"][k]["macro_ci_all17"] = ci(v)
    # training-class balance and test composition
    z = K.load_all(("t_mad",))
    bearing, Y = z["bearing"], z["Y"]
    bal = []
    for p, held in enumerate(K.PAIRS):
        tr_ids = K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held]
        tr = np.isin(bearing, tr_ids)
        te = np.isin(bearing, K.REAL_FAULTY + list(held))
        cls = np.array([K.pu.bearing_class(b) for b in bearing])
        bal.append(dict(pair=p, held=" ".join(held), train_windows=int(tr.sum()), train_healthy_win=int((tr & (cls == "healthy")).sum()),
                        train_outer_win=int((tr & (cls == "outer")).sum()), train_inner_win=int((tr & (cls == "inner")).sum()),
                        pos_rate_inner=float(Y[tr, 0].mean()), pos_rate_outer=float(Y[tr, 1].mean()),
                        test_windows=int(te.sum()), test_healthy_win=int((te & (cls == "healthy")).sum()),
                        test_outer_win=int((te & (cls == "outer")).sum()), test_inner_win=int((te & (cls == "inner")).sum()),
                        test_pos_rate_inner=float(Y[te, 0].mean()), test_pos_rate_outer=float(Y[te, 1].mean())))
    pd.DataFrame(bal).to_csv(K.OUT / "tier2_class_balance.csv", index=False)
    b = pd.DataFrame(bal)
    out["class_balance"] = {"train_windows": [int(b.train_windows.min()), int(b.train_windows.max())],
                            "train_healthy_win": [int(b.train_healthy_win.min()), int(b.train_healthy_win.max())],
                            "train_outer_win": int(b.train_outer_win.iloc[0]), "train_inner_win": int(b.train_inner_win.iloc[0]),
                            "pos_rate_inner": [float(b.pos_rate_inner.min()), float(b.pos_rate_inner.max())],
                            "pos_rate_outer": [float(b.pos_rate_outer.min()), float(b.pos_rate_outer.max())],
                            "test_healthy_win": [int(b.test_healthy_win.min()), int(b.test_healthy_win.max())],
                            "test_windows": [int(b.test_windows.min()), int(b.test_windows.max())]}
    print("balance", out["class_balance"])
    (K.OUT / "analysis_tier2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--which", nargs="+", default=["t1", "t3", "t2"])
    a = ap.parse_args()
    for w in a.which:
        {"t1": tier1, "t3": tier3, "t2": tier2}[w](a.B)
