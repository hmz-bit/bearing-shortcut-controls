"""Phase C analysis: metrics, FPR90 and paired bearing-level cluster bootstrap (B=2000, stratified, 95%) of declared variants vs LF-PNCN and vs B1.
Usage: python -I analyze_phase2.py --variants rowz rowz_drop [--tier3] [--tier2] [--B 2000]
Reads results/phase2/eval/*. Writes results/phase2/eval_metrics.json and eval_tables.csv. Per-label differences use the same bootstrap with the
label duplicated into both slots (macro of a duplicated label = that label's AUROC)."""
import argparse, glob, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p2common as P
import bootstrap as bs
import metrics as mt

W = P.WORK
EV = P.OUT / "eval"
COLS = ["split", "bearing", "y_inner", "y_outer", "s_inner", "s_outer"]


def cat(pat):
    fs = sorted(glob.glob(str(pat)))
    assert fs, pat
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def only(frame, label):
    f = frame[COLS].copy()
    other = "inner" if label == "outer" else "outer"
    f[f"y_{other}"] = f[f"y_{label}"]; f[f"s_{other}"] = f[f"s_{label}"]
    return f


def per_split(frame):
    rows = []
    for s, g in frame.groupby("split", sort=True):
        m, per = mt.macro_auroc(g[["y_inner", "y_outer"]].to_numpy(), g[["s_inner", "s_outer"]].to_numpy(), True)
        rows.append(dict(split=s, inner=per[0], outer=per[1], macro=m))
    return pd.DataFrame(rows)


def fpr90(frame, label):
    ys, ss = [], []
    for s, g in frame.groupby("split", sort=True):
        ys.append(g[f"y_{label}"].to_numpy()); ss.append(g[f"s_{label}"].to_numpy())
    return 100 * mt.mean_fpr_at_tpr(ys, ss, 0.9, "horizontal")


def metrics_row(name, f):
    ps = per_split(f)
    return dict(method=name, macro=100 * ps.macro.mean(), macro_sd=100 * ps.macro.std(ddof=0), inner=100 * ps.inner.mean(), outer=100 * ps.outer.mean(),
                fpr90_inner=fpr90(f, "inner"), fpr90_outer=fpr90(f, "outer"), n_splits=len(ps))


def paired(fa, fb, B):
    out = {}
    for lab, (a, b) in {"macro": (fa[COLS], fb[COLS]), "inner": (only(fa, "inner"), only(fb, "inner")), "outer": (only(fa, "outer"), only(fb, "outer"))}.items():
        d = bs.paired_bootstrap_multi_split(a, b, B=B, seed=0, stratify=True)["diff"]
        out[lab] = {k: d[k] for k in ("point", "ci_low", "ci_high")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--tier3", action="store_true")
    ap.add_argument("--tier2", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    res, table = {}, []
    # ---------------- Tier 1
    base = pd.read_parquet(W / "results" / "tier1" / "lfpncn_predictions_long.parquet")
    b1 = pd.read_parquet(W / "scripts" / "evaltools" / "results" / "b1_predictions_long.parquet")
    frames = {"LF-PNCN": base, "B1": b1}
    for v in a.variants:
        fs = [f for f in glob.glob(str(EV / f"{v}_r*_predictions_long.parquet")) if "_so" not in pathlib.Path(f).name]
        if fs:
            frames[v] = pd.concat([pd.read_parquet(f) for f in sorted(fs)], ignore_index=True)
    res["tier1"] = {"metrics": {}, "paired": {}}
    for k, f in frames.items():
        m = metrics_row(k, f); res["tier1"]["metrics"][k] = m; table.append(dict(tier="1", **m))
        print(f"T1 {k:14s} macro {m['macro']:.2f} +- {m['macro_sd']:.2f} | inner {m['inner']:.2f} outer {m['outer']:.2f} | FPR90 in {m['fpr90_inner']:.1f} out {m['fpr90_outer']:.1f} | splits {m['n_splits']}", flush=True)
    for v in a.variants:
        if v not in frames:
            continue
        for ref in ("LF-PNCN", "B1"):
            p = paired(frames[v], frames[ref], a.B); res["tier1"]["paired"][f"{v} - {ref}"] = p
            print(f"T1 {v} - {ref}: " + " | ".join(f"{l} {p[l]['point']:+.2f} [{p[l]['ci_low']:+.2f}, {p[l]['ci_high']:+.2f}]" for l in ("macro", "inner", "outer")), flush=True)
    # ---------------- Tier 3
    if a.tier3:
        base3 = cat(W / "results" / "tier2_3" / "t3_lfpncn_*_predictions_long.parquet")
        b13 = pd.read_parquet(W / "results" / "b1" / "t3_predictions_long.parquet")
        f3 = {"LF-PNCN": base3, "B1": b13}
        for v in a.variants:
            fs = sorted(glob.glob(str(EV / f"t3_{v}_[0-9]*_predictions_long.parquet")))
            if fs:
                f3[v] = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        res["tier3"] = {"metrics": {}, "paired": {}}
        for k, f in f3.items():
            for tc, tag in (("N15_M07_F10", "1500"), ("N09_M07_F10", "900")):
                m = metrics_row(k, f[f.test_cond == tc]); res["tier3"]["metrics"][f"{k}@{tag}"] = m; table.append(dict(tier=f"3@{tag}", **m))
                print(f"T3@{tag} {k:14s} macro {m['macro']:.2f} | inner {m['inner']:.2f} outer {m['outer']:.2f} | FPR90 out {m['fpr90_outer']:.1f}", flush=True)
        for v in a.variants:
            if v not in f3:
                continue
            for ref in ("LF-PNCN", "B1"):
                for tc, tag in (("N15_M07_F10", "1500"), ("N09_M07_F10", "900")):
                    p = paired(f3[v][f3[v].test_cond == tc], f3[ref][f3[ref].test_cond == tc], a.B); res["tier3"]["paired"][f"{v} - {ref} @{tag}"] = p
                    print(f"T3@{tag} {v} - {ref}: " + " | ".join(f"{l} {p[l]['point']:+.2f} [{p[l]['ci_low']:+.2f}, {p[l]['ci_high']:+.2f}]" for l in ("macro", "inner", "outer")), flush=True)
    # ---------------- Tier 2
    if a.tier2:
        base2 = cat(W / "results" / "tier2_3" / "t2_lfpncn_*_predictions_long.parquet")
        b12 = pd.read_parquet(W / "results" / "b1" / "t2_predictions_long.parquet")
        res["tier2"] = {"metrics": {}, "paired": {}}

        def seedsub(df, rule):
            sub = []
            for p, g in df.groupby("pair"):
                seeds = sorted(g.seed.unique()); s = seeds[0] if rule == "first" else seeds[-1]
                sub.append(g[g.seed == s].assign(split=p + 1))
            return pd.concat(sub)[COLS]
        for v in ["LF-PNCN"] + a.variants:
            if v == "LF-PNCN":
                df = base2
            else:
                fs = sorted(glob.glob(str(EV / f"t2_{v}_[0-9]*_predictions_long.parquet")))
                if not fs:
                    continue
                df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            for rule in ("first", "second"):
                fa = seedsub(df, rule)
                m = metrics_row(v, fa); res["tier2"]["metrics"][f"{v}/{rule}"] = m; table.append(dict(tier=f"2/{rule}", **m))
                print(f"T2/{rule} {v:14s} macro {m['macro']:.2f} | inner {m['inner']:.2f} outer {m['outer']:.2f} | FPR90 in {m['fpr90_inner']:.1f} out {m['fpr90_outer']:.1f}", flush=True)
                if v != "LF-PNCN":
                    for ref, fb in (("LF-PNCN", seedsub(base2, rule)), ("B1", b12[COLS])):
                        p = paired(fa, fb, a.B); res["tier2"]["paired"][f"{v} - {ref} /{rule}"] = p
                        print(f"T2/{rule} {v} - {ref}: " + " | ".join(f"{l} {p[l]['point']:+.2f} [{p[l]['ci_low']:+.2f}, {p[l]['ci_high']:+.2f}]" for l in ("macro", "inner", "outer")), flush=True)
    (P.OUT / "eval_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(table).to_csv(P.OUT / "eval_tables.csv", index=False)


if __name__ == "__main__":
    main()
