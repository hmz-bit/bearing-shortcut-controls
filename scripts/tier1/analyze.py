"""Metrics, paired bearing-level cluster bootstrap and decisions for Tier 1.
Reads results/tier1/{lfpncn,b4_tmax,b3_rf}_predictions_long.parquet and the existing evaltools B1/B2 long parquets.
Writes results/tier1/metrics.json, metrics_table.csv, paired_table.csv.
Usage: python -I analyze.py [--B 2000]
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import numpy as np
import pandas as pd
import bootstrap as bs
import metrics as mt

EVAL = C.WORK / "scripts" / "evaltools" / "results"
METHODS = {
    "LF-PNCN": C.RESULTS / "lfpncn_predictions_long.parquet",
    "B4 (t_max)": C.RESULTS / "b4_tmax_predictions_long.parquet",
    "B3 (RF, 20 feat)": C.RESULTS / "b3_rf_predictions_long.parquet",
    "B2 (RF, 15 feat)": EVAL / "b2_rf_predictions_long.parquet",
    "B1 (WDCNN envelope)": EVAL / "b1_predictions_long.parquet",
}
EXTRA = {"LF-PNCN seed+1000": C.RESULTS / "lfpncn_seed2_predictions_long.parquet",
         "B4 (t_max), own tuned config": C.RESULTS / "b4_tmax_owncfg_predictions_long.parquet"}


def per_split(frame):
    rows = []
    for s, g in frame.groupby("split", sort=True):
        ai = mt.roc_auc_score(g.y_inner, g.s_inner) if False else None
        m, per = mt.macro_auroc(g[["y_inner", "y_outer"]].to_numpy(), g[["s_inner", "s_outer"]].to_numpy(), True)
        rows.append(dict(split=s, inner=per[0], outer=per[1], macro=m))
    return pd.DataFrame(rows)


def fpr90(frame, label):
    ys, ss = [], []
    for s, g in frame.groupby("split", sort=True):
        ys.append(g[f"y_{label}"].to_numpy())
        ss.append(g[f"s_{label}"].to_numpy())
    return 100 * mt.mean_fpr_at_tpr(ys, ss, 0.9, "horizontal")


def decide_h1(d):
    pt, lo, hi = d["point"], d["ci_low"], d["ci_high"]
    if pt >= 3.0:
        return "SUPPORTED (point diff >= +3)" + ("" if lo > 0 else "; CI includes 0")
    if lo >= -1.5 and hi <= 1.5:
        return "FALSIFIED (CI within +-1.5)"
    return "INCONCLUSIVE"


def decide_h2(d):
    pt, lo, hi = d["point"], d["ci_low"], d["ci_high"]
    if pt >= 6.0:
        return "SUPPORTED (point gain >= +6)" + ("" if lo > 0 else "; CI includes 0")
    if hi < 3.0:
        return "FALSIFIED (CI upper < +3)"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    frames = {k: pd.read_parquet(p) for k, p in METHODS.items()}
    for k, p in EXTRA.items():
        if p.exists():
            frames[k] = pd.read_parquet(p)
    out, rows = {"metrics": {}, "paired": {}}, []
    for k, f in frames.items():
        ps = per_split(f)
        m = 100 * ps.macro
        e = dict(macro_mean=m.mean(), macro_sd_ddof0=m.std(ddof=0), macro_sd_ddof1=m.std(ddof=1),
                 inner=100 * ps.inner.mean(), outer=100 * ps.outer.mean(), fpr90_inner=fpr90(f, "inner"),
                 fpr90_outer=fpr90(f, "outer"), n_splits=len(ps), n_windows=len(f))
        out["metrics"][k] = e
        rows.append(dict(method=k, **e))
        print(f"{k:32s} macro {e['macro_mean']:.2f} +- {e['macro_sd_ddof0']:.2f} | inner {e['inner']:.2f} outer {e['outer']:.2f} "
              f"| FPR90 in {e['fpr90_inner']:.1f} out {e['fpr90_outer']:.1f}", flush=True)
        ps.to_csv(C.RESULTS / f"persplit_{k.split()[0].replace('-', '').lower()}_{len(rows)}.csv", index=False)
    pd.DataFrame(rows).to_csv(C.RESULTS / "metrics_table.csv", index=False)

    pairs = [("H1", "LF-PNCN", "B4 (t_max)"), ("H2", "LF-PNCN", "B1 (WDCNN envelope)"), ("H2", "LF-PNCN", "B2 (RF, 15 feat)"),
             ("H2", "LF-PNCN", "B3 (RF, 20 feat)"), ("H2", "B3 (RF, 20 feat)", "B1 (WDCNN envelope)"),
             ("aux", "B3 (RF, 20 feat)", "B2 (RF, 15 feat)"), ("aux", "B1 (WDCNN envelope)", "B2 (RF, 15 feat)")]
    for k in EXTRA:
        if k in frames:
            pairs.append(("aux", "LF-PNCN", k)) if k.startswith("LF") or k.startswith("B4") else None
    if "LF-PNCN seed+1000" in frames:
        pairs.append(("aux", "LF-PNCN seed+1000", "B1 (WDCNN envelope)"))
    pairs.append(("aux", "B4 (t_max)", "B1 (WDCNN envelope)"))
    prow = []
    for hyp, A, Bn in pairs:
        r = bs.paired_bootstrap_multi_split(frames[A], frames[Bn], B=a.B, seed=0, stratify=True)
        d = r["diff"]
        if hyp == "H1":
            dec = decide_h1(d)
        elif hyp == "H2" and A == "LF-PNCN" and Bn.startswith("B1"):
            dec = decide_h2(d)
        elif hyp == "H2" and A == "LF-PNCN" and Bn.startswith("B2"):
            dec = decide_h2(d)
        elif hyp == "H2" and A == "LF-PNCN" and Bn.startswith("B3"):
            dec = ("NETWORK ADDS NOTHING (point within +-1)" if abs(d["point"]) <= 1 else "network differs from B3 by more than 1 point") + \
                  ("; CI entirely within +-1" if d["ci_low"] >= -1 and d["ci_high"] <= 1 else "; CI not within +-1")
        elif hyp == "H2":
            dec = "(B3 - B1: descriptive; CI " + ("excludes" if d["ci_low"] > 0 or d["ci_high"] < 0 else "includes") + " 0)"
        else:
            dec = "(auxiliary)"
        out["paired"][f"{A} - {Bn}"] = dict(hyp=hyp, **d, valid_splits_per_draw=r["mean_valid_splits_per_draw"],
                                           A_ci=[r["model_A"]["ci_low"], r["model_A"]["ci_high"]],
                                           B_ci=[r["model_B"]["ci_low"], r["model_B"]["ci_high"]], decision=dec)
        prow.append(dict(hyp=hyp, comparison=f"{A} - {Bn}", diff=d["point"], ci_low=d["ci_low"], ci_high=d["ci_high"], se=d["se"],
                         p_le_0=d["p_le_0"], decision=dec))
        print(f"{hyp} {A} - {Bn}: {d['point']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] SE {d['se']:.2f} -> {dec}", flush=True)
    pd.DataFrame(prow).to_csv(C.RESULTS / "paired_table.csv", index=False)
    # model-level CIs
    out["model_ci"] = {k: None for k in frames}
    for k in frames:
        r = bs.paired_bootstrap_multi_split(frames[k], frames["B1 (WDCNN envelope)"], B=a.B, seed=0)
        out["model_ci"][k] = [r["model_A"]["ci_low"], r["model_A"]["ci_high"], r["model_A"]["se"]]
    (C.RESULTS / "metrics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
