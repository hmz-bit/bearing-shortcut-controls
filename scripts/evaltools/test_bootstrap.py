"""Tests for bootstrap.py. Run:  python test_bootstrap.py [--fast]

1. exactness : pair-matrix AUROC == sklearn roc_auc_score(sample_weight=multiplicity) and M=1 == plain per-split AUROC
2. coverage  : synthetic world with a KNOWN true paired difference
      a) single test set (17 bearings: 6 healthy, 5 outer, 6 inner), cluster bootstrap vs naive window bootstrap
      b) 100 overlapping 3:2 splits (6 test bearings, 2/2/2) with the multi-split estimator
   Reports empirical coverage of the 95% percentile CI for the true difference (delta_A > delta_B) and the
   rejection rate (CI excludes 0) in the null world (delta_A = delta_B).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bootstrap as bs
import pu_data as pu
from metrics import macro_auroc

CLASSES = ["healthy"] * 6 + ["outer"] * 5 + ["inner"] * 6
IDS = [f"H{i}" for i in range(6)] + [f"O{i}" for i in range(5)] + [f"I{i}" for i in range(6)]
LAB = {"healthy": (0, 0), "outer": (0, 1), "inner": (1, 0)}


def _id_labels(b):  # labels for synthetic ids
    return LAB[{"H": "healthy", "O": "outer", "I": "inner"}[b[0]]]


# bootstrap/pu_data derive labels from PU codes; use PU-style ids for the synthetic world
PU_IDS = [f"K00{i}" for i in range(1, 7)] + [f"KA0{i}" for i in range(1, 6)] + [f"KI0{i}" for i in range(1, 7)]
PU_LAB = {b: pu.bearing_labels(b) for b in PU_IDS}


def sim_bearings(rng, tau=1.0, tau_m=0.5, a=0.8):
    """Per-bearing effects: shared u_b and model-specific v_{m,b} (so paired errors are correlated)."""
    u = rng.normal(0, tau, 17)
    vA = rng.normal(0, tau_m, 17)
    vB = rng.normal(0, tau_m, 17)
    return a * u + vA, a * u + vB


def sim_scores(rng, bearing_idx, effA, effB, dA, dB, n_win):
    """Windows for the listed bearings. Score = delta*y + bearing effect + window noise (both labels)."""
    rows = []
    for i in bearing_idx:
        yi, yo = PU_LAB[PU_IDS[i]]
        e = rng.normal(0, 1, (n_win, 4))
        rows.append(pd.DataFrame({
            "bearing": PU_IDS[i], "y_inner": yi, "y_outer": yo,
            "sA_inner": dA * yi + effA[i] + e[:, 0], "sA_outer": dA * yo + effA[i] + e[:, 1],
            "sB_inner": dB * yi + effB[i] + e[:, 2], "sB_outer": dB * yo + effB[i] + e[:, 3],
        }))
    return pd.concat(rows, ignore_index=True)


def frames(df, split=0):
    out = []
    for m in ("A", "B"):
        out.append(pd.DataFrame({"split": split, "bearing": df["bearing"], "y_inner": df["y_inner"], "y_outer": df["y_outer"],
                                 "s_inner": df[f"s{m}_inner"], "s_outer": df[f"s{m}_outer"]}))
    return out


def pooled_diff(df):
    y = df[["y_inner", "y_outer"]].to_numpy()
    a = macro_auroc(y, df[["sA_inner", "sA_outer"]].to_numpy())
    b = macro_auroc(y, df[["sB_inner", "sB_outer"]].to_numpy())
    return 100 * (a - b)


# ------------------------------------------------------------------------------------------ test 1
def test_exactness():
    rng = np.random.default_rng(0)
    effA, effB = sim_bearings(rng)
    df = sim_scores(rng, range(17), effA, effB, 1.2, 0.8, 40)
    fa, _ = frames(df)
    t = bs.build_pair_table(fa)
    M = bs.draw_multiplicities(t.bearings, 50, rng)
    au = t.auroc(M)[:, 0, :]
    bidx = {b: i for i, b in enumerate(t.bearings)}
    worst = 0.0
    n = 0
    for d in range(50):
        w = M[d][fa["bearing"].map(bidx).to_numpy()]
        for l, lab in enumerate(("inner", "outer")):
            y = fa[f"y_{lab}"].to_numpy()
            if w[y == 1].sum() > 0 and w[y == 0].sum() > 0:
                ref = roc_auc_score(y, fa[f"s_{lab}"], sample_weight=w)
                worst = max(worst, abs(ref - au[d, l]))
                n += 1
    plain = bs.build_pair_table(fa).auroc(np.ones((1, 17)))[0, 0]
    ref_plain = [roc_auc_score(fa[f"y_{lab}"], fa[f"s_{lab}"]) for lab in ("inner", "outer")]
    assert worst < 1e-12 and np.allclose(plain, ref_plain), (worst, plain, ref_plain)
    return {"n_compared": n, "max_abs_err_weighted": worst, "max_abs_err_plain": float(np.abs(plain - ref_plain).max())}


# ------------------------------------------------------------------------------------------ test 2a
def true_single_diff(dA, dB, n_world=3000, n_win=100, seed=123):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_world):
        eA, eB = sim_bearings(rng)
        d.append(pooled_diff(sim_scores(rng, range(17), eA, eB, dA, dB, n_win)))
    return float(np.mean(d)), float(np.std(d))


def coverage_single(dA, dB, truth, R=300, B=1000, n_win=100, naive_R=60, seed=7):
    rng = np.random.default_rng(seed)
    cov, excl0, widths = 0, 0, []
    for r in range(R):
        eA, eB = sim_bearings(rng)
        df = sim_scores(rng, range(17), eA, eB, dA, dB, n_win)
        y = df[["y_inner", "y_outer"]].to_numpy()
        sa = df[["sA_inner", "sA_outer"]].to_numpy()
        sb = df[["sB_inner", "sB_outer"]].to_numpy()
        res = bs.paired_bootstrap_single_split(y, sa, sb, df["bearing"].to_numpy(), B=B, seed=r)["diff"]
        cov += res["ci_low"] <= truth <= res["ci_high"]
        excl0 += not (res["ci_low"] <= 0 <= res["ci_high"])
        widths.append(res["ci_high"] - res["ci_low"])
    out = {"R": R, "B": B, "cluster_coverage": cov / R, "cluster_excludes_zero": excl0 / R, "cluster_mean_ci_width": float(np.mean(widths))}
    # naive window bootstrap on fewer replicates (slow)
    rng = np.random.default_rng(seed + 1)
    ncov, nex, nw = 0, 0, []
    for r in range(naive_R):
        eA, eB = sim_bearings(rng)
        df = sim_scores(rng, range(17), eA, eB, dA, dB, n_win)
        y = df[["y_inner", "y_outer"]].to_numpy()
        res = bs.naive_window_bootstrap(y, df[["sA_inner", "sA_outer"]].to_numpy(), df[["sB_inner", "sB_outer"]].to_numpy(), B=150, seed=r)
        ncov += res["ci_low"] <= truth <= res["ci_high"]
        nex += not (res["ci_low"] <= 0 <= res["ci_high"])
        nw.append(res["ci_high"] - res["ci_low"])
    out.update({"naive_R": naive_R, "naive_coverage": ncov / naive_R, "naive_excludes_zero": nex / naive_R, "naive_mean_ci_width": float(np.mean(nw))})
    return out


# ------------------------------------------------------------------------------------------ test 2b
def sim_multisplit(rng, dA, dB, n_splits=100, n_win=60):
    eA, eB = sim_bearings(rng)
    cls = {c: [i for i, k in enumerate(CLASSES) if k == c] for c in ("healthy", "outer", "inner")}
    fa, fb = [], []
    for s in range(n_splits):
        test = [i for c in cls for i in rng.choice(cls[c], 2, replace=False)]
        df = sim_scores(rng, test, eA, eB, dA, dB, n_win)
        a, b = frames(df, split=s)
        fa.append(a), fb.append(b)
    return pd.concat(fa, ignore_index=True), pd.concat(fb, ignore_index=True)


def true_multi_diff(dA, dB, n_world=6000, n_win=60, seed=321):
    """E over bearing populations of the pooled 2/2/2 test-set difference = target of the split-averaged statistic."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_world):
        eA, eB = sim_bearings(rng)
        cls = {c: [i for i, k in enumerate(CLASSES) if k == c] for c in ("healthy", "outer", "inner")}
        test = [i for c in cls for i in rng.choice(cls[c], 2, replace=False)]
        d.append(pooled_diff(sim_scores(rng, test, eA, eB, dA, dB, n_win)))
    return float(np.mean(d))


def coverage_multi(dA, dB, truth, R=100, B=500, seed=11):
    rng = np.random.default_rng(seed)
    cov, excl0, widths = 0, 0, []
    for r in range(R):
        fa, fb = sim_multisplit(rng, dA, dB)
        res = bs.paired_bootstrap_multi_split(fa, fb, B=B, seed=r)["diff"]
        cov += res["ci_low"] <= truth <= res["ci_high"]
        excl0 += not (res["ci_low"] <= 0 <= res["ci_high"])
        widths.append(res["ci_high"] - res["ci_low"])
    return {"R": R, "B": B, "coverage": cov / R, "excludes_zero": excl0 / R, "mean_ci_width": float(np.mean(widths))}


def main():
    fast = "--fast" in sys.argv
    out = {}
    t0 = time.time()
    out["exactness"] = test_exactness()
    print("exactness:", out["exactness"], flush=True)

    dA, dB = 1.5, 1.0
    tS, sdS = true_single_diff(dA, dB, n_world=500 if fast else 3000)
    out["single_world"] = {"true_diff_points": tS, "sd_of_stat_across_worlds": sdS}
    out["single_alt"] = coverage_single(dA, dB, tS, R=60 if fast else 300, B=500 if fast else 1000, naive_R=20 if fast else 60)
    out["single_null"] = coverage_single(dA, dA, 0.0, R=60 if fast else 300, B=500 if fast else 1000, naive_R=20 if fast else 60, seed=17)
    print("single:", json.dumps(out, indent=1), flush=True)

    tM = true_multi_diff(dA, dB, n_world=1000 if fast else 6000)
    out["multi_world"] = {"true_diff_points": tM}
    out["multi_alt"] = coverage_multi(dA, dB, tM, R=20 if fast else 100, B=300 if fast else 500)
    out["multi_null"] = coverage_multi(dA, dA, 0.0, R=20 if fast else 100, B=300 if fast else 500, seed=19)
    out["seconds"] = time.time() - t0
    print(json.dumps(out, indent=1))
    if not fast:
        with open("results/bootstrap_tests.json", "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
