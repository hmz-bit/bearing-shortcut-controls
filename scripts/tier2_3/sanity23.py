"""Sanity checks for the fixed-Hz ablation, Tier 3 and Tier 2. Writes results/tier2_3/sanity23.json.
(a) no train/test bearing overlap in every Tier 1/Tier 3 split (100) and every Tier 2 pair (15), checked on the masks AND on the saved prediction frames;
    Tier 3 training windows are all N15_M07_F10 and test conditions are exactly the two requested ones.
(b) standardisation statistics come from training windows only (independent float64 recomputation; identical when test windows are replaced by garbage),
    for one Tier 3 split and one Tier 2 pair.
(c) shuffled-label runs (training labels permuted across windows): Tier 3 split 5 (both test speeds) and Tier 2 pair 0 -> AUROC about 0.5.
(d) timings from the per-split csv files.
Usage: python -I sanity23.py
"""
import glob, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common23 as K
import common as C


def cat(pattern):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(K.OUT / pattern)))], ignore_index=True)


def main():
    C.set_determinism()
    d = K.load_all(("t_mad",))
    X, Y, bearing, cond = d["X"]["t_mad"], d["Y"], d["bearing"], d["condition"]
    dev = K.device()
    out = {}
    # (a) overlaps
    bad = 0
    for run in range(5, 105):
        tr_ids, te_ids = C.load_split("eval", run)
        bad += len(set(tr_ids) & set(te_ids))
        tr3 = np.isin(bearing, tr_ids) & (cond == K.COND_SAME)
        bad += len(set(bearing[tr3]) & set(te_ids)) + int(not np.all(cond[tr3] == K.COND_SAME))
    out["a_tier1_tier3_split_overlaps"] = int(bad)
    bad2 = 0
    for p, held in enumerate(K.PAIRS):
        tr_ids = K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held]
        te_ids = K.REAL_FAULTY + list(held)
        bad2 += len(set(tr_ids) & set(te_ids)) + (len(tr_ids) != 16) + (len(te_ids) != 13)
        bad2 += int(np.any(np.isin(bearing, tr_ids) & np.isin(bearing, te_ids)))
    out["a_tier2_pair_overlaps"] = int(bad2)
    frames = {}
    for name, pat in (("f_tier1", "f_tier1_*_predictions_long.parquet"), ("t3_lfpncn", "t3_lfpncn_*_predictions_long.parquet"),
                      ("t3_b4", "t3_b4_*_predictions_long.parquet"), ("t3_f", "t3_f_*_predictions_long.parquet"), ("t3_b3", "t3_b3_*_predictions_long.parquet")):
        f = cat(pat)
        mism = 0
        for s, g in f.groupby("split"):
            tr_ids, te_ids = C.load_split("eval", int(s))
            mism += int(set(g.bearing) != set(te_ids)) + len(set(g.bearing) & set(tr_ids))
            if "test_cond" in g:
                mism += int(set(g.test_cond) != {K.COND_SAME, K.COND_SLOW})
        frames[name] = dict(n_splits=int(f.split.nunique()), mismatches=int(mism))
    for m in ("lfpncn", "b4", "f", "b3"):
        f = cat(f"t2_{m}_*_predictions_long.parquet")
        mism = 0
        for p, g in f.groupby("pair"):
            held = list(K.PAIRS[int(p)])
            te_ids = K.REAL_FAULTY + held
            tr_ids = K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held]
            mism += int(set(g.bearing) != set(te_ids)) + len(set(g.bearing) & set(tr_ids))
        frames[f"t2_{m}"] = dict(n_pairs=int(f.pair.nunique()), n_rows=int(len(f)), mismatches=int(mism))
    out["a_prediction_frames"] = frames
    print("(a)", out["a_tier1_tier3_split_overlaps"], out["a_tier2_pair_overlaps"], frames, flush=True)

    # (b) standardisation
    chk = {}
    tr_ids, te_ids = C.load_split("eval", 5)
    tr = np.isin(bearing, tr_ids) & (cond == K.COND_SAME)
    masks = {"same": np.isin(bearing, te_ids) & (cond == K.COND_SAME), "slow": np.isin(bearing, te_ids) & (cond == K.COND_SLOW)}
    held = list(K.PAIRS[0])
    tr2 = np.isin(bearing, K.ARTIFICIAL + [h for h in K.HEALTHY if h not in held])
    te2 = np.isin(bearing, K.REAL_FAULTY + held)
    for name, trm, tem in (("tier3_split5", tr, masks), ("tier2_pair0", tr2, {"test": te2})):
        r1 = K.train_generic(X, Y, trm, tem, seed=5, dev=dev)
        garbage = {k: X[m] * 1000.0 + 50.0 for k, m in tem.items()}
        r2 = K.train_generic(X, Y, trm, tem, seed=5, dev=dev, test_X_override=garbage)
        mean_ref = X[trm].astype(np.float64).mean(axis=(0, 3), keepdims=True)
        std_ref = X[trm].astype(np.float64).std(axis=(0, 3), ddof=1, keepdims=True)
        mean_all = X.astype(np.float64).mean(axis=(0, 3), keepdims=True)
        chk[name] = dict(n_train_windows=int(trm.sum()),
                         max_abs_diff_mean_vs_train_recompute=float(np.abs(r1["mean"] - mean_ref).max()),
                         max_rel_diff_std_vs_train_recompute=float((np.abs(r1["std"] - std_ref) / std_ref).max()),
                         max_abs_diff_mean_vs_all_windows=float(np.abs(r1["mean"] - mean_all).max()),
                         stats_identical_when_test_windows_replaced_by_garbage=bool(np.array_equal(r1["mean"], r2["mean"]) and np.array_equal(r1["std"], r2["std"])))
    out["b_standardisation"] = chk
    print("(b)", chk, flush=True)

    # (c) shuffled labels
    sh = {}
    r = K.train_generic(X, Y, tr, masks, seed=5, dev=dev, shuffle_window=True)
    sh["tier3_split5_window_shuffle"] = {k: dict(zip(("macro", "inner", "outer"), map(lambda v: 100 * float(v), C.macro_auroc_np(Y[masks[k]], S))))
                                         for k, S in r["scores"].items()}
    vals = []
    for seed in (0, 1000):
        r = K.train_generic(X, Y, tr2, {"test": te2}, seed=seed, dev=dev, shuffle_window=True)
        m, ai, ao = C.macro_auroc_np(Y[te2], r["scores"]["test"])
        vals.append(dict(seed=seed, macro=100 * float(m), inner=100 * float(ai), outer=100 * float(ao)))
    sh["tier2_pair0_window_shuffle"] = vals
    out["c_shuffled_labels"] = sh
    print("(c)", sh, flush=True)

    # (d) timings
    t = {}
    for f in sorted(glob.glob(str(K.OUT / "f_tier1_*_per_split.csv"))) :
        pass
    f1 = pd.concat([pd.read_csv(f) for f in glob.glob(str(K.OUT / "f_tier1_[0-9]*_per_split.csv"))])
    t["f_tier1_seconds_per_split"] = dict(mean=float(f1.train_seconds.mean()), median=float(f1.train_seconds.median()), max=float(f1.train_seconds.max()), total=float(f1.train_seconds.sum()))
    f3 = pd.concat([pd.read_csv(f) for f in glob.glob(str(K.OUT / "t3_*_per_split.csv"))])
    t["tier3_seconds_per_split_by_method"] = {m: dict(mean=float(g.train_seconds.mean()), total=float(g.drop_duplicates("split").train_seconds.sum())) for m, g in f3.groupby("method")}
    f2 = pd.concat([pd.read_csv(f) for f in glob.glob(str(K.OUT / "t2_*_per_pair.csv"))])
    t["tier2_seconds_per_run_by_method"] = {m: dict(mean=float(g.train_seconds.mean()), total=float(g.train_seconds.sum()), n_runs=int(len(g))) for m, g in f2.groupby("method")}
    out["d_timings"] = t
    print("(d)", t)
    (K.OUT / "sanity23.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
