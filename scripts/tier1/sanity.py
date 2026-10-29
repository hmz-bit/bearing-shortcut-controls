"""Sanity checks for Tier 1 (a)-(e). Writes results/tier1/sanity.json.
(a) shuffled-label runs (window-level on split 5; bearing-level on splits 5..14) -> AUROC should be ~0.5
(b) train vs test AUROC for splits 5..9 (train windows scored in eval mode with the standardiser of the training set)
(c) no bearing in both train and test for all 100 eval splits and 5 tuning splits; prediction frames of every method hold exactly the test bearings
(d) standardisation statistics come from training windows only (recomputed independently; perturbing test windows leaves them unchanged)
(e) training time per split (from the per-split csv)
Usage: python -I sanity.py
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import numpy as np
import pandas as pd
import torch


def main():
    C.set_determinism()
    cfg = json.loads((C.RESULTS / "chosen_config_t_mad.json").read_text())
    d = C.load_cache("t_mad")
    X, Y, bearing = d["X"], d["Y"], d["bearing"]
    dev = C.device()
    out = {"config": cfg}

    # (a) shuffled labels
    tr, te = C.load_split("eval", 5)
    r = C.train_lfpncn(X, Y, bearing, tr, cfg, seed=5, held_ids=te, dev=dev, shuffle_labels="window")
    res = r["results"][cfg["epochs"]]
    out["a_window_shuffle_split5"] = dict(macro=100 * res["macro"], inner=100 * res["inner"], outer=100 * res["outer"])
    bs = []
    for run in range(5, 15):
        tr, te = C.load_split("eval", run)
        r = C.train_lfpncn(X, Y, bearing, tr, cfg, seed=run, held_ids=te, dev=dev, shuffle_labels="bearing")
        bs.append(100 * r["results"][cfg["epochs"]]["macro"])
    out["a_bearing_label_permutation_splits5_14"] = dict(macro_mean=float(np.mean(bs)), macro_sd=float(np.std(bs, ddof=1)), values=bs)
    print("(a)", out["a_window_shuffle_split5"], out["a_bearing_label_permutation_splits5_14"]["macro_mean"], flush=True)

    # (b) train vs test AUROC
    rows = []
    for run in range(5, 10):
        tr, te = C.load_split("eval", run)
        r = C.train_lfpncn(X, Y, bearing, tr, cfg, seed=run, held_ids=te, dev=dev, return_model=True)
        res = r["results"][cfg["epochs"]]
        S_tr = C.predict(r["model"], r["Xtr"])
        m_tr, i_tr, o_tr = C.macro_auroc_np(Y[r["train_mask"]], S_tr)
        rows.append(dict(split=run, train_macro=100 * m_tr, train_inner=100 * i_tr, train_outer=100 * o_tr,
                         test_macro=100 * res["macro"], test_inner=100 * res["inner"], test_outer=100 * res["outer"]))
    out["b_train_vs_test"] = rows
    print("(b)", pd.DataFrame(rows).round(2).to_string(), flush=True)

    # (c) leakage checks
    n_bad = 0
    for kind, runs in (("eval", range(5, 105)), ("tuning", range(5))):
        for run in runs:
            tr, held = C.load_split(kind, run)
            n_bad += len(set(tr) & set(held)) + (len(tr) != 11) + (len(held) != 6)
            m_tr, m_he = np.isin(bearing, tr), np.isin(bearing, held)
            n_bad += int(np.any(m_tr & m_he))
            fn = d["fname"]
            n_bad += len({f.split("_")[3] for f in fn[m_tr]} & {f.split("_")[3] for f in fn[m_he]})
    out["c_split_overlaps_found"] = int(n_bad)
    frames = {"lfpncn": "lfpncn", "b4": "b4_tmax", "b3": "b3_rf"}
    mism = {}
    for k, tag in frames.items():
        p = C.RESULTS / f"{tag}_predictions_long.parquet"
        if not p.exists():
            mism[k] = "missing"
            continue
        f = pd.read_parquet(p)
        bad = 0
        for s, g in f.groupby("split"):
            tr, te = C.load_split("eval", int(s))
            bad += int(set(g.bearing) != set(te)) + len(set(g.bearing) & set(tr))
        mism[k] = dict(bad_splits=int(bad), n_splits=int(f.split.nunique()))
    out["c_prediction_frames"] = mism
    print("(c)", out["c_split_overlaps_found"], mism, flush=True)

    # (d) standardisation statistics
    tr, te = C.load_split("eval", 5)
    r1 = C.train_lfpncn(X, Y, bearing, tr, cfg, seed=5, held_ids=te, dev=dev)
    Xp = X[np.isin(bearing, te)] * 1000.0 + 50.0   # garbage test windows
    r2 = C.train_lfpncn(X, Y, bearing, tr, cfg, seed=5, held_ids=te, dev=dev, test_X_override=Xp)
    m_tr = np.isin(bearing, tr)
    mean_ref = X[m_tr].astype(np.float64).mean(axis=(0, 3), keepdims=True)
    std_ref = X[m_tr].astype(np.float64).std(axis=(0, 3), ddof=1, keepdims=True)
    mean_all = X.astype(np.float64).mean(axis=(0, 3), keepdims=True)
    out["d_standardisation"] = dict(
        max_abs_diff_mean_vs_train_recompute=float(np.abs(r1["mean"] - mean_ref).max()),
        max_rel_diff_std_vs_train_recompute=float((np.abs(r1["std"] - std_ref) / std_ref).max()),
        max_abs_diff_mean_vs_all_windows=float(np.abs(r1["mean"] - mean_all).max()),
        stats_identical_when_test_windows_replaced_by_garbage=bool(np.array_equal(r1["mean"], r2["mean"]) and np.array_equal(r1["std"], r2["std"])),
        channels=int(r1["mean"].size))
    print("(d)", out["d_standardisation"], flush=True)

    # (e) timing
    t = pd.read_csv(C.RESULTS / "lfpncn_per_split.csv").train_seconds
    out["e_train_seconds_per_split"] = dict(mean=float(t.mean()), min=float(t.min()), max=float(t.max()), total=float(t.sum()))
    (C.RESULTS / "sanity.json").write_text(json.dumps(out, indent=1))
    print("(e)", out["e_train_seconds_per_split"])


if __name__ == "__main__":
    main()
