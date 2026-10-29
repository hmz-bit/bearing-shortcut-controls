"""Paired bearing-level bootstrap B1 (saved WDCNN-envelope logits) vs B2 (RF) + minimum detectable effect.

Needs results/b1_predictions_long.parquet (from pu_data.logits_to_long) and results/b2_rf_predictions_long.parquet.
"""
import json

import numpy as np
import pandas as pd

import bootstrap as bs
import pu_data as pu


def main():
    a = pd.read_parquet("results/b1_predictions_long.parquet")
    b = pd.read_parquet("results/b2_rf_predictions_long.parquet")
    out = {}
    for strat in (True, False):
        r = bs.paired_bootstrap_multi_split(a, b, B=2000, seed=0, stratify=strat)
        out[f"b1_vs_b2_stratified={strat}"] = r

    ta, tb = bs.build_pair_table(a), bs.build_pair_table(b, bs.build_pair_table(a).bearings)
    one = np.ones((1, 17))
    ma, mb = ta.macro_per_split(one)[0] * 100, tb.macro_per_split(one)[0] * 100
    rho = float(np.corrcoef(ma, mb)[0, 1])
    sd_a, sd_b = float(ma.std(ddof=1)), float(mb.std(ddof=1))
    sd_d = float((ma - mb).std(ddof=1))
    out["across_splits"] = {"sd_b1": sd_a, "sd_b2": sd_b, "corr_b1_b2": rho, "sd_paired_diff": sd_d,
                            "mean_b1": float(ma.mean()), "mean_b2": float(mb.mean())}
    se_diff = out["b1_vs_b2_stratified=True"]["diff"]["se"]
    se_a = out["b1_vs_b2_stratified=True"]["model_A"]["se"]
    out["mde"] = {
        "formula_sd13_n17": {f"rho={r}": bs.minimum_detectable_effect(13.0, 17, r) for r in (0.0, 0.5, 0.8, 0.9)},
        "formula_sd13_n11_healthy_or_inner_only": {f"rho={r}": bs.minimum_detectable_effect(13.0, 11, r) for r in (0.0, 0.5, 0.8, 0.9)},
        "formula_sd13_n5_smallest_class": {f"rho={r}": bs.minimum_detectable_effect(13.0, 5, r) for r in (0.0, 0.5, 0.8, 0.9)},
        "formula_with_observed_rho": bs.minimum_detectable_effect(13.0, 17, rho),
        "empirical_from_bootstrap_se_diff": bs.empirical_mde(se_diff),
        "bootstrap_se_diff": se_diff,
        "bootstrap_se_single_model_b1": se_a,
        "empirical_unpaired_mde_from_se_b1": bs.empirical_mde(se_a * np.sqrt(2)),
    }
    # per-split single-split paired CI (6 test bearings): how often does a single split resolve B1 vs B2 ?
    ex = 0
    for s in pu.EVAL_RUNS:
        ga, gb = a[a.split == s], b[b.split == s]
        r = bs.paired_bootstrap_multi_split(ga, gb, B=500, seed=s)["diff"]
        ex += not (r["ci_low"] <= 0 <= r["ci_high"])
    out["single_split_ci_excludes_zero_fraction"] = ex / len(pu.EVAL_RUNS)
    print(json.dumps(out, indent=1))
    with open("results/analysis.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
