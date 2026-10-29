"""Collect every tuning-split run of phase 2 into results/phase2/tuning_table.csv and tuning_t3_table.csv (all variants tried, all seeds)."""
import sys, pathlib, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd, numpy as np
import p2common as P

T = P.OUT / "tuning"
rows = []
for f in sorted(glob.glob(str(T / "*_s*_persplit.csv"))):
    if "t3tune" in f:
        continue
    d = pd.read_csv(f)
    rows.append(d)
d = pd.concat(rows)
g = d.groupby(["variant", "seed_idx"])[["macro", "inner", "outer"]].mean().reset_index()
agg = g.groupby("variant").agg(n_seeds=("seed_idx", "size"), macro=("macro", "mean"), inner=("inner", "mean"), outer=("outer", "mean"),
                               macro_seed_min=("macro", "min"), macro_seed_max=("macro", "max"))
agg["per_seed_macro"] = g.groupby("variant").macro.apply(lambda s: ";".join(f"{x:.2f}" for x in s))
base = agg.loc["base", "macro"]
agg["delta_macro_vs_base"] = agg.macro - base
agg = agg.sort_values("macro", ascending=False)
agg.round(2).to_csv(P.OUT / "tuning_table.csv")
pd.set_option("display.width", 250, "display.max_columns", 30)
print("=== 5 tuning splits, mean validation Macro-AUROC (Tier 1 grid protocol at the Tier-1 configuration) ===")
print(agg.round(2).to_string())
rows = [pd.read_csv(f) for f in sorted(glob.glob(str(T / "*_t3tune_persplit.csv")))]
if rows:
    d3 = pd.concat(rows)
    g3 = d3.groupby("variant")[["same_macro", "same_inner", "same_outer", "slow_macro", "slow_inner", "slow_outer"]].mean()
    g3["loss"] = g3.same_macro - g3.slow_macro
    g3 = g3.sort_values("slow_macro", ascending=False)
    g3.round(2).to_csv(P.OUT / "tuning_t3_table.csv")
    print("\n=== Tuning-split analogue of Tier 3: train 1500 rpm of the 11 training bearings; test the 6 validation bearings at 1500 and 900 rpm (seed 0 only) ===")
    print(g3.round(2).to_string())
