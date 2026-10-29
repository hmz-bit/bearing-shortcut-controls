"""B3: Random Forest on the 20 b3 features (5 time statistics + 15 broadband log-SNR values) from the cache, 100 eval splits.
RF hyper-parameters and protocol are exactly those of evaltools/b2_rf.py (Vieira et al. Table 17, 200 trees, MultiOutputClassifier, seed 42).
Usage: python -I run_b3.py [--seed 42]
"""
import argparse, time
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import b2_rf   # evaltools (path added by common)
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="b3_rf")
    a = ap.parse_args()
    d = C.load_cache("t_mad")
    cols = [f"f{i:02d}" for i in range(20)]
    df = pd.DataFrame(d["b3"].astype(np.float64), columns=cols)
    df["bearing"] = d["bearing"]
    df["y_inner"] = d["Y"][:, 0].astype(int)
    df["y_outer"] = d["Y"][:, 1].astype(int)
    t = time.time()
    summ, pred = b2_rf.run_rf(df, cols, b2_rf.PARAMS_TABLE17, seed=a.seed)
    summ["seconds_total"] = time.time() - t
    summ.to_csv(C.RESULTS / f"{a.tag}_per_split.csv", index=False)
    pred.to_parquet(C.RESULTS / f"{a.tag}_predictions_long.parquet")
    m = 100 * summ["macro"]
    print(f"B3 RF: macro {m.mean():.2f} +- {m.std(ddof=0):.2f}; inner {100*summ.auroc_inner.mean():.2f} outer {100*summ.auroc_outer.mean():.2f}; {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
