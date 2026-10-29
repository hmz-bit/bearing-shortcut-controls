"""B2: Random Forest baseline on the 15 hand-crafted PU features, evaluated on the 100 bearing-wise splits.

Hyper-parameters are Vieira et al. 2026 Table 17 (PU): 200 trees, max_features=log2, criterion=log_loss,
max_depth=24, min_samples_split=15, min_samples_leaf=18, ccp_alpha=2.65e-2 (the authors' notebook stores the
unrounded value 0.026475639105883027; both are run and reported; the rounded Table-17 value is the default).
Model: MultiOutputClassifier over [inner, outer]. Published Macro-AUROC (Table 9): 69.76 +- 15.72.

B3 (20 features = 5 time stats + 15 envelope magnitudes at 1x-5x of f_o, f_i, f_b) cannot be built from the PU
parquet: it holds envelope peaks only for bpfo and bpfi (10 columns) -> 15 features in total. The ball-pass
(bpfb) columns are absent. ``build_b3_features`` checks for them and raises with an explanatory message.

Usage: python b2_rf.py [--ccp 0.0265 | --ccp-exact] [--runs 5-104] [--seed 42]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

import pu_data as pu

PARAMS_TABLE17 = dict(
    n_estimators=200,
    max_features="log2",
    criterion="log_loss",
    max_depth=24,
    min_samples_split=15,
    min_samples_leaf=18,
    ccp_alpha=2.65e-2,
)
CCP_NOTEBOOK = 0.026475639105883027
PUBLISHED = {"macro_mean": 69.76, "macro_std": 15.72}


def build_b3_features(df: pd.DataFrame) -> list[str]:
    """Return the 20 B3 feature names, or raise if the parquet lacks the bpfb envelope columns."""
    missing = [c for c in pu.ENV_BPFB_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            "B3 needs envelope magnitudes at 1x-5x of bpfb; the PU features parquet only has bpfo/bpfi "
            f"(missing: {missing}). They would have to be recomputed from raw PU signals + bearing geometry."
        )
    return pu.FEATURES_15 + pu.ENV_BPFB_FEATURES


def run_rf(df: pd.DataFrame, features: list[str], params: dict, runs=pu.EVAL_RUNS, seed: int = 42, n_jobs: int = -1):
    """Train/evaluate on each bearing-wise split. Returns (summary DataFrame, long per-window predictions)."""
    X = df[features].to_numpy()
    Y = df[["y_inner", "y_outer"]].to_numpy()
    rows, preds = [], []
    for r in runs:
        train_ids, test_ids = pu.load_eval_split(r)
        tr = df["bearing"].isin(train_ids).to_numpy()
        te = df["bearing"].isin(test_ids).to_numpy()
        assert not set(train_ids) & set(test_ids)
        clf = MultiOutputClassifier(RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params))
        clf.fit(X[tr], Y[tr])
        pr = clf.predict_proba(X[te])  # list per output: [inner, outer], each (n, 2)
        s_in, s_out = pr[0][:, 1], pr[1][:, 1]
        a_in = roc_auc_score(Y[te, 0], s_in)
        a_out = roc_auc_score(Y[te, 1], s_out)
        rows.append({"split": r, "auroc_inner": a_in, "auroc_outer": a_out, "macro": (a_in + a_out) / 2})
        preds.append(
            pd.DataFrame(
                {"split": r, "bearing": df.loc[te, "bearing"].to_numpy(), "y_inner": Y[te, 0], "y_outer": Y[te, 1],
                 "s_inner": s_in, "s_outer": s_out}
            )
        )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccp-exact", action="store_true", help="use the notebook's unrounded ccp_alpha")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="b2_rf")
    a = ap.parse_args()

    df = pu.load_features()
    params = dict(PARAMS_TABLE17)
    if a.ccp_exact:
        params["ccp_alpha"] = CCP_NOTEBOOK
    t = time.time()
    summ, pred = run_rf(df, pu.FEATURES_15, params, seed=a.seed)
    m = 100 * summ["macro"]
    res = {
        "tag": a.tag,
        "params": params,
        "seed": a.seed,
        "n_runs": len(summ),
        "macro_mean": m.mean(),
        "macro_std_pop": m.std(ddof=0),
        "macro_std_sample": m.std(ddof=1),
        "auroc_inner": summ["auroc_inner"].mean(),
        "auroc_outer": summ["auroc_outer"].mean(),
        "published": PUBLISHED,
        "seconds": time.time() - t,
    }
    print(json.dumps(res, indent=1))
    summ.to_csv(f"results/{a.tag}_per_split.csv", index=False)
    pred.to_parquet(f"results/{a.tag}_predictions_long.parquet")
    with open(f"results/{a.tag}_summary.json", "w") as f:
        json.dump(res, f, indent=1)
    try:
        build_b3_features(df)
    except KeyError as e:
        print("B3:", e)


if __name__ == "__main__":
    main()
