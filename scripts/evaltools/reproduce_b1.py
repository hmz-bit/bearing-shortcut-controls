"""Reproduce B1 (WDCNN-envelope, PU) from the 100 saved logit files.

Published (Vieira et al. 2026): Macro-AUROC 79.56 +- 13.07 (Table 9); mean FPR at TPR=90%: 24.00% inner, 60.15% outer.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import pu_data as pu
from metrics import mean_fpr_at_tpr, load_logit_run

PUBLISHED = {"macro_mean": 79.56, "macro_std": 13.07, "fpr90_inner": 24.00, "fpr90_outer": 60.15}


def main():
    runs = [load_logit_run(pu.logit_path(r)) for r in pu.EVAL_RUNS]
    yi = [r["y_inner"] for r in runs]
    yo = [r["y_outer"] for r in runs]
    si = [r["logit_inner"] for r in runs]  # AUROC/ROC are invariant to sigmoid, so logits suffice
    so = [r["logit_outer"] for r in runs]

    ai = np.array([roc_auc_score(y, s) for y, s in zip(yi, si)])
    ao = np.array([roc_auc_score(y, s) for y, s in zip(yo, so)])
    macro = (ai + ao) / 2
    out = {
        "n_runs": len(runs),
        "macro_mean": 100 * macro.mean(),
        "macro_std_pop": 100 * macro.std(ddof=0),
        "macro_std_sample": 100 * macro.std(ddof=1),
        "auroc_inner": ai.mean(),
        "auroc_outer": ao.mean(),
        "published": PUBLISHED,
        "fpr90": {},
    }
    for conv in ("horizontal", "horizontal_interp", "per_run_nearest", "per_run_interp", "per_run_first", "vertical", "pooled"):
        fi = 100 * mean_fpr_at_tpr(yi, si, 0.9, conv)
        fo = 100 * mean_fpr_at_tpr(yo, so, 0.9, conv)
        out["fpr90"][conv] = {"inner": fi, "outer": fo}
    print(json.dumps(out, indent=1))
    with open("results/b1_reproduction.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
