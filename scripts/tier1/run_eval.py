"""Run LF-PNCN (t_mad) or the B4 ablation (t_max) on the 100 bearing-wise evaluation splits (run 5..104) with the config
selected on the tuning splits (chosen_config_t_mad.json; the SAME config is used for both variants).
Writes results/tier1/<tag>_predictions_long.parquet (split, bearing, y_inner, y_outer, s_inner, s_outer = raw logits)
and <tag>_per_split.csv.
Usage: python -I run_eval.py --variant t_mad --tag lfpncn [--seed-offset 0] [--config chosen_config_t_mad.json]
"""
import argparse, json, time
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="t_mad")
    ap.add_argument("--tag", default="lfpncn")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--config", default="chosen_config_t_mad.json")
    ap.add_argument("--runs", default="5-104")
    a = ap.parse_args()
    lo, hi = map(int, a.runs.split("-"))
    cfg = json.loads((C.RESULTS / a.config).read_text())
    print("config", cfg, "variant", a.variant, flush=True)
    C.set_determinism()
    d = C.load_cache(a.variant)
    X, Y, bearing, fname = d["X"], d["Y"], d["bearing"], d["fname"]
    assert all(fn.split("_")[3] == b for fn, b in zip(fname, bearing)), "fname/bearing mismatch"
    dev = C.device()
    preds, rows = [], []
    t0 = time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = C.load_split("eval", run)
        assert not set(tr_ids) & set(te_ids)
        r = C.train_lfpncn(X, Y, bearing, tr_ids, cfg, seed=run + a.seed_offset, held_ids=te_ids, dev=dev)
        res = r["results"][cfg["epochs"]]
        he, tr = r["held_mask"], r["train_mask"]
        assert set(bearing[he]) == set(te_ids) and set(bearing[tr]) == set(tr_ids) and not np.any(he & tr)
        S = res["scores"]
        preds.append(pd.DataFrame({"split": run, "bearing": bearing[he], "y_inner": Y[he, 0].astype(int),
                                   "y_outer": Y[he, 1].astype(int), "s_inner": S[:, 0], "s_outer": S[:, 1]}))
        rows.append(dict(split=run, auroc_inner=res["inner"], auroc_outer=res["outer"], macro=res["macro"],
                         train_seconds=r["train_seconds"], n_train=r["n_train"], n_test=int(he.sum())))
        print(f"{a.tag} split {run}: macro {100 * res['macro']:.2f} (in {100 * res['inner']:.1f} out {100 * res['outer']:.1f}) "
              f"{r['train_seconds']:.1f}s elapsed {time.time() - t0:.0f}s", flush=True)
        if run % 20 == 0:
            pd.concat(preds, ignore_index=True).to_parquet(C.RESULTS / f"{a.tag}_predictions_partial.parquet")
    pred = pd.concat(preds, ignore_index=True)
    summ = pd.DataFrame(rows)
    pred.to_parquet(C.RESULTS / f"{a.tag}_predictions_long.parquet")
    summ.to_csv(C.RESULTS / f"{a.tag}_per_split.csv", index=False)
    m = 100 * summ["macro"]
    print(f"DONE {a.tag}: macro {m.mean():.2f} +- {m.std(ddof=0):.2f} (ddof0), total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
