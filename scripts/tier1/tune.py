"""Hyper-parameter selection on the 5 TUNING splits only (train/val). Eval splits are never touched here.

Grid: lr {1e-3,3e-4} x batch {64,256} x weight decay {0,1e-4}; epochs {20,40} evaluated at checkpoints of one 40-epoch run
(constant learning rate, so the epoch-20 checkpoint equals a 20-epoch run). Selection criterion: mean val Macro-AUROC over the 5 splits.
Usage: python -I tune.py [--variant t_mad|t_max] [--tag tune_t_mad]
"""
import argparse, itertools, json, time
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="t_mad")
    a = ap.parse_args()
    C.set_determinism()
    d = C.load_cache(a.variant)
    X, Y, bearing = d["X"], d["Y"], d["bearing"]
    dev = C.device()
    rows = []
    t0 = time.time()
    for lr, bs, wd in itertools.product([1e-3, 3e-4], [64, 256], [0.0, 1e-4]):
        per = {20: [], 40: []}
        secs = []
        for run in C.TUNING_RUNS:
            tr_ids, va_ids = C.load_split("tuning", run)
            r = C.train_lfpncn(X, Y, bearing, tr_ids, dict(lr=lr, bs=bs, wd=wd, epochs=40), seed=1000 + run,
                               checkpoints=[20, 40], held_ids=va_ids, dev=dev)
            secs.append(r["train_seconds"])
            for ep in (20, 40):
                per[ep].append(r["results"][ep])
        for ep in (20, 40):
            m = np.array([x["macro"] for x in per[ep]]) * 100
            rows.append(dict(variant=a.variant, lr=lr, bs=bs, wd=wd, epochs=ep, val_macro_mean=m.mean(), val_macro_sd=m.std(ddof=1),
                             val_inner=100 * np.mean([x["inner"] for x in per[ep]]), val_outer=100 * np.mean([x["outer"] for x in per[ep]]),
                             per_split=";".join(f"{v:.2f}" for v in m), sec_per_run_40ep=np.mean(secs)))
            print(a.variant, rows[-1]["lr"], bs, wd, ep, f"{m.mean():.2f} +- {m.std(ddof=1):.2f}", f"{np.mean(secs):.1f}s", flush=True)
    df = pd.DataFrame(rows).sort_values("val_macro_mean", ascending=False).reset_index(drop=True)
    df.to_csv(C.RESULTS / f"tuning_table_{a.variant}.csv", index=False)
    best = df.iloc[0]
    cfg = dict(lr=float(best.lr), bs=int(best.bs), wd=float(best.wd), epochs=int(best.epochs),
               val_macro_mean=float(best.val_macro_mean), selected_on=f"5 tuning splits, variant {a.variant}")
    (C.RESULTS / f"chosen_config_{a.variant}.json").write_text(json.dumps(cfg, indent=1))
    print("BEST", cfg, f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
