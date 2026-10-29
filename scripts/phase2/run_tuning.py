"""Tuning-split evaluation of phase 2 variants (5 tuning splits ONLY; evaluation splits are never loaded here).
Protocol = Tier 1 grid protocol at the Tier-1-chosen configuration (lr 1e-3, bs 64, wd 1e-4, 40 epochs; override per variant is possible
through VARIANTS[name]['cfg']): train on the 11 training bearings, score the 6 validation bearings, seed = 1000 + run + 5000*seed_index.
Usage: python -I run_tuning.py base L2 ML14 --seeds 1 [--tag x]  -> results/phase2/tuning/<name>_s<k>.parquet + tuning_results.csv (appended)
"""
import argparse, json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p2common as P
import trainer as T
import variants as V

TUN = P.OUT / "tuning"


def run_variant(name, seeds, dev, meta, cache_X, quiet=False, tier3=False):
    v = V.VARIANTS[name]
    X = cache_X.setdefault(v["feat"], V.load_feat(v["feat"]))
    opt = dict(v["opt"])
    if opt.get("extra") == "tstats":
        opt["extra"] = V.load_extra("tstats", meta)
    cfg = dict(P.BASE_CFG); cfg.update(v.get("cfg", {}))
    out = []
    for k in seeds:
        preds, rows = [], []
        for run in P.TUNING_RUNS:
            tr_ids, va_ids = P.load_split("tuning", run)
            if tier3:
                b, cnd = meta["bearing"], meta["condition"]
                trm = np.isin(b, tr_ids) & (cnd == P.CONDS[0])
                m15 = np.isin(b, va_ids) & (cnd == P.CONDS[0])
                m09 = np.isin(b, va_ids) & (cnd == P.CONDS[1])
                r = T.train_eval(X, meta["Y"], b, None, None, cfg, seed=1000 + run + 5000 * k, opt=opt, dev=dev,
                                 train_mask=trm, held_mask=m15, extra_held={"n09": m09})
                e = r["extra"]["n09"]
                rows.append(dict(variant=name, seed_idx=k, split=run, same_macro=100 * r["macro"], same_inner=100 * r["inner"], same_outer=100 * r["outer"],
                                 slow_macro=100 * e["macro"], slow_inner=100 * e["inner"], slow_outer=100 * e["outer"], sec=r["seconds"]))
                print(f"T3tune {name} s{k} split {run}: 1500 {100*r['macro']:.1f} (out {100*r['outer']:.1f}) | 900 {100*e['macro']:.1f} (in {100*e['inner']:.1f} out {100*e['outer']:.1f})", flush=True)
                continue
            r = T.train_eval(X, meta["Y"], meta["bearing"], tr_ids, va_ids, cfg, seed=1000 + run + 5000 * k, opt=opt, dev=dev)
            he = r["held_mask"]
            preds.append(pd.DataFrame({"split": run, "bearing": meta["bearing"][he], "condition": meta["condition"][he],
                                       "fname": meta["fname"][he], "win": meta["win"][he],
                                       "y_inner": meta["Y"][he, 0].astype(int), "y_outer": meta["Y"][he, 1].astype(int),
                                       "s_inner": r["scores"][:, 0], "s_outer": r["scores"][:, 1]}))
            rows.append(dict(variant=name, seed_idx=k, split=run, macro=100 * r["macro"], inner=100 * r["inner"], outer=100 * r["outer"],
                             sec=r["seconds"]))
            if not quiet:
                print(f"{name} s{k} split {run}: macro {100*r['macro']:.2f} in {100*r['inner']:.1f} out {100*r['outer']:.1f} {r['seconds']:.0f}s", flush=True)
        TUN.mkdir(parents=True, exist_ok=True)
        if tier3:
            df = pd.DataFrame(rows)
            df.to_csv(TUN / f"{name}_s{k}_t3tune_persplit.csv", index=False)
            print(f"== T3tune {name} seed{k}: 1500 macro {df.same_macro.mean():.2f} | 900 macro {df.slow_macro.mean():.2f} (inner {df.slow_inner.mean():.2f}, outer {df.slow_outer.mean():.2f}) | loss {df.same_macro.mean()-df.slow_macro.mean():.2f}", flush=True)
            out.append(df)
            continue
        pd.concat(preds, ignore_index=True).to_parquet(TUN / f"{name}_s{k}.parquet")
        df = pd.DataFrame(rows)
        df.to_csv(TUN / f"{name}_s{k}_persplit.csv", index=False)
        print(f"== {name} seed{k}: val macro {df.macro.mean():.2f}  inner {df.inner.mean():.2f}  outer {df.outer.mean():.2f}  "
              f"per split {[round(x,1) for x in df.macro]}", flush=True)
        out.append(df)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tier3", action="store_true", help="tuning-split analogue of Tier 3: train 1500 rpm only, test val bearings at 1500 and 900 rpm")
    a = ap.parse_args()
    import common as C
    C.set_determinism()
    dev = P.device()
    meta = P.load_meta()
    cache_X = {}
    for nm in a.names:
        run_variant(nm, list(range(a.seed_start, a.seed_start + a.seeds)), dev, meta, cache_X, tier3=a.tier3)


if __name__ == "__main__":
    main()
