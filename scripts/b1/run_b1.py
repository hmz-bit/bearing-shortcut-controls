"""Run B1 (WDCNN, envelope spectrum) for: t1val (validation on 6 Tier-1 splits), t1shuf (shuffled-label sanity),
t3 (cross-speed), t2 (artificial -> real). Resumable: one parquet shard + json per split in results/b1/shards/<tag>/.
usage: python -I run_b1.py --tier t1val|t1shuf|t3|t2 [--splits 6,8,...] [--workers 8] [--epochs 30] [--tag name] [--mult 1]
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import pu_data as pu

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--splits", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=core.EPOCHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tag = a.tier + (("_" + a.tag) if a.tag else "")
    SH = os.path.join(core.OUT, "shards", tag)
    os.makedirs(SH, exist_ok=True)
    logf = open(os.path.join(core.OUT, f"log_{tag}.txt"), "a", buffering=1)

    def log(s):
        print(s, flush=True); logf.write(s + "\n")

    data = core.Data()
    dev = "cuda"
    pool = core.make_pool(data, a.workers)
    T3_CONDS = ["N15_M07_F10", "N09_M07_F10"]

    jobs = []
    if a.tier in ("t1val", "t1shuf", "t3"):
        if a.splits:
            ids = [int(s) for s in a.splits.split(",")]
        elif a.tier == "t1val":
            ids = [5, 25, 45, 65, 85, 104]
        elif a.tier == "t1shuf":
            ids = [5, 25, 45]
        else:
            ids = list(range(6, 105, 2))
        for r in ids:
            tr, te = pu.load_eval_split(r)
            jobs.append(dict(id=r, train=tr, test=te, tconds=T3_CONDS[:1] if a.tier == "t3" else None,
                             shuffle=(a.tier == "t1shuf")))
    elif a.tier == "t2":
        pairs = list(itertools.combinations(core.HEALTHY, 2))
        ids = [int(s) for s in a.splits.split(",")] if a.splits else list(range(1, 16))
        for k in ids:
            held = list(pairs[k - 1])
            tr = core.ARTIFICIAL + [h for h in core.HEALTHY if h not in held]
            te = core.REAL_OUTER + core.REAL_INNER + held
            jobs.append(dict(id=k, train=tr, test=te, tconds=None, shuffle=False, held=",".join(held)))

    log(f"== {tag}: {len(jobs)} splits, epochs {a.epochs}, workers {a.workers}, mult {a.mult}, seed {a.seed}")
    t_all = time.time()
    for j in jobs:
        out = os.path.join(SH, f"split_{j['id']}.parquet")
        if os.path.exists(out):
            continue
        t0 = time.time()
        tr, te = j["train"], j["test"]
        assert not (set(tr) & set(te)), "train/test bearing overlap"
        tf = data.files(tr, j["tconds"])
        lab = data.fY[tf].copy()
        if j["shuffle"]:
            lab = lab[np.random.default_rng(1000 + j["id"]).permutation(len(lab))]
        std = data.train_std(tf)           # training files only
        assert set(data.meta.bearing.values[tf]) <= set(tr)
        log(f"split {j['id']}: train bearings {len(tr)}, files {len(tf)}, std_train {std:.5f}")
        model = core.train_model(data, pool, tf, lab, std, seed=a.seed, epochs=a.epochs, device=dev, log=log, mult=a.mult)
        t_train = time.time() - t0
        rows = data.windows(te, T3_CONDS) if a.tier == "t3" else data.windows(te)
        assert not (set(data.bearing[rows]) & set(tr))
        lg = core.score_windows(model, data, rows, std, device=dev)
        extra = {"held_healthy": j["held"]} if "held" in j else None
        df = core.long_frame(data, rows, lg, j["id"], extra)
        df.to_parquet(out, index=False)
        info = dict(split=j["id"], n_train_files=int(len(tf)), n_test_windows=int(len(rows)), std_train=std,
                    train_seconds=round(t_train, 1), total_seconds=round(time.time() - t0, 1),
                    overlap=0, train_bearings=tr, test_bearings=te)
        json.dump(info, open(os.path.join(SH, f"split_{j['id']}.json"), "w"))
        log(f"split {j['id']} done in {time.time() - t0:.1f}s (train {t_train:.1f}s)")
        del model
        torch.cuda.empty_cache()
    log(f"== {tag} finished in {time.time() - t_all:.0f}s")
    pool.close()
