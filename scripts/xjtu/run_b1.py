"""B1 (WDCNN on the fixed-Hz envelope spectrum) on the XJTU-SY windows; recipe and deviations in results/xjtu/PREDECLARED_XJTU.txt item 8.
Usage: python -I run_b1.py <first> <last> [perm|sens5]
Output: results/xjtu/pred_[sens5_]b1_r<first>-<last>.parquet (or per_split_perm_b1_*) + per_split_*.csv (held-out and train macro AUROC)."""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN
import xjtu_common as C

EPOCHS, BS, LR, GAIN_STD, PER_EPOCH, SCALE = 30, 32, 1e-3, 0.3, 864, 100.0


def macro(Y, S):
    a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    return (a + b) / 2, a, b


@torch.no_grad()
def score(model, E, rows, scale, dev):
    model.eval()
    out = [model((E[torch.from_numpy(rows[i:i + 128]).to(dev), 0].float() / scale).unsqueeze(1)) for i in range(0, len(rows), 128)]
    return torch.cat(out).float().cpu().numpy()


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else ""
    perm, sens = mode == "perm", mode == "sens5"
    if not (C.RES / "PREDECLARED_XJTU.txt").exists():
        raise SystemExit("pre-declaration missing")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    w = C.load_windows()
    keep = w["sens5"] if sens else np.ones(len(w["Y"]), bool)
    Y, exp, stage, fileno = w["Y"][keep], w["exp"][keep], w["stage"][keep], w["fileno"][keep]
    raw_n, raw_sum, raw_sumsq = w["raw_n"][keep], w["raw_sum"][keep], w["raw_sumsq"][keep]
    E = torch.from_numpy(np.load(C.DATA / "xjtu_env.npy")[keep]).to(dev)          # (N, 3, 21000) float16 x SCALE
    splits = C.load_splits()
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = splits[run]
        assert not set(tr_ids) & set(te_ids)
        for k in ((0, 1) if perm else (-1,)):
            Yr = Y
            if perm:
                Yr = C.perm_Y(Y, exp, stage, run, k, tr_ids, te_ids)
                if Yr is None:
                    rows.append(dict(split=run, perm=k, skipped=1)); continue
            tr_idx, he_idx = np.where(np.isin(exp, tr_ids))[0], np.where(np.isin(exp, te_ids))[0]
            n = raw_n[tr_idx].sum(); mean = raw_sum[tr_idx].sum() / n
            std = float(np.sqrt(raw_sumsq[tr_idx].sum() / n - mean * mean))
            seed = run + (1000 * (k + 1) if perm else 0)
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            pool = np.array([(i, o) for i in tr_idx for o in range(3)])
            per_epoch = min(PER_EPOCH, (len(pool) // BS) * BS)
            model = WDCNN(21000, 2).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            Yt = torch.from_numpy(Yr).to(dev)
            for ep in range(EPOCHS):
                model.train()
                sel = pool[rng.permutation(len(pool))[:per_epoch]]
                gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(sel))).astype(np.float32)
                g_t = torch.from_numpy(gains / (SCALE * std)).to(dev)
                i_t, o_t = torch.from_numpy(sel[:, 0]).to(dev), torch.from_numpy(sel[:, 1]).to(dev)
                for b in range(len(sel) // BS):
                    sl = slice(b * BS, (b + 1) * BS)
                    xb = (E[i_t[sl], o_t[sl]].float() * g_t[sl, None]).unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(model(xb), Yt[i_t[sl]])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            S_te = score(model, E, he_idx, SCALE * std, dev)
            S_tr = score(model, E, tr_idx, SCALE * std, dev)
            hm, hi_, ho = macro(Yr[he_idx], S_te)
            tm, _, _ = macro(Yr[tr_idx], S_tr)
            if not perm:
                preds.append(pd.DataFrame({"split": run, "bearing": exp[he_idx], "stage": stage[he_idx], "fileno": fileno[he_idx], "y_inner": Yr[he_idx, 0].astype(int),
                                           "y_outer": Yr[he_idx, 1].astype(int), "s_inner": S_te[:, 0], "s_outer": S_te[:, 1]}))
            rows.append(dict(split=run, perm=k, skipped=0, auroc_inner=hi_, auroc_outer=ho, macro=hm, train_macro=tm))
            print(f"b1{' perm'+str(k) if perm else ''}{' sens5' if sens else ''} split {run}: held {100*hm:.1f} (IR {100*hi_:.1f} OR {100*ho:.1f}) train {100*tm:.1f} std {std:.3f} t={time.time()-t0:.0f}s", flush=True)
    tag = f"{'perm_' if perm else ''}{'sens5_' if sens else ''}b1_r{lo}-{hi}"
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(C.RES / f"pred_{tag}.parquet")
    pd.DataFrame(rows).to_csv(C.RES / f"per_split_{tag}.csv", index=False)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
