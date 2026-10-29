"""B1 (WDCNN on the fixed-Hz envelope spectrum) on the Paderborn run-to-failure windows; recipe and deviations in results/rtf/PREDECLARED_RTF.txt item 7.
Usage: python -I run_rtf_b1.py <first> <last> [perm]   (perm = bearing-label permutation control, 2 permutations per split, splits 5..24)
Output: results/rtf/pred_b1_r<first>-<last>.parquet (or pred_perm_b1_*) + per_split_*.csv (held-out and train macro AUROC)."""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN
import rtf_common as C

EPOCHS, BS, LR, GAIN_STD, PER_EPOCH = 30, 32, 1e-3, 0.3, 800


def macro(Y, S):
    a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    return (a + b) / 2, a, b


@torch.no_grad()
def score(model, E, rows, std, dev):
    model.eval()
    out = [model((E[torch.from_numpy(rows[i:i + 128]).to(dev)] / std).unsqueeze(1)) for i in range(0, len(rows), 128)]
    return torch.cat(out).float().cpu().numpy()


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else ""
    perm = mode in ("perm", "permw")
    if not (C.RES / "PREDECLARED_RTF.txt").exists():
        raise SystemExit("pre-declaration missing")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    w = C.load_windows()
    Y, exp, stage, M = w["Y"], w["exp"], w["stage"], w["M"]
    E = torch.from_numpy(np.load(C.DATA / "rtf_env.npy")).to(dev)
    splits = C.load_splits()
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        tr_ids, te_ids = splits[run]
        assert not set(tr_ids) & set(te_ids)
        for k in ((0, 1) if perm else (-1,)):
            Yr = Y
            if perm:
                Yr = (C.perm_Y_whole if mode == "permw" else C.perm_Y)(Y, exp, stage, run, k, tr_ids, te_ids)
                if Yr is None:
                    rows.append(dict(split=run, perm=k, skipped=1)); continue
            tr_idx, he_idx = np.where(np.isin(exp, tr_ids))[0], np.where(np.isin(exp, te_ids))[0]
            n = w["raw_n"][tr_idx].sum(); mean = w["raw_sum"][tr_idx].sum() / n
            std = float(np.sqrt(w["raw_sumsq"][tr_idx].sum() / n - mean * mean))
            seed = run + (1000 * (k + 1) if perm else 0)
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = WDCNN(21000, 2).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            Yt = torch.from_numpy(Yr).to(dev)
            for ep in range(EPOCHS):
                model.train()
                sel = rng.permutation(len(tr_idx))[:PER_EPOCH]
                rows_ep = tr_idx[sel]
                gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(sel))).astype(np.float32)
                g_t = torch.from_numpy(gains / std).to(dev)
                r_t = torch.from_numpy(rows_ep).to(dev)
                for b in range(len(sel) // BS):
                    sl = slice(b * BS, (b + 1) * BS)
                    xb = (E[r_t[sl]] * g_t[sl, None]).unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(model(xb), Yt[r_t[sl]])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            S_te = score(model, E, he_idx, std, dev)
            S_tr = score(model, E, tr_idx, std, dev)
            hm, hi_, ho = macro(Yr[he_idx], S_te)
            tm, _, _ = macro(Yr[tr_idx], S_tr)
            if not perm:
                preds.append(pd.DataFrame({"split": run, "bearing": exp[he_idx], "stage": stage[he_idx], "M": M[he_idx], "y_inner": Yr[he_idx, 0].astype(int),
                                           "y_outer": Yr[he_idx, 1].astype(int), "s_inner": S_te[:, 0], "s_outer": S_te[:, 1]}))
            rows.append(dict(split=run, perm=k, skipped=0, auroc_inner=hi_, auroc_outer=ho, macro=hm, train_macro=tm))
            print(f"b1{' perm'+str(k) if perm else ''} split {run}: held {100*hm:.1f} (IR {100*hi_:.1f} OR {100*ho:.1f}) train {100*tm:.1f} std {std:.3f} t={time.time()-t0:.0f}s", flush=True)
    tag = f"{(mode + '_') if perm else ''}b1_r{lo}-{hi}"
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(C.RES / f"pred_{tag}.parquet")
    pd.DataFrame(rows).to_csv(C.RES / f"per_split_{tag}.csv", index=False)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
