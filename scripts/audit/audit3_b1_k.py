"""Part K, B1 addendum of results/audit3/PREDECLARED_AUDIT3.txt: B1 on the faulty-stage windows only.
Usage: python -I audit3_b1_k.py <xjtu|rtf>    Output: results/audit3/pred_k_<ds>_b1.parquet, per_split_k_<ds>_b1.csv"""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (WORK / "scripts" / "xjtu", WORK / "scripts" / "rtf", WORK / "scripts" / "b1"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN

OUT = WORK / "results" / "audit3"
EPOCHS, BS, LR, GAIN_STD = 30, 32, 1e-3, 0.3


def macro(Y, S):
    a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    return 100 * (a + b) / 2, 100 * a, 100 * b


@torch.no_grad()
def score(model, getx, rows, dev):
    model.eval()
    return torch.cat([model(getx(torch.from_numpy(rows[i:i + 128]).to(dev)).unsqueeze(1)) for i in range(0, len(rows), 128)]).float().cpu().numpy()


def main():
    ds = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if ds == "xjtu":
        import xjtu_common as C
        SCALE, PER_EPOCH = 100.0, 864
        E = torch.from_numpy(np.load(C.DATA / "xjtu_env.npy")).to(dev)        # (N,3,21000)
    else:
        import rtf_common as C
        SCALE, PER_EPOCH = 1.0, 800
        E = torch.from_numpy(np.load(C.DATA / "rtf_env.npy")).to(dev)         # (N,21000)
    w = C.load_windows(); sp = C.load_splits()
    Y, exp, stage = w["Y"], w["exp"], w["stage"]
    fm = stage == "faulty"
    preds, rows, skipped, t0 = [], [], 0, time.time()
    for run in range(5, 105):
        tr_ids, te_ids = sp[run]
        tr_idx = np.where(np.isin(exp, tr_ids) & fm)[0]; he_idx = np.where(np.isin(exp, te_ids) & fm)[0]
        if any(len(np.unique(Y[he_idx][:, c])) < 2 or len(np.unique(Y[tr_idx][:, c])) < 2 for c in (0, 1)):
            skipped += 1; continue
        n = w["raw_n"][tr_idx].sum(); mean = w["raw_sum"][tr_idx].sum() / n
        std = float(np.sqrt(w["raw_sumsq"][tr_idx].sum() / n - mean * mean))
        torch.manual_seed(run); rng = np.random.default_rng(run)
        model = WDCNN(21000, 2).to(dev); opt = torch.optim.Adam(model.parameters(), lr=LR); Yt = torch.from_numpy(Y).to(dev)
        if ds == "xjtu":
            pool = np.array([(i, o) for i in tr_idx for o in range(3)])
            per_epoch = min(PER_EPOCH, (len(pool) // BS) * BS)
            getx_te = lambda r: E[r, 0].float() / (SCALE * std)
        else:
            per_epoch = PER_EPOCH
            getx_te = lambda r: E[r] / std
        for ep in range(EPOCHS):
            model.train()
            if ds == "xjtu":
                sel = pool[rng.permutation(len(pool))[:per_epoch]]
                gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(sel))).astype(np.float32)
                g_t = torch.from_numpy(gains / (SCALE * std)).to(dev)
                i_t, o_t = torch.from_numpy(sel[:, 0]).to(dev), torch.from_numpy(sel[:, 1]).to(dev)
                for b in range(len(sel) // BS):
                    sl = slice(b * BS, (b + 1) * BS)
                    xb = (E[i_t[sl], o_t[sl]].float() * g_t[sl, None]).unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(model(xb), Yt[i_t[sl]])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            else:
                sel = rng.permutation(len(tr_idx))[:per_epoch]
                rows_ep = tr_idx[sel]
                gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(sel))).astype(np.float32)
                g_t = torch.from_numpy(gains / std).to(dev); r_t = torch.from_numpy(rows_ep).to(dev)
                for b in range(len(sel) // BS):
                    sl = slice(b * BS, (b + 1) * BS)
                    xb = (E[r_t[sl]] * g_t[sl, None]).unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(model(xb), Yt[r_t[sl]])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        S = score(model, getx_te, he_idx, dev)
        m, a, b_ = macro(Y[he_idx], S)
        preds.append(pd.DataFrame({"split": run, "bearing": exp[he_idx], "stage": stage[he_idx], "y_inner": Y[he_idx, 0].astype(int), "y_outer": Y[he_idx, 1].astype(int),
                                   "s_inner": S[:, 0], "s_outer": S[:, 1]}))
        rows.append(dict(split=run, inner=a, outer=b_, macro=m))
        print(f"b1 K {ds} split {run}: macro {m:.1f} (IR {a:.1f} OR {b_:.1f}) t={time.time()-t0:.0f}s", flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_k_{ds}_b1.parquet")
    pd.DataFrame(rows).to_csv(OUT / f"per_split_k_{ds}_b1.csv", index=False)
    print(f"DONE K {ds} b1: macro {np.mean([r['macro'] for r in rows]):.2f} inner {np.mean([r['inner'] for r in rows]):.2f} outer {np.mean([r['outer'] for r in rows]):.2f} defined {len(rows)} skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
