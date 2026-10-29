"""B1 (WDCNN on envelope spectrum) on UORED, recipe and deviations pre-declared in results/uored/PREDECLARED_B1_UORED.txt.
Usage: python -I run_b1_uored.py [first last]   (default splits 5..104). Output: results/uored/pred_b1.parquet, per_split_b1.csv."""
import sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN

OUT = WORK / "results" / "uored"
SPLITS = WORK / "bearing-data-leakage" / "data" / "splits" / "uored-vafcls" / "eval"
EPOCHS, BS, LR, GAIN_STD, MULT, SCALE = 30, 16, 1e-4, 0.3, 20, 1000.0


def main():
    if not (OUT / "PREDECLARED_B1_UORED.txt").exists():
        raise SystemExit("pre-declaration missing")
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (5, 104)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    z = np.load(WORK / "data" / "uored" / "uored_windows.npz", allow_pickle=False)
    fname, bearing, cls, state, Y = z["fname"], z["bearing"], z["cls"], z["state"], z["Y"]
    bank = torch.from_numpy(np.load(WORK / "data" / "uored" / "b1_env_bank.npy")).to(dev)           # (N, 21000) float16 x1000
    st = pd.read_csv(WORK / "data" / "uored" / "b1_file_stats.csv").set_index("fname")
    files = list(dict.fromkeys(fname.tolist()))
    f_rows = {f: np.where(fname == f)[0] for f in files}
    f_bearing = np.array([int(bearing[f_rows[f][0]]) for f in files])
    f_cls = np.array([cls[f_rows[f][0]] for f in files])
    f_lab = np.stack([(f_cls == c) for c in "IOBC"], 1).astype(np.float32)                           # inner, outer, ball, cage
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        sp = pd.read_parquet(SPLITS / f"run_{run}.parquet").iloc[0]
        tr_ids, te_ids = [int(i) for i in sp["train_ids"]], [int(i) for i in sp["test_ids"]]
        assert not set(tr_ids) & set(te_ids)
        tf = np.where(np.isin(f_bearing, tr_ids))[0]
        n = st.loc[[files[i] for i in tf]]
        mean = n.raw_sum.sum() / n.n_raw.sum()
        std = float(np.sqrt(n.raw_sumsq.sum() / n.n_raw.sum() - mean * mean))
        torch.manual_seed(run); rng = np.random.default_rng(run)
        model = WDCNN(21000, 4).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lab_t = torch.from_numpy(f_lab[tf]).to(dev)
        for ep in range(EPOCHS):
            model.train()
            perm = rng.permutation(len(tf) * MULT) % len(tf)
            nb = len(perm) // BS
            perm = perm[: nb * BS]
            win = np.array([rng.integers(0, len(f_rows[files[tf[p]]])) for p in perm])
            grow = np.array([f_rows[files[tf[p]]][w] for p, w in zip(perm, win)])
            gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(perm))).astype(np.float32)
            gains_t = torch.from_numpy(gains / (SCALE * std)).to(dev)
            rows_t = torch.from_numpy(grow).to(dev)
            perm_t = torch.from_numpy(perm).to(dev)
            for b in range(nb):
                sl = slice(b * BS, (b + 1) * BS)
                xb = (bank[rows_t[sl]].float() * gains_t[sl, None]).unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(model(xb), lab_t[perm_t[sl]])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        he = np.where(np.isin(bearing, te_ids))[0]
        with torch.no_grad():
            S = torch.cat([model((bank[torch.from_numpy(he[i:i + 128]).to(dev)].float() / (SCALE * std)).unsqueeze(1)) for i in range(0, len(he), 128)]).float().cpu().numpy()
        assert set(bearing[he].tolist()) == set(te_ids)
        a_in, a_out = roc_auc_score(Y[he, 0], S[:, 0]), roc_auc_score(Y[he, 1], S[:, 1])
        preds.append(pd.DataFrame({"split": run, "bearing": bearing[he].astype(int), "cls": cls[he], "state": state[he].astype(int),
                                   "y_inner": Y[he, 0].astype(int), "y_outer": Y[he, 1].astype(int), "s_inner": S[:, 0], "s_outer": S[:, 1]}))
        rows.append(dict(split=run, auroc_inner=a_in, auroc_outer=a_out, macro=(a_in + a_out) / 2))
        print(f"b1 split {run}: macro {50*(a_in+a_out):.1f} (in {100*a_in:.1f} out {100*a_out:.1f}) std {std:.1f} t={time.time()-t0:.0f}s", flush=True)
        if run % 10 == 0 or run == hi:
            pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_b1_partial.parquet")
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "pred_b1.parquet")
    pd.DataFrame(rows).to_csv(OUT / "per_split_b1.csv", index=False)
    print(f"DONE b1: macro {100*np.mean([r['macro'] for r in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
