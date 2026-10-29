"""B1 (WDCNN on the fixed-Hz envelope spectrum) on CWRU; recipe and deviations pre-declared in results/cwru/PREDECLARED_B1_CWRU.txt.
Usage: python -I b1_run_cwru.py --runs 3-52 [--tag ""]   (each run has scenarios DE and FE; each (run, scenario) trained once).
Output: results/cwru/pred_b1_r<lo>-<hi><tag>.parquet (same columns and row order as the LF-PNCN prediction files) and per_split_b1_r<lo>-<hi><tag>.csv."""
import argparse, sys, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN

OUT = WORK / "results" / "cwru"
SPL = WORK / "bearing-data-leakage" / "data" / "splits" / "cwru" / "eval"
EPOCHS, BS, LR, GAIN_STD, MULT = 30, 32, 1e-2, 0.3, 5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--tag", default=""); a = ap.parse_args()
    if not (OUT / "PREDECLARED_B1_CWRU.txt").exists():
        raise SystemExit("pre-declaration missing")
    lo, hi = map(int, a.runs.split("-"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    z = np.load(WORK / "data" / "cwru" / "cwru_windows.npz", allow_pickle=False)
    Y, wid, grp, sig, load = z["Y"], z["wid"], z["group"], z["sig_loc"], z["load"]
    bank = torch.from_numpy(np.load(WORK / "data" / "cwru" / "b1_env_bank.npy")).to(dev)         # (5328, 6000) float32
    assert bank.shape[0] == len(wid)
    faulty_loc = np.array([g[:2] for g in grp])
    healthy = sig != faulty_loc
    unit = np.where(healthy, np.char.add("H_", grp), grp)
    typ = np.array([g.split("_")[1] for g in grp])
    ball = ((~healthy) & (typ == "B")).astype(np.float32)
    lab = np.stack([Y[:, 0], Y[:, 1], ball], 1).astype(np.float32)                               # inner, outer, ball
    preds, rows, t0 = [], [], time.time()
    for run in range(lo, hi + 1):
        for scen in ("DE", "FE"):
            tr_w = set(pd.read_parquet(SPL / f"train_{scen}_run_{run}.parquet")["waveform_id"])
            te_w = set(pd.read_parquet(SPL / f"test_{scen}_run_{run}.parquet")["waveform_id"])
            tr, te = np.isin(wid, list(tr_w)), np.isin(wid, list(te_w))
            assert not (tr_w & te_w) and not np.any(tr & te)
            assert not (set(grp[tr]) & set(grp[te])), "configuration shared between train and test"
            assert not (set(sig[tr & healthy]) & set(sig[te & healthy])), "healthy bearing shared between train and test"
            sigs = [w for w in dict.fromkeys(wid[tr].tolist())]                                  # training signals, cache order
            s_rows = [np.where(wid == w)[0] for w in sigs]
            s_lab = torch.from_numpy(np.stack([lab[r[0]] for r in s_rows])).to(dev)
            seed = run if scen == "DE" else run + 1000
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = WDCNN(6000, 3).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            n = len(sigs)
            for ep in range(EPOCHS):
                model.train()
                perm = rng.permutation(n * MULT) % n
                nb = len(perm) // BS
                perm = perm[: nb * BS]
                grow = np.array([s_rows[p][rng.integers(0, len(s_rows[p]))] for p in perm])
                gains = np.abs(rng.normal(1.0, GAIN_STD, size=len(perm))).astype(np.float32)
                rows_t, gains_t, perm_t = (torch.from_numpy(v).to(dev) for v in (grow, gains, perm))
                for b in range(nb):
                    sl = slice(b * BS, (b + 1) * BS)
                    xb = (bank[rows_t[sl]] * gains_t[sl, None]).unsqueeze(1)
                    loss = F.binary_cross_entropy_with_logits(model(xb), s_lab[perm_t[sl]])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            he = np.where(te)[0]
            with torch.no_grad():
                S = torch.cat([model(bank[torch.from_numpy(he[i:i + 256]).to(dev)].unsqueeze(1)) for i in range(0, len(he), 256)]).float().cpu().numpy()
            a_in, a_out = roc_auc_score(Y[he, 0], S[:, 0]), roc_auc_score(Y[he, 1], S[:, 1])
            preds.append(pd.DataFrame({"run": run, "scenario": scen, "split": f"{run}_{scen}", "unit": unit[he], "group": grp[he], "healthy": healthy[he],
                                       "sig_loc": sig[he], "load": load[he].astype(int), "y_inner": Y[he, 0].astype(int), "y_outer": Y[he, 1].astype(int),
                                       "s_inner": S[:, 0], "s_outer": S[:, 1]}))
            rows.append(dict(run=run, scenario=scen, auroc_inner=a_in, auroc_outer=a_out, macro=(a_in + a_out) / 2, n_train_signals=n))
            print(f"b1 run {run} {scen}: macro {50*(a_in+a_out):.1f} (in {100*a_in:.1f} out {100*a_out:.1f}) n_train_sig {n} t={time.time()-t0:.0f}s", flush=True)
    tag = f"b1_r{lo}-{hi}{a.tag}"
    pd.concat(preds, ignore_index=True).to_parquet(OUT / f"pred_{tag}.parquet")
    pd.DataFrame(rows).to_csv(OUT / f"per_split_{tag}.csv", index=False)
    print(f"DONE {tag}: macro {100*np.mean([r['macro'] for r in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
