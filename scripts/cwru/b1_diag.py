"""Bug diagnostic for B1 on CWRU (not a tuning step; nothing is changed): (1) amplitude of the envelope spectra by signal location and health state,
(2) train-set AUROC and loss of the same training code on three splits, (3) label/score alignment check."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent; WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from wdcnn import WDCNN
SPL = WORK / "bearing-data-leakage/data/splits/cwru/eval"
z = np.load(WORK / "data/cwru/cwru_windows.npz", allow_pickle=False)
Y, wid, grp, sig = z["Y"], z["wid"], z["group"], z["sig_loc"]
bank = np.load(WORK / "data/cwru/b1_env_bank.npy")
fl = np.array([g[:2] for g in grp]); healthy = sig != fl; typ = np.array([g.split("_")[1] for g in grp])
amp = np.sqrt((bank.astype(np.float64) ** 2).sum(1))
d = pd.DataFrame(dict(sig=sig, healthy=healthy, typ=np.where(healthy, "H", typ), amp=amp))
print("median spectrum L2 norm by signal location / class:"); print(d.groupby(["sig", "typ"]).amp.median().unstack().round(4).to_string())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bk = torch.from_numpy(bank).to(dev)
ball = ((~healthy) & (typ == "B")).astype(np.float32)
lab = np.stack([Y[:, 0], Y[:, 1], ball], 1).astype(np.float32)
for run, scen in ((3, "DE"), (3, "FE"), (40, "DE")):
    tr_w = set(pd.read_parquet(SPL / f"train_{scen}_run_{run}.parquet")["waveform_id"]); te_w = set(pd.read_parquet(SPL / f"test_{scen}_run_{run}.parquet")["waveform_id"])
    tr, te = np.isin(wid, list(tr_w)), np.isin(wid, list(te_w))
    sigs = list(dict.fromkeys(wid[tr].tolist())); s_rows = [np.where(wid == w)[0] for w in sigs]
    s_lab = torch.from_numpy(np.stack([lab[r[0]] for r in s_rows])).to(dev)
    seed = run if scen == "DE" else run + 1000
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = WDCNN(6000, 3).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-2); n = len(sigs); losses = []
    for ep in range(30):
        m.train(); perm = rng.permutation(n * 5) % n; nb = len(perm) // 32; perm = perm[: nb * 32]
        grow = np.array([s_rows[p][rng.integers(0, len(s_rows[p]))] for p in perm]); g = np.abs(rng.normal(1, .3, len(perm))).astype(np.float32)
        r_t, g_t, p_t = (torch.from_numpy(v).to(dev) for v in (grow, g, perm)); tot = 0
        for b in range(nb):
            sl = slice(b * 32, (b + 1) * 32); xb = (bk[r_t[sl]] * g_t[sl, None]).unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(m(xb), s_lab[p_t[sl]]); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        losses.append(tot / nb)
    m.eval()
    def sc(mask):
        ix = np.where(mask)[0]
        with torch.no_grad(): return ix, torch.cat([m(bk[torch.from_numpy(ix[i:i+256]).to(dev)].unsqueeze(1)) for i in range(0, len(ix), 256)]).cpu().numpy()
    ixt, St = sc(tr); ixe, Se = sc(te)
    tr_in, tr_out = roc_auc_score(Y[ixt, 0], St[:, 0]), roc_auc_score(Y[ixt, 1], St[:, 1])
    te_in, te_out = roc_auc_score(Y[ixe, 0], Se[:, 0]), roc_auc_score(Y[ixe, 1], Se[:, 1])
    print(f"run {run} {scen}: loss ep1 {losses[0]:.3f} ep10 {losses[9]:.3f} ep30 {losses[-1]:.3f} | train AUROC in {100*tr_in:.1f} out {100*tr_out:.1f} | test in {100*te_in:.1f} out {100*te_out:.1f}")
    # test negatives: which negatives score highest for the outer head?
    dfe = pd.DataFrame(dict(typ=np.where(healthy[ixe], "H_" + sig[ixe], typ[ixe]), y=Y[ixe, 1], s=Se[:, 1]))
    print("  mean outer-head logit on test windows by class:", dfe.groupby("typ").s.mean().round(2).to_dict())
