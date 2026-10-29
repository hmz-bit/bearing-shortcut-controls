"""Shared code for the Tier 1 experiment (bearing-wise, real damage, Paderborn PU).

Works with `python -I`: the folders of the helper modules are added to sys.path here.
Nothing from bearing-data-leakage is executed or unpickled; only parquet split files are read (via evaltools.pu_data).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (HERE, WORK / "scripts" / "evaltools", WORK / "scripts" / "lfpncn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import pu_data as pu          # evaltools
from model import LFPNCN      # lfpncn

CACHE = WORK / "data" / "cache" / "pu_windows.npz"
RESULTS = WORK / "results" / "tier1"
TUNING_RUNS = [0, 1, 2, 3, 4]
VIEIRA_BEARINGS = ["K001", "K002", "K003", "K004", "K005", "K006",
                   "KA04", "KA15", "KA16", "KA22", "KA30",
                   "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
EPOCH_CHOICES = (20, 40)


def load_cache(variant: str = "t_mad"):
    """Return dict with X (N,3,15,64) float32 for `variant`, b3 (N,20), Y (N,2)=[inner,outer], bearing, fname, win.
    Only the 17 Vieira et al. bearings are kept."""
    z = np.load(CACHE, allow_pickle=False)
    bearing = z["bearing"]
    keep = np.isin(bearing, VIEIRA_BEARINGS)
    d = {"bearing": bearing[keep], "fname": z["fname"][keep], "win": z["win"][keep], "b3": z["b3"][keep]}
    d["X"] = z[variant][keep] if variant in z.files else None
    lab = np.array([pu.bearing_labels(b) for b in d["bearing"]], dtype=np.float32)
    d["Y"] = lab
    assert set(np.unique(d["bearing"])) == set(VIEIRA_BEARINGS)
    return d


def load_split(kind: str, run: int):
    """(train_ids, held_ids) for a tuning (val) or eval (test) split."""
    if kind == "eval":
        return pu.load_eval_split(run)
    row = pd.read_parquet(pu.TUNING_DIR / f"run_{run}.parquet").iloc[0]
    return list(row["train_ids"]), list(row["val_ids"])


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_standardiser(Xtr: torch.Tensor):
    """Per-channel (mode x row = 45 channels) mean/std over TRAINING windows and the 64 positions only."""
    mean = Xtr.mean(dim=(0, 3), keepdim=True)
    std = Xtr.std(dim=(0, 3), keepdim=True).clamp_min(1e-6)
    return mean, std


def macro_auroc_np(Y, S):
    a_in = roc_auc_score(Y[:, 0], S[:, 0])
    a_out = roc_auc_score(Y[:, 1], S[:, 1])
    return (a_in + a_out) / 2, a_in, a_out


@torch.no_grad()
def predict(model, X, bs=1024):
    """Logits reordered to [s_inner, s_outer] (network outputs are OR, IR, Ball)."""
    model.eval()
    out = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)], dim=0)
    return torch.stack([out[:, 1], out[:, 0]], dim=1).cpu().numpy().astype(np.float64)


def train_lfpncn(X, Y, bearing, train_ids, cfg, seed, checkpoints=None, held_ids=None, dev=None,
                 shuffle_labels=None, return_model=False, test_X_override=None):
    """Train LF-PNCN on the windows of `train_ids` bearings; evaluate on `held_ids` bearings.

    X: numpy (N,3,15,64); Y: numpy (N,2)=[inner,outer]; cfg: dict(lr, bs, wd, epochs).
    The standardiser is fitted on the training windows only. The Ball output is excluded from the loss
    (PU has no ball faults). checkpoints: epochs at which to evaluate on held windows (default: [cfg.epochs]).
    shuffle_labels: None | 'window' (permute training labels across windows) | 'bearing' (permute labels across bearings).
    Returns dict(results={epoch: (macro, inner, outer, scores)}, train_seconds, mean, std, held_mask, ...).
    """
    dev = dev or device()
    tr = np.isin(bearing, train_ids)
    he = np.isin(bearing, held_ids) if held_ids is not None else np.zeros(len(bearing), bool)
    assert not (set(train_ids) & set(held_ids or [])), "train/held bearing overlap"
    assert not np.any(tr & he)
    Ytr = Y[tr].copy()
    rng = np.random.default_rng(seed)
    if shuffle_labels == "window":
        Ytr = Ytr[rng.permutation(len(Ytr))]
    elif shuffle_labels == "bearing":
        btr = bearing[tr]
        ub = np.array(sorted(set(btr)))
        perm = rng.permutation(len(ub))
        lab_of = {b: Y[bearing == b][0] for b in ub}
        newlab = {b: lab_of[ub[p]] for b, p in zip(ub, perm)}
        Ytr = np.stack([newlab[b] for b in btr])
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    Xtr = torch.from_numpy(X[tr]).to(dev)
    mean, std = fit_standardiser(Xtr)           # TRAINING windows only
    Xtr = (Xtr - mean) / std
    Xh = None
    if he.any():
        Xh_raw = torch.from_numpy(X[he] if test_X_override is None else test_X_override).to(dev)
        Xh = (Xh_raw - mean) / std
    # network outputs: col0 = OR (outer label), col1 = IR (inner label); Ball column never enters the loss
    T = torch.from_numpy(Ytr[:, [1, 0]]).to(dev)
    model = LFPNCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n, bs = len(Xtr), cfg["bs"]
    checkpoints = sorted(set(checkpoints or [cfg["epochs"]]))
    results = {}
    t0 = time.time()
    for ep in range(1, max(checkpoints) + 1):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n - bs + 1, bs):          # drop_last: avoids a size-1 batch in BatchNorm
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])[:, :2]
            loss = F.binary_cross_entropy_with_logits(logits, T[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in checkpoints and Xh is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            S = predict(model, Xh)
            m, a_in, a_out = macro_auroc_np(Y[he], S)
            results[ep] = dict(macro=m, inner=a_in, outer=a_out, scores=S)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    out = dict(results=results, train_seconds=time.time() - t0, mean=mean.cpu().numpy(), std=std.cpu().numpy(),
               held_mask=he, train_mask=tr, n_train=int(tr.sum()))
    if return_model:
        out.update(model=model, Xtr=Xtr, Ytr=Ytr)
    return out


def set_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
