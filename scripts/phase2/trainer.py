"""Generalised training loop for LF-PNCN variants (phase 2).

With default options it is the Tier 1 loop line for line (same init order, drop_last, Adam, BCE on OR/IR columns only,
standardiser from training windows only) and reproduces `tier1/common.train_lfpncn`.

Options (dict `opt`):
  rows        : number of tensor rows (channels of the first conv); 15 for the original input, 30 for multi-resolution
  rowz        : like rowcenter but also divide by 1.4826*MAD+0.05 of the row (per-window row z-score)
  rowcenter   : subtract the per-window, per-row median over the 64 xi positions (test-time per-window baseline removal; applied identically to train and test)
  jitter_pts  : max integer shift (in xi grid points, uniform in [-k,k]) of the whole window along the order axis in training (speed jitter;
                the grid spans +-3 % of h*f_m, so +-2 % speed error = +-21 points)
  gain        : (lo, hi) random multiplicative gain on the raw log-SNR before standardisation in training, or None
  pos_weight  : True -> class-balanced BCE (pos_weight = n_neg/n_pos per label from the training windows)
  extra       : None or numpy (N,k) side features (e.g. time-domain statistics); standardised on training windows, concatenated before the heads
  dropout     : dropout probability on the 96-d embedding before the heads (0 = none)
"""
import time
import numpy as np
import torch
import torch.nn.functional as TF
from torch import nn

import p2common as P
import common as C


def _stem(rows):
    return nn.Sequential(
        nn.Conv1d(rows, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    )


class Net(nn.Module):
    def __init__(self, rows=15, n_extra=0, dropout=0.0):
        super().__init__()
        self.stems = nn.ModuleList([_stem(rows) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(96 + n_extra, 1) for _ in range(3)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, e=None):
        z = torch.cat([s(x[:, i]) for i, s in enumerate(self.stems)], dim=1)
        z = self.drop(z)
        if e is not None:
            z = torch.cat([z, e], dim=1)
        return torch.cat([h(z) for h in self.heads], dim=1)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _rowcenter(x, z=False):
    """Per-window, per-row baseline removal: x - median over xi. z=True also divides by 1.4826*MAD + 0.05 (per-window row z-score)."""
    med = x.median(dim=-1, keepdim=True).values
    y = x - med
    if z:
        mad = 1.4826 * y.abs().median(dim=-1, keepdim=True).values
        y = y / (mad + 0.05)
    return y


def _shift(x, k):
    """x (B,3,R,64); k (B,) integer shifts in [-K,K]; edge-replicated shift along the last axis."""
    B, _, _, W = x.shape
    idx = (torch.arange(W, device=x.device)[None, :] - k[:, None]).clamp(0, W - 1)      # (B,W)
    return torch.gather(x, 3, idx[:, None, None, :].expand(-1, x.shape[1], x.shape[2], -1))


def train_eval(X, Y, bearing, train_ids, held_ids, cfg, seed, opt=None, dev=None, test_X=None, test_extra=None,
               train_mask=None, held_mask=None, return_model=False, score_train=False, extra_held=None):
    """Train on windows of `train_ids` (or train_mask), score `held_ids` windows (or held_mask).
    Returns dict(scores (n_held,2) [s_inner, s_outer] logits, macro, inner, outer, seconds, ...)."""
    opt = opt or {}
    dev = dev or P.device()
    tr = np.isin(bearing, train_ids) if train_mask is None else train_mask
    he = np.isin(bearing, held_ids) if held_mask is None else held_mask
    assert not np.any(tr & he), "train/held overlap"
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    Xtr = torch.from_numpy(X[tr]).to(dev)
    Xh = torch.from_numpy(X[he] if test_X is None else test_X).to(dev)
    zr = bool(opt.get("rowz"))
    if opt.get("rowcenter") or zr:
        Xtr, Xh = _rowcenter(Xtr, zr), _rowcenter(Xh, zr)
    mean, std = C.fit_standardiser(Xtr)
    Ytr = torch.from_numpy(Y[tr][:, [1, 0]]).to(dev)         # OR (outer), IR (inner)
    ex_tr = ex_h = None
    if opt.get("extra") is not None:
        E = opt["extra"]
        etr = torch.from_numpy(E[tr]).to(dev)
        em, es = etr.mean(0, keepdim=True), etr.std(0, keepdim=True).clamp_min(1e-6)
        ex_tr = (etr - em) / es
        eh = torch.from_numpy(E[he] if test_extra is None else test_extra).to(dev)
        ex_h = (eh - em) / es
    model = Net(X.shape[2], 0 if ex_tr is None else ex_tr.shape[1], opt.get("dropout", 0.0)).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pw = None
    if opt.get("pos_weight"):
        npos = Ytr.sum(0)
        pw = ((len(Ytr) - npos) / npos.clamp_min(1)).to(dev)
    jit = int(opt.get("jitter_pts", 0))
    gain = opt.get("gain")
    n, bs = len(Xtr), cfg["bs"]
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]
            if gain is not None:
                g = torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(gain[0], gain[1], generator=gen)
                xb = xb * g
            if jit > 0:
                k = torch.randint(-jit, jit + 1, (len(idx),), device=dev, generator=gen)
                xb = _shift(xb, k)
            xb = (xb - mean) / std
            logits = model(xb, None if ex_tr is None else ex_tr[idx])[:, :2]
            loss = TF.binary_cross_entropy_with_logits(logits, Ytr[idx], pos_weight=pw)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()

    @torch.no_grad()
    def score(Xq, Eq):
        Xq = (Xq - mean) / std
        out = torch.cat([model(Xq[i:i + 1024], None if Eq is None else Eq[i:i + 1024]) for i in range(0, len(Xq), 1024)], 0)
        return torch.stack([out[:, 1], out[:, 0]], 1).cpu().numpy().astype(np.float64)   # [s_inner, s_outer]

    S = score(Xh, ex_h)
    r = dict(scores=S, seconds=secs, held_mask=he, train_mask=tr, n_train=int(tr.sum()))
    if held_ids is not None or held_mask is not None:
        m, ai, ao = C.macro_auroc_np(Y[he], S)
        r.update(macro=m, inner=ai, outer=ao)
    if score_train:
        r["train_scores"] = score(Xtr, ex_tr)
    if extra_held:      # additional held-out masks (e.g. Tier-3-style evaluation at another speed): name -> mask
        r["extra"] = {}
        for nm, m in extra_held.items():
            assert not np.any(m & tr), "train/extra-held overlap"
            xq = torch.from_numpy(X[m]).to(dev)
            if opt.get("rowcenter") or zr:
                xq = _rowcenter(xq, zr)
            eq = None
            if ex_tr is not None:
                eq = (torch.from_numpy(opt["extra"][m]).to(dev) - em) / es
            Sq = score(xq, eq)
            mq, aiq, aoq = C.macro_auroc_np(Y[m], Sq)
            r["extra"][nm] = dict(scores=Sq, macro=mq, inner=aiq, outer=aoq)
    if return_model:
        r["model"] = model
    return r
