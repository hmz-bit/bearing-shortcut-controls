"""Shared code for the fixed-Hz ablation (F), Tier 3 (cross-speed) and Tier 2 (artificial to real).

Reuses tier1/common.py (read-only import) and mirrors its training loop line for line (seed handling, drop_last,
optimiser, BatchNorm eval at scoring, Ball column excluded from the loss, standardiser from training windows only).
Works with `python -I`. Nothing from bearing-data-leakage is executed or unpickled.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (HERE, WORK / "scripts" / "tier1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF

import common as C            # tier1 (adds evaltools and lfpncn to sys.path)
import pu_data as pu
from model import LFPNCN

CACHE = WORK / "data" / "cache" / "pu_windows.npz"
CACHE_FIX = WORK / "data" / "cache" / "pu_windows_fixed.npz"
OUT = WORK / "results" / "tier2_3"
CFG = json.loads((WORK / "results" / "tier1" / "chosen_config_t_mad.json").read_text())
COND_SAME, COND_SLOW = "N15_M07_F10", "N09_M07_F10"

ART_OUTER = "KA01 KA03 KA05 KA06 KA07 KA08 KA09".split()
ART_INNER = "KI01 KI03 KI05 KI07 KI08".split()
ARTIFICIAL = ART_OUTER + ART_INNER
HEALTHY = "K001 K002 K003 K004 K005 K006".split()
REAL_OUTER = "KA04 KA15 KA16 KA22 KA30".split()
REAL_INNER = "KI04 KI14 KI16 KI17 KI18 KI21".split()
REAL_FAULTY = REAL_OUTER + REAL_INNER
PAIRS = list(itertools.combinations(HEALTHY, 2))        # 15 held-out healthy pairs, fixed order
NEURAL = {"lfpncn": "t_mad", "b4": "t_max", "f": "t_fix"}


def load_all(variants=("t_mad",), need_b3=False):
    """All 29 bearings (the 17 Vieira ones + 12 artificial). Returns dict with X[variant], b3, Y (inner, outer), bearing, condition, fname, win."""
    z = np.load(CACHE, allow_pickle=False)
    d = {"bearing": z["bearing"], "condition": z["condition"], "fname": z["fname"], "win": z["win"], "X": {}}
    if need_b3:
        d["b3"] = z["b3"]
    zf = None
    for v in variants:
        if v == "t_fix":
            zf = np.load(CACHE_FIX, allow_pickle=False)
            for k in ("fname", "win", "bearing", "condition"):
                assert np.array_equal(zf[k], z[k]), f"fixed cache row order differs ({k})"
            d["X"][v] = zf["t_fix"]
        else:
            d["X"][v] = z[v]
    d["Y"] = np.array([pu.bearing_labels(b) for b in d["bearing"]], dtype=np.float32)
    assert all(fn.split("_")[3] == b for fn, b in zip(d["fname"], d["bearing"])), "fname/bearing mismatch"
    return d


def device():
    """GPU by default; T23_DEVICE=cpu forces CPU (T23_THREADS torch threads, default 1). Used because the single 4 GB GPU is time-sliced
    with other jobs and parallel GPU processes slow each other down by more than 10x."""
    import os
    dev = os.environ.get("T23_DEVICE")
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("T23_THREADS", "1")))
        return torch.device("cpu")
    return C.device()


def train_generic(X, Y, train_mask, test_masks: dict, seed, cfg=CFG, dev=None, shuffle_window=False, return_model=False,
                  test_X_override: dict | None = None):
    """Train LF-PNCN on windows `train_mask`; return dict(scores={name: (n,2) [s_inner, s_outer] logits}, mean, std, seconds).

    Same loop as tier1/common.train_lfpncn. The standardiser uses the training windows only. Ball column excluded from the loss.
    """
    dev = dev or device()
    for m in test_masks.values():
        assert not np.any(m & train_mask), "train/test window overlap"
    Ytr = Y[train_mask].copy()
    rng = np.random.default_rng(seed)
    if shuffle_window:
        Ytr = Ytr[rng.permutation(len(Ytr))]
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    Xtr = torch.from_numpy(X[train_mask]).to(dev)
    mean, std = C.fit_standardiser(Xtr)
    Xtr = (Xtr - mean) / std
    T = torch.from_numpy(Ytr[:, [1, 0]]).to(dev)
    model = LFPNCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n, bs = len(Xtr), cfg["bs"]
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])[:, :2]
            loss = TF.binary_cross_entropy_with_logits(logits, T[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    scores = {}
    for name, m in test_masks.items():
        src = X[m] if not test_X_override or name not in test_X_override else test_X_override[name]
        Xh = (torch.from_numpy(src).to(dev) - mean) / std
        scores[name] = C.predict(model, Xh)
        del Xh
    out = dict(scores=scores, mean=mean.cpu().numpy(), std=std.cpu().numpy(), seconds=seconds, n_train=int(train_mask.sum()))
    if return_model:
        out.update(model=model, Xtr=Xtr)
    del Xtr, T
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def long_frame(split, bearing, Y, S, **extra):
    df = pd.DataFrame({"split": split, "bearing": bearing, "y_inner": Y[:, 0].astype(int), "y_outer": Y[:, 1].astype(int),
                       "s_inner": S[:, 0], "s_outer": S[:, 1]})
    for k, v in extra.items():
        df[k] = v
    return df


def rf_scores(b3, Y, train_mask, test_mask, seed=42, n_jobs=4):
    """B3: RF exactly as evaltools.b2_rf (Vieira Table 17 hyper-parameters, MultiOutputClassifier, seed 42)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    import b2_rf
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **b2_rf.PARAMS_TABLE17))
    clf.fit(b3[train_mask].astype(np.float64), Y[train_mask].astype(int))
    pr = clf.predict_proba(b3[test_mask].astype(np.float64))
    return np.stack([pr[0][:, 1], pr[1][:, 1]], axis=1)


def parse_range(s):
    lo, hi = map(int, s.split("-"))
    return lo, hi
