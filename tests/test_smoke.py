"""Smoke test on synthetic data only (no dataset, no network, CPU only).

Run from the repository root:   python -I tests/test_smoke.py

Checks
  1. the order-tensor pipeline turns a synthetic 1-s record with a periodic outer-race impulse train into a finite
     (3, 15, 64) tensor whose outer-race rows peak at the outer-race order;
  2. the three-branch network accepts the tensor and has the documented 22,947 parameters;
  3. the pair-count bootstrap AUROC equals sklearn's weighted roc_auc_score on a synthetic prediction frame.
The longer existing tests are scripts/lfpncn/test_lfpncn.py and scripts/evaltools/test_bootstrap.py.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("lfpncn", "evaltools"):
    sys.path.insert(0, os.path.join(ROOT, "scripts", sub))

from sklearn.metrics import roc_auc_score  # noqa: E402

import bootstrap as bs  # noqa: E402
import features as F  # noqa: E402
from geometry import PU_6203, fault_frequencies  # noqa: E402
from model import LFPNCN, count_parameters  # noqa: E402


def synthetic_window(f_r=25.0, seed=0):
    """3 kHz resonance excited by an impulse train at the outer-race frequency, plus noise (42 kHz, 1 s)."""
    rng = np.random.default_rng(seed)
    fs = F.FS_TARGET
    n = F.WIN
    f_o = fault_frequencies(f_r, PU_6203)["f_o"]
    x = rng.normal(0, 0.05, n)
    t = np.arange(0, 0.01, 1 / fs)
    ring = np.exp(-t * 800) * np.sin(2 * np.pi * 3000 * t)
    for k in range(int(f_o * n / fs)):
        i = int(round(k * fs / f_o))
        seg = ring[: n - i]
        x[i:i + len(seg)] += 3.0 * seg
    return x


def test_pipeline_and_model():
    w = synthetic_window()
    arr = F.window_to_array(w, 25.0, PU_6203)
    assert arr.shape == (3, 15, 64) and arr.dtype == np.float32 and np.isfinite(arr).all()
    # outer-race mode (index 0), band 2 (rows 5..9): the centre of the order axis carries the line
    centre = arr[0, 5:10, 30:34].mean()
    edge = np.r_[arr[0, 5:10, :8].ravel(), arr[0, 5:10, -8:].ravel()].mean()
    assert centre > edge, (centre, edge)
    import torch
    net = LFPNCN().eval()
    assert count_parameters(net) == 22947
    with torch.no_grad():
        out = net(torch.from_numpy(arr[None]))
    assert out.shape == (1, 3) and torch.isfinite(out).all()


def test_bootstrap_matches_sklearn():
    rng = np.random.default_rng(1)
    ids = [f"K00{i}" for i in range(1, 7)] + [f"KA0{i}" for i in range(1, 6)] + [f"KI0{i}" for i in range(1, 7)]
    rows = []
    for b in ids:
        yi = int(b.startswith("KI"))
        yo = int(b.startswith("KA"))
        e = rng.normal(0, 1)
        n = 40
        rows.append(pd.DataFrame({"split": 0, "bearing": b, "y_inner": yi, "y_outer": yo,
                                  "s_inner": 1.2 * yi + e + rng.normal(0, 1, n),
                                  "s_outer": 1.0 * yo + e + rng.normal(0, 1, n)}))
    frame = pd.concat(rows, ignore_index=True)
    tab = bs.build_pair_table(frame)
    M = bs.draw_multiplicities(tab.bearings, 5, np.random.default_rng(2))
    got = tab.auroc(M)[:, 0, :]                 # (draws, 2)
    for d in range(M.shape[0]):
        w = frame["bearing"].map(dict(zip(tab.bearings, M[d]))).to_numpy()
        for l, lab in enumerate(("inner", "outer")):
            ref = roc_auc_score(frame[f"y_{lab}"], frame[f"s_{lab}"], sample_weight=w)
            assert abs(ref - got[d, l]) < 1e-9, (d, lab, ref, got[d, l])


if __name__ == "__main__":
    test_pipeline_and_model()
    test_bootstrap_matches_sklearn()
    print("smoke test passed")
