"""Metrics for multilabel bearing-fault evaluation (Macro-AUROC, FPR at fixed TPR, logit loading).

Conventions follow Vieira et al. (2026) and the notebook ``notebooks/results.ipynb`` of their repository
(read as text only, never executed).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

LABELS = ("inner", "outer")


def load_logit_run(path) -> dict:
    """Load one saved test-logit file (torch tensor, weights_only=True, CPU only).

    Tensor columns: [index, inner logit, outer logit, inner label, outer label].
    Returns dict of numpy arrays: idx, logit_inner, logit_outer, y_inner, y_outer.
    """
    import torch  # imported lazily so the sklearn-only modules work without torch

    t = torch.load(str(path), map_location="cpu", weights_only=True).numpy()
    if t.ndim != 2 or t.shape[1] != 5:
        raise ValueError(f"unexpected logit tensor shape {t.shape} in {path}")
    return {
        "idx": t[:, 0].astype(np.int64),
        "logit_inner": t[:, 1].astype(np.float64),
        "logit_outer": t[:, 2].astype(np.float64),
        "y_inner": t[:, 3].astype(np.int64),
        "y_outer": t[:, 4].astype(np.int64),
    }


def macro_auroc(y_true, scores, return_per_label: bool = False):
    """Macro-AUROC = unweighted mean of the per-label binary AUROCs.

    y_true, scores: arrays of shape (n, L) (L=2 for [inner, outer]).
    Returns the macro value in [0, 1] (or (macro, per_label_array) when return_per_label).
    A label with a single class in y_true makes AUROC undefined and raises ValueError.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if y.ndim == 1:
        y, s = y[:, None], s[:, None]
    per = np.array([roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1])])
    return (per.mean(), per) if return_per_label else per.mean()


# ----------------------------------------------------------------------------- FPR at fixed TPR
def fpr_at_tpr(y, s, tpr: float = 0.90, method: str = "nearest") -> float:
    """FPR at a target TPR for a single binary ROC curve.

    method:
      'nearest' : ROC vertex whose TPR is closest to ``tpr`` (what the authors' notebook does, argmin |tpr-0.9|)
      'interp'  : linear interpolation of FPR as a function of TPR (np.interp)
      'first'   : smallest FPR with TPR >= target (operating-point / conservative reading)
    """
    fpr, tp, _ = roc_curve(y, s)
    if method == "nearest":
        return float(fpr[np.abs(tp - tpr).argmin()])
    if method == "interp":
        return float(np.interp(tpr, tp, fpr))
    if method == "first":
        return float(fpr[np.argmax(tp >= tpr)])
    raise ValueError(method)


def horizontal_average_roc(curves, points: int = 10000):
    """Horizontal (TPR-grid) average of ROC curves, as in Vieira's results notebook.

    curves: list of (fpr, tpr) arrays. For each curve FPR is interpolated on a common TPR grid; the grid
    values are averaged over runs. Returns (mean_fpr, tpr_grid, std_fpr, all_fprs).
    """
    grid = np.linspace(0.0, 1.0, points)
    fprs = np.array([np.interp(grid, tp, fp) for fp, tp in curves])
    return fprs.mean(axis=0), grid, fprs.std(axis=0), fprs


def vertical_average_roc(curves, points: int = 1000):
    """Vertical (FPR-grid) average: mean TPR at each FPR. Returns (fpr_grid, mean_tpr, std_tpr)."""
    grid = np.linspace(0.0, 1.0, points)
    tprs = np.array([np.interp(grid, fp, tp) for fp, tp in curves])
    return grid, tprs.mean(axis=0), tprs.std(axis=0)


def mean_fpr_at_tpr(ys, ss, tpr: float = 0.90, convention: str = "horizontal", points: int = 10000) -> float:
    """Mean FPR at the target TPR over a list of runs (ys[i], ss[i] = labels/scores of run i, one label).

    convention:
      'horizontal'       : horizontal-average ROC on a 10000-point TPR grid, then the grid point nearest to ``tpr``
                           (exactly the procedure in the authors' notebook; this is the Vieira et al. number)
      'horizontal_interp': same average, read at exactly ``tpr`` by interpolation
      'per_run_nearest'  : per-run ``fpr_at_tpr(method='nearest')`` then mean over runs
      'per_run_interp'   : per-run linear-interp FPR@TPR then mean
      'per_run_first'    : per-run smallest FPR with TPR>=target then mean
      'vertical'         : vertical-average ROC (mean TPR over runs on an FPR grid); FPR where it first reaches tpr
      'pooled'           : one ROC on all windows of all runs concatenated
    Returned as a fraction in [0, 1].
    """
    if convention.startswith("per_run"):
        m = convention.split("_", 2)[2]
        return float(np.mean([fpr_at_tpr(y, s, tpr, m) for y, s in zip(ys, ss)]))
    if convention == "pooled":
        return fpr_at_tpr(np.concatenate(ys), np.concatenate(ss), tpr, "nearest")
    curves = [roc_curve(y, s)[:2] for y, s in zip(ys, ss)]
    if convention in ("horizontal", "horizontal_interp"):
        mean_fpr, grid, _, _ = horizontal_average_roc(curves, points)
        if convention == "horizontal":
            return float(mean_fpr[np.abs(grid - tpr).argmin()])
        return float(np.interp(tpr, grid, mean_fpr))
    if convention == "vertical":
        fg, mt, _ = vertical_average_roc(curves, 10000)
        return float(fg[np.argmax(mt >= tpr)])
    raise ValueError(convention)
