"""Bearing-level cluster bootstrap for paired Macro-AUROC differences.

Why clusters: windows from one bearing are strongly dependent (same physical defect, same operating
conditions); the independent sampling unit is the BEARING. A window-level bootstrap treats ~1900 windows
as independent and gives intervals that are far too narrow. Likewise the 100 evaluation splits are draws of
6 test bearings out of the same 17 bearings, so they are NOT 100 independent replicates.

Prediction tables ("pred frames") are long DataFrames with columns
    split, bearing, y_inner, y_outer, s_inner, s_outer
(one row per test window; ``split`` may be constant for a single test set).

Exact, fast computation: for a split s and label l let P be the bearings with l=1 and N those with l=0
(a bearing has a constant label). Define the pair-count matrix
    C[i, j] = sum_{p in window(i), n in window(j)} 1[s_p > s_n] + 0.5 * 1[s_p == s_n]
Resampling bearings with replacement gives multiplicities m (m_i = number of times bearing i is drawn) and
the AUROC of the pooled resampled test set is exactly
    AUROC(m) = (m^T C m) / ((m . Npos) (m . Nneg)),
where Npos/Nneg are the positive/negative window counts per bearing. This equals sklearn's
roc_auc_score(..., sample_weight=m[bearing]) (verified in test_bootstrap.py) but costs microseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import pu_data as pu

LABELS = ("inner", "outer")


# ------------------------------------------------------------------------------------------ core structure
@dataclass
class PairTable:
    """Pair-count tensors for one model. Shapes: C (S, 2, Bn, Bn); Npos, Nneg (S, 2, Bn); in_split (S, Bn)."""

    splits: list
    bearings: list
    C: np.ndarray
    Npos: np.ndarray
    Nneg: np.ndarray
    in_split: np.ndarray

    def auroc(self, M: np.ndarray) -> np.ndarray:
        """M: (D, Bn) bearing multiplicities -> AUROC array (D, S, 2); NaN where undefined."""
        out = np.empty((M.shape[0], len(self.splits), 2))
        for l in range(2):
            num = np.einsum("di,sij,dj->ds", M, self.C[:, l], M, optimize=True)
            dp = M @ self.Npos[:, l].T
            dn = M @ self.Nneg[:, l].T
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, :, l] = np.where((dp > 0) & (dn > 0), num / (dp * dn), np.nan)
        return out

    def macro_per_split(self, M: np.ndarray) -> np.ndarray:
        """(D, S) macro AUROC per split; NaN if either label is undefined in that split."""
        return self.auroc(M).mean(axis=2)


def _pair_counts(pos_scores: list[np.ndarray], neg_scores: list[np.ndarray]) -> np.ndarray:
    C = np.zeros((len(pos_scores), len(neg_scores)))
    for j, sn in enumerate(neg_scores):
        sn_sorted = np.sort(sn)
        for i, sp in enumerate(pos_scores):
            lo = np.searchsorted(sn_sorted, sp, side="left")  # negatives strictly below
            hi = np.searchsorted(sn_sorted, sp, side="right")
            C[i, j] = lo.sum() + 0.5 * (hi - lo).sum()
    return C


def build_pair_table(frame: pd.DataFrame, bearings: list[str] | None = None) -> PairTable:
    """Precompute the pair-count tensors of a pred frame (any number of splits)."""
    bearings = bearings or sorted(frame["bearing"].unique(), key=lambda b: (pu.bearing_labels(b), b))
    bidx = {b: i for i, b in enumerate(bearings)}
    splits = sorted(frame["split"].unique())
    Bn, S = len(bearings), len(splits)
    C = np.zeros((S, 2, Bn, Bn))
    Npos = np.zeros((S, 2, Bn))
    Nneg = np.zeros((S, 2, Bn))
    ins = np.zeros((S, Bn), dtype=bool)
    for si, (sp, g) in enumerate(frame.groupby("split", sort=True)):
        per = {b: gg for b, gg in g.groupby("bearing")}
        for b in per:
            ins[si, bidx[b]] = True
        for l, lab in enumerate(LABELS):
            pos_b = [b for b in per if per[b][f"y_{lab}"].iloc[0] == 1]
            neg_b = [b for b in per if per[b][f"y_{lab}"].iloc[0] == 0]
            for b in per:  # a bearing must have a constant label
                if per[b][f"y_{lab}"].nunique() != 1:
                    raise ValueError(f"bearing {b} has mixed {lab} labels in split {sp}")
            if not pos_b or not neg_b:
                continue
            ps = [per[b][f"s_{lab}"].to_numpy() for b in pos_b]
            ns = [per[b][f"s_{lab}"].to_numpy() for b in neg_b]
            Csub = _pair_counts(ps, ns)
            for a, b in enumerate(pos_b):
                Npos[si, l, bidx[b]] = len(ps[a])
                for c, d in enumerate(neg_b):
                    C[si, l, bidx[b], bidx[d]] = Csub[a, c]
            for c, d in enumerate(neg_b):
                Nneg[si, l, bidx[d]] = len(ns[c])
    return PairTable(splits, bearings, C, Npos, Nneg, ins)


# ------------------------------------------------------------------------------------------ resampling
def _bearing_class_index(bearings: list[str], present: np.ndarray | None = None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, b in enumerate(bearings):
        if present is None or present[i]:
            out.setdefault(pu.bearing_class(b), []).append(i)
    return out


def draw_multiplicities(bearings: list[str], B: int, rng: np.random.Generator, stratify: bool = True,
                        present: np.ndarray | None = None) -> np.ndarray:
    """(B, Bn) bearing multiplicities from a with-replacement draw of bearings.

    stratify=True keeps the class counts fixed (n_healthy, n_outer, n_inner drawn within class), which matches the
    design (every test set contains all three classes). stratify=False draws len(bearings) bearings freely.
    """
    Bn = len(bearings)
    M = np.zeros((B, Bn))
    if stratify:
        for idx in _bearing_class_index(bearings, present).values():
            idx = np.array(idx)
            draws = rng.integers(0, len(idx), size=(B, len(idx)))
            for k in range(len(idx)):
                np.add.at(M, (np.arange(B), idx[draws[:, k]]), 1)
    else:
        pool = np.array([i for i in range(Bn) if present is None or present[i]])
        draws = rng.integers(0, len(pool), size=(B, len(pool)))
        for k in range(len(pool)):
            np.add.at(M, (np.arange(B), pool[draws[:, k]]), 1)
    return M


def _summarise(point: float, boot: np.ndarray, alpha: float = 0.05) -> dict:
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "boot_mean": float(boot.mean()), "se": float(boot.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi), "n_valid": int(len(boot)),
            "p_le_0": float((boot <= 0).mean())}


# ------------------------------------------------------------------------------------------ public API
def paired_bootstrap_multi_split(frame_a: pd.DataFrame, frame_b: pd.DataFrame, B: int = 2000, seed: int = 0,
                                 stratify: bool = True, alpha: float = 0.05, return_draws: bool = False) -> dict:
    """Cluster bootstrap over bearings for the mean-over-splits Macro-AUROC difference A - B.

    Both frames must cover the same splits and bearings (windows may differ, e.g. RF has 320/bearing, the CNN
    ~319). Each replicate: draw the bearing multiplicities m once (shared by A and B -> paired), then for every
    split compute the AUROC of the pooled resampled test set (bearing i counted m_i times), average over
    the splits where it is defined, and take A - B. Splits in which a label loses all its positive (or all its
    negative) bearings are skipped for that replicate.
    Values are in AUROC points x100.
    """
    ta = build_pair_table(frame_a)
    tb = build_pair_table(frame_b, ta.bearings)
    assert ta.splits == tb.splits and ta.bearings == tb.bearings
    rng = np.random.default_rng(seed)
    M = draw_multiplicities(ta.bearings, B, rng, stratify)
    ma, mb = ta.macro_per_split(M), tb.macro_per_split(M)
    valid = ~np.isnan(ma) & ~np.isnan(mb)
    with np.errstate(invalid="ignore"):
        ca = np.where(valid, ma, np.nan)
        cb = np.where(valid, mb, np.nan)
        mean_a, mean_b = np.nanmean(ca, axis=1), np.nanmean(cb, axis=1)
    ones = np.ones((1, len(ta.bearings)))
    pa, pb = ta.macro_per_split(ones)[0], tb.macro_per_split(ones)[0]
    res = {
        "model_A": _summarise(100 * np.nanmean(pa), 100 * mean_a, alpha),
        "model_B": _summarise(100 * np.nanmean(pb), 100 * mean_b, alpha),
        "diff": _summarise(100 * (np.nanmean(pa) - np.nanmean(pb)), 100 * (mean_a - mean_b), alpha),
        "mean_valid_splits_per_draw": float(valid.sum(axis=1).mean()),
        "n_boot": B, "stratify": stratify,
    }
    if return_draws:
        res["draws"] = {"diff": 100 * (mean_a - mean_b), "A": 100 * mean_a, "B": 100 * mean_b}
    return res


def paired_bootstrap_single_split(y, scores_a, scores_b, bearing_ids, B: int = 2000, seed: int = 0,
                                  stratify: bool = True, alpha: float = 0.05) -> dict:
    """Paired cluster bootstrap on ONE test set.

    y: (n, 2) labels [inner, outer]; scores_a, scores_b: (n, 2) scores of the two models on the SAME windows;
    bearing_ids: (n,) physical bearing id of each window. Resamples test bearings (stratified by class by default),
    recomputes Macro-AUROC of A and B on the pooled resampled set and returns mean difference and the 95%
    percentile CI (x100).
    """
    y, sa, sb = map(np.asarray, (y, scores_a, scores_b))
    bid = np.asarray(bearing_ids)
    def mk(s):
        return pd.DataFrame({"split": 0, "bearing": bid, "y_inner": y[:, 0], "y_outer": y[:, 1],
                             "s_inner": s[:, 0], "s_outer": s[:, 1]})
    return paired_bootstrap_multi_split(mk(sa), mk(sb), B=B, seed=seed, stratify=stratify, alpha=alpha)


def naive_window_bootstrap(y, scores_a, scores_b, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Window-level (iid) paired bootstrap, for comparison only: ignores clustering, intervals are too narrow."""
    from metrics import macro_auroc
    y, sa, sb = map(np.asarray, (y, scores_a, scores_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            d.append(100 * (macro_auroc(y[i], sa[i]) - macro_auroc(y[i], sb[i])))
        except ValueError:
            pass
    point = 100 * (macro_auroc(y, sa) - macro_auroc(y, sb))
    return _summarise(point, np.array(d), alpha)


# ------------------------------------------------------------------------------------------ MDE
def minimum_detectable_effect(sd_points: float = 13.0, n_units: int = 17, rho: float = 0.0, alpha: float = 0.05,
                              power: float = 0.80) -> float:
    """Minimum detectable paired difference (AUROC points) from a normal approximation.

    delta = (z_{1-alpha/2} + z_{power}) * sd_diff / sqrt(n_units),   sd_diff = sd_points * sqrt(2 * (1 - rho))
    where sd_points is the SD of ONE model's per-split Macro-AUROC (observed: ~13), rho the correlation of the two
    models' per-split AUROCs (rho=0: independent; large rho when both models see the same bearings, which is
    the paired case), n_units the number of independent units (bearings, NOT 100 splits, NOT ~1900 windows).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * sd_points * np.sqrt(2 * (1 - rho)) / np.sqrt(n_units))


def empirical_mde(se_diff_points: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """MDE implied by a bootstrap standard error of the paired difference: (z_a/2 + z_power) * SE."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se_diff_points)
