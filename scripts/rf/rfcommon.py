"""Shared code for the random-forest baseline (declared in results/rf/PREDECLARED_RF.txt).
Features = the 20 'B3' features of scripts/build_cache.py (time_stats, broadband_logsnr), generalised to any sampling rate and geometry.
Classifier and tie-aware weighted-AUROC helpers. Works with `python -I` (paths relative to this file)."""
import pathlib, sys
import numpy as np
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "lfpncn"))
import geometry as G

RES = WORK / "results" / "rf"
DATA = WORK / "data" / "rf"
B = 2000

# ---------------------------------------------------------------- features
def time_stats(w):
    rms = float(np.sqrt(np.mean(w ** 2)))
    return [rms, float(w.max() - w.min()), float(stats.kurtosis(w, fisher=False)), float(stats.skew(w)), float(np.abs(w).max() / rms)]


def broadband_logsnr(w, f_r, fs, bearing):
    """15 values; identical to build_cache.broadband_logsnr except fs and bearing are arguments."""
    sos = signal.butter(4, [500.0, min(10000.0, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, w)
    e = np.abs(signal.hilbert(xb)) ** 2
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e)) / len(e)
    fr = np.fft.rfftfreq(len(e), 1 / fs)
    sel = (fr >= 1.0) & (fr <= 9500.0)
    m = np.median(E[sel])
    s = 1.4826 * np.median(np.abs(E[sel] - m))
    Et = np.log10(1.0 + np.maximum(0.0, E - m) / (s + 1e-12))
    ff = G.fault_frequencies(f_r, bearing)
    out = []
    for key, fmod in (("f_o", 0.0), ("f_i", f_r), ("f_b", ff["f_c"])):
        for h in range(1, 6):
            c = h * ff[key]
            dlt = max(0.03 * h * ff[key], 1.2 * fmod)
            idx = (fr >= c - dlt) & (fr <= c + dlt)
            out.append(float(Et[idx].max()))
    return out


def window_features(w, f_r, fs, bearing):
    w = np.asarray(w, dtype=np.float64)
    return np.array(time_stats(w) + broadband_logsnr(w, f_r, fs, bearing), dtype=np.float64)


# ---------------------------------------------------------------- classifier
PARAMS_PRIMARY = dict(n_estimators=200, max_features="log2", criterion="log_loss", max_depth=24, min_samples_split=15,
                      min_samples_leaf=18, ccp_alpha=2.65e-2)
PARAMS_U = dict(n_estimators=200, max_features="sqrt")
VARIANTS = {"rf": PARAMS_PRIMARY, "rfu": PARAMS_U}
SEED = 42


def fit_scores(Xtr, Ytr, Xte, params, n_jobs=4):
    """MultiOutputClassifier over [inner, outer]; returns (n_te, 2) probabilities of class 1."""
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **params))
    clf.fit(Xtr, Ytr.astype(int))
    pr = clf.predict_proba(Xte)
    return np.stack([pr[0][:, 1], pr[1][:, 1]], 1)


# ---------------------------------------------------------------- tie-aware weighted AUROC
def prep_sorted(y, unit, score):
    o = np.argsort(score, kind="stable")
    sc = score[o]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    return y[o].astype(float), unit[o], sc, starts


def wauc(M, y, u, starts):
    """M (D, n_units) multiplicities; y, u sorted by score; starts = first index of each tie group. Tie-aware (half credit)."""
    W = M[:, u]
    pos, neg = W * y, W * (1 - y)
    gneg = np.add.reduceat(neg, starts, axis=1)
    gpos = np.add.reduceat(pos, starts, axis=1)
    before = np.cumsum(gneg, axis=1) - gneg
    num = (gpos * (before + 0.5 * gneg)).sum(1)
    dp, dn = pos.sum(1), neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((dp > 0) & (dn > 0), num / (dp * dn), np.nan)
