"""Phase 2 input builders (plug-in architecture).

A *feature spec* is a dict: {"L": context length in s (1, 2 or 4), "cpw": bool (cepstrum pre-whitening before the envelope),
"bands": tuple of (lo, hi) Hz, "n_harm": int}.  `build_file(path, spec)` returns an array (4, 3, rows, 64): one tensor per ORIGINAL
1-s window of the file (row-aligned with pu_windows.npz); for L>1 the 1-s windows inside the same L-s block share the same tensor.
So every variant has exactly the same rows/labels/bearings as the Tier 1 cache and the same evaluation code and paired bootstrap apply.

L=1, cpw=False, default bands reproduces the cache `t_mad` exactly (checked in `check_equal_cache`).
Cepstrum pre-whitening (Smith and Randall 2015): x_w = IFFT( X / |X| ) (amplitude cepstrum removed, phase kept), then the same
three-band squared-envelope pipeline.  All code is a thin variant of lfpncn/features.py (imported, not modified).
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (HERE, WORK / "scripts", WORK / "scripts" / "lfpncn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np
from scipy import signal
import features as F
import geometry as G
from pu_loader import load_vibration

FS = F.FS_TARGET
BANDS3 = F.BANDS_HZ
BANDS6 = ((500.0, 1500.0), (1500.0, 2500.0), (2500.0, 3750.0), (3750.0, 5000.0), (5000.0, 7500.0), (7500.0, 10000.0))
SPECS = {
    "L1": dict(L=1, cpw=False, bands=BANDS3),
    "L2": dict(L=2, cpw=False, bands=BANDS3),
    "L4": dict(L=4, cpw=False, bands=BANDS3),
    "L1cpw": dict(L=1, cpw=True, bands=BANDS3),
    "L1comb": dict(L=1, cpw="comb", bands=BANDS3),
    "L4cpw": dict(L=4, cpw=True, bands=BANDS3),
    "L1b6": dict(L=1, cpw=False, bands=BANDS6),
}


def cepstrum_prewhiten(w, eps=1e-12):
    X = np.fft.rfft(w)
    return np.fft.irfft(X / (np.abs(X) + eps), n=len(w))


def cepstrum_comb_edit(w, f_r, fs=FS, k_max=None, half_width=2):
    """Cepstrum editing (Smith and Randall 2015 style, discrete-component removal): zero the real-cepstrum components at the
    rahmonics k*fs/f_r (+-half_width samples), divide the spectrum by the exp of that removed log-amplitude part.
    Keeps the broadband spectral shape (unlike full whitening) but removes shaft-harmonic combs."""
    n = len(w)
    X = np.fft.rfft(w)
    logmag = np.log(np.abs(X) + 1e-12)
    c = np.fft.irfft(logmag, n=n)
    q0 = fs / f_r
    k_max = k_max or int((n / 2) // q0)
    mask = np.zeros(n)
    for k in range(1, k_max + 1):
        q = int(round(k * q0))
        for dq in range(-half_width, half_width + 1):
            if 0 < q + dq < n // 2:
                mask[q + dq] = 1.0
                mask[n - (q + dq)] = 1.0          # real cepstrum is symmetric
    removed = np.fft.rfft(c * mask).real
    return np.fft.irfft(X * np.exp(-removed), n=n)


def spectra(w, spec, fs=FS, f_r=None):
    """Squared-envelope spectra of the bands of `spec` for one block w: (freqs, E (n_bands, n_freq)), 1/L Hz bins."""
    w = np.asarray(w, dtype=np.float64)
    if spec.get("cpw") == "comb":
        w = cepstrum_comb_edit(w, f_r, fs)
    elif spec.get("cpw"):
        w = cepstrum_prewhiten(w)
    n = len(w)
    out = []
    for lo, hi in spec["bands"]:
        hi = min(hi, fs / 2 * 0.999)
        sos = signal.butter(F.BUTTER_ORDER, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xb = signal.sosfiltfilt(sos, w)
        e = np.abs(signal.hilbert(xb)) ** 2
        e = e - e.mean()
        out.append(np.abs(np.fft.rfft(e)) / n)
    return np.fft.rfftfreq(n, 1.0 / fs), np.stack(out)


def normalise(freqs, E, spec, fs=FS):
    """Median/MAD log-SNR over 1/L .. band-width Hz (same rule as lfpncn.features, 'band' statistic range)."""
    out = np.empty_like(E)
    for b, (lo, hi) in enumerate(spec["bands"]):
        hi = min(hi, fs / 2 * 0.999)
        sel = (freqs > 0) & (freqs <= (hi - lo))
        ref = E[b, sel]
        m = np.median(ref)
        sigma = 1.4826 * np.median(np.abs(ref - m))
        out[b] = np.log10(1.0 + np.maximum(0.0, E[b] - m) / (sigma + F.EPS))
    return out


def block_tensor(w, f_r, spec, bearing=G.PU_6203, n_harm=5):
    """One L-s block -> (3, n_bands*n_harm, 64) float32 (OR, IR, Ball)."""
    freqs, E = spectra(w, spec, f_r=f_r)
    Et = normalise(freqs, E, spec)
    ff = G.fault_frequencies(f_r, bearing)
    sp = {"OR": (ff["f_o"], 0.0), "IR": (ff["f_i"], f_r), "Ball": (ff["f_b"], ff["f_c"])}
    return np.stack([F.order_resample(freqs, Et, *sp[m], n_harm=n_harm) for m in F.MODES]).astype(np.float32)


def build_file(path, spec, f_r=None):
    """(4, 3, rows, 64) for the 4 original 1-s windows of a file."""
    meta, x = load_vibration(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if f_r is None:
        f_r = G.shaft_frequency_from_name(stem)
    x42 = F.resample_to_target(x)
    nwin = min(4, len(x42) // F.WIN)          # a few files are shorter than 4 s (cache has 3 windows for them)
    assert nwin >= 3, (path, nwin)
    L = int(spec["L"])
    out = []
    for blk in range(0, nwin, L):
        end = min(blk + L, nwin)
        t = block_tensor(x42[blk * F.WIN:end * F.WIN], f_r, spec)
        out += [t] * (end - blk)
    return np.stack(out[:nwin])
