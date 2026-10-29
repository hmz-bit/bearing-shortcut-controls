"""LF-PNCN input pipeline: multi-band squared-envelope spectrum -> robust log-SNR ->
fault-frequency-order resampling. Output per mode: float32 (15, 64) = 3 bands x 5 harmonics."""
import numpy as np
from scipy import signal

from geometry import fault_frequencies

FS_TARGET = 42000
WIN = 42000                      # 1 s
BANDS_HZ = ((500.0, 2500.0), (2500.0, 5000.0), (5000.0, 10000.0))
MODES = ("OR", "IR", "Ball")
N_HARM = 5
N_XI = 64
EPS = 1e-12
BUTTER_ORDER = 4                 # not fixed by the method definition


def resample_to_target(x, fs_in=64000, fs_out=FS_TARGET):
    """Polyphase resample (64 kHz -> 42 kHz uses 21/32)."""
    from math import gcd
    g = gcd(int(fs_out), int(fs_in))
    return signal.resample_poly(np.asarray(x, dtype=np.float64), fs_out // g, fs_in // g)


def make_windows(x, win=WIN):
    """Non-overlapping 1-s windows, shape (n_win, win). Incomplete tail is dropped."""
    n = len(x) // win
    return np.asarray(x[: n * win]).reshape(n, win)


def _bands(fs):
    hi_cap = fs / 2.0
    return tuple((lo, min(hi, hi_cap * 0.999)) if hi > hi_cap else (lo, hi) for lo, hi in BANDS_HZ)


def squared_envelope_spectra(w, fs=FS_TARGET, nfft_factor=1):
    """Return (freqs, E) where E[b] is the one-sided FFT magnitude of the mean-removed squared
    envelope of band b. nfft_factor>1 zero-pads (finer grid, same resolution)."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    nfft = n * int(nfft_factor)
    out = []
    for lo, hi in _bands(fs):
        sos = signal.butter(BUTTER_ORDER, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xb = signal.sosfiltfilt(sos, w)
        e = np.abs(signal.hilbert(xb)) ** 2          # |x + j H{x}|^2
        e = e - e.mean()
        out.append(np.abs(np.fft.rfft(e, nfft)) / n)
    return np.fft.rfftfreq(nfft, 1.0 / fs), np.stack(out)


def normalise_spectra(freqs, E, normalize="mad", stat_range="band", fs=FS_TARGET):
    """Robust log-SNR (normalize='mad') or divide-by-max (normalize='max', Zhai et al. B4).

    stat_range: 'band' computes median/MAD/max over 1 Hz..(band width) Hz, the support of the
    squared envelope of a band-limited signal; 'full' uses every bin up to fs/2 (literal reading
    of the original definition; see README for why 'band' is the default).
    """
    out = np.empty_like(E)
    for b, (lo, hi) in enumerate(_bands(fs)):
        fmax = (hi - lo) if stat_range == "band" else freqs[-1]
        sel = (freqs > 0) & (freqs <= fmax)
        ref = E[b, sel]
        if normalize == "mad":
            m = np.median(ref)
            sigma = 1.4826 * np.median(np.abs(ref - m))
            out[b] = np.log10(1.0 + np.maximum(0.0, E[b] - m) / (sigma + EPS))
        elif normalize == "max":
            out[b] = E[b] / (ref.max() + EPS)
        else:
            raise ValueError("normalize must be 'mad' or 'max'")
    return out


def order_resample(freqs, Et, f_m, f_mod, n_harm=N_HARM, n_xi=N_XI):
    """Interpolate Et (n_bands, n_freq) onto xi in [-1,1] around h*f_m for h=1..n_harm.
    Returns (n_bands*n_harm, n_xi); row index = band*n_harm + (h-1)."""
    xi = np.linspace(-1.0, 1.0, n_xi)
    rows = []
    for b in range(Et.shape[0]):
        for h in range(1, n_harm + 1):
            c = h * f_m
            delta = max(0.03 * h * f_m, 1.2 * f_mod)
            rows.append(np.interp(c + xi * delta, freqs, Et[b]))
    return np.stack(rows)


def window_to_tensors(w, f_r, bearing, fs=FS_TARGET, normalize="mad", stat_range="band", nfft_factor=1):
    """One 42 kHz window -> dict mode -> float32 (15, 64)."""
    ff = fault_frequencies(f_r, bearing)
    freqs, E = squared_envelope_spectra(w, fs, nfft_factor)
    Et = normalise_spectra(freqs, E, normalize, stat_range, fs)
    spec = {"OR": (ff["f_o"], 0.0), "IR": (ff["f_i"], f_r), "Ball": (ff["f_b"], ff["f_c"])}
    return {m: order_resample(freqs, Et, *spec[m]).astype(np.float32) for m in MODES}


def window_to_array(w, f_r, bearing, **kw):
    """Same as window_to_tensors but stacked (3, 15, 64) in order OR, IR, Ball."""
    t = window_to_tensors(w, f_r, bearing, **kw)
    return np.stack([t[m] for m in MODES]).astype(np.float32)


def signal_to_array(x64k, f_r, bearing, fs_in=64000, **kw):
    """Raw 64 kHz record -> (n_win, 3, 15, 64) float32."""
    wins = make_windows(resample_to_target(x64k, fs_in))
    return np.stack([window_to_array(w, f_r, bearing, **kw) for w in wins])
