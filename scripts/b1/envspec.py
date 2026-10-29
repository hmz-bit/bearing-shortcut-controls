"""Envelope-spectrum front end of the B1 baseline, re-implemented from the text of the Vieira et al. repository.

Source (read as text, never executed):
  bearing-data-leakage/src/data/transforms/signal_representations.py : get_envelope, array_filter, bandpass_filter
  bearing-data-leakage/src/data/transforms/signal_processing/power_cepstrum.py : get_fft
  configs/transform/envelope.yaml, configs/experiment/paderborn/eval_envelope.yaml : cutoff [500, 10000], fs 42000
Steps: Butterworth band-pass (order 12 as in get_envelope default `order=12`, cutoff 500-10000 Hz, zero-phase
sosfiltfilt with padlen=int(12/2), padtype even) -> |Hilbert| -> remove mean -> one-sided FFT magnitude
(|FFT|/n, bins 0..n/2-1, bins 1.. doubled). For n = 42000 samples: 21000 bins, 1 Hz spacing, 0..20999 Hz.
"""
import numpy as np
from scipy import signal
from scipy.fft import fft

FS = 42000
CUTOFF = (500.0, 10000.0)
ORDER = 12
_SOS = signal.butter(ORDER, np.asarray(CUTOFF, dtype=np.float64) / (0.5 * FS), btype="bandpass", output="sos")


def envelope_spectrum(x):
    """x: 1-D array (42000 samples). Returns float32 array (21000,)."""
    x = np.asarray(x, dtype=np.float64)
    xf = signal.sosfiltfilt(_SOS, x, padlen=int(ORDER / 2), padtype="even")
    env = np.abs(signal.hilbert(xf))
    env = env - env.mean()
    n = len(env)
    sp = np.abs(fft(env)) / n
    out = sp[: n // 2].copy()
    out[1:] *= 2.0
    return out.astype(np.float32)
