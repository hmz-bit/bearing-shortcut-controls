"""Envelope-spectrum bank for B1 on CWRU (see results/cwru/PREDECLARED_B1_CWRU.txt). Same signal and window order as build_cwru_cache.py
(asserted against cwru_windows.npz). Front end of Vieira et al. read as text: cutoff [500, 6000] at fs 12000 is a 12th-order high-pass at 500 Hz
(array_filter second branch), zero-phase sosfiltfilt padlen 6 even, |Hilbert|, mean removed, one-sided |FFT|/n, bins >= 1 doubled -> 6000 bins.
Output: data/cwru/b1_env_bank.npy (5328, 6000) float32 (raw sensor scale)."""
import os, re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import signal
from scipy.fft import fft
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import build_cwru_cache as BC          # meta_universe(), file map; its __main__ block is not run

FS, WIN, HOP, ORDER, CUT = 12000, 12000, 3000, 12, 500.0
SOS = signal.butter(ORDER, CUT / (0.5 * FS), btype="highpass", output="sos")


def envelope_spectrum(x):
    xf = signal.sosfiltfilt(SOS, np.asarray(x, dtype=np.float64), padlen=int(ORDER / 2), padtype="even")
    env = np.abs(signal.hilbert(xf))
    env = env - env.mean()
    n = len(env)
    sp = np.abs(fft(env)) / n
    out = sp[: n // 2].copy()
    out[1:] *= 2.0
    return out.astype(np.float32)


def proc(row):
    w = row["waveform_id"]
    m = re.fullmatch(r"(IR|B|OR@6|OR@3)_(\d+)_(DE|FE)_(DE|FE)_12_(\d)HP", w)
    typ, size, fl, sl, load = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
    f = BC.FM[(BC.FM.fault_loc == fl) & (BC.FM.typ == typ) & (BC.FM["size"] == size) & (BC.FM.load == load)]
    assert len(f) == 1
    n = int(f.iloc[0]["file"])
    mat = loadmat(os.path.join(WORK, "data", "cwru", "raw", f"{n}.mat"))
    x = np.ravel(mat[f"X{n:03d}_{sl}_time"]).astype(np.float64)
    return w, np.stack([envelope_spectrum(x[s:s + WIN]) for s in range(0, len(x) - WIN + 1, HOP)])


if __name__ == "__main__":
    u = BC.meta_universe()
    assert len(u) == 144
    with Pool(6) as pool:
        res = pool.map(proc, [r for _, r in u.iterrows()], chunksize=4)
    z = np.load(os.path.join(WORK, "data", "cwru", "cwru_windows.npz"), allow_pickle=False)
    wid = np.concatenate([[r[0]] * len(r[1]) for r in res])
    assert (wid == z["wid"]).all(), "window order differs from the LF-PNCN cache"
    bank = np.concatenate([r[1] for r in res])
    assert bank.shape == (len(wid), 6000) and np.isfinite(bank).all()
    np.save(os.path.join(WORK, "data", "cwru", "b1_env_bank.npy"), bank)
    print("bank", bank.shape, "max", float(bank.max()), "median of window max", float(np.median(bank.max(1))))
