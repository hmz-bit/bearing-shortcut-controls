"""Build (i) the resampled full signals used for random-crop training and (ii) the envelope spectra of the
fixed 1-s evaluation windows, for all 29 bearings.

Same files, resampling and windowing as scripts/build_cache.py: pu_loader.load_vibration (scipy.io.loadmat),
scipy.signal.resample_poly 64 kHz -> 42 kHz (21/32), non-overlapping 1-s windows (first 4), row order identical to
data/cache/pu_windows.npz (fname, win). As in the original pipeline (preprocess_paderborn) the per-signal mean is
removed before resampling.

Outputs in results/b1/:
  signals_42k.npy      float32 (n_files, 196100) zero padded, resampled full signals, file order = order of first appearance in the cache
  signals_meta.csv     file index, fname, bearing, condition, n_samples(42k), n_raw(64k), raw_sum, raw_sumsq (float64, raw 64 kHz signal)
  env_spectra.npy      float16 (n_windows, 21000), = envelope spectrum (raw sensor units, mean-removed) x 1000, rows aligned to pu_windows.npz
  env_index.csv        row, fname, win, bearing, condition (copy of the cache keys, checked equal)
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORK, "scripts"))
sys.path.insert(0, HERE)
from pu_loader import load_vibration
import envspec

OUT = os.path.join(WORK, "results", "b1")
CACHE = os.path.join(WORK, "data", "cache", "pu_windows.npz")
PU = os.path.join(WORK, "data", "pu")
WIN = 42000
SCALE = 1000.0
MAXLEN = 196100


def work(args):
    i, fname, bearing, nwin = args
    meta, x = load_vibration(os.path.join(PU, bearing, fname))
    x = x.astype(np.float64)
    raw = (len(x), float(x.sum()), float((x * x).sum()))
    y = signal.resample_poly(x - x.mean(), 21, 32).astype(np.float32)
    wins = y[: nwin * WIN].reshape(nwin, WIN)
    spec = np.stack([envspec.envelope_spectrum(w) for w in wins])
    return i, y, raw, spec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    z = np.load(CACHE, allow_pickle=False)
    fname, win, bearing, cond = z["fname"], z["win"], z["bearing"], z["condition"]
    # unique files in order of first appearance
    first = np.unique(fname, return_index=True)[1]
    first.sort()
    files = [(k, str(fname[j]), str(bearing[j]), int((fname == fname[j]).sum())) for k, j in enumerate(first)]
    nF = len(files)
    print("files", nF, "windows", len(fname), flush=True)
    sigf = open(os.path.join(OUT, "signals_42k.f32"), "wb")
    offset = 0
    spec_all = np.lib.format.open_memmap(os.path.join(OUT, "env_spectra.npy"), mode="w+", dtype=np.float16, shape=(len(fname), 21000))
    # row offset of each file in the cache (rows of a file are contiguous, win ascending)
    row0 = {str(fname[j]): j for j in first}
    meta_rows = [None] * nF
    t0 = time.time()
    with Pool(5) as pool:
        for n, (i, y, raw, spec) in enumerate(pool.imap(work, files, chunksize=2)):
            k, fn, b, nw = files[i]
            sigf.write(y.tobytes())
            r = row0[fn]
            assert np.array_equal(win[r:r + nw], np.arange(nw)) and (fname[r:r + nw] == fn).all()
            spec_all[r:r + nw] = (spec * SCALE).astype(np.float16)
            meta_rows[i] = dict(file_idx=i, fname=fn, bearing=b, condition=str(cond[r]), offset=offset, n_samples=len(y), n_raw=raw[0],
                                raw_sum=raw[1], raw_sumsq=raw[2], row0=r, nwin=nw)
            offset += len(y)
            if n % 200 == 0:
                print(n, round(time.time() - t0), "s", flush=True)
    sigf.close(); spec_all.flush()
    pd.DataFrame(meta_rows).to_csv(os.path.join(OUT, "signals_meta.csv"), index=False)
    pd.DataFrame(dict(row=np.arange(len(fname)), fname=fname, win=win, bearing=bearing, condition=cond)).to_csv(
        os.path.join(OUT, "env_index.csv"), index=False)
    with open(os.path.join(OUT, "env_spectra_scale.txt"), "w") as f:
        f.write(f"env_spectra.npy = envelope spectrum x {SCALE} (float16). divide by {SCALE} to get raw sensor units.\n")
    print("done", round(time.time() - t0), "s", flush=True)
