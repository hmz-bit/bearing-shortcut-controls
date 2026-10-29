"""Envelope-spectrum bank for B1 on UORED (see results/uored/PREDECLARED_B1_UORED.txt). Same file and window order as build_uored_cache.py.
Output: data/uored/b1_env_bank.npy (2220, 21000) float16 = spectrum x 1000; b1_file_stats.csv (fname, n_raw, raw_sum, raw_sumsq)."""
import glob, os, sys, re
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORK, "scripts", "b1"))
import envspec
HOP, WIN, SCALE = 10500, 42000, 1000.0


def proc(path):
    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), encoding="utf-8-sig")
    spec = np.stack([envspec.envelope_spectrum(x[s:s + WIN]) for s in range(0, len(x) - WIN + 1, HOP)])
    return os.path.basename(path), (spec * SCALE).astype(np.float16), len(x), float(x.sum()), float((x.astype(np.float64) ** 2).sum())


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(WORK, "data", "uored", "csv", "*.csv")))
    with Pool(6) as p:
        res = p.map(proc, files, chunksize=2)
    z = np.load(os.path.join(WORK, "data", "uored", "uored_windows.npz"), allow_pickle=False)
    names = np.concatenate([[r[0]] * len(r[1]) for r in res])
    assert (names == z["fname"]).all()
    bank = np.concatenate([r[1] for r in res])
    assert bank.shape == (len(z["fname"]), 21000) and np.isfinite(bank.astype(np.float32)).all()
    np.save(os.path.join(WORK, "data", "uored", "b1_env_bank.npy"), bank)
    pd.DataFrame(dict(fname=[r[0] for r in res], n_raw=[r[2] for r in res], raw_sum=[r[3] for r in res], raw_sumsq=[r[4] for r in res])).to_csv(
        os.path.join(WORK, "data", "uored", "b1_file_stats.csv"), index=False)
    print("bank", bank.shape, "max float16", float(np.abs(bank.astype(np.float32)).max()))
