"""Window cache for UORED-VAFCLS (pre-declared in results/uored/PREDECLARED_UORED.txt).
Accelerometer column, 42 kHz (no resampling), 1-s windows with 75% overlap (hop 10500), f_r = 29.75 Hz for all files,
UORED 6203 geometry. Output: data/uored/uored_windows.npz with t_mad (N,3,15,64), bearing (N,) int 1..20, state (N,) int 0/1/2,
cls (N,) str H/I/O/B/C, fname (N,), Y (N,2) [inner, outer]. Works with `python -I`.
"""
import glob, os, re, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORK, "scripts", "lfpncn"))
import features as F
import geometry as G

UORED_6203 = G.Bearing("UORED_6203", 8, 6.77, 28.50, 0.0)
F_R = 29.75
HOP = 10500


def process(path):
    base = os.path.basename(path)
    m = re.fullmatch(r"([HIOBC])_(\d+)_([012])\.csv", base)
    cls, k, st = m.group(1), int(m.group(2)), int(m.group(3))
    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), encoding="utf-8-sig")
    assert len(x) == 420000, (base, len(x))
    starts = range(0, len(x) - F.WIN + 1, HOP)
    T = np.stack([F.window_to_array(x[s:s + F.WIN], F_R, UORED_6203) for s in starts])
    return base, cls, k, st, T


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(WORK, "data", "uored", "csv", "*.csv")))
    assert len(files) == 60, len(files)
    t0 = time.time()
    with Pool(6) as pool:
        res = pool.map(process, files, chunksize=2)
    T = np.concatenate([r[4] for r in res])
    n = [len(r[4]) for r in res]
    cls = np.concatenate([[r[1]] * len(r[4]) for r in res])
    Y = np.stack([(cls == "I"), (cls == "O")], 1).astype(np.float32)
    out = dict(t_mad=T, bearing=np.concatenate([[r[2]] * len(r[4]) for r in res]).astype(np.int16),
               state=np.concatenate([[r[3]] * len(r[4]) for r in res]).astype(np.int8), cls=cls,
               fname=np.concatenate([[r[0]] * len(r[4]) for r in res]), Y=Y)
    np.savez(os.path.join(WORK, "data", "uored", "uored_windows.npz"), **out)
    print("windows", len(T), "per file", set(n), "finite", bool(np.isfinite(T).all()), "seconds", round(time.time() - t0))
