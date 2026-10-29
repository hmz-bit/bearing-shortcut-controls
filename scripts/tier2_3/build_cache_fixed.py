"""Fixed-Hz ablation cache (method F).

Same median/MAD log-SNR order tensor as t_mad, but f_o, f_i, f_b, f_c and f_r are taken from the REFERENCE shaft
frequency 25 Hz (1500 rpm, PU 6203 geometry) for every file, whatever its true speed. At 1500 rpm t_fix == t_mad.
Output: data/cache/pu_windows_fixed.npz with t_fix (N,3,15,64) + bearing, condition, fname, win copied from
pu_windows.npz (row order asserted identical to the files re-read here).
Usage: python -I build_cache_fixed.py
"""
import glob, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, ".."))
for p in (SCRIPTS, os.path.join(SCRIPTS, "lfpncn")):
    if p not in sys.path:
        sys.path.insert(0, p)
import build_cache as BC          # reuse BEARINGS, DATA, FS (module-level code only; main is guarded)
from pu_loader import load_vibration
import features as F
import geometry as G

F_REF = 25.0                      # 1500 rpm


def process_fixed(path):
    try:
        meta, x = load_vibration(path)
        wins = F.make_windows(F.resample_to_target(x))[:4]
        ffq = G.fault_frequencies(F_REF, G.PU_6203)          # reference frequencies for every file
        out = []
        for w in wins:
            freqs, E = F.squared_envelope_spectra(w, BC.FS)
            Et = F.normalise_spectra(freqs, E, normalize="mad")
            modes = [F.order_resample(freqs, Et, ffq["f_o"], 0.0),
                     F.order_resample(freqs, Et, ffq["f_i"], F_REF),
                     F.order_resample(freqs, Et, ffq["f_b"], ffq["f_c"])]
            out.append(np.stack(modes).astype(np.float32))
        return os.path.basename(path), meta["bearing_id"], meta["condition"], np.stack(out), len(wins), None
    except Exception as e:
        return os.path.basename(path), None, None, None, 0, repr(e)[:80]


if __name__ == "__main__":
    files = []
    for b in BC.BEARINGS:
        files += sorted(glob.glob(os.path.join(BC.DATA, "pu", b, "*.mat")))
    print("files", len(files), flush=True)
    t0 = time.time()
    res = []
    with Pool(max(1, (os.cpu_count() or 4) - 8)) as pool:
        for i, r in enumerate(pool.imap(process_fixed, files, chunksize=4)):
            res.append(r)
            if i % 200 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    bad = [(r[0], r[5]) for r in res if r[5]]
    ok = [r for r in res if not r[5]]
    t_fix = np.concatenate([r[3] for r in ok])
    bearing = np.concatenate([[r[1]] * r[4] for r in ok])
    cond = np.concatenate([[r[2]] * r[4] for r in ok])
    fname = np.concatenate([[r[0]] * r[4] for r in ok])
    win = np.concatenate([np.arange(r[4], dtype=np.int8) for r in ok])
    ref = np.load(os.path.join(BC.DATA, "cache", "pu_windows.npz"), allow_pickle=False)
    assert np.array_equal(fname, ref["fname"]) and np.array_equal(win, ref["win"]), "row order differs from pu_windows.npz"
    assert np.array_equal(bearing, ref["bearing"]) and np.array_equal(cond, ref["condition"])
    # at 1500 rpm the fixed-Hz tensor must equal t_mad; at other speeds it must differ
    n15 = np.array([c.startswith("N15") for c in cond])
    d15 = float(np.abs(t_fix[n15] - ref["t_mad"][n15]).max())
    d09 = float(np.abs(t_fix[~n15] - ref["t_mad"][~n15]).max())
    print("max |t_fix - t_mad| at N15:", d15, " at N09:", d09)
    assert d15 < 1e-4 and d09 > 1e-2
    np.savez(os.path.join(BC.DATA, "cache", "pu_windows_fixed.npz"), t_fix=t_fix, bearing=bearing, condition=cond, fname=fname, win=win)
    print("windows", len(bearing), "unreadable", bad, "seconds", round(time.time() - t0), flush=True)
