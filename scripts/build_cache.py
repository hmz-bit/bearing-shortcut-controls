"""Build the window cache for LF-PNCN and B3 from the raw PU .mat files (no pickle).

Output: data/cache/pu_windows.npz with, for N windows (4 non-overlapping 1-s windows per file, 42 kHz):
  t_mad  (N,3,15,64) float32  LF-PNCN input, median/MAD log-SNR (modes OR, IR, Ball)
  t_max  (N,3,15,64) float32  same grid, max-peak normalisation (ablation B4)
  b3     (N,20) float32       5 time statistics + 15 broadband median/MAD log-SNR values (1x-5x of f_o, f_i, f_b)
  bearing (N,) str, condition (N,) str, fname (N,) str, win (N,) int8
A file data/cache/DONE is written at the end. Bearings: 17 real/healthy (Vieira) + 12 artificial; KB* excluded.
"""
import glob, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lfpncn"))
from pu_loader import load_vibration
import features as F
import geometry as G

DATA = os.path.join(HERE, "..", "data")
BEARINGS = ("K001 K002 K003 K004 K005 K006 KA04 KA15 KA16 KA22 KA30 KI04 KI14 KI16 KI17 KI18 KI21 "
            "KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08").split()
FS = F.FS_TARGET


def broadband_logsnr(w, f_r):
    """15 values: median/MAD log-SNR of the 500-10000 Hz squared-envelope spectrum at h*f_m (h=1..5), max within +-Delta."""
    sos = signal.butter(4, [500.0, min(10000.0, FS / 2 - 1)], btype="band", fs=FS, output="sos")
    xb = signal.sosfiltfilt(sos, w)
    e = np.abs(signal.hilbert(xb)) ** 2
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e)) / len(e)
    fr = np.fft.rfftfreq(len(e), 1 / FS)
    sel = (fr >= 1.0) & (fr <= 9500.0)
    m = np.median(E[sel])
    s = 1.4826 * np.median(np.abs(E[sel] - m))
    Et = np.log10(1.0 + np.maximum(0.0, E - m) / (s + 1e-12))
    ff = G.fault_frequencies(f_r, G.PU_6203)
    out = []
    for key, fmod in (("f_o", 0.0), ("f_i", f_r), ("f_b", ff["f_c"])):
        for h in range(1, 6):
            c = h * ff[key]
            dlt = max(0.03 * h * ff[key], 1.2 * fmod)
            idx = (fr >= c - dlt) & (fr <= c + dlt)
            out.append(float(Et[idx].max()))
    return out


def time_stats(w):
    rms = float(np.sqrt(np.mean(w ** 2)))
    return [rms, float(w.max() - w.min()), float(stats.kurtosis(w, fisher=False)), float(stats.skew(w)), float(np.abs(w).max() / rms)]


def process(path):
    try:
        meta, x = load_vibration(path)
        f_r = G.shaft_frequency_from_name(os.path.splitext(os.path.basename(path))[0])
        wins = F.make_windows(F.resample_to_target(x))[:4]
        t_mad, t_max, b3 = [], [], []
        for w in wins:
            freqs, E = F.squared_envelope_spectra(w, FS)
            ffq = G.fault_frequencies(f_r, G.PU_6203)
            arrs = []
            for norm in ("mad", "max"):
                Et = F.normalise_spectra(freqs, E, normalize=norm)
                modes = [F.order_resample(freqs, Et, ffq["f_o"], 0.0),
                         F.order_resample(freqs, Et, ffq["f_i"], f_r),
                         F.order_resample(freqs, Et, ffq["f_b"], ffq["f_c"])]
                arrs.append(np.stack(modes).astype(np.float32))
            t_mad.append(arrs[0]); t_max.append(arrs[1])
            b3.append(time_stats(w) + broadband_logsnr(w, f_r))
        n = len(wins)
        return (os.path.basename(path), meta["bearing_id"], meta["condition"], np.stack(t_mad), np.stack(t_max),
                np.array(b3, dtype=np.float32), n, None)
    except Exception as e:  # corrupt source file (e.g. KA08 N15_M01_F10_2)
        return (os.path.basename(path), None, None, None, None, None, 0, repr(e)[:80])


if __name__ == "__main__":
    os.makedirs(os.path.join(DATA, "cache"), exist_ok=True)
    files = []
    for b in BEARINGS:
        files += sorted(glob.glob(os.path.join(DATA, "pu", b, "*.mat")))
    print("files", len(files), flush=True)
    t0 = time.time()
    res = []
    with Pool(max(1, (os.cpu_count() or 4) - 4)) as pool:
        for i, r in enumerate(pool.imap(process, files, chunksize=4)):
            res.append(r)
            if i % 200 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    bad = [(r[0], r[7]) for r in res if r[7]]
    ok = [r for r in res if not r[7]]
    out = dict(
        t_mad=np.concatenate([r[3] for r in ok]), t_max=np.concatenate([r[4] for r in ok]), b3=np.concatenate([r[5] for r in ok]),
        bearing=np.concatenate([[r[1]] * r[6] for r in ok]), condition=np.concatenate([[r[2]] * r[6] for r in ok]),
        fname=np.concatenate([[r[0]] * r[6] for r in ok]), win=np.concatenate([np.arange(r[6], dtype=np.int8) for r in ok]),
    )
    np.savez(os.path.join(DATA, "cache", "pu_windows.npz"), **out)
    print("windows", len(out["bearing"]), "unreadable", bad, "seconds", round(time.time() - t0), flush=True)
    open(os.path.join(DATA, "cache", "DONE"), "w").write("ok")
