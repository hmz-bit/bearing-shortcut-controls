"""Phase A (extra): is there outer-race information for KA15/KA22/KA30 in bands the model does not see?
Raw 64 kHz records (no resampling), full 4 s, 0.25 Hz bins, squared-envelope MAD-SNR at f_o (max over +-0.25 Hz) per band; also with cepstrum pre-whitening (CPW).
Bands: 100-500, 500-1500, 1500-2500, 2500-5000 (model band 2), 5000-10000 (model band 3), 10-15k, 15-20k, 20-30k.  Descriptive only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from multiprocessing import Pool
import p2common as P
import features2 as F2
import geometry as G

BANDS = ((100., 500.), (500., 1500.), (1500., 2500.), (2500., 5000.), (5000., 10000.), (10000., 15000.), (15000., 20000.), (20000., 30000.))
meta = P.load_meta()
FS = 64000


def job(a):
    path, fr, cpw = a
    from pu_loader import load_vibration
    _, x = load_vibration(path)
    x = x[:4 * FS]
    spec = dict(L=4, cpw=cpw, bands=BANDS)
    if cpw:
        # whiten only inside 100-30000 Hz (amplitude 1, phase kept), as in the cepstrum pre-whitening literature
        X = np.fft.rfft(x.astype(np.float64)); f = np.fft.rfftfreq(len(x), 1 / FS)
        w = (f >= 100) & (f <= 30000)
        X2 = np.where(w, X / (np.abs(X) + 1e-12), 0)
        x = np.fft.irfft(X2, n=len(x)); spec["cpw"] = False
    freqs, E = F2.spectra(x, spec, fs=FS)
    ff = G.fault_frequencies(fr, G.PU_6203)
    out = []
    for b, (lo, hi) in enumerate(BANDS):
        sel = (freqs > 0) & (freqs <= hi - lo)
        m = np.median(E[b, sel]); s = 1.4826 * np.median(np.abs(E[b, sel] - m))
        snr = np.maximum(0, E[b] - m) / (s + 1e-12)
        k = np.abs(freqs - ff["f_o"]) <= 0.25
        out.append(float(snr[k].max()))
    return out


if __name__ == "__main__":
    rows = []
    for cond, fr in (("N15_M07_F10", 25.0), ("N09_M07_F10", 15.0)):
        files = []
        for b in P.HEALTHY + P.OUTER:
            fs_ = np.unique(meta["fname"][(meta["bearing"] == b) & (meta["condition"] == cond)])
            files += [(b, f) for f in fs_[:10]]            # 10 files per bearing
        for cpw in (False, True):
            with Pool(8) as pool:
                res = pool.map(job, [(str(P.WORK / "data" / "pu" / b / f), fr, cpw) for b, f in files], chunksize=2)
            for (b, f), r in zip(files, res):
                rows.append(dict(cond=cond, cpw=cpw, bearing=b, **{f"{int(lo/1000) if lo>=1000 else lo}-{int(hi)}": v for (lo, hi), v in zip(BANDS, r)}))
    d = pd.DataFrame(rows)
    d.to_csv(P.OUT / "diag5_band_snr_at_fo.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for (cond, cpw), g in d.groupby(["cond", "cpw"]):
        print(f"\n=== {cond}  CPW={cpw}: median over 10 files of linear SNR at f_o, per band (Hz) ===")
        print(g.groupby("bearing").median(numeric_only=True).round(1).drop(columns=["cpw"], errors="ignore").to_string())
