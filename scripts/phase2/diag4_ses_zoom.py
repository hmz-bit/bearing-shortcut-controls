"""Phase A (hypothesis): at 900 rpm f_o = 45.81 Hz is 0.81 Hz from 3*f_r = 45.0 Hz (1500 rpm: 76.36 vs 75.0, gap 1.36 Hz).
Zoomed squared-envelope spectrum (full 4 s record, 0.25 Hz bins, MAD-normalised SNR) of healthy vs outer-race (and inner-race for f_i vs 5 f_r)
recordings. Reports, per bearing and speed, the SNR at the bin of 3 f_r and at the bin of f_o (max over +-0.25 Hz), per band.
Uses the 17 Vieira bearings, operating conditions N09_M07_F10 and N15_M07_F10. Descriptive only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from multiprocessing import Pool
import p2common as P
import features2 as F2
import geometry as G
import features as F
from pu_loader import load_vibration

SPEC = dict(L=4, cpw=False, bands=F2.BANDS3)
meta = P.load_meta()


def snr_spectrum(path):
    m, x = load_vibration(path)
    x42 = F.resample_to_target(x)[:4 * F.WIN]
    freqs, E = F2.spectra(x42, SPEC)
    out = []
    for b, (lo, hi) in enumerate(SPEC["bands"]):
        sel = (freqs > 0) & (freqs <= hi - lo)
        mm = np.median(E[b, sel]); sg = 1.4826 * np.median(np.abs(E[b, sel] - mm))
        out.append(np.maximum(0, E[b] - mm) / (sg + 1e-12))     # linear SNR
    return np.stack(out)[:, :int(400 / 0.25) + 1]


def at(snr, freqs, f0, tol=0.25):
    sel = np.abs(freqs - f0) <= tol
    return snr[:, sel].max(1)


if __name__ == "__main__":
    freqs = np.arange(int(400 / 0.25) + 1) * 0.25
    rows = []
    for cond, fr in (("N09_M07_F10", 15.0), ("N15_M07_F10", 25.0)):
        ff = G.fault_frequencies(fr, G.PU_6203)
        files = []
        for b in P.BEARINGS:
            fs_ = np.unique(meta["fname"][(meta["bearing"] == b) & (meta["condition"] == cond)])
            files += [(b, f) for f in fs_]
        with Pool(8) as pool:
            res = pool.map(snr_spectrum, [str(P.WORK / "data" / "pu" / b / f) for b, f in files], chunksize=4)
        for (b, f), s in zip(files, res):
            cls = "healthy" if b.startswith("K0") else "outer" if b.startswith("KA") else "inner"
            rows.append(dict(cond=cond, bearing=b, cls=cls, file=f,
                             **{f"b{k+1}_3fr": v for k, v in enumerate(at(s, freqs, 3 * fr))},
                             **{f"b{k+1}_fo": v for k, v in enumerate(at(s, freqs, ff["f_o"]))},
                             **{f"b{k+1}_5fr": v for k, v in enumerate(at(s, freqs, 5 * fr))},
                             **{f"b{k+1}_fi": v for k, v in enumerate(at(s, freqs, ff["f_i"]))}))
        # mean zoomed spectrum around f_o per class, band 2 (2.5-5 kHz): print peak positions
        for cls, names in (("healthy", P.HEALTHY), ("outer", P.OUTER)):
            idx = [i for i, (b, f) in enumerate(files) if b in names]
            m = np.mean([res[i][1] for i in idx], axis=0)
            lo_f, hi_f = ff["f_o"] - 3, ff["f_o"] + 3
            sel = (freqs >= lo_f) & (freqs <= hi_f)
            top = np.argsort(m[sel])[::-1][:3]
            print(f"{cond} {cls}: band2 mean SNR spectrum in f_o+-3 Hz: top bins (Hz, SNR) =",
                  [(round(float(freqs[sel][t]), 2), round(float(m[sel][t]), 2)) for t in top],
                  "| at 3fr", round(float(at(m[None], freqs, 3 * fr)[0]), 2), " at f_o", round(float(at(m[None], freqs, ff["f_o"])[0]), 2))
    d = pd.DataFrame(rows)
    d.to_csv(P.OUT / "diag4_ses_zoom_per_file.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nmean linear SNR at the 3*f_r bin and the f_o bin (0.25 Hz resolution, 4-s record), by speed, class and band")
    g = d.groupby(["cond", "cls"])[[f"b{k}_{s}" for s in ("3fr", "fo") for k in (1, 2, 3)]].mean().round(1)
    print(g.to_string())
    print("\nper outer/healthy bearing, 900 rpm: mean SNR at f_o (bands 1,2,3) / at 3fr")
    g = d[d.cond == "N09_M07_F10"].groupby("bearing")[[f"b{k}_{s}" for s in ("fo", "3fr") for k in (1, 2, 3)]].mean().round(1)
    print(g.to_string())
    print("\ninner race: mean SNR at f_i vs 5 f_r, by speed and class")
    print(d.groupby(["cond", "cls"])[[f"b{k}_{s}" for s in ("fi", "5fr") for k in (1, 2, 3)]].mean().round(1).to_string())
