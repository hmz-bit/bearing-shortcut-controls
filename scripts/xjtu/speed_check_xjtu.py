"""Label-free shaft-line check for XJTU-SY (PREDECLARED_XJTU.txt item 5). For every selected snapshot: Welch spectrum of the horizontal channel
(one 32768-sample segment, resolution 0.78 Hz), strongest line within +-5% of the nominal shaft frequency of the condition folder.
Writes results/xjtu/speed_check.txt and speed_check.csv; prints the folder medians and the rule outcome."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy import signal
import xjtu_common as C

sn = C.select_snapshots()
rows = []
for r in sn.itertuples():
    x = C.load_snapshot(r.bearing, r.fileno)
    f, P = signal.welch(x - x.mean(), fs=25600, window="hann", nperseg=32768)
    nom = C.NOMINAL_HZ[r.cond]
    sel = (f >= 0.95 * nom) & (f <= 1.05 * nom)
    pk = float(f[sel][np.argmax(P[sel])])
    prom = float(P[sel].max() / np.median(P[(f >= 0.5 * nom) & (f <= 1.5 * nom)]))
    rows.append(dict(bearing=r.bearing, stage=r.stage, fileno=r.fileno, cond=r.cond, nominal=nom, peak_hz=pk, diff_pct=100 * (pk - nom) / nom, prominence=prom))
d = pd.DataFrame(rows)
d.to_csv(C.RES / "speed_check.csv", index=False)
lines = ["XJTU-SY label-free shaft-line check (Welch, 0.78 Hz resolution, +-5% window around the nominal frequency)", f"snapshots checked: {len(d)}", ""]
for c in (1, 2, 3):
    g = d[d.cond == c]
    med = float(g.peak_hz.median()); nom = C.NOMINAL_HZ[c]
    lines.append(f"condition {c} nominal {nom} Hz: median peak {med:.2f} Hz ({100*(med-nom)/nom:+.2f}%), share within 2.5% of nominal {100*(np.abs(g.diff_pct) <= 2.5).mean():.0f}%, median prominence {g.prominence.median():.1f}")
    trig = abs(100 * (med - nom) / nom) > 2.5
    lines.append(f"  rule (replace nominal by the folder median if |difference| > 2.5%): {'TRIGGERED' if trig else 'not triggered, nominal frequency is used'}")
lines.append("")
lines.append("per bearing (median peak Hz, share of snapshots with prominence > 5):")
for b, g in d.groupby("bearing"):
    lines.append(f"  {b}: nominal {g.nominal.iloc[0]}, median peak {g.peak_hz.median():.2f} Hz, healthy-stage {g[g.stage=='healthy'].peak_hz.median():.2f}, faulty-stage {g[g.stage=='faulty'].peak_hz.median():.2f}, share prominent {100*(g.prominence>5).mean():.0f}%")
(C.RES / "speed_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
