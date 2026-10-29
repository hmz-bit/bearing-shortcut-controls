"""Label-free snapshot selection for Paderborn RTF (rule in results/rtf/PREDECLARED_RTF.txt). Writes data/rtf/selection.csv.
In-range = SET speed inside the Table 2 range (+-1 rpm); healthy = first 10% of the in-range snapshots, faulty = last 10%; at most CAP per end,
evenly spaced by index; f_r from the measured speed if it agrees with the set speed within 3%, else from the set speed."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
import numpy as np, pandas as pd

RANGE = dict(B01=(2400, 3000), B02=(2400, 3000), B03=(2400, 3000), B04=(1500, 3000), B05=(1500, 3000), B06=(1500, 3600), B07=(1500, 3000),
             B08=(1500, 3000), B09=(1500, 3000), B10=(1500, 2700), B11=(1500, 2700), B12=(1500, 2700), B13=(1500, 2700), B14=(2700, 2700),
             B15=(2700, 2700), B16=(2700, 2700), B17=(2700, 2700))
CAP = 100
FRAC = 0.10


def main():
    rows, summ = [], []
    for e, (lo, hi) in RANGE.items():
        oc = pd.read_csv(WORK / "data" / "rtf" / "aux" / f"{e}_operatingConditions.csv")
        st, me = oc["setSpeed / rpm"].values, oc["meanAbs_speed / rpm"].values
        ok = np.where((st >= lo - 1) & (st <= hi + 1))[0]
        k = int(round(FRAC * len(ok)))
        first, last = ok[:k], ok[len(ok) - k:]
        for stage, sel in (("healthy", first), ("faulty", last)):
            pick = sel[np.unique(np.round(np.linspace(0, len(sel) - 1, min(CAP, len(sel)))).astype(int))]
            for i in pick:
                agree = abs(me[i] - st[i]) / st[i] <= 0.03
                rows.append(dict(exp=e, row=int(i), M=int(i) + 1, stage=stage, set_rpm=float(st[i]), meas_rpm=float(me[i]),
                                 f_r=float((me[i] if agree else st[i]) / 60.0), speed_source="measured" if agree else "set"))
        summ.append(dict(exp=e, n_total=len(oc), n_inrange=len(ok), k_10pct=k, healthy_M_range=f"{first[0]+1}-{first[-1]+1}", faulty_M_range=f"{last[0]+1}-{last[-1]+1}",
                         n_per_end=min(CAP, k), disagree_inrange=int((np.abs(me[ok] - st[ok]) / st[ok] > 0.03).sum())))
    pd.DataFrame(rows).to_csv(WORK / "data" / "rtf" / "selection.csv", index=False)
    s = pd.DataFrame(summ); s.to_csv(WORK / "results" / "rtf" / "selection_summary.csv", index=False)
    print(s.to_string()); print("windows", len(rows))


if __name__ == "__main__":
    main()
