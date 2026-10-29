"""Collects results/rf/rf_results_<dataset>.json into results/rf/rf_summary_table.csv and rf_summary_table.txt (Macro-AUROC with 95% intervals and differences)."""
import json, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pandas as pd
import rfcommon as R

rows, lines = [], []
for ds in ("cwru", "uored", "xjtu", "rtf"):
    f = R.RES / f"rf_results_{ds}.json"
    if not f.exists():
        continue
    r = json.load(open(f))
    for tag in [t for t in ("primary", "stage_matched") if t in r]:
        lines.append(f"== {ds} {tag}")
        for n, v in r[tag].items():
            if "_minus_" in n:
                if n.endswith("_macro"):
                    lines.append(f"   {n:34s} {v['point']:+6.2f} [{v['ci'][0]:+6.2f}, {v['ci'][1]:+6.2f}]")
                    rows.append(dict(dataset=ds, analysis=tag, item=n, point=v["point"], ci_lo=v["ci"][0], ci_hi=v["ci"][1]))
            else:
                lines.append(f"   {n:10s} macro {v['macro']:6.2f} [{v['macro_ci'][0]:6.2f}, {v['macro_ci'][1]:6.2f}] inner {v['inner']:6.2f} outer {v['outer']:6.2f} valid {v['n_valid_splits']}")
                rows.append(dict(dataset=ds, analysis=tag, item=n, point=v["macro"], ci_lo=v["macro_ci"][0], ci_hi=v["macro_ci"][1], inner=v["inner"], outer=v["outer"], n_valid=v["n_valid_splits"]))
    lines.append("   decisions: " + json.dumps(r["decisions"]))
pd.DataFrame(rows).to_csv(R.RES / "rf_summary_table.csv", index=False)
(R.RES / "rf_summary_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
