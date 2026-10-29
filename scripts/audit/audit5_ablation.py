"""Component ablation (results/ablation/PREDECLARED_ABLATION.txt): contrasts among base, rowz (row z-score only), base_drop (dropout only), rowz_drop.
Usage: PYTHONUTF8=1 python -I audit5_ablation.py   Output: results/ablation/ablation_results.json"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "evaltools")); sys.path.insert(0, str(WORK / "scripts" / "rf"))
import numpy as np, pandas as pd
import audit2_ab as A
import audit3_e_i as E

R = WORK / "results"
OUT = R / "ablation"
PAIRS = (("rowz", "base"), ("base_drop", "base"), ("rowz_drop", "base"), ("rowz_drop", "rowz"), ("rowz_drop", "base_drop"))
res = {}


def run(frames, keeps=None, labels=("macro",)):
    clusters = sorted(set(frames["base"].cluster)); types = A.cluster_types(frames["base"], clusters)
    M = A.draws(clusters, types, np.random.default_rng(A.SEED)); ones = np.ones((1, len(clusters)))
    tabs = {k: A.pair_table(f, clusters, keeps[k] if keeps else None) for k, f in frames.items()}
    return E.contrast(tabs, M, ones, list(labels), PAIRS)


# UORED
raw = {"base": pd.read_parquet(R / "uored/pred_base.parquet"), "rowz_drop": pd.read_parquet(R / "uored/pred_rowz_drop.parquet"),
       "rowz": pd.read_parquet(R / "uored/pred_rowz.parquet"), "base_drop": pd.read_parquet(R / "uored/pred_base_drop.parquet")}
fr = {}
for k, v in raw.items():
    d = A.std(v, "uored"); d["bn"] = v["bearing"].astype(int).to_numpy(); fr[k] = d
res["uored_original"] = run(fr, labels=("macro",))
keep3 = {k: [(d.y_inner == 1) | ((d.y_inner == 0) & (d.bn >= 6)), (d.y_outer == 1) | ((d.y_outer == 0) & (d.bn >= 6))] for k, d in fr.items()}
res["uored_outer_E3"] = run(fr, keep3, labels=(1,))
print("UORED done", flush=True)

# CWRU
cw = {"base": A.cat(R / "cwru/pred_base_r*.parquet"), "rowz_drop": A.cat(R / "cwru/pred_rowz_drop_r*.parquet"),
      "rowz": A.cat(R / "cwru/pred_rowz_r*.parquet"), "base_drop": A.cat(R / "cwru/pred_base_drop_r*.parquet")}
fr, keeps = {}, {}
for k, v in cw.items():
    fr[k] = A.std(v, "cwru")
    hl = v[v.healthy].groupby("split").sig_loc.agg(lambda s: s.unique()[0])
    m = pd.Series(v.sig_loc.to_numpy() == v.split.map(hl).to_numpy())
    keeps[k] = [m, m]
res["cwru_original"] = run(fr)
res["cwru_channel_matched"] = run(fr, keeps)
print("CWRU done", flush=True)

# stage-free
for ds in ("xjtu", "rtf"):
    fr = {k: A.std(pd.read_parquet(R / f"audit3/pred_k_{ds}_{k}.parquet"), ds) for k in ("base", "rowz", "base_drop", "rowz_drop")}
    res[f"{ds}_stage_free"] = run(fr, labels=("macro", 0, 1))
    print(ds, "done", flush=True)
OUT.mkdir(exist_ok=True)
(OUT / "ablation_results.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    for lab, x in v.items():
        print(k, lab, {a: round(b, 1) for a, b in x["points"].items()})
        print("    ", {a: [round(t, 1) for t in d] for a, d in x["diff"].items()})
