"""Audit round 4 (results/audit4/PREDECLARED_AUDIT4.txt): P1 paired difference of contrasts on PU, P2 pooled stage-free primary contrast.
Usage: PYTHONUTF8=1 python -I audit4_primary.py"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "evaltools"))
import numpy as np, pandas as pd
import audit2_ab as A

OUT = WORK / "results" / "audit4"
OUT.mkdir(parents=True, exist_ok=True)
R = WORK / "results"
res = {}

# P1: PU
m = A.load_models("pu")
frames = {k: m[k] for k in ("base", "rowz_drop")}
clusters = sorted(set(frames["base"].cluster)); types = A.cluster_types(frames["base"], clusters)
M = A.draws(clusters, types, np.random.default_rng(A.SEED)); ones = np.ones((1, len(clusters)))
def contrast(keep_fn):
    out = {}
    for k, f in frames.items():
        T = A.pair_table(f, clusters, keep_fn(f) if keep_fn else None)
        out[k] = (A.macro(T, ones)[0][0], A.macro(T, M)[0])
    return out[("rowz_drop")][0] - out["base"][0], out["rowz_drop"][1] - out["base"][1]
keepA = lambda f: [(f.y_inner == 1) | f.healthy, (f.y_outer == 1) | f.healthy]
p_all, b_all = contrast(None)
p_hon, b_hon = contrast(keepA)
dd = b_hon - b_all
res["P1_PU"] = dict(contrast_all_negatives=[float(p_all), float(np.percentile(b_all, 2.5)), float(np.percentile(b_all, 97.5))],
                    contrast_healthy_only=[float(p_hon), float(np.percentile(b_hon, 2.5)), float(np.percentile(b_hon, 97.5))],
                    difference_of_contrasts=[float(p_hon - p_all), float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))])
print("P1", res["P1_PU"], flush=True)

# P2: stage-free pooled contrast
pts, boots = [], []
for i, ds in enumerate(("xjtu", "rtf")):
    fr = {k: A.std(pd.read_parquet(R / f"audit3/pred_k_{ds}_{k}.parquet"), ds) for k in ("base", "rowz_drop")}
    cl = sorted(set(fr["base"].cluster)); ty = A.cluster_types(fr["base"], cl)
    Md = A.draws(cl, ty, np.random.default_rng(A.SEED + i)); o = np.ones((1, len(cl)))
    T = {k: A.pair_table(f, cl) for k, f in fr.items()}
    pts.append(A.macro(T["rowz_drop"], o)[0][0] - A.macro(T["base"], o)[0][0])
    boots.append(A.macro(T["rowz_drop"], Md)[0] - A.macro(T["base"], Md)[0])
pooled = (boots[0] + boots[1]) / 2
res["P2_stage_free_pooled"] = dict(per_dataset_points=[float(x) for x in pts], pooled=[float((pts[0] + pts[1]) / 2), float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))])
print("P2", res["P2_stage_free_pooled"], flush=True)
(OUT / "primary_results.json").write_text(json.dumps(res, indent=1))
