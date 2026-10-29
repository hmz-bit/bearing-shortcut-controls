"""Parts E, F, G, I of results/audit3/PREDECLARED_AUDIT3.txt (no training except the metadata forest of F).
Needs results/audit3/pred_uored_{rf_dm,rfu_dm}.parquet from audit3_rf_dc.py for E.
Usage: PYTHONUTF8=1 python -I audit3_e_i.py [E|F|G|I ...]    Output: results/audit3/{e,f,g,i}_results.json"""
import sys, json, pathlib, re, time
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "rf")); sys.path.insert(0, str(WORK / "scripts" / "evaltools"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import audit2_ab as A

OUT = WORK / "results" / "audit3"
R = WORK / "results"


def macro_l(T, M, l):
    """mean over defined splits of the AUROC of label l (0 inner, 1 outer) for multiplicities M (D,Bn)."""
    num = np.einsum("di,sij,dj->ds", M, T["C"][:, l], M, optimize=True)
    dp, dn = M @ T["Np"][:, l].T, M @ T["Nn"][:, l].T
    good = (dp > 0) & (dn > 0)
    per = np.where(good, num / np.where(good, dp * dn, 1), np.nan)
    return np.nanmean(per, axis=1) * 100, good.sum(1)


def contrast(tabs, M, ones, labels, pairs):
    """tabs: model -> pair table. labels: list of 0/1/'macro'. Returns dict label -> {points, ndef, diff}."""
    out = {}
    for lab in labels:
        pts, nd, boots = {}, {}, {}
        for k, T in tabs.items():
            if lab == "macro":
                p, n = A.macro(T, ones); b = A.macro(T, M)[0]
            else:
                p, n = macro_l(T, ones, lab); b = macro_l(T, M, lab)[0]
            pts[k], nd[k], boots[k] = float(p[0]), int(n[0]), b
        diff = {}
        for a, b_ in pairs:
            if a in pts and b_ in pts:
                dd = boots[a] - boots[b_]
                diff[f"{a}-{b_}"] = [pts[a] - pts[b_], float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
        out[str(lab)] = dict(points=pts, n_splits_defined=nd, diff=diff)
    return out


PAIRS = (("rowz_drop", "base"), ("rowz_drop", "b1"), ("rowz_drop", "rf"), ("rowz_drop", "rf_dm"), ("base", "b1"))


def part_e():
    import warnings; warnings.filterwarnings("ignore")
    raw = {k: pd.read_parquet(R / f"uored/pred_{k}.parquet") for k in ("base", "rowz_drop", "b1")}
    raw["rf"] = pd.read_parquet(R / "rf/pred_uored_rf.parquet")
    for nm in ("rf_dm", "rfu_dm"):
        raw[nm] = pd.read_parquet(OUT / f"pred_uored_{nm}.parquet")
    raw["rfu"] = pd.read_parquet(R / "rf/pred_uored_rfu.parquet")
    fr = {}
    for k, v in raw.items():
        d = A.std(v, "uored")
        d["cls"] = v["cls"].to_numpy(); d["state"] = v["state"].to_numpy(); d["bn"] = v["bearing"].astype(int).to_numpy()
        fr[k] = d
    clusters = sorted(set(fr["base"].cluster)); types = A.cluster_types(fr["base"], clusters)
    M = A.draws(clusters, types, np.random.default_rng(A.SEED)); ones = np.ones((1, len(clusters)))

    def masks(d, kind):
        pos = [d.y_inner == 1, d.y_outer == 1]
        allt = pd.Series(True, index=d.index)
        out = []
        for l in (0, 1):
            p, neg = pos[l], ~pos[l]
            if kind == "E1": m = allt
            elif kind == "E2": m = p | (d.cls == "H")
            elif kind == "E3": m = p | (neg & (d.bn >= 6))
            elif kind == "E4": m = (p & (d.state == 2)) | (neg & (d.state == 2))
            elif kind == "E5": m = (p & (d.state == 2)) | (neg & (d.state == 2) & (d.bn >= 6))
            out.append(m)
        return out

    res = {}
    for kind in ("E1", "E2", "E3", "E4", "E5"):
        tabs = {k: A.pair_table(d, clusters, masks(d, kind)) for k, d in fr.items()}
        res[kind] = contrast(tabs, M, ones, [0, 1], PAIRS + (("rowz_drop", "rfu_dm"),))
        print("E", kind, {l: {k: round(v, 1) for k, v in r["points"].items()} for l, r in res[kind].items()}, flush=True)
    (OUT / "e_results.json").write_text(json.dumps(res, indent=1))


def part_i():
    cm = {"base": A.cat(R / "cwru/pred_base_r*.parquet"), "rowz_drop": A.cat(R / "cwru/pred_rowz_drop_r*.parquet"),
          "b1": pd.read_parquet(R / "cwru/pred_b1_r3-52.parquet"), "rf": pd.read_parquet(R / "rf/pred_cwru_rf.parquet")}
    fr, keeps = {}, {}
    for k, v in cm.items():
        d = A.std(v, "cwru")
        hl = v[v.healthy].groupby("split").sig_loc.agg(lambda s: s.unique()[0])
        m = (v.sig_loc.to_numpy() == v.split.map(hl).to_numpy())
        fr[k] = d; keeps[k] = [pd.Series(m), pd.Series(m)]
    clusters = sorted(set(fr["base"].cluster)); types = A.cluster_types(fr["base"], clusters)
    M = A.draws(clusters, types, np.random.default_rng(A.SEED)); ones = np.ones((1, len(clusters)))
    tabs = {k: A.pair_table(fr[k], clusters, keeps[k]) for k in fr}
    res = {"channel_matched": contrast(tabs, M, ones, ["macro", 0, 1], PAIRS)}
    (OUT / "i_results.json").write_text(json.dumps(res, indent=1))
    print("I", {l: {k: round(v, 2) for k, v in r["points"].items()} for l, r in res["channel_matched"].items()}, flush=True)


def part_f():
    import rfcommon as RC
    z = np.load(WORK / "data/cache/pu_windows.npz", allow_pickle=False)
    V = ["K001", "K002", "K003", "K004", "K005", "K006", "KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
    keep = np.isin(z["bearing"], V)
    b, cond, fn = z["bearing"][keep], z["condition"][keep], z["fname"][keep]
    sm = pd.read_csv(R / "b1/signals_meta.csv")
    sm["key"] = sm.fname.str.replace(".mat", "", regex=False)
    sm = sm.set_index("key")
    key = np.array([f.replace(".mat", "") for f in fn])
    miss = set(key) - set(sm.index)
    assert not miss, list(miss)[:5]
    s = sm.loc[key]
    dc = (s.raw_sum / s.n_raw).to_numpy(float); nraw = s.n_raw.to_numpy(float)
    cnum = np.array([[int(x) for x in re.findall(r"\d+", c)] for c in cond], float)
    import pu_data as pu
    Y = np.array([pu.bearing_labels(x) for x in b], float)
    sets = {"F1_condition": cnum, "F2_dc": dc[:, None], "F3_dc_len": np.stack([dc, nraw], 1), "F4_cond_dc_len": np.concatenate([cnum, dc[:, None], nraw[:, None]], 1)}
    res = {}
    for nm, X in sets.items():
        vi, vo = [], []
        for run in range(5, 105):
            tr_ids, te_ids = pu.load_eval_split(run)
            tr, te = np.isin(b, tr_ids), np.isin(b, te_ids)
            S = RC.fit_scores(X[tr], Y[tr], X[te], RC.PARAMS_U)
            try:
                vi.append(100 * roc_auc_score(Y[te, 0], S[:, 0])); vo.append(100 * roc_auc_score(Y[te, 1], S[:, 1]))
            except ValueError:
                pass
        res[nm] = dict(inner=float(np.mean(vi)), outer=float(np.mean(vo)), macro=float((np.mean(vi) + np.mean(vo)) / 2), n_splits=len(vi), above_56=bool((np.mean(vi) + np.mean(vo)) / 2 > 56))
        print("F", nm, res[nm], flush=True)
    bm = pd.DataFrame({"bearing": b, "dc": dc}).groupby("bearing").dc.agg(["mean", "std"]).round(4)
    res["per_bearing_dc"] = bm.to_dict("index")
    (OUT / "f_results.json").write_text(json.dumps(res, indent=1))


def part_g():
    HIGH = "KA04 KA15 KA16 KA22 KI14 KI16 KI17 KI18 KI21".split(); LOW = ["KA30", "KI04"]
    models = A.load_models("pu")
    res = {}
    for k, d in models.items():
        d = d.copy(); d["dmg"] = np.maximum(d.s_inner, d.s_outer)
        u = d.groupby(["split", "cluster"]).agg(score=("dmg", "mean"), dam=("y_inner", "max"), dam2=("y_outer", "max")).reset_index()
        u["dam"] = (u.dam | u.dam2).astype(int)
        out = {"high": [], "low": []}
        share = {}
        for s, g in u.groupby("split"):
            h = g[g.dam == 0]
            if h.empty: continue
            for nm, grp in (("high", HIGH), ("low", LOW)):
                p = g[g.cluster.isin(grp)]
                if p.empty: continue
                out[nm].append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(h))], np.r_[p.score.to_numpy(), h.score.to_numpy()]))
            for _, r in g[g.dam == 1].iterrows():
                share.setdefault(r.cluster, []).append(float((h.score < r.score).mean()))
        res[k] = dict(auroc_high_dc_damaged_vs_healthy=100 * float(np.mean(out["high"])), n_high=len(out["high"]),
                      auroc_low_dc_damaged_vs_healthy=100 * float(np.mean(out["low"])), n_low=len(out["low"]),
                      per_bearing_share_healthy_below={b: round(100 * float(np.mean(v)), 1) for b, v in sorted(share.items())})
        print("G", k, {a: (round(v, 1) if isinstance(v, float) else v) for a, v in res[k].items() if a != "per_bearing_share_healthy_below"}, flush=True)
    (OUT / "g_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["I", "G", "F", "E"]
    for p in parts:
        t0 = time.time(); {"E": part_e, "F": part_f, "G": part_g, "I": part_i}[p](); print(p, "done", round(time.time() - t0), "s", flush=True)
