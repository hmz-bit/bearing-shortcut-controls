"""One-shot supplementary evaluation on KAIST part 1 (results/kaist/PREDECLARED_KAIST.txt and its addendum).
Usage: python -I run_kaist.py build     (windows and features from data/kaist/raw -> data/kaist/kaist_windows.npz)
       python -I run_kaist.py run       (all models on the 9 primary splits -> results/kaist/)
Nothing is tuned; the configuration of the other datasets is used unchanged."""
import sys, json, pathlib, time, itertools
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (WORK / "scripts" / "phase2", WORK / "scripts" / "rf", WORK / "scripts" / "lfpncn"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd
import scipy.io as sio
from sklearn.metrics import roc_auc_score

RAW = WORK / "data" / "kaist" / "raw"
NPZ = WORK / "data" / "kaist" / "kaist_windows.npz"
OUT = WORK / "results" / "kaist"
FS, WIN, HOP, FR = 25600, 25600, 6400, 3010 / 60.0
SIZES = ("03", "10", "30")


def build():
    import features as F, geometry as G, rfcommon as R
    geom = G.Bearing("KAIST_6205", 9, 7.90, 38.5, 0.0)
    t_all, f_all, ts_all, meta = [], [], [], []
    for load in (0, 2, 4):
        for race in ("BPFI", "BPFO"):
            for size in SIZES:
                name = f"{load}Nm_{race}_{size}.mat"
                s = sio.loadmat(RAW / name)["Signal"][0, 0]
                inc = float(np.ravel(s["x_values"][0, 0]["increment"][0, 0])[0])
                assert abs(1.0 / inc - FS) < 1e-3, inc
                x = np.asarray(s["y_values"][0, 0]["values"][:, 0], dtype=np.float64)       # column 0 = x_direction_housing_A (article order, see addendum)
                assert len(x) == 1536000 and np.isfinite(x).all()
                for st in range(0, len(x) - WIN + 1, HOP):
                    w = x[st:st + WIN]
                    t_all.append(F.window_to_array(w, FR, geom, fs=FS))
                    f_all.append(R.window_features(w, FR, FS, geom))
                    ts_all.append(R.time_stats(w) + [float(w.mean())])
                    meta.append((race, size, load))
                print("built", name, len(meta), flush=True)
    race = np.array([m[0] for m in meta]); size = np.array([m[1] for m in meta]); load = np.array([m[2] for m in meta])
    unit = np.array([f"{r}_{s}" for r, s, _ in meta])
    Y = np.stack([(race == "BPFI"), (race == "BPFO")], 1).astype(np.float32)
    np.savez(NPZ, t_mad=np.stack(t_all).astype(np.float32), feat=np.stack(f_all), ts=np.array(ts_all), unit=unit, race=race, size=size, load=load, Y=Y)
    print("saved", NPZ, Y.shape, flush=True)


def auc(Y, S):
    a, b = roc_auc_score(Y[:, 0], S[:, 0]), roc_auc_score(Y[:, 1], S[:, 1])
    return 100 * a, 100 * b, 100 * (a + b) / 2


def run():
    import p2common as P, trainer as T, variants as V, common as CC, rfcommon as R
    CC.set_determinism(); dev = P.device()
    z = dict(np.load(NPZ, allow_pickle=False))
    X, Y, unit, load = z["t_mad"], z["Y"], z["unit"], z["load"]
    feat, ts = z["feat"], z["ts"]
    inner_units = [f"BPFI_{s}" for s in SIZES]; outer_units = [f"BPFO_{s}" for s in SIZES]
    splits = list(itertools.product(inner_units, outer_units))           # 9 primary splits, index 1..9
    matched = {i + 1 for i, (a, b) in enumerate(splits) if a.split("_")[1] == b.split("_")[1]}
    models = ["base", "rowz_drop", "rf", "rfu", "ref_meta_load", "ref_amplitude"]
    feats = {"rf": feat, "rfu": feat, "ref_meta_load": load[:, None].astype(float), "ref_amplitude": ts}
    params = {"rf": R.PARAMS_PRIMARY, "rfu": R.PARAMS_U, "ref_meta_load": R.PARAMS_U, "ref_amplitude": R.PARAMS_U}
    rows, preds, t0 = [], [], time.time()
    for k, (ui, uo) in enumerate(splits, start=1):
        te = np.isin(unit, [ui, uo]); tr = ~te
        assert not np.any(tr & te)
        for m in models:
            if m in ("base", "rowz_drop"):
                S = T.train_eval(X, Y, unit, None, None, dict(P.BASE_CFG), seed=k, opt=dict(V.VARIANTS[m]["opt"]), dev=dev, train_mask=tr, held_mask=te)["scores"]
            else:
                S = R.fit_scores(feats[m][tr], Y[tr], feats[m][te], params[m])
            ai, ao, am = auc(Y[te], S)
            rows.append(dict(split=k, test_inner=ui, test_outer=uo, size_matched=k in matched, model=m, inner=ai, outer=ao, macro=am))
            preds.append(pd.DataFrame({"split": k, "model": m, "unit": unit[te], "load": load[te], "y_inner": Y[te, 0].astype(int), "y_outer": Y[te, 1].astype(int),
                                       "s_inner": S[:, 0], "s_outer": S[:, 1]}))
            print(f"split {k} ({ui} vs {uo}) {m}: macro {am:.1f} (inner {ai:.1f} outer {ao:.1f}) t={time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_split.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "pred_all_models.parquet")
    summ = {}
    for m in models:
        d = df[df.model == m]
        summ[m] = dict(primary_mean_macro=float(d.macro.mean()), primary_mean_inner=float(d.inner.mean()), primary_mean_outer=float(d.outer.mean()),
                       size_matched_mean_macro=float(d[d.size_matched].macro.mean()), size_unmatched_mean_macro=float(d[~d.size_matched].macro.mean()),
                       n_splits=int(len(d)), n_size_matched=int(d.size_matched.sum()))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
