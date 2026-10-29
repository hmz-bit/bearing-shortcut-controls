"""Windows for the XJTU-SY experiments (PREDECLARED_XJTU.txt items 2-6, 8, 12).
Output: data/xjtu/xjtu_windows.npz (t_mad (N,3,15,64), Y (N,2) [IR, OR], exp, stage, fileno, N_files, cond, sens5 (bool), raw_n, raw_sum, raw_sumsq)
and data/xjtu/xjtu_env.npy (N,3,21000) float16 = B1 envelope spectrum x 100 for the three crop starts (0, 3584, 7168 raw samples)."""
import sys, pathlib, math, time
from multiprocessing import Pool
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "lfpncn")); sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd
from scipy import signal
import features as F
import geometry as G
import envspec
import xjtu_common as C

XJTU_UER204 = G.Bearing("XJTU_UER204", 8, 7.92, 34.55, 0.0)
STARTS = (0, 3584, 7168)
SCALE = 100.0


def to42k(x25600):
    return signal.resample_poly(np.asarray(x25600, dtype=np.float64), 105, 64)


def proc(args):
    b, stage, fileno, cond = args
    x = C.load_snapshot(b, fileno)
    crops = [to42k(x[s:s + 25600]) for s in STARTS]
    assert all(len(c) == 42000 for c in crops)
    t = F.window_to_array(crops[0], C.NOMINAL_HZ[cond], XJTU_UER204)
    env = np.stack([envspec.envelope_spectrum(c) for c in crops]) * SCALE
    w = crops[0]
    return t, env, len(w), float(w.sum()), float((w ** 2).sum())


if __name__ == "__main__":
    sn = C.select_snapshots()
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(proc, [(r.bearing, r.stage, r.fileno, r.cond) for r in sn.itertuples()], chunksize=8)
    T = np.stack([r[0] for r in res]).astype(np.float32)
    E = np.stack([r[1] for r in res])
    assert np.isfinite(T).all() and np.isfinite(E).all() and np.abs(E).max() < 6.0e4, float(np.abs(E).max())
    E16 = E.astype(np.float16)
    Y = np.zeros((len(sn), 2), np.float32)
    faulty = (sn.stage == "faulty").to_numpy()
    Y[:, 0] = faulty & sn.bearing.isin(C.IRPOS).to_numpy()
    Y[:, 1] = faulty & sn.bearing.isin(C.ORPOS).to_numpy()
    # 5% sensitivity rule: first / last ceil(5% N) files (at least 3)
    m5 = np.maximum(3, np.ceil(0.05 * sn.N.to_numpy())).astype(int)
    sens5 = np.where(sn.stage.to_numpy() == "healthy", sn.fileno.to_numpy() <= m5, sn.fileno.to_numpy() > sn.N.to_numpy() - m5)
    np.savez(C.DATA / "xjtu_windows.npz", t_mad=T, Y=Y, exp=sn.bearing.to_numpy().astype(str), stage=sn.stage.to_numpy().astype(str),
             fileno=sn.fileno.to_numpy().astype(np.int32), N_files=sn.N.to_numpy().astype(np.int32), cond=sn.cond.to_numpy().astype(np.int8), sens5=sens5,
             raw_n=np.array([r[2] for r in res]), raw_sum=np.array([r[3] for r in res]), raw_sumsq=np.array([r[4] for r in res]))
    np.save(C.DATA / "xjtu_env.npy", E16)
    sn.assign(y_ir=Y[:, 0], y_or=Y[:, 1], sens5=sens5).to_csv(C.RES / "selection_summary_windows.csv", index=False)
    print("windows", len(sn), "t_mad", T.shape, "env", E16.shape, "max env x100", float(np.abs(E).max()), "sens5 windows", int(sens5.sum()), "seconds", round(time.time() - t0))
    print(sn.groupby(["bearing", "stage"]).size().unstack().T.to_string())
