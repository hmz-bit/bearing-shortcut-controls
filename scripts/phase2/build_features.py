"""Build a row-aligned feature array for one spec from the raw .mat files (17 Vieira bearings, same row order as load_meta()).
Usage: python -I build_features.py L4 [--procs 8] [--check]
Output: results/phase2/cache/feat_<name>.npy  (N,3,rows,64) float32.   --check also verifies L1 == cache t_mad on 40 files.
"""
import argparse, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
import p2common as P
import features2 as F2

CACHE = P.OUT / "cache"


def job(args):
    path, name = args
    return F2.build_file(path, F2.SPECS[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = P.load_meta()
    fn = meta["fname"]
    # file order = order of first appearance; rows of a file are contiguous and win = 0..3
    uniq, first = np.unique(fn, return_index=True)
    order = uniq[np.argsort(first)]
    paths = [str(P.WORK / "data" / "pu" / f.split("_")[3] / (f if f.endswith(".mat") else f + ".mat")) for f in order]
    if a.check:
        paths_, order_ = paths[::34][:40], order[::34][:40]
    else:
        paths_, order_ = paths, order
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, [(p, a.name) for p in paths_], chunksize=4)
    print("built", len(res), "files in", round(time.time() - t0), "s", flush=True)
    if a.check:
        X = P.load_t_mad()
        mx = 0.0
        for f, r in zip(order_, res):
            idx = np.nonzero(fn == f)[0]
            assert list(meta["win"][idx]) == list(range(len(idx))), f
            mx = max(mx, float(np.abs(X[idx] - r[:len(idx)]).max()))
        print("max |built - cache t_mad| =", mx)
        return
    X = np.zeros((len(fn), 3, res[0].shape[2], 64), np.float32)
    for f, r in zip(order, res):
        idx = np.nonzero(fn == f)[0]
        assert list(meta["win"][idx]) == list(range(len(idx))), f
        X[idx] = r[:len(idx)]
    np.save(CACHE / f"feat_{a.name}.npy", X)
    print("saved", X.shape)


if __name__ == "__main__":
    main()
