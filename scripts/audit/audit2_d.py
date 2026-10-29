"""Part D of results/audit2/PREDECLARED_AUDIT2.txt: test-invariance fingerprint.
Train the phase 2 trainer (rowz_drop options) for 3 epochs on CPU, 1 thread, seed 0, on the training bearings of PU tuning split 0, with
(i) the real validation bearings as held-out, (i') the same again (determinism check of the test itself), (ii) held-out windows replaced by Gaussian noise,
(iii) a single different held-out bearing. The SHA-256 of all model parameters and buffers must be identical in all runs.
Usage: P2_DEVICE=cpu P2_THREADS=1 PYTHONUTF8=1 python -I audit2_d.py"""
import os, sys, pathlib, hashlib, json
os.environ.setdefault("P2_DEVICE", "cpu"); os.environ.setdefault("P2_THREADS", "1")
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "scripts" / "phase2"))
import numpy as np, torch
import p2common as P
import trainer as T
import common as C
import variants as V

C.set_determinism()
torch.use_deterministic_algorithms(True)


def digest(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


meta = P.load_meta()
v = V.VARIANTS["rowz_drop"]
X = V.load_feat(v["feat"])
opt = dict(v["opt"])
cfg = dict(P.BASE_CFG); cfg.update(epochs=3)
tr_ids, va_ids = P.load_split("tuning", 0)
other = [b for b in va_ids][:1]
rng = np.random.default_rng(0)
runs = {}
for name, held, tx in (("i_real_val", va_ids, None), ("i_repeat", va_ids, None),
                       ("ii_noise", va_ids, None), ("iii_single_other", other, None)):
    test_X = None
    if name == "ii_noise":
        n = int(np.isin(meta["bearing"], va_ids).sum())
        test_X = rng.normal(size=(n,) + X.shape[1:]).astype(np.float32) * 5.0
    r = T.train_eval(X, meta["Y"], meta["bearing"], tr_ids, held, cfg, seed=0, opt=opt, dev=torch.device("cpu"),
                     test_X=test_X, return_model=True)
    runs[name] = dict(sha256=digest(r["model"]), n_train=r["n_train"])
    print(name, runs[name], flush=True)
ref = runs["i_real_val"]["sha256"]
res = dict(runs=runs, repeat_deterministic=runs["i_repeat"]["sha256"] == ref,
           noise_identical=runs["ii_noise"]["sha256"] == ref, single_other_identical=runs["iii_single_other"]["sha256"] == ref)
print(res)
(WORK / "results" / "audit2" / "d_results.json").write_text(json.dumps(res, indent=1))
