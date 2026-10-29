"""Shared training/evaluation machinery for B1 (WDCNN on the envelope spectrum), own implementation of the recipe in
bearing-data-leakage/configs/experiment/paderborn/eval_envelope.yaml (+ configs/train.yaml defaults).

Recipe (YAML lines):
  lr 1e-3, batch_size 32, epochs 30, seed 0, Adam, no scheduler, no validation (no_val: True), last epoch evaluated
  normalization_strategy: global  -> src/data/data_pipeline.py apply_normalization: (x - mean_train)/std_train on the RAW
                                     time signals of the training set, BEFORE cropping/envelope (envelope is homogeneous of degree 1,
                                     so this equals dividing the envelope spectrum by std_train).
  augmentation pre_repr train: RandomCrop1D(crop_size 42000) on the full ~4 s signal (42 kHz after resampling),
                               then RandomGainOnFFT(mean 1, std 0.3) = one scalar gain ~ N(1, 0.3) on the (time) signal.
  segmentation_strategy: full_signal (train: one random 1-s crop per file per epoch, dataset_multiplier 1, drop_last True, shuffle)
  segmentation_strategy_eval: overlap, overlap_pct_eval 0 -> non-overlapping 1-s windows (4 per file), no augmentation.
  transform: envelope, cutoff [500, 10000], sampling_rate 42000 (see envspec.py).
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "scripts", "evaltools"))
import envspec
from wdcnn import WDCNN

OUT = os.path.join(WORK, "results", "b1")
CACHE = os.path.join(WORK, "data", "cache", "pu_windows.npz")
CROP = 42000
EPOCHS, BS, LR, STD_GAIN = 30, 32, 1e-3, 0.3
SCALE = 1000.0

ARTIFICIAL = "KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08".split()
HEALTHY = "K001 K002 K003 K004 K005 K006".split()
REAL_OUTER = "KA04 KA15 KA16 KA22 KA30".split()
REAL_INNER = "KI04 KI14 KI16 KI17 KI18 KI21".split()


def labels_of(b):
    """(inner, outer): K = healthy, KA = outer ring, KI = inner ring (also for the artificial bearings)."""
    if b.startswith("KA"): return (0, 1)
    if b.startswith("KI"): return (1, 0)
    return (0, 0)


class Data:
    def __init__(self):
        z = np.load(CACHE, allow_pickle=False)
        self.fname, self.win, self.bearing, self.cond = z["fname"], z["win"], z["bearing"], z["condition"]
        self.meta = pd.read_csv(os.path.join(OUT, "signals_meta.csv"))
        self.E = np.load(os.path.join(OUT, "env_spectra.npy"), mmap_mode="r")      # float16, x1000
        self.Y = np.array([labels_of(b) for b in self.bearing], dtype=np.float32)
        self.fY = np.array([labels_of(b) for b in self.meta.bearing], dtype=np.float32)
        # sanity: window rows of the cache and the file table agree
        assert (self.fname[self.meta.row0.values] == self.meta.fname.values).all()

    def files(self, bearings, conds=None):
        m = self.meta.bearing.isin(bearings)
        if conds is not None:
            m &= self.meta.condition.isin(conds)
        return np.where(m.values)[0]

    def windows(self, bearings, conds=None):
        m = np.isin(self.bearing, bearings)
        if conds is not None:
            m &= np.isin(self.cond, conds)
        return np.where(m)[0]

    def train_std(self, file_idx):
        """np.std of the concatenated RAW 64 kHz training signals (train files only), from stored per-file sums."""
        m = self.meta.iloc[file_idx]
        n, s, ss = m.n_raw.sum(), m.raw_sum.sum(), m.raw_sumsq.sum()
        mean = s / n
        return float(np.sqrt(ss / n - mean * mean))


# ---------------------------------------------------------------- crop workers (cropworker.py: numpy/scipy only)
import cropworker


def make_pool(data, n_workers):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    return ctx.Pool(n_workers, initializer=cropworker.init_worker,
                    initargs=(os.path.join(OUT, "signals_42k.f32"), data.meta.offset.values))


def epoch_batch_inputs(pool, data, file_idx, rng, std):
    """One random crop per file (RandomCrop1D) + gain (RandomGainOnFFT), envelope spectrum, normalised. Returns float32 (n, 21000)."""
    L = data.meta.n_samples.values[file_idx]
    starts = np.array([rng.integers(0, l - CROP) for l in L])          # torch.randint(0, length-crop): high exclusive
    gains = np.abs(rng.normal(1.0, STD_GAIN, size=len(file_idx))).astype(np.float32)   # |g|: envelope(g*x) = |g|*envelope(x)
    specs = pool.map(cropworker.crop_env, list(zip(file_idx.tolist(), starts.tolist())), chunksize=8)
    X = np.stack(specs) * (gains / std)[:, None]
    return X.astype(np.float32)


def train_model(data, pool, train_files, labels, std, seed=0, epochs=EPOCHS, device="cuda", log=None, mult=1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WDCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(train_files)
    nb = n // BS                                           # drop_last=True
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n * mult) % n if mult > 1 else rng.permutation(n)
        nb_ep = (len(perm)) // BS
        sel = perm[: nb_ep * BS]
        X = epoch_batch_inputs(pool, data, train_files[sel], rng, std)
        Y = torch.from_numpy(labels[sel])
        tot = 0.0
        for b in range(nb_ep):
            xb = torch.from_numpy(X[b * BS:(b + 1) * BS]).unsqueeze(1).to(device)
            yb = Y[b * BS:(b + 1) * BS].to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss)
        if log is not None and (ep % 5 == 4 or ep == 0):
            log(f"    epoch {ep + 1}/{epochs} loss {tot / nb_ep:.4f}")
    return model


@torch.no_grad()
def score_windows(model, data, rows, std, device="cuda"):
    """Logits (n, 2) [inner, outer] for the cached evaluation windows `rows` (float16 storage, /SCALE, /std_train)."""
    model.eval()
    out = []
    for i in range(0, len(rows), 128):
        r = rows[i:i + 128]
        x = torch.from_numpy(np.stack([data.E[j] for j in r]).astype(np.float32) / (SCALE * std))
        out.append(model(x.unsqueeze(1).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def long_frame(data, rows, logits, split, extra=None):
    df = pd.DataFrame({"split": split, "bearing": data.bearing[rows], "y_inner": data.Y[rows, 0].astype(int),
                       "y_outer": data.Y[rows, 1].astype(int), "s_inner": logits[:, 0], "s_outer": logits[:, 1],
                       "test_cond": data.cond[rows], "fname": data.fname[rows], "win": data.win[rows].astype(int)})
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df
