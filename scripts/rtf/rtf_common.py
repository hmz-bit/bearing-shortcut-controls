"""Shared helpers for the Paderborn run-to-failure experiments (see results/rtf/PREDECLARED_RTF.txt)."""
import itertools, pathlib, json
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
DATA, RES = WORK / "data" / "rtf", WORK / "results" / "rtf"
ALL = [f"B{i:02d}" for i in range(1, 18)]
IRPOS = set("B01 B02 B03 B04 B06 B07 B08 B11 B12 B13 B15 B16".split())
ORPOS = set("B01 B09 B10 B11 B16".split())
SEED = 20261028


def satisfying_combos():
    out = []
    for te in itertools.combinations(ALL, 6):
        tr = [b for b in ALL if b not in te]
        if (sum(b in ORPOS for b in te) >= 1 and sum(b in IRPOS for b in te) >= 1 and
                sum(b in ORPOS for b in tr) >= 2 and sum(b in IRPOS for b in tr) >= 2):
            out.append(te)
    return out


def make_splits():
    combos = satisfying_combos()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(combos), 100, replace=False)
    rows = []
    for i, k in enumerate(pick):
        te = list(combos[k]); tr = [b for b in ALL if b not in te]
        rows.append(dict(split=5 + i, test=" ".join(te), train=" ".join(tr)))
    return pd.DataFrame(rows), len(combos)


def load_splits():
    d = pd.read_csv(RES / "splits.csv")
    return {int(r.split): (r.train.split(), r.test.split()) for r in d.itertuples()}


def load_windows():
    z = np.load(DATA / "rtf_windows.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def perm_Y(Y, exp, stage, split, k, train_ids, test_ids):
    """Bearing-label permutation: the faulty-end (IR, OR) label pair of each bearing is reassigned to another bearing (healthy windows stay 0).
    Redrawn (at most 20 times) until train and test both contain both classes for both labels. Returns permuted Y or None."""
    rng = np.random.default_rng(7000 + 10 * split + k)
    pair = {b: Y[(exp == b) & (stage == "faulty")][0] for b in ALL}
    tr_m, te_m = np.isin(exp, train_ids), np.isin(exp, test_ids)
    for _ in range(20):
        perm = rng.permutation(len(ALL))
        Yp = Y.copy()
        for b, j in zip(ALL, perm):
            Yp[(exp == b) & (stage == "faulty")] = pair[ALL[j]]
        if all(len(np.unique(Yp[m][:, c])) == 2 for m in (tr_m, te_m) for c in (0, 1)):
            return Yp
    return None


def perm_Y_whole(Y, exp, stage, split, k, train_ids, test_ids):
    """Whole-bearing permutation (addendum 1): the faulty-end label pair of each bearing is reassigned to another bearing and applied to ALL
    windows of the bearing (healthy-end and faulty-end), so labels are constant per bearing and carry no life-stage information."""
    rng = np.random.default_rng(8000 + 10 * split + k)
    pair = {b: Y[(exp == b) & (stage == "faulty")][0] for b in ALL}
    tr_m, te_m = np.isin(exp, train_ids), np.isin(exp, test_ids)
    for _ in range(20):
        perm = rng.permutation(len(ALL))
        Yp = Y.copy()
        for b, j in zip(ALL, perm):
            Yp[exp == b] = pair[ALL[j]]
        if all(len(np.unique(Yp[m][:, c])) == 2 for m in (tr_m, te_m) for c in (0, 1)):
            return Yp
    return None
