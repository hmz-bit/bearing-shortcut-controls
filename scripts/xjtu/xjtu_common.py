"""Shared helpers for the XJTU-SY experiments (see results/xjtu/PREDECLARED_XJTU.txt)."""
import itertools, pathlib, json, math
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
DATA, RES = WORK / "data" / "xjtu", WORK / "results" / "xjtu"
RAW = DATA / "raw_selected" / "XJTU-SY_Bearing_Datasets"
ALL = [f"B{c}_{k}" for c in (1, 2, 3) for k in range(1, 6)]
COND_DIR = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
NOMINAL_HZ = {1: 35.0, 2: 37.5, 3: 40.0}
# Failure location per bearing (Table 3 of the dataset authors' interpretation document); IR = inner race, OR = outer race
IRPOS = set("B2_1 B3_3 B3_4 B1_5 B3_2".split())
ORPOS = set("B1_1 B1_2 B1_3 B2_2 B2_4 B2_5 B3_1 B3_5 B1_5 B3_2".split())
CAGE_ONLY = set("B1_4 B2_3".split())
SEED = 20261029


def bearing_dir(b):
    c, k = int(b[1]), int(b[3])
    return RAW / COND_DIR[c] / f"Bearing{c}_{k}"


def satisfying_combos():
    out = []
    for te in itertools.combinations(ALL, 5):
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
    z = np.load(DATA / "xjtu_windows.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def select_snapshots(max_per_end=100):
    """Label-free snapshot table (PREDECLARED_XJTU.txt item 2): healthy = first m files, faulty = last m files (m as extracted, selection json),
    at most max_per_end per end evenly spaced by file index. Columns: bearing, stage, fileno, N, m, cond."""
    sel = json.load(open(DATA / "selection_first_last_10pct.json"))
    rows = []
    for b in ALL:
        c, k = int(b[1]), int(b[3])
        v = sel[f"{COND_DIR[c]}/Bearing{c}_{k}"]
        m, n = v["m"], v["n"]
        for stage, lo in (("healthy", 1), ("faulty", n - m + 1)):
            idx = np.arange(lo, lo + m)
            if len(idx) > max_per_end:
                idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, max_per_end)).astype(int))]
            for i in idx:
                rows.append(dict(bearing=b, stage=stage, fileno=int(i), N=n, m=m, cond=c))
    return pd.DataFrame(rows)


def load_snapshot(b, fileno):
    """Horizontal channel (column 1) of one snapshot as float64, 32768 samples."""
    a = np.loadtxt(bearing_dir(b) / f"{fileno}.csv", delimiter=",", skiprows=1, usecols=(0,))
    assert a.shape == (32768,) and np.isfinite(a).all()
    return a


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


if __name__ == "__main__":
    combos = satisfying_combos()
    print("test sets C(15,5) =", math.comb(15, 5), "satisfying the constraints:", len(combos))
    df, n = make_splits()
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "splits.csv", index=False)
    print("splits written:", len(df), "of", n)
    cnt = pd.Series([b for r in df.itertuples() for b in r.test.split()]).value_counts().sort_index()
    print("test-set frequency per bearing:", cnt.to_dict())
