"""Leakage guard: mechanical checks on the split definitions used in this project.

Read-only. Needs numpy, pandas and pyarrow (use work/.venv). Run:
    PYTHONUTF8=1 python -I tools/leakage_guard.py [--out PATH]

What it checks (each is a hard FAIL if violated, an INFO line otherwise):
  L1  train and test bearing sets are disjoint in every split of every dataset (PU, UORED, CWRU, XJTU-SY, Paderborn run-to-failure)
  L2  CWRU: waveform ids AND fault-configuration groups are disjoint between train and test
  L3  the same test set of bearings is not accidentally identical across "different" splits (effective number of distinct test sets)
  S1  selection contamination: bearings that served as validation bearings during tuning and as test bearings later (PU)
  C1  condition sharing (XJTU-SY): how often a test bearing has training bearings recorded under the same operating condition
  C2  leave-one-condition-out feasibility (XJTU-SY): would a condition-wise protocol keep both labels in the training set
It does NOT prove absence of leakage. It only catches identity-level mistakes; the preprocessing and label-construction
questions are listed in docs/METHOD_AND_PROTOCOL.md and has to be answered by reading the code.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
SPL = WORK / "bearing-data-leakage" / "data" / "splits"   # clone made by tools/fetch_third_party.sh

FAILS: list[str] = []
LINES: list[str] = []


def say(msg: str) -> None:
    LINES.append(msg)
    print(msg)


def fail(msg: str) -> None:
    FAILS.append(msg)
    say("FAIL  " + msg)


def pu() -> None:
    say("== PU (Paderborn, 17 bearings of Vieira et al.)")
    ev, tu = {}, {}
    for run in range(5, 105):
        r = pd.read_parquet(SPL / "paderborn" / "eval" / f"run_{run}.parquet").iloc[0]
        ev[run] = (set(r["train_ids"]), set(r["test_ids"]))
        if ev[run][0] & ev[run][1]:
            fail(f"PU eval run {run}: train/test overlap {sorted(ev[run][0] & ev[run][1])}")
    for run in range(0, 5):
        r = pd.read_parquet(SPL / "paderborn" / "tuning" / f"run_{run}.parquet").iloc[0]
        tu[run] = (set(r["train_ids"]), set(r["val_ids"]))
        if tu[run][0] & tu[run][1]:
            fail(f"PU tuning run {run}: train/val overlap")
    say(f"L1  PU: {len(ev)} eval + {len(tu)} tuning splits checked")
    distinct = len({frozenset(t) for _, t in ev.values()})
    say(f"L3  PU: {distinct} distinct test sets among {len(ev)} eval splits (17 bearings in total, so the effective sample is the bearing, not the split)")
    val_seen = Counter(b for _, v in tu.values() for b in v)
    test_seen = Counter(b for _, t in ev.values() for b in t)
    all_b = sorted(set(test_seen) | set(val_seen))
    contaminated = sorted(b for b in test_seen if b in val_seen)
    say(f"S1  PU: {len(contaminated)} of {len(test_seen)} bearings that appear as eval-test bearings were also validation bearings during tuning "
        f"({', '.join(contaminated)})")
    only_train = sorted(b for b in all_b if b not in val_seen)
    say(f"S1  PU: bearings never used as validation in the 5 tuning splits: {', '.join(only_train) or 'none'}")
    say("    -> every PU bearing influenced design decisions; PU numbers are development-set numbers, not confirmatory evidence")


def uored() -> None:
    say("== UORED-VAFCLS")
    n = 0
    for run in range(5, 105):
        r = pd.read_parquet(SPL / "uored-vafcls" / "eval" / f"run_{run}.parquet").iloc[0]
        a, b = set(map(int, r["train_ids"])), set(map(int, r["test_ids"]))
        n += 1
        if a & b:
            fail(f"UORED run {run}: overlap {sorted(a & b)}")
    say(f"L1  UORED: {n} eval splits checked")
    npz = WORK / "data" / "uored" / "uored_windows.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        cls, Y = z["cls"], z["Y"]
        tab = pd.crosstab(pd.Series(cls, name="class"), pd.Series([f"inner={int(a)} outer={int(b)}" for a, b in Y], name="label"))
        say("    label table of WINDOWS (class H/I/O/B/C versus the two binary labels). Class H windows are the state-0 (early-life) windows of the SAME 20 bearings, not separate healthy bearings; other-fault windows count as negatives:")
        say(tab.to_string())


def cwru() -> None:
    say("== CWRU")
    d = SPL / "cwru" / "eval"
    bad = checked = 0
    for f in sorted(d.glob("train_*_run_*.parquet")):
        t = f.name.replace("train_", "test_")
        tr, te = pd.read_parquet(f), pd.read_parquet(d / t)
        checked += 1
        if set(tr["waveform_id"]) & set(te["waveform_id"]):
            fail(f"CWRU {f.name}: waveform overlap")
            bad += 1
        if set(tr["group"]) & set(te["group"]):
            fail(f"CWRU {f.name}: fault-configuration group shared between train and test")
            bad += 1
    say(f"L2  CWRU: {checked} train/test pairs checked, {bad} violations")
    say("    note: all CWRU configurations come from one test rig, one motor, one sensor mounting; rig identity cannot be split away")


def run_to_failure(name: str, csv: Path, cond_of=None) -> pd.DataFrame | None:
    say(f"== {name}")
    if not csv.exists():
        say(f"    {csv} missing, skipped")
        return None
    d = pd.read_csv(csv)
    sets = []
    for r in d.itertuples():
        tr, te = set(r.train.split()), set(r.test.split())
        sets.append(frozenset(te))
        if tr & te:
            fail(f"{name} split {r.split}: overlap {sorted(tr & te)}")
    say(f"L1  {name}: {len(d)} splits checked")
    say(f"L3  {name}: {len(set(sets))} distinct test sets")
    cnt = Counter(b for s in sets for b in s)
    say(f"    test-set frequency per bearing: min {min(cnt.values())}, max {max(cnt.values())}")
    return d


def xjtu() -> None:
    d = run_to_failure("XJTU-SY", WORK / "results" / "xjtu" / "splits.csv")
    if d is None:
        return
    shared, total = 0, 0
    for r in d.itertuples():
        tr_c = Counter(int(b[1]) for b in r.train.split())
        for b in r.test.split():
            total += 1
            shared += tr_c[int(b[1])] > 0
    say(f"C1  XJTU-SY: {shared}/{total} test-bearing instances have training bearings from the SAME operating condition "
        f"(same nominal speed and load, same rig session type)")
    ir = set("B2_1 B3_3 B3_4 B1_5 B3_2".split())
    orr = set("B1_1 B1_2 B1_3 B2_2 B2_4 B2_5 B3_1 B3_5 B1_5 B3_2".split())
    for c in (1, 2, 3):
        tr = [f"B{k}_{i}" for k in (1, 2, 3) if k != c for i in range(1, 6)]
        ok = sum(b in ir for b in tr) >= 1 and sum(b in orr for b in tr) >= 1
        say(f"C2  hold out condition {c}: training bearings {len(tr)}, inner-race and outer-race positives present in training: {ok}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the report to this path")
    a = ap.parse_args()
    pu(); uored(); cwru(); xjtu()
    run_to_failure("Paderborn run-to-failure", WORK / "results" / "rtf" / "splits.csv")
    say("")
    say(f"RESULT: {len(FAILS)} identity-level violations" + ("" if FAILS else " (none found)"))
    if a.out:
        Path(a.out).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
