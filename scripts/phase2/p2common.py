"""Shared helpers for phase 2 (outer-race / cross-speed improvement of LF-PNCN).

Works with `python -I`. Reuses (read-only) tier1/common.py, lfpncn/*, evaltools/*.
Protocol guard: `load_split("eval", r)` is refused unless the environment variable P2_ALLOW_EVAL=1 is set AND
results/phase2/PREDECLARED.txt exists. Tuning-phase scripts never set that variable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (HERE, WORK / "scripts" / "tier1", WORK / "scripts" / "tier2_3", WORK / "scripts" / "lfpncn",
          WORK / "scripts" / "evaltools", WORK / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch

import common as C            # tier1 (data loading, standardiser, predict)
import pu_data as pu

OUT = WORK / "results" / "phase2"
CACHE_DIR = WORK / "data" / "cache"
PRED = OUT / "PREDECLARED.txt"
BASE_CFG = json.loads((WORK / "results" / "tier1" / "chosen_config_t_mad.json").read_text())
TUNING_RUNS = C.TUNING_RUNS
BEARINGS = C.VIEIRA_BEARINGS
OUTER = "KA04 KA15 KA16 KA22 KA30".split()
INNER = "KI04 KI14 KI16 KI17 KI18 KI21".split()
HEALTHY = "K001 K002 K003 K004 K005 K006".split()
CONDS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]


def load_split(kind, run):
    if kind == "eval":
        if os.environ.get("P2_ALLOW_EVAL") != "1" or not PRED.exists():
            raise RuntimeError("evaluation splits are locked until PREDECLARED.txt exists and P2_ALLOW_EVAL=1")
    return C.load_split(kind, run)


def load_meta():
    """Vieira-17 meta (aligned to the row order used everywhere below): bearing, condition, fname, win, Y, row index into cache."""
    z = np.load(C.CACHE, allow_pickle=False)
    bearing = z["bearing"]
    keep = np.isin(bearing, BEARINGS)
    d = {"bearing": bearing[keep], "condition": z["condition"][keep], "fname": z["fname"][keep], "win": z["win"][keep],
         "row": np.nonzero(keep)[0]}
    d["Y"] = np.array([pu.bearing_labels(b) for b in d["bearing"]], dtype=np.float32)
    return d


def load_t_mad():
    z = np.load(C.CACHE, allow_pickle=False)
    keep = np.isin(z["bearing"], BEARINGS)
    return z["t_mad"][keep]


def device():
    d = os.environ.get("P2_DEVICE")
    if d == "cpu":
        torch.set_num_threads(int(os.environ.get("P2_THREADS", "1")))
        return torch.device("cpu")
    return C.device()


def per_label_auroc(Y, S):
    return C.macro_auroc_np(Y, S)
