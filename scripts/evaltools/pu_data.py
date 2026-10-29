"""Loaders for the Paderborn (PU) artefacts shipped in bearing-data-leakage/data (READ-ONLY).

Nothing here executes third-party code or unpickles anything. We read only
parquet files and torch tensors saved with ``weights_only=True``.

Schemas (verified, see README.md):
  features/paderborn_features_segmented.parquet : 5440 rows x 17 cols
      waveform_id, waveform_id_seg (str) + 15 float64 features.
      Bearing id is the 4th '_' token of waveform_id (e.g. N15_M07_F10_KA15_16 -> KA15).
      17 bearings x 320 windows each, stored in bearing-contiguous blocks.
  splits/paderborn/eval/run_{5..104}.parquet : 1 row, columns train_ids (11 bearings), test_ids (6).
  splits/paderborn/tuning/run_{0..4}.parquet  : 1 row, columns train_ids (11), val_ids (6).
  saved_logits/pu/<exp>_run_{5..104}/test/epoch_30_run_0.pt : float32 (n~1915, 5)
      [row index, inner logit, outer logit, inner label, outer label].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[2]
DATA = WORK / "bearing-data-leakage" / "data"
FEATURES_PQ = DATA / "features" / "paderborn_features_segmented.parquet"
EVAL_DIR = DATA / "splits" / "paderborn" / "eval"
TUNING_DIR = DATA / "splits" / "paderborn" / "tuning"
LOGIT_ROOT = DATA / "saved_logits" / "pu"
LOGIT_TEMPLATE = "paderborn_eval_envelope_lr_0.001_bs_32_norm_global_run_{run}"

EVAL_RUNS = list(range(5, 105))  # the 100 evaluation splits (run ids 5..104)

TIME_FEATURES = [
    "acceleration/rms/global",
    "acceleration/pk-pk/global",
    "acceleration/kurt/global",
    "acceleration/skewness/global",
    "acceleration/fc/global",
]
ENV_FEATURES = [
    f"envelope/spectralPeak/{h}.0x-{f}/500-10000" for f in ("bpfo", "bpfi") for h in range(1, 6)
]
FEATURES_15 = TIME_FEATURES + ENV_FEATURES  # exactly what the original shallow_models notebook used
ENV_BPFB_FEATURES = [f"envelope/spectralPeak/{h}.0x-bpfb/500-10000" for h in range(1, 6)]


def bearing_labels(bearing_id: str) -> tuple[int, int]:
    """(inner, outer) multilabel target from the PU bearing code: K=healthy, KA=outer, KI=inner."""
    if bearing_id.startswith("KA"):
        return (0, 1)
    if bearing_id.startswith("KI"):
        return (1, 0)
    if bearing_id.startswith("K"):
        return (0, 0)
    raise ValueError(f"unexpected bearing id {bearing_id!r}")


def bearing_class(bearing_id: str) -> str:
    return {(0, 0): "healthy", (0, 1): "outer", (1, 0): "inner"}[bearing_labels(bearing_id)]


def load_features() -> pd.DataFrame:
    """Feature table + 'bearing', 'y_inner', 'y_outer', 'win' (position within bearing block)."""
    f = pd.read_parquet(FEATURES_PQ)
    f["bearing"] = f["waveform_id"].str.split("_").str[3]
    lab = f["bearing"].map(bearing_labels)
    f["y_inner"] = lab.str[0].astype(int)
    f["y_outer"] = lab.str[1].astype(int)
    f["win"] = f.groupby("bearing").cumcount()
    return f


def bearing_order() -> list[str]:
    """Global bearing order of the dataset = order of first appearance in the features parquet.
    The saved test logits follow this same order (verified in infer_bearing_counts)."""
    f = pd.read_parquet(FEATURES_PQ, columns=["waveform_id"])
    b = f["waveform_id"].str.split("_").str[3]
    return list(dict.fromkeys(b))


def load_eval_split(run: int) -> tuple[list[str], list[str]]:
    row = pd.read_parquet(EVAL_DIR / f"run_{run}.parquet").iloc[0]
    return list(row["train_ids"]), list(row["test_ids"])


def logit_path(run: int) -> Path:
    return LOGIT_ROOT / LOGIT_TEMPLATE.format(run=run) / "test" / "epoch_30_run_0.pt"


def _class_groups(ordered_bearings: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for b in ordered_bearings:
        if groups and bearing_labels(groups[-1][0]) == bearing_labels(b):
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


def _label_run_lengths(y_inner, y_outer) -> list[int]:
    lab = list(zip(y_inner.tolist(), y_outer.tolist()))
    out, prev, n = [], None, 0
    for l in lab:
        if l == prev:
            n += 1
        else:
            if prev is not None:
                out.append(n)
            prev, n = l, 1
    out.append(n)
    return out


def infer_bearing_counts(runs=EVAL_RUNS) -> dict[str, int]:
    """Number of windows per bearing in the saved logit files.

    Logit files do not store bearing ids. They are ordered by the global bearing order and each bearing
    contributes ~320 windows (a few are dropped by the original pipeline). A bearing is directly measurable
    whenever it forms an isolated label run; pairs of adjacent same-class bearings are resolved with the
    already-known count of the other member. Consistency across all splits is asserted.
    """
    from metrics import load_logit_run  # local import (torch only needed here)

    order = bearing_order()
    known: dict[str, int] = {}
    pending = []
    for r in runs:
        _, test_ids = load_eval_split(r)
        run = load_logit_run(logit_path(r))
        groups = _class_groups([b for b in order if b in test_ids])
        runs_len = _label_run_lengths(run["y_inner"], run["y_outer"])
        if len(runs_len) != len(groups):
            raise RuntimeError(f"run {r}: {len(runs_len)} label runs vs {len(groups)} bearing groups")
        for g, total in zip(groups, runs_len):
            if len(g) == 1:
                if known.setdefault(g[0], total) != total:
                    raise RuntimeError(f"inconsistent window count for {g[0]}: {known[g[0]]} vs {total}")
            else:
                pending.append((tuple(g), total))
    changed = True
    while pending and changed:
        changed = False
        rest = []
        for g, total in pending:
            unk = [b for b in g if b not in known]
            if len(unk) == 0:
                if sum(known[b] for b in g) != total:
                    raise RuntimeError(f"group {g}: known counts do not sum to {total}")
            elif len(unk) == 1:
                known[unk[0]] = total - sum(known[b] for b in g if b in known)
                changed = True
            else:
                rest.append((g, total))
        pending = rest
    if pending:
        raise RuntimeError(f"unresolved bearing counts: {pending[:3]}")
    return known


def logits_to_long(runs=EVAL_RUNS) -> pd.DataFrame:
    """Per-window long table for the saved B1 logits with inferred bearing ids.

    columns: split, bearing, y_inner, y_outer, s_inner, s_outer (raw logits).
    Bearing assignment is validated: labels implied by the inferred bearing must match the stored labels
    for every window of every run.
    """
    from metrics import load_logit_run

    order = bearing_order()
    counts = infer_bearing_counts(runs)
    parts = []
    for r in runs:
        _, test_ids = load_eval_split(r)
        run = load_logit_run(logit_path(r))
        seq = [b for b in order if b in test_ids]
        bearing = np.concatenate([[b] * counts[b] for b in seq])
        if len(bearing) != len(run["y_inner"]):
            raise RuntimeError(f"run {r}: assigned {len(bearing)} windows vs {len(run['y_inner'])}")
        exp = np.array([bearing_labels(b) for b in bearing])
        if not (np.array_equal(exp[:, 0], run["y_inner"]) and np.array_equal(exp[:, 1], run["y_outer"])):
            raise RuntimeError(f"run {r}: inferred bearing labels disagree with stored labels")
        parts.append(
            pd.DataFrame(
                {
                    "split": r,
                    "bearing": bearing,
                    "y_inner": run["y_inner"].astype(int),
                    "y_outer": run["y_outer"].astype(int),
                    "s_inner": run["logit_inner"],
                    "s_outer": run["logit_outer"],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
