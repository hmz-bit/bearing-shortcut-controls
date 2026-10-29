Fixed-Hz ablation (F), Tier 3 (cross-speed) and Tier 2 (artificial to real) with LF-PNCN, B4, F and B3.
Outputs: results/tier2_3/ (prediction parquet files in long format, per-split csv, analysis_*.json, tier*_table.csv, sanity23.json, logs).

Files
  build_cache_fixed.py  data/cache/pu_windows_fixed.npz (t_fix, f_r = 25 Hz for every file; row order asserted equal to pu_windows.npz)
  common23.py           data loading + train_generic (mirrors tier1/common.train_lfpncn; verified to reproduce Tier 1 logits exactly on GPU, splits 5-6)
  run_tier1_fixed.py    100 Tier 1 evaluation splits for F (--variant t_mad reproduces Tier 1 LF-PNCN, used for the equivalence check)
  run_tier3.py          train on N15_M07_F10 of the training bearings, test at N15_M07_F10 and N09_M07_F10 (lfpncn, b4, f, b3)
  run_tier2.py          15 healthy-pair splits, 12 artificial + 4 healthy bearings -> 11 real faulty + 2 healthy bearings (2 seeds for neural methods)
  analyze23.py          metrics and paired bearing-level cluster bootstraps (evaltools bootstrap.py), class balance
  sanity23.py, sanity_shuffle.py   sanity checks

Device note: the GPU is time-sliced with other jobs; with several GPU processes one training step took >0.2 s instead of ~0.01 s.
Tier 1 F ran on the GPU (single process); Tier 2 and Tier 3 (all four methods) ran on the CPU with 1 torch thread per process
(T23_DEVICE=cpu). Consequently LF-PNCN, B4 and F inside Tier 2 / Tier 3 are mutually comparable (same device), and Tier 1 F is
comparable to the GPU Tier 1 LF-PNCN.

Rerun (from this folder, Git Bash)
  PY=python
  export PYTHONUTF8=1
  $PY -I build_cache_fixed.py
  $PY -I run_tier1_fixed.py --runs 5-104                                   # GPU, about 65 min
  export T23_DEVICE=cpu T23_THREADS=1
  for r in 5-29 30-54 55-79 80-104; do $PY -I run_tier3.py --runs $r & done   # about 38 min in parallel
  for r in 0-4 5-9 10-14; do $PY -I run_tier2.py --pairs $r & done            # about 71 min in parallel
  wait
  $PY -I analyze23.py --B 2000
  T23_THREADS=4 $PY -I sanity23.py
  for r in 5-14 15-24 25-34; do T23_THREADS=2 $PY -I sanity_shuffle.py --tier t3 --range $r & done
  for r in 0-4 5-9 10-14; do T23_THREADS=2 $PY -I sanity_shuffle.py --tier t2 --range $r & done
(the repeated shuffled-label summary was merged into sanity23.json by a small inline script)
