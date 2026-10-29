Phase 2: outer-race / cross-speed improvement of LF-PNCN (development on the 5 tuning splits only; evaluation after pre-declaration).
All code in scripts/phase2, all outputs in results/phase2. Existing code is imported read-only.

Files
  p2common.py          paths, meta (17 Vieira bearings, same row order as the Tier 1 cache); load_split("eval") is locked unless PREDECLARED.txt exists and P2_ALLOW_EVAL=1
  features2.py         input builders: window length L (1, 2, 4 s; rows aligned to the 1-s cache rows), cepstrum pre-whitening (full whitening, cepstrum comb editing), six bands
  build_features.py    builds results/phase2/cache/feat_<name>.npy from the raw .mat files (L1 reproduces the Tier 1 cache exactly, checked)
  trainer.py           generalised training loop (reproduces Tier 1 exactly by default): rowcenter / rowz (per-window row normalisation), jitter, gain, dropout, balanced loss, side features
  variants.py          registry name -> (input, options); add new input variants here (plug-in point)
  run_tuning.py        tuning-split runs (also --tier3: tuning-split analogue of Tier 3)
  summarize_tuning.py  tuning_table.csv, tuning_t3_table.csv
  diag1_harmonics.py .. diag5_bands.py   Phase A diagnostics (logs diag*_log.txt)
  run_eval.py, run_tier23.py, analyze_phase2.py   Phase C (locked until PREDECLARED.txt exists)

Rerun (Git Bash)
  PY=python
  export PYTHONUTF8=1; cd <repository root>
  for n in L4 L2 L1cpw L4cpw L1b6 L1comb; do $PY -I scripts/phase2/build_features.py $n --procs 6; done     # about 1 min each
  $PY -I scripts/phase2/diag1_harmonics.py; $PY -I scripts/phase2/diag2_windows.py; $PY -I scripts/phase2/diag4_ses_zoom.py; $PY -I scripts/phase2/diag5_bands.py
  $PY -I scripts/phase2/run_tuning.py base rowc rowz rowz_drop cpw1 comb ML14 L4 b6 aug --seeds 3          # about 2.5 min per variant and seed on GPU
  $PY -I scripts/phase2/run_tuning.py base rowc rowz rowz_drop --tier3 --seeds 1
  $PY -I scripts/phase2/diag3_false_alarms.py base_s0                                                       # needs tuning/base_s0.parquet
  $PY -I scripts/phase2/summarize_tuning.py
  # evaluation (only after results/phase2/PREDECLARED.txt exists)
  export P2_ALLOW_EVAL=1
  $PY -I scripts/phase2/run_eval.py rowz_drop --runs 5-104                       # GPU, about 30-55 min; CPU: P2_DEVICE=cpu P2_THREADS=1 with --runs 5-29 etc. in parallel
  $PY -I scripts/phase2/run_tier23.py rowz_drop --tier 3 --runs 5-104
  $PY -I scripts/phase2/run_tier23.py rowz_drop --tier 2 --pairs 0-14
  $PY -I scripts/phase2/analyze_phase2.py --variants rowz_drop rowz rowc rowz_b6_drop --tier3 --tier2 --B 2000
Notes: seeds 1 and 2 of base, rowc, rowz were run on CPU (1 thread), everything else on GPU; the per-window row transform is computed inside the trainer
(no statistics are shared between windows).
