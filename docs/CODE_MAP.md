CODE MAP
========

The code keeps the folder layout of the experiments (scripts/<component or data set>/), because every script finds the repository root from its own location
and several scripts import sibling modules by adding their folder to sys.path.  Moving modules into one package would change imports and therefore the code; it was not done.
The table maps the usual roles to the files.

Role                                   Files
-------------------------------------  ---------------------------------------------------------------------------------------------------------
Geometry and fault frequencies         scripts/lfpncn/geometry.py
Input features (order tensor)          scripts/lfpncn/features.py (PU 42 kHz pipeline); scripts/phase2/features2.py (window length, pre-whitening, six bands)
Model                                  scripts/lfpncn/model.py; the variant network with dropout is in scripts/phase2/trainer.py (class Net)
Trainer                                scripts/tier1/common.py (train_lfpncn), scripts/tier2_3/common23.py (train_generic), scripts/phase2/trainer.py (train_eval, row transforms, dropout)
Variants                               scripts/phase2/variants.py
Evaluation tools                       scripts/evaltools/metrics.py (Macro-AUROC, FPR at fixed TPR), bootstrap.py (cluster bootstrap), pu_data.py (loaders), analysis.py, test_bootstrap.py
B1 baseline (WDCNN on envelope spectra) scripts/b1/ (envspec.py, wdcnn.py, core.py, cropworker.py, build_signals.py, run_b1.py, analyze_b1.py); per data set: scripts/cwru/b1_*.py, scripts/uored/*b1*.py, scripts/rtf/run_rtf_b1.py, scripts/xjtu/run_b1.py
Random-forest baselines B2, B3          scripts/evaltools/b2_rf.py, scripts/tier1/run_b3.py
PU loaders and cache builders          scripts/pu_loader.py, build_pu_metadata.py, build_cache.py, download_pu.sh, scripts/tier2_3/build_cache_fixed.py
PU experiment drivers                  scripts/tier1/ (tuning, evaluation, sanity, analysis), scripts/tier2_3/ (fixed-frequency ablation, cross-speed, artificial to real), scripts/phase2/ (variant development, evaluation, analysis), scripts/final/compare_vs_b1.py
Audit and reference detectors          scripts/audit/ (overfit_audit.py, analyze_a3.py, stage_only_reference.py, stage_matched_robustness.py)
CWRU                                   scripts/cwru/ (download, validation, speed check, cache, run, analysis, B1 run and analysis)
UORED-VAFCLS                           scripts/uored/
XJTU-SY                                scripts/xjtu/
Paderborn run-to-failure               scripts/rtf/ (remote range reading of the archives, selection, windows, runs, analysis)
Tests                                  scripts/lfpncn/test_lfpncn.py, scripts/evaltools/test_bootstrap.py (existing), tests/test_smoke.py (synthetic data), tests/run_tests.sh
Random-forest baseline on the external data sets   scripts/rf/ (feature builders from the raw files, run_rf.py, analysis, ablations); pre-declarations protocols/PREDECLARED_RF*.txt
Shortcut controls (audit rounds 2 and 3)  scripts/audit/audit2_ab.py (healthy-only negatives, unit-level metric), audit2_c.py (metadata-only forest), audit2_d.py (test-invariance check), audit3_rf_dc.py (forest with the window mean removed), audit3_e_i.py (UORED per label, PU metadata with DC, DC sensitivity of the networks, CWRU channel matching), audit3_jk.py and audit3_b1_k.py (leave-one-condition-out, stage-free fault-type evaluation), audit4_primary.py, audit5_ablation.py, weak_outer_bearings.py (per-bearing outer-race AUROC)
KAIST part 1 (supplementary)               scripts/kaist/ (fetch_kaist.py, run_kaist.py); protocols/PREDECLARED_KAIST.txt
