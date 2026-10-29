HOW TO REPRODUCE EACH TABLE
===========================

Conventions
  - All commands are run from the repository root in a POSIX shell (Git Bash on Windows was used originally).
    PY is the Python interpreter of an environment built from requirements.txt (Python 3.11).
        export PYTHONUTF8=1
        export PY=python
  - Code is run with "python -I" (isolated mode); each script adds its own folder to sys.path.  Scripts locate the
    repository root from their own file position, so the layout must stay as shipped (scripts/<folder>/..., results/<folder>/...).
  - Results are written into results/<folder>/ and would overwrite the shipped copies; work on a copy if the shipped files should be kept.
  - Run times below are wall-clock times observed in the original runs (single laptop with a 4 GB GPU, several processes sharing it, the
    LF-PNCN family partly on CPU).  They are indications, not guarantees.  "chunk" means one of several processes run in parallel over disjoint split ranges.
  - Pre-declaration guards: the scripts that evaluate frozen methods refuse to run unless results/<folder>/PREDECLARED*.txt exists
    (these files are shipped; identical copies are in protocols/).  Phase-2 evaluation additionally needs P2_ALLOW_EVAL=1.
  - Sequence numbers show the order in which the stages were originally run.  A stage can only start after the stages it reads from.

Step 0.  Environment and third-party material
    pip install -r requirements.txt
    bash tools/fetch_third_party.sh               # clone of the benchmark repository at commit 4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d
    bash scripts/download_pu.sh                   # PU archives into data/pu (large download; run time not recorded)
    (other data sets: docs/DATA.md)
    python -I scripts/build_pu_metadata.py        # data/pu_metadata.csv
    python -I scripts/build_cache.py              # data/cache/pu_windows.npz (order tensors t_mad, t_max and 20 features b3); run time not recorded
    python -I scripts/check_a_reproduce_table9.py # recomputes the published PU B1 Macro-AUROC from their saved logits

Table A.  PU, Tier 1 (bearing-wise, 100 evaluation splits): columns "base (LF-PNCN)", "B1", and the B2/B3/B4 reference rows
    cd scripts/evaltools
    $PY -I reproduce_b1.py                        # published B1 from saved logits -> results/b1_reproduction.json      (seconds)
    $PY -I -c "import pu_data as pu; pu.logits_to_long().to_parquet('results/b1_predictions_long.parquet')"
                                                  # window-level B1 scores (used by every later bootstrap); NOT VERIFIED that this one-liner
                                                  # equals what was run originally (the original call was not recorded as a script)
    $PY -I b2_rf.py ; $PY -I b2_rf.py --ccp-exact --tag b2_rf_ccpexact ; $PY -I b2_rf.py --seed 1 --tag b2_rf_seed1     # RF baseline (a few minutes)
    $PY test_bootstrap.py                         # bootstrap exactness and coverage tests (about 3 min with --fast)
    cd ../tier1
    $PY -I tune.py --variant t_mad                # 16-cell grid on the 5 tuning splits, writes chosen_config_t_mad.json        (578 s)
    $PY -I tune.py --variant t_max                # sensitivity only
    $PY -I run_eval.py --variant t_mad --tag lfpncn                         # 100 evaluation splits, GPU                     (3598 s)
    $PY -I run_eval.py --variant t_max --tag b4_tmax
    $PY -I run_eval.py --variant t_mad --tag lfpncn_seed2 --seed-offset 1000
    $PY -I run_b3.py
    $PY -I sanity.py
    $PY -I analyze.py --B 2000                    # results/tier1/metrics.json, metrics_table.csv, paired_table.csv (needs the B1 and B2 parquet files)
  The row "rowz_drop" and the paired differences of this table come from phase 2 (Table B).

Table B.  PU, row-normalised variant (rowz_drop) and the ablations of the development phase
    $PY -I scripts/phase2/build_features.py L4 --procs 6          # also L2 L1cpw L4cpw L1b6 L1comb; about 1 min each; writes results/phase2/cache/*.npy
    $PY -I scripts/phase2/diag1_harmonics.py ; diag2_windows.py ; diag4_ses_zoom.py ; diag5_bands.py       # descriptive diagnostics, no model
    $PY -I scripts/phase2/run_tuning.py base rowc rowz rowz_drop cpw1 comb ML14 L4 b6 aug --seeds 3        # tuning splits only; about 2.5 min per variant and seed on GPU
    $PY -I scripts/phase2/run_tuning.py base rowc rowz rowz_drop --tier3 --seeds 1
    $PY -I scripts/phase2/diag3_false_alarms.py base_s0 ; $PY -I scripts/phase2/summarize_tuning.py        # results/phase2/tuning_table.csv, tuning_t3_table.csv
    (protocols/PREDECLARED.txt was written here, before any evaluation split was used by phase-2 code; see docs/DEVELOPMENT_LOG.md)
    export P2_ALLOW_EVAL=1
    $PY -I scripts/phase2/run_eval.py rowz_drop --runs 5-104      # GPU about 55 min (3319 s); CPU: P2_DEVICE=cpu P2_THREADS=1 with --runs 5-29 ... in parallel
    $PY -I scripts/phase2/run_eval.py rowz rowc rowz_b6_drop ...  # one variant per call, each --runs 5-104 (see the file names in results/phase2/eval)
    $PY -I scripts/phase2/run_tier23.py rowz_drop --tier 3 --runs 5-104 ; --tier 2 --pairs 0-14
    $PY -I scripts/phase2/analyze_phase2.py --variants rowz_drop rowz rowc rowz_b6_drop --tier3 --tier2 --B 2000    # results/phase2/eval_metrics.json, eval_tables.csv
    $PY -I scripts/phase2/post_hoc_eval.py                       # post hoc, not used for any selection
  Table A and B columns "rowz_drop", "rowz_drop - base", "rowz_drop - B1" for PU: results/phase2/eval_metrics.json (tier1).

Table C.  PU, fixed-frequency ablation, Tier 3 (cross-speed) and Tier 2 (artificial to real); B1 re-implementation (own B1 on PU)
    export T23_DEVICE=cpu T23_THREADS=1
    $PY -I scripts/tier2_3/build_cache_fixed.py
    $PY -I scripts/tier2_3/run_tier1_fixed.py --runs 5-104                           # GPU, 65 min
    for r in 5-29 30-54 55-79 80-104; do $PY -I scripts/tier2_3/run_tier3.py --runs $r & done; wait       # 4 chunks, about 38 min (2277 s each)
    for r in 0-4 5-9 10-14; do $PY -I scripts/tier2_3/run_tier2.py --pairs $r & done; wait                # 3 chunks, about 71 min
    $PY -I scripts/tier2_3/analyze23.py --B 2000
    $PY -I scripts/tier2_3/sanity23.py ; sanity_shuffle.py --tier t3 --range 5-14 (and the other ranges listed in scripts/tier2_3/README.md)
    B1:  $PY scripts/b1/build_signals.py ; $PY -I scripts/b1/run_b1.py --tier t1val --workers 6 (11 min) ; analyze_b1.py val ; run_b1.py --tier t3 (15 min per 50 splits)
         ; run_b1.py --tier t2 ; run_b1.py --tier t1shuf ; analyze_b1.py t3|t2|t1shuf       (details: scripts/b1/README.md)
    $PY -I scripts/final/compare_vs_b1.py                                             # results/b1/compare_vs_b1.json
  (not in the cross-data-set table, but the source of the PU claims about order resampling and cross-speed.)

Table D.  Audit on PU (tuning splits and artificial bearings only)
    $PY -I scripts/audit/overfit_audit.py a12        # about 16 min (982 s)
    $PY -I scripts/audit/overfit_audit.py a3         # about 9 min (522 s)
    $PY -I scripts/audit/analyze_a3.py

Table E.  CWRU (artificial faults), 100 splits = 50 runs x 2 scenarios
    $PY -I scripts/cwru/download_cwru.py ; validate_raw.py ; speed_check_cwru.py ; build_cwru_cache.py
    for r in 3-12 13-22 23-32 33-42 43-52; do $PY -I scripts/cwru/run_cwru.py --runs $r base rowz_drop & done; wait         # 5 chunks on CPU, about 19 min each for two variants
    $PY -I scripts/cwru/analyze_cwru.py                                  # results/cwru/cwru_results.json, per_bearing_auroc.csv
    $PY -I scripts/cwru/b1_build_bank.py ; $PY -I scripts/cwru/b1_run_cwru.py --runs 3-52          # about 4 min on GPU
    $PY -I scripts/cwru/analyze_b1_cwru.py                               # results/cwru/cwru_results_b1.json  (columns base, rowz_drop, B1 and the differences)
    $PY -I scripts/cwru/b1_diag.py

Table F.  UORED-VAFCLS (naturally developed damage, constant speed), 100 splits
    $PY -I scripts/uored/speed_check.py ; build_uored_cache.py
    $PY -I scripts/uored/run_uored.py base rowz_drop                      # about 10 min per variant on CPU (602 s for rowz_drop)
    $PY -I scripts/uored/analyze_uored.py
    $PY -I scripts/uored/build_b1_bank.py ; run_b1_uored.py ; analyze_uored_b1.py       # B1: about 16 min on GPU (937 s)

Table G.  XJTU-SY (run to failure, constant speed per bearing), 100 splits
    $PY -I scripts/xjtu/speed_check_xjtu.py ; build_windows.py
    bash scripts/xjtu/run_all.sh                                          # 5 parallel chains, about 20 min each for the primary runs; then permutation and 5% sensitivity runs
    $PY -I scripts/xjtu/analyze_xjtu.py                                  # results/xjtu/xjtu_results.json
    $PY -I scripts/xjtu/null_life_stage.py                               # post hoc life-stage reference, results/xjtu/null_life_stage.json

Table H.  Paderborn run-to-failure under time-varying speed and load, 100 splits
    $PY -I scripts/rtf/list_manifest.py ; fetch_aux.py ; speed_check_rtf.py ; select_snapshots.py
    $PY -I scripts/rtf/build_rtf_windows.py                               # range-reads the selected members from Zenodo; about 48 min (2869 s)
    for r in "5 24" "25 44" "45 64" "65 84" "85 104"; do $PY -I scripts/rtf/run_rtf_lfpncn.py base $r & done; wait         # CPU chunks, about 23 min
    (same for rowz_drop) ; $PY -I scripts/rtf/run_rtf_b1.py 5 104                     # B1 about 18 min on GPU
    $PY -I scripts/rtf/run_rtf_lfpncn.py <base|rowz_drop> 5 7 perm  (chunks 5-7 ... 23-24) ; run_rtf_b1.py 5 24 perm       # declared label-pair permutation control
    $PY -I scripts/rtf/run_rtf_lfpncn.py <base|rowz_drop> 5 7 permw ; run_rtf_b1.py 5 24 permw                       # whole-bearing permutation (addendum, post hoc)
    $PY -I scripts/rtf/analyze_rtf.py                                    # results/rtf/rtf_results.json, per_bearing_auroc.csv

Table I.  results/CROSS_DATASET_SUMMARY and results/TABLE_SOURCES.csv
    $PY -I scripts/audit/stage_only_reference.py                          # UORED and Paderborn run-to-failure stage-only reference (post hoc)
    $PY -I scripts/audit/stage_matched_robustness.py                      # leave-one-negative-bearing-out ranges (point estimates)
    python tools/make_table_sources.py                                    # rebuilds results/TABLE_SOURCES.csv from the shipped result files (seconds)

Table J.  Random-forest baseline (CWRU, UORED-VAFCLS, XJTU-SY, Paderborn run-to-failure), protocols/PREDECLARED_RF.txt
    $PY -I scripts/rf/build_features.py <cwru|uored|xjtu>      # 20 features from the raw files, row order of the existing caches (needs the raw data, docs/DATA.md)
    $PY -I scripts/rf/build_features_rtf.py
    $PY -I scripts/rf/run_rf.py <cwru|uored|xjtu|rtf>          # results/rf/pred_*.parquet and per_split_*.csv; run_rf_ablation.py, analyze_rf.py and make_summary.py follow (see the docstrings)

Table K.  Controls of audit rounds 2 to 4 (evaluation only), protocols/PREDECLARED_AUDIT2.txt, PREDECLARED_AUDIT3.txt, PREDECLARED_AUDIT4.txt
    $PY -I scripts/audit/audit2_ab.py                          # healthy-only negatives and unit-level metric -> results/audit2/ab_results.json
    $PY -I scripts/audit/audit2_c.py                           # metadata-only forest -> results/audit2/c_results.json
    P2_DEVICE=cpu P2_THREADS=1 $PY -I scripts/audit/audit2_d.py   # test-invariance check -> results/audit2/d_results.json
    $PY -I scripts/audit/audit3_rf_dc.py                       # forest on UORED with the window mean removed (rebuilds features from the raw csv files)
    $PY -I scripts/audit/audit3_e_i.py E F G I                 # UORED per label, PU metadata with DC, DC sensitivity of the networks, CWRU channel matching
    $PY -I scripts/audit/audit3_jk.py J                        # leave-one-condition-out on XJTU-SY
    $PY -I scripts/audit/audit3_jk.py K <xjtu|rtf> <base|rowz_drop|rf|rfu|meta>   # stage-free setting, one model per call
    $PY -I scripts/audit/audit3_b1_k.py <xjtu|rtf>             # B1 in the stage-free setting (addendum 1 of PREDECLARED_AUDIT3.txt)
    $PY -I scripts/audit/audit3_jk.py Kan                      # contrasts and intervals of the stage-free setting -> results/audit3/k_results.json
    $PY -I scripts/audit/audit4_primary.py                     # primary comparison and difference of contrasts -> results/audit4/primary_results.json

Table L.  KAIST part 1, supplementary one-shot evaluation, protocols/PREDECLARED_KAIST.txt
    $PY -I scripts/kaist/fetch_kaist.py                        # 18 members of vibration.zip by range requests (about 722 MB)
    $PY -I scripts/kaist/run_kaist.py build                    # windows and features -> data/kaist/kaist_windows.npz
    $PY -I scripts/kaist/run_kaist.py run                      # nine splits, all models and references -> results/kaist/

Table M.  Component ablation (row z-score alone, dropout alone), protocols/PREDECLARED_ABLATION.txt
    $PY -I scripts/uored/run_uored.py rowz base_drop           # UORED-VAFCLS, 100 splits
    $PY -I scripts/cwru/run_cwru.py --runs 3-52 rowz base_drop # CWRU, 100 (run, scenario) splits
    $PY -I scripts/audit/audit3_jk.py K <xjtu|rtf> <rowz|base_drop>   # stage-free setting
    $PY -I scripts/audit/audit5_ablation.py                    # contrasts and intervals -> results/ablation/ablation_results.json

PAPER TABLES AND THE STEPS ABOVE
================================

The lettered tables of this file are steps of the pipeline; the tables of the paper are assembled from their outputs.

    Paper table / figure                        Steps that produce it            Main input files shipped in the repository
    Table 1  datasets                           docs/DATA.md                     (descriptive)
    Table 2  headline, original tasks           A, B, E, F, G, H, I, J           results/CROSS_DATASET_SUMMARY.md, results/rf/SUMMARY.md, results/*/pred_*.parquet
    Table 3  UORED outer label, negative sets   K (audit3_e_i.py E)              results/audit3/ (UORED per-label outputs), results/uored/pred_*.parquet
    Table 4  stage-free setting                 K (audit3_jk.py K, Kan)          results/audit3/pred_k_*.parquet, results/audit3/k_results.json
    Table 8  weak outer-race bearings           weak_outer_bearings.py           results/*/pred_*.parquet (PU, UORED, CWRU, XJTU-SY, RTF)
    Table 5  component ablation                 M                                results/ablation/ablation_results.json
    Table 6  status of the analyses             protocols/TIMESTAMPS.csv         protocols/*.txt
    Table 7  leave-one-condition-out            K (audit3_jk.py J)               results/audit3/j_results.json
    Figure 3 contrasts with the base model      I, K, M                          results/audit2/ab_results.json, results/audit3/, results/audit4/
    KAIST subsection                            L                                results/kaist/
    Every number of the paper                   results/PAPER_TABLE_SOURCES.csv  (table or sentence, source file, key)

Recomputing the intervals from the shipped predictions (minutes, CPU only; no raw data and no training)
    Contrasts and bootstrap intervals of Tables 3 to 5 and of the stage-free and unit-level results are computed from the shipped prediction files by
        $PY -I scripts/audit/audit3_e_i.py E       # UORED per label and negative sets (reads results/uored and results/rf predictions)
        $PY -I scripts/audit/audit3_jk.py Kan      # stage-free contrasts
        $PY -I scripts/audit/audit4_primary.py     # summary contrast and difference of contrasts
        $PY -I scripts/audit/audit5_ablation.py    # ablation contrasts
        $PY -I scripts/audit/weak_outer_bearings.py  # per-bearing outer-race AUROC (results/audit/weak_outer_bearings.csv; post hoc)
    The unit-level and healthy-only rows for PU (scripts/audit/audit2_ab.py) also need the saved logits of the third-party benchmark (step 0), which are not shipped.
    These scripts write into results/<folder>/ and would overwrite the shipped copies; run them on a copy of the repository.
    Retraining a model (steps A to H, K, M) needs the raw data and, for the PU baseline B1, the third-party clone (step 0); the run times are given in each step.

Checks of the repository itself (seconds)
    python tools/leakage_guard.py                                         # identity-level checks of all split definitions
    bash tests/run_tests.sh                                               # smoke test and two existing tests (the bootstrap test takes about 3 min)
    sha256sum -c MANIFEST.sha256                                          # file integrity; paths in the manifest are relative to the repository root
