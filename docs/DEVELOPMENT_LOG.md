DEVELOPMENT LOG
===============

Facts only.  Sources: modification times of the original code and result files (UTC+03:00, local time of the machine), the pre-declaration files in
protocols/, and the logs in results/.  A modification time is the time of the last write of a file; a file that is rewritten during a run (per-split
csv files, summary tables) therefore shows the end of the run, not its start.  The creation times of log files, which are opened when a run starts,
are given in protocols/TIMESTAMPS.csv.  All dates are 2026-10-27 to 2026-10-29 (the entries of the last section are dated 2026-10-29).  "Evaluation" means the evaluation splits (runs 5-104) of the data set concerned.


Day 1, 27 October
-----------------
21:25-21:29  The third-party benchmark repository is present as a local clone (files dated 21:25, git metadata 21:29), HEAD commit 4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d.
21:27        scripts/check_a_reproduce_table9.py (recomputation of the published PU B1 Macro-AUROC from their saved logits).
21:33-22:33  scripts/download_pu.sh, pu_loader.py, build_pu_metadata.py; data/pu_metadata.csv written 22:33.
22:05-22:06  scripts/lfpncn/ (geometry, features, model, test).  Bearing geometry: pitch circle 28.55 mm (a value of 29.05 mm is kept for sensitivity checks only).
22:08-22:16  scripts/evaltools/: loaders, metrics, B1 reproduction (results/b1_reproduction.json 22:08:40: Macro-AUROC 79.58 on the 100 evaluation splits, published 79.56), random-forest
             baseline B2 (22:10-22:15, three runs: rounded and exact ccp_alpha, second seed), bootstrap (22:11) and its tests (22:12; bootstrap_tests.json 22:28).
23:48-23:57  scripts/build_cache.py; data/cache/pu_windows.npz written 23:49 (order tensors t_mad and t_max, 20 features b3).  scripts/tier1/ common.py (23:52), tune.py, run_eval.py,
             run_b3.py (23:53), sanity.py (23:54).  B3 random forest on the 20 features, 100 evaluation splits: results/tier1/b3_rf_predictions_long.parquet 23:57.
             Existing evaluation results at this time: B1 (published logits), B2, B3.  No LF-PNCN evaluation prediction exists yet.

Day 2, 28 October
-----------------
00:03        Tier 1 tuning on the five tuning splits finished: chosen configuration lr 1e-3, batch 64, weight decay 1e-4, 40 epochs, validation Macro-AUROC 89.57
             (tuning_table_t_mad.csv, chosen_config_t_mad.json).  No LF-PNCN evaluation prediction existed at that time.
00:16        Separate tuning of the B4 variant, for sensitivity (chosen_config_t_max.json).  The configuration of t_mad was used for B4 in the evaluation.
01:03-01:08  Tier 1 evaluation of LF-PNCN (01:03:46), B4 (01:03:55) and a second seed (01:05:16); metrics.json 01:06 (LF-PNCN 85.63, B1 79.58, B4 75.72, B3 66.76, B2 69.72);
             sanity checks sanity.json 01:08.
04:25-04:28  Fixed-frequency ablation F and the Tier 2/Tier 3 code (scripts/tier2_3/); equivalence check of the re-implemented training loop against Tier 1 logits at 04:28.
04:26-04:47  B1 re-implementation (scripts/b1/); envelope-spectrum bank results/b1/env_spectra.npy built 04:33.  Existing results at this time included the Tier 1 evaluation of all methods.
04:56-05:08  B1 run on six PU splits (5, 25, 45, 65, 85, 104) and compared with the published per-split values (three seeds; mean difference +2.95 Macro-AUROC points; val_3seed_table.csv).
05:27-05:57  B1 Tier 3 (even-numbered splits, then others; summaries 05:27-05:45; a variant with four crop draws per epoch 05:51), Tier 3 for LF-PNCN, B4, F, B3 (predictions 05:29, analysis 05:30),
             B1 Tier 2 (05:41), Tier 1 F (05:57).
06:02-06:16  Tier 2 predictions (06:02), shuffled-label repetitions (06:09-06:14), sanity23.json (06:14), paired comparisons against B1 (compare_vs_b1.json 06:16).
             Result at this point: LF-PNCN loses 14.5 points when trained at 1500 rpm and tested at 900 rpm; its outer-race AUROC at 900 rpm is 55.2.
10:20-10:32  Phase 2 begins (scripts/phase2/): shared helpers 10:20; descriptive diagnostics diag1-diag5 (logs 10:21-10:32) on the 17 benchmark bearings: strength of the fault-order lines,
             effect of longer windows, false alarms of the outer-race head on tuning splits, zoomed spectra, additional bands.  Existing results: Tier 1, 2, 3 for all methods.
10:26-10:42  Feature caches for variants (results/phase2/cache, six files), trainer (10:35), run_tuning.py (10:28), features2.py (10:41), summarize_tuning.py (10:49).
10:26-10:59  Tuning-split runs of the variants (70 files in results/phase2/tuning), including three seeds for base, rowc, rowz, rowz_drop; scripts/phase2/variants.py last changed 10:56:18.
11:00:01     protocols/PREDECLARED.txt written: variants rowz_drop, rowz, rowc, rowz_b6_drop declared; Tier 2 and Tier 3 for rowz_drop only.  Selection basis: tuning splits.
             All scripts of scripts/phase2 except analyze_phase2.py and post_hoc_eval.py (both 11:57:18) had their last modification before 11:00:01.
11:37-12:12  Phase 2 evaluation outputs (first prediction file 11:37:44, last 12:12:43; log of the first run created 11:00:14).  rowz_drop: Tier 1 88.85, difference to LF-PNCN +3.22 [-0.88, +8.72].
11:57        scripts/phase2/post_hoc_eval.py: per-bearing behaviour on the evaluation splits, run after the first evaluation outputs; not used for selection.
12:13        analyze_phase2 outputs (eval_metrics.json 12:13:22) and the tuning summary tables (tuning_table.csv, tuning_t3_table.csv 12:13:25).
13:22        protocols/PREDECLARED_AUDIT.txt (13:22:31); audit code 13:22:32; A1/A2 results 13:39; A3 results 13:47; analysis script 13:48.  Uses tuning splits and the 12 artificial bearings only.
14:10        protocols/PREDECLARED_UORED.txt (14:10:34); results/uored/speed_check.txt is dated 14:10:54 (label-free); cache builder 14:11; first prediction file (base) 14:26:54; log of the run created 14:11:45.
             Statement in the file: B1 not run on UORED.
14:37        UORED analysis: base 80.92, rowz_drop 91.55 (uored_results.json 14:37:50).
14:51        protocols/PREDECLARED_B1_UORED.txt (14:51:00), after the UORED results of base and rowz_drop existed; B1 bank, run (log created 14:52:20) and analysis; results 15:08 (B1 94.21).
             The per-bearing table (per_bearing_auroc.csv, 15:08:46) was corrected after the run: a first version had counted a bearing's healthy recordings as positives in the "all" rows.
14:55-14:59  CWRU: raw-file validation and speed check (14:55), protocols/PREDECLARED_CWRU.txt (14:56:16), cache builder 14:56, run script 14:57, one-split timing run on run 3 (prediction file 14:58:54;
             evaluation run 3 is part of the evaluation set; not used for any decision), chunk runs started from 14:59 (log creation times).  Statement in the file: B1 not run on CWRU.
15:38        CWRU analysis: base 88.40, rowz_drop 95.29 (cwru_results.json 15:38:07).
15:49        protocols/PREDECLARED_B1_CWRU.txt (15:49:59), after the CWRU results of base and rowz_drop existed; B1 run 15:50-15:54 (71.75); analysis 15:56; diagnostic of the B1 pipeline
             (scripts/cwru/b1_diag.py, 15:55; train-set AUROC of the training code, amplitude and label-alignment checks; "nothing is changed").
16:00-16:05  Paderborn run-to-failure: manifest and auxiliary files (16:00-16:01), shaft-line speed check (speed_check.txt 16:04:52), label-free snapshot selection (16:05:47, selection_summary.csv 16:05:48).
16:06:30     protocols/PREDECLARED_RTF.txt.  splits.csv written 16:07:25; window build 16:10 onward (log created 16:07:00; 2869 s).  First prediction file 17:16:42 (B1).
16:42-16:46  XJTU-SY: splits.csv 16:42:46; protocols/PREDECLARED_XJTU.txt 16:43:18; speed check output 16:44:11; window build and run scripts 16:44-16:46 (analysis script 16:46:28); first run log created 16:46:03;
             first prediction file 17:03:54.
17:13-17:29  XJTU-SY label-permutation runs (17:13-17:21), 5% sensitivity runs (17:16-17:29); xjtu_results.json 17:30:30.  Result: permutation control means base 63.1, rowz_drop 58.2, B1 55.8, threshold 56.
17:31        scripts/xjtu/null_life_stage.py (17:31:04), written after the permutation results, marked post hoc in its docstring: life-stage-only detector 85.5 on the permuted draws and 84.9 on the true labels.
17:32:15     protocols/PREDECLARED_RTF_ADDENDUM1.txt.  The same minute: scripts/rtf/rtf_common.py, run_rtf_b1.py and run_rtf_lfpncn.py were modified (they contain the whole-bearing permutation mode);
             analysis script analyze_rtf.py 17:33.  At that time RTF primary predictions existed (17:16-17:39) and B1 label-pair permutation results had been seen; log of the first LF-PNCN permutation run created 17:39:38,
             log of the first whole-bearing permutation run (B1) created 17:32:23.
17:35-18:01  RTF label-pair permutation results (base 65.2, rowz_drop 64.2, B1 63.6) and whole-bearing permutation results (about 49 for all three), interim analysis log 17:41, final analysis 18:03:14.
18:04        results/CROSS_DATASET_SUMMARY had the modification time 18:04:28 when it was first read (it was modified again at 20:28:16, see below).
18:28-18:29  scripts/audit/stage_only_reference.py (18:28:49) and its log (18:29:10): UORED 70.0, Paderborn run-to-failure 84.1.
20:26-20:28  scripts/audit/stage_matched_robustness.py (20:26:36) and its log (20:27:44); results/CROSS_DATASET_SUMMARY modified 20:28:16 (section "Corrections and additions" added).
20:33        A random-forest baseline on CWRU, UORED, XJTU-SY and Paderborn run-to-failure was pre-declared (results/rf/PREDECLARED_RF.txt, 20:33:55) and the runs were completed afterwards; the forest scripts are in scripts/rf/ and its predictions and tables in results/rf/
             (PREDECLARED_RF_ADDENDUM1.txt records the later additions to the protocol).


Summary of which choices were made before which results
-------------------------------------------------------
 - LF-PNCN configuration (learning rate, batch size, weight decay, epochs): chosen on PU tuning splits at 00:03, before any LF-PNCN evaluation prediction existed (01:03).
 - rowz_drop: selected among the phase-2 variants on PU tuning splits and declared at 11:00:01.  It was designed after the PU Tier 3 results (06:02-06:16) and the diagnostics (10:21-10:32) existed.
   The Tier 3 gain of rowz_drop is therefore a post-diagnosis result, not a hypothesis fixed before Tier 3.
 - All runs on CWRU, UORED, XJTU-SY, Paderborn run-to-failure used the PU configuration and the declared variants unchanged.  Their pre-declarations precede their first prediction files and logs (protocols/TIMESTAMPS.csv).
 - The B1 comparisons on UORED (14:51) and CWRU (15:49) were declared after the LF-PNCN results on those data sets existed; on XJTU-SY and Paderborn run-to-failure B1 was part of the first pre-declaration.
 - Analyses added after results were seen: whole-bearing permutation (RTF, 17:32), life-stage-only reference detectors (XJTU-SY 17:31, UORED and RTF 18:29), leave-one-negative-bearing-out robustness (20:26),
   the post-hoc per-bearing look at PU evaluation splits (11:57), and the role given to the late-life-negatives analysis in the addendum (the analysis itself was declared as descriptive before the runs).

Later additions (after the entries above; exact file times are in protocols/TIMESTAMPS.csv)
-----------------------------------------------------------------------------------------
 - Random-forest baseline on CWRU, UORED-VAFCLS, XJTU-SY and Paderborn run-to-failure: completed; the pre-declared prediction (forest at or below the networks where speed varies) was contradicted on Paderborn run-to-failure.
 - Audit round 2 (protocols/PREDECLARED_AUDIT2.txt, results/audit2/): healthy-only negatives, unit-level metric, metadata-only forest, test-invariance check of the training code. The UORED label structure
   (every bearing contributes early-life windows with label 0) was corrected during this round and is recorded in an addendum of the pre-declaration.
 - Audit round 3 (protocols/PREDECLARED_AUDIT3.txt, results/audit3/): UORED per label with bearing-model and life-stage matched negatives, forest with the window mean removed, PU metadata control with the recording DC offset,
   DC sensitivity of the networks, CWRU channel matching, leave-one-condition-out on XJTU-SY, stage-free fault-type evaluation on XJTU-SY and Paderborn run-to-failure (B1 added by an addendum written before its run).
   These analyses were chosen after earlier results existed and are post hoc in that sense; every variant that was run is reported.
 - KAIST part 1 (constant speed, artificial inner- and outer-race cracks), one-shot supplementary evaluation of the stage-free fault-type setting (protocols/PREDECLARED_KAIST.txt, results/kaist/): the protocol was saved before any data
   file was downloaded, one addendum on the channel mapping was saved after the first download and before any window or model output.  Both networks reach 100 in all nine splits, the dataset does not discriminate between them, and a
   time-statistics reference shows that an amplitude cue exists; the result is reported as supplementary evidence with that limit.
