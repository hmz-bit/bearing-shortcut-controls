METHOD AND EVALUATION PROTOCOL
==============================

This description is derived from the code in scripts/ and from the pre-declaration files in protocols/ only.  Where a statement rests on
one of them the source is named.  Numbers are in AUROC points (x100) unless stated otherwise.  Names: "LF-PNCN" (the order-spectrum network, also
called "base" in the result files), "rowz_drop" (its row-normalised variant with dropout), "B1" (the envelope-spectrum baseline), "B2", "B3", "B4", "F" (see section 5).


1. Input representation: order-resampled envelope spectra from three bands
---------------------------------------------------------------------------
Code: scripts/lfpncn/features.py, geometry.py.  One window = 1 s of vibration at 42 kHz (records at other rates are resampled with
scipy.signal.resample_poly; PU 64 kHz uses 21/32; UORED is recorded at 42 kHz; CWRU is used at its native 12 kHz, see section 7).

 a) Three band-pass filters (Butterworth, order 4, zero-phase sosfiltfilt): 500-2500 Hz, 2500-5000 Hz, 5000-10000 Hz (upper edge capped just below Nyquist
    when the sampling rate is lower; CWRU: third band 5000-5994 Hz).
 b) Squared envelope per band: |Hilbert transform of the band signal| squared, mean removed, one-sided FFT magnitude divided by the window length
    (1 Hz bins for a 1-s window).
 c) Robust log-SNR per band: with m = median and s = 1.4826 x MAD of the spectrum over 1 Hz .. (band width) Hz,
        value = log10(1 + max(0, E - m) / (s + 1e-12)).
    (the "max" ablation B4 divides by the maximum of that range instead).
 d) Order resampling.  For the characteristic frequencies of the bearing (geometry z, d, D, contact angle; shaft frequency f_r):
        f_o = z f_r/2 (1 - (d/D) cos th)   outer race            f_i = z f_r/2 (1 + (d/D) cos th)   inner race
        f_b = D f_r/(2d) (1 - ((d/D) cos th)^2)   rolling element   f_c = f_r/2 (1 - (d/D) cos th)   cage
    three modes are formed: outer race (centre f_o, modulation frequency 0), inner race (centre f_i, modulation f_r), rolling element (centre f_b,
    modulation f_c).  For each mode, band and harmonic h = 1..5 the spectrum is linearly interpolated on 64 points xi in [-1, 1] around h x centre,
    with half-width Delta = max(0.03 h x centre, 1.2 x modulation).  Each mode gives a (3 bands x 5 harmonics = 15, 64) array; a window is a
    float32 tensor (3, 15, 64) in the order outer, inner, rolling element.
 e) Shaft frequency f_r: nominal speed of the condition (PU file name, CWRU load condition, XJTU-SY condition folder), a constant derived from the
    spectrum (UORED, 29.75 Hz), or the measured speed of the snapshot when within 3% of the set speed (Paderborn run-to-failure).  Rules per data set are
    in the pre-declarations.  Geometries: scripts/lfpncn/geometry.py and the pre-declarations.


2. Network and training
-----------------------
Code: scripts/lfpncn/model.py (architecture), scripts/tier1/common.py and scripts/phase2/trainer.py (training).
 - Three identical but separate stems (outer, inner, rolling element); each: Conv1d(15->16, k 7) BN ReLU, Conv1d(16->32, k 5) BN ReLU,
   Conv1d(32->32, k 3) BN ReLU, global average pooling over the 64 positions -> 32 features per mode.  Concatenation (96) -> three linear heads
   (outer, inner, rolling element) producing logits.  22,947 trainable parameters.
 - Loss: binary cross-entropy with logits on the outer-race and inner-race outputs only; the rolling-element head never enters the loss.
 - Standardisation: per channel (3 modes x 15 rows = 45) mean and standard deviation over the training windows and the 64 positions only, re-fitted in every split.
 - Optimiser: Adam, learning rate 1e-3, batch size 64, weight decay 1e-4 (coupled, as torch.optim.Adam), 40 epochs, constant learning rate,
   drop_last batches, no early stopping, no validation use, BatchNorm in evaluation mode for scoring.  Seed of evaluation split r is r (PU Tier 1);
   the seed rules of the other data sets are in their pre-declarations.  Scores are raw logits (AUROC is rank based).
 - Variant "rowz_drop" (scripts/phase2/trainer.py, scripts/phase2/variants.py): before the standardiser, every window is transformed row by row
   (each of the 3 x 15 rows over its 64 positions): x -> (x - median(x)) / (1.4826 x MAD(x) + 0.05).  The transform has no learned parameter and
   shares no statistics across windows.  In addition, dropout with probability 0.3 is applied to the 96-dimensional embedding before the heads.
   Everything else is identical to LF-PNCN ("base").
 - Other variants (descriptive or rejected on the tuning splits): rowz (row z-score, no dropout), rowc (centring only), rowz_b6_drop (six bands, 30 rows),
   longer windows (L2, L4), cepstrum pre-whitening and comb editing (cpw1, cpw4, comb), jitter and gain augmentation, class-balanced loss,
   time-statistic side features.  Registry: scripts/phase2/variants.py; all tried variants are in results/phase2/tuning_table.csv.


3. Baselines
------------
 - B1: a one-dimensional CNN (WDCNN) on the fixed-frequency envelope spectrum.  Own re-implementation (scripts/b1/) of the recipe of the third-party
   benchmark study, whose code was read as text only: band-pass 500-10000 Hz (order 12, zero-phase) on the 42 kHz window, |Hilbert|, mean removed, one-sided
   |FFT|/n, giving 21000 bins at 1 Hz; WDCNN with 290,262 parameters for that input; BCE with logits; Adam, lr 1e-3, batch 32, 30 epochs, random 1-s crop with a scalar gain
   |N(1, 0.3)| per training draw, normalisation by the standard deviation of the raw training signals (PU recipe).  The recipe values per data set are in the
   pre-declarations (CWRU: lr 1e-2, 5 draws per epoch, no normalisation, fixed 500 Hz high-pass reading of the cutoff; UORED: lr 1e-4, batch 16, 20 draws per epoch;
   XJTU-SY and Paderborn run-to-failure: the PU recipe, with the deviations stated there).  On PU the "B1" entries of the cross-data-set table are the benchmark
   authors' own saved logits scored with scripts/evaltools; the own re-implementation is used on Tier 2/3 (PU) and on the four external data sets.
   Own B1 on six PU splits was 2.95 points above the published per-split values on average (3 seeds; results/b1/val_3seed_table.csv).
 - B2: random forest on the 15 hand-crafted features of the benchmark study (hyper-parameters of its Table 17: 200 trees, max_features log2, log-loss criterion, max depth 24,
   min_samples_split 15, min_samples_leaf 18, ccp_alpha 0.0265, MultiOutputClassifier, seed 42): scripts/evaltools/b2_rf.py.
 - B3: the same forest on 20 features (5 time-domain statistics + 15 broadband median/MAD log-SNR peak values at 1x-5x of f_o, f_i, f_b), scripts/tier1/run_b3.py.
 - B4: LF-PNCN with max-peak instead of median/MAD normalisation.  F: LF-PNCN tensor with all frequencies fixed to the 1500 rpm reference for every file
   (scripts/tier2_3/build_cache_fixed.py).


4. Evaluation protocol
----------------------
 - Splits are bearing-wise: the unit of splitting is the physical bearing, so no bearing contributes windows to both training and test sets of a split
   (asserted in the code and checked in scripts/tier1/sanity.py and scripts/tier2_3/sanity23.py).  PU, CWRU and UORED use the splits of the third-party benchmark study
   (PU: tuning runs 0-4 with 11 training and 6 validation bearings; evaluation runs 5-104 with 11 training and 6 test bearings, two healthy, two outer-race, two inner-race;
   CWRU: runs 3-52 x scenarios DE and FE; UORED: runs 5-104 with 8 test bearings of 20).  XJTU-SY (10 train / 5 test bearings) and Paderborn run-to-failure (11 train / 6 test)
   use 100 splits drawn with fixed seeds under the constraints written in the pre-declarations (results/xjtu/splits.csv, results/rtf/splits.csv).
   Segment-wise random splits and condition-wise splits are not used anywhere.
 - Metric: for every split, window-pooled ROC AUC of the inner-race score against the inner-race label and of the outer-race score against the outer-race label over all
   windows of the test bearings; Macro-AUROC = mean of the two; reported value = mean over the 100 splits.  FPR at 90% TPR (horizontal-average convention of the benchmark study)
   is computed for PU (scripts/evaltools/metrics.py).
 - Cluster bootstrap (scripts/evaltools/bootstrap.py and the dataset analysis scripts): the independent unit is the physical bearing, not the window and not the split.
   In every replicate (B = 2000, numpy default_rng(0)) the bearings are resampled with replacement within strata (class counts fixed), the same draw is used for all compared methods (paired),
   the AUROC of the pooled resampled test set of each split is computed exactly from precomputed bearing-pair counts (AUROC(m) = m'Cm / ((m.Npos)(m.Nneg)), checked against
   sklearn.metrics.roc_auc_score with sample weights to 2e-16 in scripts/evaltools/test_bootstrap.py and in the dataset scripts), averaged over the splits in which both labels
   still exist, and the difference of means is taken.  Intervals are 95% percentile intervals.  Strata and cluster definitions per data set are in the pre-declarations.
   A synthetic coverage test (scripts/evaltools/test_bootstrap.py, results in scripts/evaltools/results/bootstrap_tests.json) found 87-90% coverage for nominal 95% intervals
   with 17 bearings, so the intervals behave like roughly 90% intervals; the window-level bootstrap (kept only to show the contrast) had 42% coverage.
   The bootstrap covers test-bearing sampling only, not training or seed variance.
 - Decision rules: per data set they are written in the pre-declaration (supported if difference > 0 and lower bound > 0; inconclusive if the point estimate >= 0 and the interval
   contains 0; not supported if the point estimate < 0; the literal +6 point threshold is reported separately).  They were fixed before the corresponding runs (protocols/TIMESTAMPS.csv).


5. What was tuned where, what was frozen
----------------------------------------
Tuned on the five PU tuning splits (runs 0-4) only:
 - Tier 1 grid for LF-PNCN: learning rate {1e-3, 3e-4} x batch {64, 256} x weight decay {0, 1e-4} x epochs {20, 40}; selection by mean validation Macro-AUROC; chosen lr 1e-3, batch 64,
   weight decay 1e-4, 40 epochs (validation Macro 89.57); file results/tier1/chosen_config_t_mad.json, table tuning_table_t_mad.csv.  The same configuration was used unchanged for B4 and F.
   B4 was also tuned separately for sensitivity only (tuning_table_t_max.csv).
 - Selection of rowz_drop among the variants listed in results/phase2/tuning_table.csv (mean over 3 seeds where run; one seed for several others) and of the tuning-split analogue of
   the cross-speed test (tuning_t3_table.csv).  rowz_drop, rowz, rowc and rowz_b6_drop were then declared for evaluation (protocols/PREDECLARED.txt); the other variants were tried and not declared.
Not tuned, frozen at the values above:
 - Every evaluation on PU evaluation splits (runs 5-104), Tier 2 and Tier 3, the audit, and the whole of CWRU, UORED, XJTU-SY and Paderborn run-to-failure.  Hyper-parameters, the variants
   and the seed rules were not changed after seeing results on these data (stated in each pre-declaration; where a pipeline error was found the pre-declarations require rerunning everything).
Descriptive analysis that looked at PU bearings without using the splits: the phase-2 diagnostics (scripts/phase2/diag1-5) used the 17 benchmark bearings, which also occur in the
evaluation splits as test bearings; they computed no model output and selected no hyper-parameter, but they informed which variants were tried.
B1 recipes for PU, UORED and CWRU are the values of the benchmark authors' own configuration files (read as text), which they may have tuned on those data; for XJTU-SY and Paderborn
run-to-failure the PU recipe was used untuned.
Development and evaluation of PU use the same 17 bearings (the benchmark's own protocol).  PU is therefore the development data set; the four others are the external evaluations.


6. Controls and analyses, with the moment they were fixed
---------------------------------------------------------
 Declared before the runs they belong to (protocols/):
  - label permutation across bearings on PU (audit A2), train-versus-held-out gap (A1), untouched artificial bearings (A3): PREDECLARED_AUDIT.txt;
  - label-pair permutation (faulty-end labels reassigned between bearings, healthy windows stay 0), train-versus-held-out gap, per-bearing AUROC, and a "late-life negatives" analysis
    (negatives restricted to faulty-end windows of bearings lacking the defect) on XJTU-SY (PREDECLARED_XJTU.txt items 13, 14) and Paderborn run-to-failure (PREDECLARED_RTF.txt items 11, 12),
    the latter two as descriptive analyses without decision rule.
 Added after results had been seen (post hoc):
  - Whole-bearing label permutation on Paderborn run-to-failure (all windows of a bearing carry the permuted label pair): PREDECLARED_RTF_ADDENDUM1.txt (17:32), written after the first
    B1 label-pair permutation results were seen and before any LF-PNCN permutation run or whole-bearing permutation run.  It records that the declared permutation does not control
    for life stage and that the expectation written in item 12 was wrong.
  - In the addendum the late-life-negatives analysis (already declared as descriptive) is named the more conservative fault-type measure to be reported next to the primary result.
  - Life-stage-only reference detector (score = window comes from a faulty life stage): XJTU-SY scripts/xjtu/null_life_stage.py (results/xjtu/null_life_stage.json, run after the
    permutation results), UORED and Paderborn run-to-failure scripts/audit/stage_only_reference.py.  No pre-declaration.
  - Leave-one-negative-bearing-out robustness of the stage-matched analysis: scripts/audit/stage_matched_robustness.py (point estimates only).  No pre-declaration.
  - A first version of the per-bearing AUROC table for UORED counted a bearing's healthy recordings as positives in the "all" rows; it was corrected after the run (predictions unaffected).
  - Phase-2 post-hoc per-bearing look at the 100 evaluation splits: scripts/phase2/post_hoc_eval.py, not used for any selection.


7. Data-set specific deviations (from the pre-declarations)
-----------------------------------------------------------
 - CWRU: artificial EDM defects; 12 kHz sampling, hence third band truncated to 5000-5994 Hz; windows 1 s with 75% overlap; two bearing geometries by signal location;
   healthy-side clusters by recording configuration.
 - UORED-VAFCLS: single constant shaft frequency 29.75 Hz for all files; windows with 75% overlap; the three states of a bearing are life stages of one physical bearing.
 - XJTU-SY: first 25600 samples of a 1.28 s snapshot resampled 25.6 -> 42 kHz; labels from the dataset providers' failure-location table; healthy = first and failure = last 10% of the files (at least 5);
   B1 with three fixed crop offsets.
 - Paderborn run-to-failure: first second of channel A of selected snapshots; healthy = first and faulty = last 10% of the snapshots whose set speed lies in the experiment's speed range (at most 100 per end);
   contact angle 0 assumed; no outer-race-only bearing exists.
 - Both run-to-failure data sets define labels by early versus late life; a life-stage-only detector scores 84.9 (XJTU-SY) and 84.1 (Paderborn run-to-failure) with the true labels, above all models there.
