# evaltools: evaluation and statistics tools for the PU bearing benchmark

Everything here reads `bearing-data-leakage/data/` (third-party clone, no license) as **read-only data**. No
third-party code is imported or executed, nothing is unpickled (`files_metadata.bz2` is never opened; bearing
labels are derived from the bearing code), torch tensors are read with `torch.load(..., map_location="cpu",
weights_only=True)`. CPU only. Interpreter: `python`; run scripts from this folder.

## Data schemas (verified)

| File | Content |
|---|---|
| `features/paderborn_features_segmented.parquet` | 5440 rows x 17 columns: `waveform_id`, `waveform_id_seg` (str) + 15 float64 features, no NaN. Bearing id = 4th `_` token of `waveform_id` (`N15_M07_F10_KA15_16` -> `KA15`). 17 bearings x 320 windows (80 waveforms x 4 segments), stored in bearing-contiguous blocks (order: KA15, KI17, K006, K003, KI04, K004, KI21, KI14, KI18, K005, KA16, KI16, K002, KA30, K001, KA22, KA04). Features: `acceleration/{rms,pk-pk,kurt,skewness,fc}/global` (5 time statistics, `fc` = crest factor) and `envelope/spectralPeak/{1..5}.0x-{bpfo,bpfi}/500-10000` (10 envelope magnitudes). **No `bpfb` columns, no label columns, no bearing column.** |
| `splits/paderborn/eval/run_{5..104}.parquet` | 100 files, 1 row each: `train_ids` (list of 11 bearing ids), `test_ids` (list of 6). Train/test is marked only by these id lists; each test set has 2 healthy + 2 outer + 2 inner bearings. 100 distinct test sets, every one of the 17 bearings appears in 25 to 48 of them, so the splits overlap heavily. |
| `splits/paderborn/tuning/run_{0..4}.parquet` | 5 files: `train_ids` (11), `val_ids` (6). Used only for hyper-parameter search. |
| `saved_logits/pu/paderborn_eval_envelope_lr_0.001_bs_32_norm_global_run_{5..104}/test/epoch_30_run_0.pt` | float32 tensor (n, 5), n = 1907..1918: `[row index, inner logit, outer logit, inner label, outer label]`. **No bearing id.** |

Labels: `K***` healthy (0,0), `KA**` outer fault (inner=0, outer=1), `KI**` inner fault (1,0). Bearings: 6 healthy
(K001-K006), 5 outer (KA04, KA15, KA16, KA22, KA30), 6 inner (KI04, KI14, KI16, KI17, KI18, KI21).

**Recovering bearing ids for the logits.** Logit rows follow the global bearing order of the features file
(verified: the label-run sequence of every one of the 100 files matches the order implied by that sequence), with
315 to 320 windows per bearing (a few dropped upstream). `pu_data.infer_bearing_counts` measures every bearing's
window count from isolated label runs, asserts the count is identical across all splits, and
`pu_data.logits_to_long` asserts that the labels implied by the inferred bearing equal the stored labels for every
window of every run. Counts: K001 320, K002 318, K003 320, K004 320, K005 320, K006 320, KA04 319, KA15 317,
KA16 319, KA22 315, KA30 319, KI04 320, KI14 320, KI16 319, KI17 319, KI18 317, KI21 318. Logit windows and feature
windows cannot be matched one to one (the dropped windows are unknown), so paired comparisons across models are
paired at the bearing level, not the window level.

## Files and functions

* `pu_data.py` loaders: `load_features()`, `bearing_order()`, `load_eval_split(run)`, `logit_path(run)`,
  `bearing_labels(id)`, `bearing_class(id)`, `infer_bearing_counts()`, `logits_to_long()` (long table with columns
  `split, bearing, y_inner, y_outer, s_inner, s_outer`). Constants `FEATURES_15`, `ENV_BPFB_FEATURES`, `EVAL_RUNS = 5..104`.
* `metrics.py`
  * `load_logit_run(path)` -> dict `idx, logit_inner, logit_outer, y_inner, y_outer` (weights_only load).
  * `macro_auroc(y_true, scores, return_per_label=False)`: mean of per-label sklearn AUROCs over the columns of (n, L) arrays.
  * `fpr_at_tpr(y, s, tpr=0.9, method)`: single ROC; `nearest` (authors' notebook: ROC vertex with TPR closest to 0.9), `interp`, `first` (smallest FPR with TPR >= target).
  * `horizontal_average_roc(curves, points=10000)`, `vertical_average_roc(curves, points)`: Vieira's two averaging schemes (code in their `results.ipynb`, read as text).
  * `mean_fpr_at_tpr(ys, ss, tpr, convention)`: conventions `horizontal` (authors' procedure), `horizontal_interp`, `per_run_nearest|interp|first`, `vertical`, `pooled`.
* `reproduce_b1.py`: writes `results/b1_reproduction.json`.
* `b2_rf.py`: `run_rf(df, features, params, runs, seed)`, `build_b3_features(df)`; CLI flags `--ccp-exact`, `--seed`, `--tag`. Writes `results/<tag>_per_split.csv`, `_predictions_long.parquet`, `_summary.json`.
* `bootstrap.py`:
  * `build_pair_table(frame)`: precomputes, per split and label, the matrix C[i, j] of (positive window of bearing i, negative window of bearing j) pair counts, so the AUROC of any bearing resample is exact and instantaneous: `AUROC(m) = m'Cm / ((m.Npos)(m.Nneg))` (m = bearing multiplicities). Checked against `roc_auc_score(sample_weight=m)` to 2e-16.
  * `draw_multiplicities(bearings, B, rng, stratify=True)`: with-replacement bearing draw; stratified keeps the class sizes (6 healthy / 5 outer / 6 inner, i.e. every resample still has all three classes).
  * `paired_bootstrap_single_split(y, scores_a, scores_b, bearing_ids, B=2000)`: one test set; resamples test bearings, recomputes Macro-AUROC of A and B on the same resample, returns mean difference, SE and 95% percentile CI (AUROC points x100).
  * `paired_bootstrap_multi_split(frame_a, frame_b, B=2000, stratify=True)`: the 100-split version. The 17 bearings are resampled once per replicate (shared by A and B), each split's pooled test set uses the resampled multiplicities, AUROC is computed per split and averaged over the splits where both labels are still defined (about 84 of 100 per replicate when stratified), then A - B. This correctly treats the overlapping splits: the replicate unit is the bearing, not the split.
  * `naive_window_bootstrap(...)`: window-level iid bootstrap, kept only to demonstrate how wrong it is.
  * `minimum_detectable_effect(sd_points, n_units, rho, alpha, power)` and `empirical_mde(se)`.
* `analysis.py`: B1 vs B2 paired comparison and MDE -> `results/analysis.json`.
* `test_bootstrap.py`: exactness + synthetic coverage tests -> `results/bootstrap_tests.json`.

## Reproduced numbers (nothing tuned)

| Quantity | Published (Vieira et al.) | Reproduced | Note |
|---|---|---|---|
| B1 Macro-AUROC (WDCNN-envelope, 100 saved logit files) | 79.56 +- 13.07 (Tab. 9) | **79.58 +- 12.96** (ddof=0; 13.03 with ddof=1) | +0.02 mean, std within 0.1 |
| B1 per-label AUROC inner / outer | 0.898 / 0.693 (Fig. 11) | 0.8987 / 0.6930 | |
| B1 mean FPR@TPR=90% inner | 24.00% | **24.00%** (24.0019) | horizontal average, 10000-point TPR grid, nearest-to-0.9 grid point |
| B1 mean FPR@TPR=90% outer | 60.15% | **60.15%** (60.1545) | same |
| B2 RF Macro-AUROC (Tab. 17 params, seed 42) | 69.76 +- 15.72 (Tab. 9) | **69.72 +- 15.76** (ddof=0; 15.84 ddof=1) | -0.04 mean |
| B2 with notebook's unrounded ccp_alpha | | 69.72 +- 15.76 | no effect |
| B2 seed 1 | | 69.53 +- 15.83 | seed noise ~0.2 |
| B2 per-label inner / outer | | 0.8315 / 0.5630 | |

FPR@90 averaging convention (B1): only the **horizontal** average (the authors' `horizontal_average` + `fpr_score`)
matches both numbers. Per-run interp / first-point means also give 24.01/60.15 and 24.00/60.15 (nearly the same
quantity), per-run nearest gives 23.82/60.00, whereas vertical averaging (38.0/72.2) and pooling all windows
(33.1/74.1) do not match.

B3 (20 features) **cannot be built from the available parquet**: it has envelope peaks only for `bpfo` and `bpfi`
(10 columns, 15 features in total, which is exactly what the 15-feature RF of the paper used). The five `bpfb`
(ball-spin) columns are missing; B3 would need the raw PU signals (not downloaded) plus the 6203 bearing geometry to
recompute envelope peaks at 1x-5x of f_b. `cwru_features_segmented20.parquet` is a 20-feature file for CWRU only.

## Paired comparison B1 vs B2 (bearing-level cluster bootstrap, B=2000, stratified)

B1 79.58 [70.9, 86.3] (SE 4.0), B2 69.72 [57.3, 82.3] (SE 6.8); **difference 9.86 points, 95% CI [-1.90, +20.92]**,
SE 5.83, bootstrap P(diff <= 0) = 0.053. Unstratified resampling gives [-2.45, +23.41]. So the headline gap between
the CNN and the RF is **not distinguishable from zero at the bearing level** (95%), although it looks decisive
against the across-split SDs if the 100 splits are wrongly treated as independent. Per single 6-bearing test set,
the paired CI excludes 0 in only 34% of the splits (B=500 each).
Across splits: SD(B1) 13.03, SD(B2) 15.84, correlation of per-split macro-AUROC between B1 and B2 only 0.41,
SD of the paired per-split difference 15.84.

## Bootstrap unit tests (synthetic world, known truth; `results/bootstrap_tests.json`)

World: 17 bearings (6/5/6), score = delta * label + shared bearing effect + model-specific bearing effect + window noise;
A has delta 1.5, B has delta 1.0 (true paired pooled-AUROC difference 8.29 points on single sets, 8.22 for the
split-averaged statistic; both computed by Monte Carlo over thousands of independent bearing populations).

| Test | Result |
|---|---|
| Exactness: pair-matrix AUROC vs sklearn weighted AUROC (100 comparisons) | max abs error 2.2e-16 |
| Single set, **cluster** bootstrap, B=1000, 300 replicates | 95% CI covers the true difference in **89.3%**; mean width 13.6; excludes 0 in 60% (power) |
| Single set, null world (true difference 0) | coverage 89.0%; false rejection **11.0%** (nominal 5%) |
| Single set, **naive window** bootstrap, 60 replicates | coverage **41.7%**; mean width 3.9; null false rejection 58% |
| 100 overlapping splits, multi-split estimator, B=500, 100 replicates | coverage **90.0%**; mean width 10.0; excludes 0 in 85% |
| 100 splits, null world | coverage 87.0%; false rejection **13%** |

Reading: the cluster bootstrap is mildly anti-conservative with only 17 (or 6) clusters (coverage 87 to 90% instead
of 95%, the usual small-cluster percentile-bootstrap behaviour), so treat its CIs as roughly 90% intervals and its
"significant" calls with caution; the window-level bootstrap is wildly overconfident (42% coverage) and must not be used.

## Minimum detectable effect (80% power, two-sided 5%)

Normal approximation `delta = 2.80 * SD_diff / sqrt(n)`, `SD_diff = 13 * sqrt(2 (1 - rho))`, n = independent bearings:

| rho (correlation of the two models' per-split AUROC) | n = 17 | n = 11 | n = 5 (smallest class) |
|---|---|---|---|
| 0.0 | 12.5 | 15.5 | 23.0 |
| 0.5 | 8.8 | 11.0 | 16.3 |
| 0.8 | 5.6 | 6.9 | 10.3 |
| 0.9 | 4.0 | 4.9 | 7.3 |

Observed B1/B2 correlation is 0.41 (formula gives 9.6 points at n = 17 with SD 13). The data-driven estimate from
the bootstrap SE of the paired difference (5.83) is `2.8 * 5.83 = 16.3` points, larger because the outer-fault AUROC
rests on only 5 bearings and the effective number of independent units is smaller than 17 (this explanation is a guess, not tested). **Practical MDE for a
paired Macro-AUROC comparison on this benchmark: about 10 to 16 points** (about 12.5 if the models are independent).
Differences of a few points between architectures cannot be resolved with 17 bearings; use per-bearing reporting,
bearing-level bootstrap CIs, and effect-size language rather than significance claims.

## Not reproduced / caveats

* Macro-AUROC means differ from the paper by 0.02 (B1) and 0.04 (B2); the std by 0.1 (B1, partly ddof) and 0.04 (B2). The
  most likely causes are ddof/rounding conventions and, for RF, scikit-learn version and RNG differences (seed noise ~0.2).
  Not investigated further, nothing tuned.
* The synthetic coverage is below nominal (87 to 90%), see above.
* The bootstrap of B1 uses 100 distinct trained networks (one per split) with 6 test bearings each; resampling bearings
  does not re-train, so it captures test-bearing sampling variance only, not training-set or seed variance.
