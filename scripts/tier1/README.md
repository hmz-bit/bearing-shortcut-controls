# Tier 1: bearing-wise, real-damage evaluation on Paderborn (PU), LF-PNCN vs baselines

Input: `data/cache/pu_windows.npz` (read with `allow_pickle=False`). Only the 17 bearings of Vieira et al.
are used (K001-K006 healthy, KA04/15/16/22/30 outer, KI04/14/16/17/18/21 inner). Splits and labels come from
`evaltools/pu_data.py` (tuning runs 0..4, evaluation runs 5..104). Nothing in `bearing-data-leakage` is executed or unpickled.
Output: `results/tier1/`.

## Files
| File | Purpose |
|---|---|
| `common.py` | data loading, split loading, training function (`train_lfpncn`), standardiser, determinism flags |
| `tune.py` | grid search on the 5 tuning splits only (train/val); writes `tuning_table_<variant>.csv`, `chosen_config_<variant>.json` |
| `run_eval.py` | trains LF-PNCN (`--variant t_mad`) or B4 (`--variant t_max`) on the 100 evaluation splits with `chosen_config_t_mad.json` |
| `run_b3.py` | B3 Random Forest on the 20 `b3` features, `evaltools.b2_rf` hyper-parameters, `run_rf` protocol |
| `analyze.py` | metrics, FPR@90, bearing-level paired cluster bootstrap (evaltools `bootstrap.py`), decisions |
| `sanity.py` | sanity checks (a)-(e) |

## Protocol
* Network: `lfpncn/model.py` `LFPNCN()` (22,947 parameters). Logit columns are (OR, IR, Ball). Labels `[inner, outer]` map to
  IR and OR. PU has no ball fault, so the loss is `BCEWithLogitsLoss` on columns OR and IR only; the Ball column never enters the loss.
  Scores saved are raw logits (`s_inner` = IR logit, `s_outer` = OR logit); AUROC and ROC are rank-based.
* Optimiser: Adam (L2 weight decay coupled, as in `torch.optim.Adam`), constant learning rate, no scheduler, no early stopping, fixed epochs,
  `drop_last` batches, BatchNorm in eval mode at scoring. cuDNN deterministic flags on. Seed of split `r` is `r` (tuning: `1000 + r`).
* Standardisation: per-channel (3 modes x 15 rows = 45 channels) mean/std over the training windows and the 64 positions only, fitted again in every split.
* Selection: mean validation Macro-AUROC over the 5 tuning splits; grid lr {1e-3, 3e-4} x batch {64, 256} x weight decay {0, 1e-4} x epochs {20, 40}
  (the 20-epoch value is the epoch-20 checkpoint of the 40-epoch run; legitimate because the learning rate is constant).
  The configuration chosen for `t_mad` is reused unchanged for B4 (`t_max`). B4 was also tuned separately (sensitivity only, `tuning_table_t_max.csv`).
* Statistics: `evaltools.bootstrap.paired_bootstrap_multi_split` (bearing-level, stratified, B = 2000, seed 0). FPR@TPR=90% uses
  `evaltools.metrics.mean_fpr_at_tpr(..., convention="horizontal")` per label over the 100 splits.

## Rerun (from this folder, about 1.5 to 2 hours on the RTX 3050 Ti when run sequentially; runs can be started in parallel)
```
PY=python
export PYTHONUTF8=1
cd scripts/tier1
$PY -I tune.py --variant t_mad                         # writes chosen_config_t_mad.json
$PY -I tune.py --variant t_max                         # sensitivity only
$PY -I run_eval.py --variant t_mad --tag lfpncn
$PY -I run_eval.py --variant t_max --tag b4_tmax       # same config as t_mad
$PY -I run_eval.py --variant t_mad --tag lfpncn_seed2 --seed-offset 1000   # seed robustness (supplementary)
$PY -I run_b3.py
$PY -I sanity.py
$PY -I analyze.py --B 2000
```
`analyze.py` reads B1 and B2 from `evaltools/results/b1_predictions_long.parquet` and `b2_rf_predictions_long.parquet` (read-only).
