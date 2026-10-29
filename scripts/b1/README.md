B1 baseline (WDCNN on the fixed-Hz envelope spectrum, Vieira et al. 2026), own implementation.
Code in scripts/b1, outputs in results/b1.
Nothing from bearing-data-leakage was imported, executed or unpickled; its Python and YAML files were read as text.

Files
  envspec.py        envelope spectrum (band-pass 500-10000 Hz order 12 sosfiltfilt padlen 6 even, |Hilbert|, mean removed,
                    one-sided |FFT|/n with bins>=1 doubled; 42000 samples -> 21000 bins, 1 Hz).  Source: signal_representations.py
                    get_envelope / array_filter, power_cepstrum.py get_fft, configs/transform/envelope.yaml, eval_envelope.yaml.
  wdcnn.py          WDCNN (290,262 parameters for the 21000-bin input).  Source: src/models/components/backbone/wdcnn.py.
  cropworker.py     numpy/scipy-only pool worker (random 1-s crop -> envelope spectrum).
  core.py           data access, training loop (Adam 1e-3, batch 32, 30 epochs, BCE-with-logits), scoring.
  build_signals.py  builds results/b1/signals_42k.f32 (+signals_meta.csv) and results/b1/env_spectra.npy.
  run_b1.py         runs a tier (t1val, t1shuf, t3, t2); resumable, one parquet shard per split in results/b1/shards/<tag>/.
  analyze_b1.py     merges shards, writes *_predictions_long.parquet, *_per_split.csv, *_summary.json.

env_spectra.npy: float16 (9258, 21000), rows identical to data/cache/pu_windows.npz (fname, win, bearing, condition;
  copy in env_index.csv).  Values are envelope spectrum x 1000 (raw sensor units), see env_spectra_scale.txt.
signals_42k.f32: flat float32 of the mean-removed, resample_poly(21/32) 42 kHz full signals; offsets/lengths and raw 64 kHz sums
  (for the training-set std) in signals_meta.csv.

Rerun (from the repository root, PYTHONUTF8=1, OPENBLAS_NUM_THREADS=1):
  python scripts/b1/build_signals.py
  python -I scripts/b1/run_b1.py --tier t1val --workers 6 [--seed 1 --tag seed1]
  python -I scripts/b1/analyze_b1.py val [seed1]
  python -I scripts/b1/run_b1.py --tier t3 --workers 6 [--splits 5,7,...]   (default: even splits 6..104)
  python -I scripts/b1/analyze_b1.py t3
  python -I scripts/b1/run_b1.py --tier t2 --workers 6
  python -I scripts/b1/analyze_b1.py t2
  python -I scripts/b1/run_b1.py --tier t1shuf --workers 6      (shuffled training labels, splits 5, 25, 45)
  python -I scripts/b1/analyze_b1.py t1shuf
  sensitivity: run_b1.py --tier t3 --mult 4 --tag mult4 ; analyze_b1.py t3 mult4
Long-format outputs: t3_predictions_long.parquet (split, bearing, y_inner, y_outer, s_inner, s_outer [logits], test_cond, fname, win),
  t2_predictions_long.parquet (same + held_healthy; split = pair index 1..15 of itertools.combinations(K001..K006, 2)).
