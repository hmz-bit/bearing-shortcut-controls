# LF-PNCN input pipeline and model

Run the tests (global interpreter, CPU only, no network):

    python -I test_lfpncn.py

Files: `geometry.py` (fault frequencies), `features.py` (input pipeline), `model.py` (network), `test_lfpncn.py`.
Scripts add their own folder to `sys.path` (needed because `-I` removes it); import from this folder.

## API

geometry.py
- `Bearing(name, z, d, D, theta_deg)`; presets `PU_6203` (D = 28.55 mm), `PU_6203_SPEC` (D = 29.05 mm), `CWRU_6205`.
- `fault_frequencies(f_r, bearing) -> {"f_o","f_i","f_b","f_c"}` in Hz.
- `shaft_frequency_from_name("N15_M07_F10_K001_1") -> 25.0`, `shaft_frequency_from_rpm(rpm)`.

features.py
- `resample_to_target(x, fs_in=64000)`: `resample_poly` 21/32 to 42 kHz.
- `make_windows(x42k)`: non-overlapping 1-s windows, shape `(n, 42000)`.
- `squared_envelope_spectra(w, fs, nfft_factor=1) -> (freqs, E[3, n_freq])`.
- `normalise_spectra(freqs, E, normalize="mad"|"max", stat_range="band"|"full")`.
- `order_resample(freqs, Et, f_m, f_mod)`: `(15, 64)`, row = band*5 + (h-1).
- `window_to_tensors(w, f_r, bearing, ...) -> {"OR","IR","Ball": float32 (15,64)}`;
  `window_to_array(...) -> (3,15,64)`; `signal_to_array(x64k, f_r, bearing) -> (n_win,3,15,64)`.
- f_mod = 0 (OR), f_r (IR), f_c (Ball). Delta = max(0.03 h f_m, 1.2 f_mod).

model.py
- `LFPNCN()`: input `(B,3,15,64)` (order OR, IR, Ball), output logits `(B,3)`; train with `BCEWithLogitsLoss`.
- `count_parameters(model)` = 22,947 (3 x 7,552 stems + 3 x 97 heads).

## Choices not fixed by the method definition
- Butterworth order 4 (`sosfiltfilt`, zero phase). Band 3 upper edge = min(10000, fs/2) = 10 kHz at 42 kHz.
- Envelope spectrum = |rfft| / N of the mean-removed squared envelope (1 Hz bins for 1-s windows).
- Median/MAD (and the max for `normalize="max"`) are computed over 1 Hz .. (band width) Hz, the support of the squared
  envelope of a band-limited signal, not over 0..fs/2 (see Deviations). `stat_range="full"` gives the literal version.
- `nfft_factor` (default 1) optionally zero-pads the FFT. It interpolates the grid but does not improve the true 1 Hz resolution.

## Deviations
1. Statistics range: with median/MAD over all bins up to 21 kHz the noise floor is dominated by near-empty bins, so
   sigma is tiny and pure noise gives tensor mean about 3.0 instead of about 0.26. The default restricts to the envelope support.
2. PU geometry: Lessmeier et al. 2016 Table 1 gives pitch circle diameter 28.55 mm (not 29.05 mm); z = 8, d = 6.75 mm and
   nominal pressure angle 0 are confirmed. f_o/f_r = 3.054 (28.55) vs 3.071 (29.05). `PU_6203` uses 28.55.
