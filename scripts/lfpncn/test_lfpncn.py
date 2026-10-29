"""Plain-assert tests. Run:  python -I test_lfpncn.py"""
import glob
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

from features import (FS_TARGET, make_windows, normalise_spectra, resample_to_target,
                      squared_envelope_spectra, window_to_array, window_to_tensors)
from geometry import CWRU_6205, PU_6203, PU_6203_SPEC, fault_frequencies, shaft_frequency_from_name
from model import LFPNCN, count_parameters

XI = np.linspace(-1.0, 1.0, 64)
BAND2 = slice(5, 10)          # rows of band [2500,5000] Hz (contains the 3 kHz resonance)


def synth_fault(rpm, f_fault, fs=64000, f_res=3000.0, beta=800.0, noise=0.5, seed=0):
    """3 kHz damped resonance excited by an impulse train at f_fault (fractional timing), plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(fs) / fs
    x = np.zeros(fs)
    for tk in np.arange(0.01, 1.0 - 0.02, 1.0 / f_fault):
        i0 = int(tk * fs)
        tt = t[i0:i0 + 1500] - tk
        x[i0:i0 + 1500] += np.exp(-beta * tt) * np.sin(2 * np.pi * f_res * tt)
    return x + noise * x.std() * rng.standard_normal(fs)


def peak_xi(T):
    """xi of the maximum for each of the 5 harmonics in band 2."""
    return [float(XI[T[BAND2][h].argmax()]) for h in range(5)]


def delta(h, f_m):
    return 0.03 * h * f_m


def test_geometry():
    r = fault_frequencies(1.0, PU_6203)
    r2 = fault_frequencies(1.0, PU_6203_SPEC)
    rc = fault_frequencies(1.0, CWRU_6205)
    print(f"[geom] PU 6203 (D=28.55, Lessmeier Table 1): f_o/f_r={r['f_o']:.4f} f_i/f_r={r['f_i']:.4f} "
          f"f_b/f_r={r['f_b']:.4f} f_c/f_r={r['f_c']:.4f}")
    print(f"[geom] PU 6203 (D=29.05, alternative):        f_o/f_r={r2['f_o']:.4f} f_i/f_r={r2['f_i']:.4f} "
          f"f_b/f_r={r2['f_b']:.4f}")
    print(f"[geom] CWRU 6205: f_o/f_r={rc['f_o']:.4f} f_i/f_r={rc['f_i']:.4f} f_b/f_r={rc['f_b']:.4f}")
    assert abs(r["f_o"] - 3.054) < 0.005 and abs(r2["f_o"] - 3.07) < 0.005
    assert abs(rc["f_o"] - 3.585) < 0.005
    assert abs(r["f_o"] + r["f_i"] - 8 * 1.0) < 1e-9          # f_o + f_i = z*f_r
    assert abs(shaft_frequency_from_name("N15_M07_F10_K001_1") - 25.0) < 1e-9
    assert abs(shaft_frequency_from_name("N09_M07_F10_K001_1") - 15.0) < 1e-9


def test_a_synthetic_fault():
    rpm = 1500
    f_r = rpm / 60.0
    ff = fault_frequencies(f_r, PU_6203)
    w = make_windows(resample_to_target(synth_fault(rpm, ff["f_o"])))[0]
    t = window_to_tensors(w, f_r, PU_6203)
    for m in t:
        assert t[m].shape == (15, 64) and t[m].dtype == np.float32
    p_or, p_ir = peak_xi(t["OR"]), peak_xi(t["IR"])
    print(f"[a] BPFO={ff['f_o']:.3f} Hz @1500 rpm. OR peak xi (band 2, h=1..5): {np.round(p_or, 3).tolist()}")
    print(f"[a] IR peak xi (band 2, h=1..5): {np.round(p_ir, 3).tolist()}")
    allb = [[float(XI[t['OR'][b * 5 + h].argmax()]) for h in range(5)] for b in range(3)]
    print(f"[a] OR peak xi per band (b1,b2,b3): {np.round(allb, 3).tolist()}")
    for h in range(1, 6):
        tol = 0.5 / delta(h, ff["f_o"]) + 0.05          # half a 1-Hz bin expressed in xi
        assert abs(p_or[h - 1]) <= tol, (h, p_or[h - 1], tol)
    n_ir_center = sum(abs(p) < 0.15 for p in p_ir)
    cen_or = float(np.mean(t["OR"][BAND2][:, 30:34]))
    cen_ir = float(np.mean(t["IR"][BAND2][:, 30:34]))
    print(f"[a] mean value at xi~0 (band 2): OR={cen_or:.3f}, IR={cen_ir:.3f}; "
          f"IR peaks with |xi|<0.15: {n_ir_center}/5")
    assert n_ir_center <= 1 and cen_or > 2.0 * cen_ir


def test_b_healthy_noise():
    rng = np.random.default_rng(1)
    means, tmax_mad, tmax_max, smax_max, smax_mad, lit_means = [], [], [], [], [], []
    for _ in range(5):
        w = rng.standard_normal(FS_TARGET)
        a = window_to_array(w, 25.0, PU_6203, normalize="mad")
        b = window_to_array(w, 25.0, PU_6203, normalize="max")
        lit = window_to_array(w, 25.0, PU_6203, normalize="mad", stat_range="full")
        fq, E = squared_envelope_spectra(w)
        smax_max.append(float(normalise_spectra(fq, E, "max").max()))
        smax_mad.append(float(normalise_spectra(fq, E, "mad").max()))
        means.append(float(a.mean()))
        tmax_mad.append(float(a.max()))
        tmax_max.append(float(b.max()))
        lit_means.append(float(lit.mean()))
    print(f"[b] noise, MAD log-SNR (stats over 1..bandwidth Hz): tensor mean={np.mean(means):.3f}, "
          f"tensor max={np.max(tmax_mad):.3f}, full-spectrum max={np.max(smax_mad):.3f}")
    print(f"[b] noise, MAD log-SNR with literal full-range (0..fs/2) stats: tensor mean={np.mean(lit_means):.3f}")
    print(f"[b] noise, normalize='max': spectrum max in [{np.min(smax_max):.6f}, {np.max(smax_max):.6f}]; "
          f"sampled-tensor max={np.max(tmax_max):.3f}")
    assert np.mean(means) < 0.4 and np.max(tmax_mad) < 1.5
    assert np.allclose(smax_max, 1.0, atol=1e-6)
    assert np.max(tmax_max) <= 1.0 + 1e-6
    ff = fault_frequencies(25.0, PU_6203)
    w = make_windows(resample_to_target(synth_fault(1500, ff["f_o"])))[0]
    fpeak = float(window_to_array(w, 25.0, PU_6203)[0][BAND2].max())
    print(f"[b] fault OR peak (MAD)={fpeak:.3f} vs noise tensor max={np.max(tmax_mad):.3f}")
    assert fpeak > 2.0 * np.max(tmax_mad)


def test_c_speed_invariance():
    res = {}
    for nf in (1, 4):
        pk = {}
        for rpm in (900, 1500):
            f_r = rpm / 60.0
            ff = fault_frequencies(f_r, PU_6203)
            w = make_windows(resample_to_target(synth_fault(rpm, ff["f_o"])))[0]
            pk[rpm] = peak_xi(window_to_tensors(w, f_r, PU_6203, nfft_factor=nf)["OR"])
        shift = [abs(a - b) for a, b in zip(pk[900], pk[1500])]
        res[nf] = (pk, shift)
        print(f"[c] nfft_factor={nf}: peak xi @900={np.round(pk[900], 3).tolist()}, "
              f"@1500={np.round(pk[1500], 3).tolist()}, |shift|={np.round(shift, 3).tolist()}, max={max(shift):.3f}")
    f_o9 = fault_frequencies(15.0, PU_6203)["f_o"]
    for h in range(1, 6):       # both peaks lie within half a bin of the line => shift <= 1 bin in xi
        assert res[1][1][h - 1] <= 1.0 / delta(h, f_o9) + 0.1, (h, res[1][1][h - 1])
    for rpm in (900, 1500):
        fo = fault_frequencies(rpm / 60.0, PU_6203)["f_o"]
        for h in range(1, 6):
            assert abs(res[1][0][rpm][h - 1]) <= 0.5 / delta(h, fo) + 0.05


def test_d_param_count():
    m = LFPNCN()
    n = count_parameters(m)
    stem = sum(p.numel() for p in m.stems[0].parameters())
    head = sum(p.numel() for p in m.heads[0].parameters())
    print(f"[d] params total={n}, stem={stem} (x3), head={head} (x3); expected 22947, diff={n - 22947}")
    assert stem == 7552 and head == 97 and n == 22947
    x = torch.randn(4, 3, 15, 64)
    y = m(x)
    assert y.shape == (4, 3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, torch.randint(0, 2, (4, 3)).float())
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert m.stems[0][0].weight.data_ptr() != m.stems[1][0].weight.data_ptr()
    print(f"[d] forward/backward OK, logits shape {tuple(y.shape)}")


def test_e_real_k001():
    cands = sorted(glob.glob(os.path.join(HERE, "..", "..", "data", "pu", "K001", "N09_*K001_1.mat")))
    if not cands:
        print("[e] SKIPPED: no K001 file found")
        return
    sys.path.insert(0, os.path.join(HERE, ".."))
    import pu_loader
    meta, x = pu_loader.load_vibration(cands[0])
    f_nom = shaft_frequency_from_name(os.path.basename(cands[0]))
    f_meas = meta["rpm_measured"] / 60.0
    w = make_windows(resample_to_target(x, meta["fs"]))
    print(f"[e] {os.path.basename(cands[0])}: raw n={len(x)}, windows={len(w)}, nominal f_r={f_nom:.3f} Hz, "
          f"measured rpm={meta['rpm_measured']:.1f} (f_r={f_meas:.3f} Hz)")
    arr = window_to_array(w[0], f_nom, PU_6203)
    assert arr.shape == (3, 15, 64) and arr.dtype == np.float32 and np.isfinite(arr).all()
    print(f"[e] real K001 window tensor: mean={arr.mean():.3f}, max={arr.max():.3f}, "
          f"per-mode max OR/IR/Ball={arr[0].max():.3f}/{arr[1].max():.3f}/{arr[2].max():.3f}")
    m = LFPNCN().eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(arr)[None]))
    print(f"[e] untrained model probs (OR,IR,Ball)={np.round(p[0].numpy(), 3).tolist()}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASSED")
