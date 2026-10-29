"""Registry of phase 2 variants: name -> dict(feat=<'+'-joined input names>, opt=<trainer options>, cfg=<optional override of lr/bs/wd/epochs>).
Input names: 'L1' = cache t_mad (Tier 1 input); anything else = results/phase2/cache/feat_<name>.npy (build_features.py).
'T1' as a side-feature name would be time statistics; use opt['extra']='tstats'.
Plug in new inputs: add an entry to features2.SPECS (or a new .npy in results/phase2/cache), then one entry here.
"""
import numpy as np
import p2common as P

CACHE = P.OUT / "cache"

VARIANTS = {
    "base":       dict(feat="L1", opt={}),
    # prototypes (names fixed here; the tuning table reports all of them)
    "L2":         dict(feat="L2", opt={}),
    "L4":         dict(feat="L4", opt={}),
    "ML14":       dict(feat="L1+L4", opt={}),
    "ML124":      dict(feat="L1+L2+L4", opt={}),
    "cpw1":       dict(feat="L1cpw", opt={}),
    "cpw4":       dict(feat="L4cpw", opt={}),
    "comb":       dict(feat="L1comb", opt={}),
    "rowz_comb":  dict(feat="L1comb", opt=dict(rowz=True)),
    "rowz_drop":  dict(feat="L1", opt=dict(rowz=True, dropout=0.3)),
    "rowz_aug":   dict(feat="L1", opt=dict(rowz=True, jitter_pts=5, gain=(0.8, 1.25))),
    "base_drop":  dict(feat="L1", opt=dict(dropout=0.3)),
    "rowz_drop5": dict(feat="L1", opt=dict(rowz=True, dropout=0.5)),
    "rowc_drop":  dict(feat="L1", opt=dict(rowcenter=True, dropout=0.3)),
    "rowz_b6_drop": dict(feat="L1b6", opt=dict(rowz=True, dropout=0.3)),
    "b6":         dict(feat="L1b6", opt={}),
    "jit2":       dict(feat="L1", opt=dict(jitter_pts=21)),
    "jit05":      dict(feat="L1", opt=dict(jitter_pts=5)),
    "aug":        dict(feat="L1", opt=dict(jitter_pts=5, gain=(0.8, 1.25))),
    "gain":       dict(feat="L1", opt=dict(gain=(0.8, 1.25))),
    "rowz":       dict(feat="L1", opt=dict(rowz=True)),
    "rowc_aug":   dict(feat="L1", opt=dict(rowcenter=True, jitter_pts=5, gain=(0.8, 1.25))),
    "rowc_b6":    dict(feat="L1b6", opt=dict(rowcenter=True)),
    "rowc_ML14":  dict(feat="L1+L4", opt=dict(rowcenter=True)),
    "rowc_L4":    dict(feat="L4", opt=dict(rowcenter=True)),
    "rowc":       dict(feat="L1", opt=dict(rowcenter=True)),
    "tstats":     dict(feat="L1", opt=dict(extra="tstats")),
    "balanced":   dict(feat="L1", opt=dict(pos_weight=True)),
}


def load_feat(spec):
    parts = []
    for nm in spec.split("+"):
        if nm == "L1":
            parts.append(P.load_t_mad())
        else:
            parts.append(np.load(CACHE / f"feat_{nm}.npy"))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=2)


def load_extra(kind, meta):
    if kind == "tstats":
        z = np.load(P.C.CACHE, allow_pickle=False)
        b3 = z["b3"][meta["row"]][:, :5].astype(np.float64)       # rms, pk-pk, kurtosis, skewness, crest
        b3[:, 0] = np.log(b3[:, 0]); b3[:, 1] = np.log(b3[:, 1])  # log scale for the two amplitude statistics
        return b3.astype(np.float32)
    raise ValueError(kind)
