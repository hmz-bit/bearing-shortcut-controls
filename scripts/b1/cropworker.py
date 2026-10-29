"""Light-weight crop/envelope worker (numpy + scipy only, no torch) for the training pipeline."""
import numpy as np
import envspec

CROP = 42000
_SIG = None
_META = None


def init_worker(path, offsets):
    global _SIG, _META
    _SIG = np.memmap(path, dtype=np.float32, mode="r")
    _META = offsets


def crop_env(job):
    fi, start = job
    o = int(_META[fi]) + start
    return envspec.envelope_spectrum(np.asarray(_SIG[o:o + CROP]))
