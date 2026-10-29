"""Safe loader for the Paderborn (KAt) bearing .mat files (no pickle; scipy.io.loadmat only).

File name pattern: N{rpm}_M{torque}_F{force}_{bearing}_{repetition}.mat
Operating conditions: N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm),
N15_M01_F10 (0.1 Nm), N15_M07_F04 (400 N).
"""
import os, re
import numpy as np
from scipy.io import loadmat

NAME = re.compile(r"^(N\d\d)_(M\d\d)_(F\d\d)_(K[A-Z]?\d{2,3})_(\d+)$")
FS_RAW = 64000


def parse_name(stem):
    m = NAME.match(stem)
    if not m:
        raise ValueError(f"unexpected PU file name: {stem}")
    n, mo, f, bearing, rep = m.groups()
    return {"speed": n, "torque": mo, "force": f, "bearing_id": bearing, "rep": int(rep), "condition": f"{n}_{mo}_{f}"}


def load_vibration(path):
    """Return (meta, vibration_1 as float32 array at 64 kHz)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    rec = mat[stem]
    chans = {str(c.Name): np.asarray(c.Data, dtype=np.float32) for c in rec.Y}
    meta = parse_name(stem)
    meta["fs"] = FS_RAW
    meta["rpm_measured"] = float(np.mean(np.abs(chans["speed"])))
    return meta, chans["vibration_1"]
