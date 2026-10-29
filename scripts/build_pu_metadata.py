"""Check B: read every PU .mat with our safe loader and write a metadata table (no pickle)."""
import glob, os, sys, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pu_loader import load_vibration

root = os.path.join(os.path.dirname(__file__), "..", "data", "pu")
files = sorted(glob.glob(os.path.join(root, "*", "*.mat")))
print("files:", len(files), flush=True)
out = os.path.join(os.path.dirname(__file__), "..", "data", "pu_metadata.csv")
bad = []
t0 = time.time()
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bearing_id", "condition", "rep", "n_samples", "seconds", "rpm_measured", "rms", "finite"])
    for i, f in enumerate(files):
        try:
            meta, x = load_vibration(f)
            w.writerow([meta["bearing_id"], meta["condition"], meta["rep"], len(x), round(len(x) / 64000, 3),
                        round(meta["rpm_measured"], 2), round(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 5), bool(np.isfinite(x).all())])
        except Exception as e:
            bad.append((os.path.basename(f), repr(e)[:80]))
        if i % 400 == 0:
            print(i, round(time.time() - t0), "s", flush=True)
print("done; unreadable:", len(bad), bad[:5])
