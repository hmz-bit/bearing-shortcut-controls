"""Check A: recompute Macro-AUROC of the saved PU WDCNN-envelope logits (Vieira et al. 2026, Table 9)."""
import glob, os, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "data/saved_logits/pu/paderborn_eval_envelope_*/test/epoch_30_run_0.pt")))
print("runs found:", len(files))
rows = []
for f in files:
    t = torch.load(f, map_location="cpu", weights_only=True).numpy()
    s1, s2, y1, y2 = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
    rows.append([roc_auc_score(y1, s1), roc_auc_score(y2, s2), roc_auc_score(y1, s2), roc_auc_score(y2, s1)])
a = np.array(rows)
print("shape", t.shape, "label sums (last run):", y1.sum(), y2.sum())
for name, cols in (("pairing col1-y1, col2-y2", (0, 1)), ("pairing col1-y2, col2-y1", (3, 2))):
    m = a[:, list(cols)].mean(axis=1)
    print(f"{name}: per-label means {a[:, cols[0]].mean():.4f} / {a[:, cols[1]].mean():.4f} | Macro-AUROC {100*m.mean():.2f} +- {100*m.std():.2f}")
print("Vieira Table 9 (WDCNN envelope): 79.56 +- 13.07 | Fig. 11: inner 0.898, outer 0.693")
