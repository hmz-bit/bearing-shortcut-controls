#!/usr/bin/env python3
"""Write MANIFEST.sha256 (sha256sum format, paths relative to the repository root, sorted, MANIFEST itself excluded).
Check with:  sha256sum -c MANIFEST.sha256   (from the repository root)."""
import hashlib
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "__pycache__")]
    for f in fn:
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        if rel == "MANIFEST.sha256":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        out.append((rel, h.hexdigest()))
out.sort()
import sys
if "--check" in sys.argv[1:]:
    # verify only; do not rewrite the manifest
    bad = 0
    with open(os.path.join(root, "MANIFEST.sha256")) as fh:
        shipped = {}
        for ln in fh:
            if ln.strip():
                digest, rel = ln.rstrip().split("  ", 1)
                shipped[rel] = digest
    now = dict(out)
    for rel in sorted(set(shipped) | set(now)):
        if shipped.get(rel) != now.get(rel):
            print("MISMATCH", rel)
            bad += 1
    print("manifest OK" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)
with open(os.path.join(root, "MANIFEST.sha256"), "w", newline="\n") as fh:
    for rel, h in out:
        fh.write(f"{h}  {rel}\n")
print(len(out), "files")
