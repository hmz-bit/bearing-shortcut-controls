#!/bin/bash
# Clones the third-party benchmark repository (splits, features, saved logits) next to the code, at the commit used
# for all experiments.  The repository has no licence; nothing from it is redistributed here, and nothing from it is
# imported or executed by this project (its files are read as data or, for the B1 baseline, as text).
set -e
cd "$(dirname "$0")/.."
URL=https://github.com/gama-ufsc/bearing-data-leakage.git
COMMIT=4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d
if [ ! -d bearing-data-leakage/.git ]; then git clone "$URL" bearing-data-leakage; fi
git -C bearing-data-leakage checkout "$COMMIT"
git -C bearing-data-leakage log -1 --format='%H %cd'
mkdir -p data
