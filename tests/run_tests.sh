#!/bin/bash
# Runs the smoke test and the two existing tests.  Run from the repository root.  The existing bootstrap test writes
# results/bootstrap_tests.json relative to its working directory, so it is run from a temporary directory here.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
$PY -I "$ROOT/tests/test_smoke.py"
(cd "$ROOT/scripts/lfpncn" && $PY -I test_lfpncn.py)
TMP="$(mktemp -d)"; mkdir -p "$TMP/results"
(cd "$TMP" && $PY "$ROOT/scripts/evaltools/test_bootstrap.py" --fast)
rm -rf "$TMP"
