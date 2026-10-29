# bearing-order-spectra

Code, protocols and result tables for a study of bearing fault diagnosis with order-resampled envelope spectra
(three frequency bands, a three-branch one-dimensional CNN) evaluated with strictly bearing-wise splits and a
bearing-level cluster bootstrap.  Paper: Shortcut controls for bearing-wise fault-diagnosis benchmarks: a five-dataset case study with an order-resampled envelope-spectrum network (see CITATION.cff).

Author, affiliation and copyright holder: see LICENSE and CITATION.cff.
Parts of the code were written with the assistance of a generative AI tool under the author's direction; see the statement in the paper.
Licence of the code and of the result tables produced by it: MIT (LICENSE).  The data sets it uses have their own licences (docs/DATA.md).

## What is here

    scripts/      all code, one folder per component or data set (see docs/CODE_MAP.md)
    protocols/    the pre-declaration files written before each frozen evaluation, EDITS.txt, TIMESTAMPS.csv
    results/      predictions (parquet), per-split and per-bearing tables, bootstrap outputs, logs, TABLE_SOURCES.csv
    tests/        smoke test on synthetic data, runner for the existing tests
    tools/        identity-level split check, manifest and table-source builders, third-party fetch script
    docs/         METHOD_AND_PROTOCOL.md, DEVELOPMENT_LOG.md, DATA.md, REPRODUCE.md, CODE_MAP.md, EXCLUDED_FILES.md
    data/         empty placeholder for locally fetched data (not versioned)

The folder layout is part of the code: scripts find the repository root from their own file location and read data/ and bearing-data-leakage/ relative to it.

## Not included

Raw data, derived signal arrays, and anything from the third-party benchmark repository (no licence): splits, features, saved logits and paper PDFs.
docs/DATA.md lists every data set with licence, URL and version, and says how to obtain it.  tools/fetch_third_party.sh clones the benchmark repository at
commit 4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d.  Large caches are listed with sha256 in results/LARGE_FILES.txt; files left out for other reasons in docs/EXCLUDED_FILES.md.

## Quick start

    pip install -r requirements.txt            # Python 3.11
    bash tests/run_tests.sh                    # smoke test (seconds) and the two existing tests
    python tools/leakage_guard.py              # identity-level checks of all split definitions (needs the third-party clone and the shipped split files)
    python tools/make_table_sources.py         # rebuild results/TABLE_SOURCES.csv from the shipped result files

Full command order per table with expected run times: docs/REPRODUCE.md.

## Main results (Macro-AUROC means; bootstrap over physical bearings)

results/CROSS_DATASET_SUMMARY.md summarises five data sets: Paderborn PU (development set), CWRU, UORED-VAFCLS, XJTU-SY and the Paderborn run-to-failure
data.  results/TABLE_SOURCES.csv maps every number of that summary to its source file and key.  The two run-to-failure data sets define labels by early versus late life; the summary and
docs/METHOD_AND_PROTOCOL.md state what that limits.

results/rf/ holds a random-forest baseline on the four external data sets, results/audit2/, results/audit3/ and results/audit4/ hold evaluation-only controls (healthy-only negatives, unit-level metric, metadata-only forest,
test-invariance check, forest without the DC level, UORED per label, CWRU channel matching, leave-one-condition-out, stage-free fault-type evaluation).  Their protocols are in protocols/PREDECLARED_RF*.txt and protocols/PREDECLARED_AUDIT2.txt, PREDECLARED_AUDIT3.txt, PREDECLARED_AUDIT4.txt.  results/ablation/ holds the component ablation (PREDECLARED_ABLATION.txt) and results/kaist/ a supplementary evaluation on the KAIST data (PREDECLARED_KAIST.txt).  results/PAPER_TABLE_SOURCES.csv maps the numbers of the paper to their source files.

## Reproducibility notes

 - Pre-declaration files (protocols/PREDECLARED*.txt) are byte-identical copies of the originals except for the few wording changes recorded with hashes in protocols/EDITS.txt.
   Identical copies sit in results/<folder>/ because the run scripts check for them there.  protocols/TIMESTAMPS.csv gives file modification times of each pre-declaration and of the first outputs of the run it governs.
 - Seeds, split lists and hyper-parameters are in the code and in the pre-declarations; GPU runs use deterministic cuDNN flags but results can still differ in the last digits across hardware and library versions.
