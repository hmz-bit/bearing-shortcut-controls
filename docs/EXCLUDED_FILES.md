FILES OF THE DEVELOPMENT TREE THAT ARE NOT IN THIS REPOSITORY

Paths are relative to the original results/ folder unless they start with scripts/.  Large caches that are rebuilt by the scripts are listed with their sha256 in results/LARGE_FILES.txt.
Not included: raw data and derived signal arrays of the data sets (docs/DATA.md), the third-party repository clone, the Python virtual environment, compiled caches (__pycache__), notebooks and logs of package installation,
and the earlier per-folder write-ups and dataset notes, whose facts are re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md and docs/DEVELOPMENT_LOG.md.

results/b1/shards/ : 288 files, 3578421 bytes, per-split shards (one parquet and one json per split) merged into results/b1/*_predictions_long.parquet and *_per_split.csv.
results/datasets/ : 3 earlier notes on data-set selection and provenance (the facts are in docs/DATA.md).

path | bytes | reason
audit/SUMMARY.md | 2807 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
audit2/SUMMARY.md | 6196 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
audit3/SUMMARY.md | 8275 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
b1/stdout_t1shuf.txt | 981 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t1val.txt | 1877 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t1val_seed1.txt | 1887 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t1val_seed2.txt | 1887 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t2.txt | 4578 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t3.txt | 14983 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t3_mult4.txt | 14368 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
b1/stdout_t3_odd.txt | 14978 | duplicate console capture of b1/log_*.txt (one copy contains an aborted multiprocessing attempt with machine paths)
cwru/SUMMARY.md | 4101 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
kaist/SUMMARY.md | 1800 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
phase2/SUMMARY.md | 3671 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
rf/SUMMARY.md | 4041 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
rtf/SUMMARY.md | 2961 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
tier1/SUMMARY.md | 3388 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
tier2_3/SUMMARY.md | 3654 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
uored/DATASET_PROVENANCE.md | 2349 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
uored/pred_b1_partial.parquet | 1078040 | byte-identical duplicate of uored/pred_b1.parquet
uored/SUMMARY.md | 4540 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
xjtu/SUMMARY.md | 3206 | earlier write-up of the same results, superseded by docs/ and the result files (facts re-stated in docs/DATA.md, docs/METHOD_AND_PROTOCOL.md, docs/DEVELOPMENT_LOG.md)
scripts/evaltools/results/b1_predictions_long.parquet | 2844838 | window-level scores derived from the third-party saved logits (no licence); regenerate with pu_data.logits_to_long after cloning the third-party repository

Not copied because they are not under results/ or scripts/ : data/ (all data sets and caches), bearing-data-leakage/ (third-party clone, no licence), .venv/, the earlier notebook and configuration folders.
data/pu_metadata.csv (file-level metadata of the PU archives, rebuilt by scripts/build_pu_metadata.py) is not copied.
