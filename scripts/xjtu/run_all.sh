#!/bin/bash
# Runs every pre-declared XJTU-SY training job (results/xjtu/PREDECLARED_XJTU.txt): 5 parallel chains (4 CPU LF-PNCN chains, 1 GPU B1 chain).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
PY=${PY:-python}
L=results/xjtu
cpu() { P2_DEVICE=cpu P2_THREADS=2 $PY -I scripts/xjtu/run_lfpncn.py "$@"; }

( cpu base 5 54      > $L/log_base_5-54.txt 2>&1;      cpu base 5 24 perm  > $L/log_perm_base.txt 2>&1;      cpu base 5 54 sens5  > $L/log_sens5_base_5-54.txt 2>&1 ) &
( cpu base 55 104    > $L/log_base_55-104.txt 2>&1;                                                          cpu base 55 104 sens5 > $L/log_sens5_base_55-104.txt 2>&1 ) &
( cpu rowz_drop 5 54 > $L/log_rowz_drop_5-54.txt 2>&1; cpu rowz_drop 5 24 perm > $L/log_perm_rowz_drop.txt 2>&1; cpu rowz_drop 5 54 sens5 > $L/log_sens5_rowz_drop_5-54.txt 2>&1 ) &
( cpu rowz_drop 55 104 > $L/log_rowz_drop_55-104.txt 2>&1;                                                   cpu rowz_drop 55 104 sens5 > $L/log_sens5_rowz_drop_55-104.txt 2>&1 ) &
( $PY -I scripts/xjtu/run_b1.py 5 104 > $L/log_b1.txt 2>&1; $PY -I scripts/xjtu/run_b1.py 5 24 perm > $L/log_perm_b1.txt 2>&1; $PY -I scripts/xjtu/run_b1.py 5 104 sens5 > $L/log_sens5_b1.txt 2>&1 ) &
wait
echo ALL_DONE > $L/ALL_DONE.txt
