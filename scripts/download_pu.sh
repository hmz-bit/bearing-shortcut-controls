#!/bin/bash
# Downloads and extracts the Paderborn bearing archives (CC BY-NC 4.0; cite Lessmeier et al. 2016).
cd "$(dirname "$0")/../data" || exit 1
mkdir -p pu_raw pu
IDS="K001 K002 K003 K004 K005 K006 KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21"
for id in $IDS; do
  if [ -d "pu/$id" ] && [ "$(ls pu/$id | wc -l)" -ge 80 ]; then echo "skip $id"; continue; fi
  [ -s "pu_raw/$id.rar" ] || curl -sS -m 900 --retry 3 -o "pu_raw/$id.rar" "https://groups.uni-paderborn.de/kat/BearingDataCenter/$id.rar" || { echo "DOWNLOAD FAIL $id"; continue; }
  ${TAR:-tar} -xf "pu_raw/$id.rar" -C pu && echo "ok $id $(ls pu/$id | wc -l) files" || echo "EXTRACT FAIL $id"
done
echo DONE
