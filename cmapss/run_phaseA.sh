#!/bin/zsh
# Phase A: 망 종류 4개 × spectral radius {0.9, 1.2} × 누설률 {0.1, 0.5}. 각자 검증 RMSE로 설정을 고른다.
cd "$(dirname $0)/.."
PY=${PY:-.venv/bin/python}
for v in connectome random degree_shuffle weight_shuffle; do
  for rho in 0.9 1.2; do
    for leak in 0.1 0.5; do
      key="FD001_${v}_glu-1_rho${rho}_leak${leak}_mapsemantic_s0"
      echo "=== $(date '+%H:%M:%S') $key"
      $PY -u cmapss/reservoir_states.py --variant $v --rho $rho --leak $leak || echo "FAILED states $key"
      $PY -u cmapss/readout.py $key || echo "FAILED readout $key"
    done
  done
done
echo "=== PHASE A DONE $(date '+%H:%M:%S')"
