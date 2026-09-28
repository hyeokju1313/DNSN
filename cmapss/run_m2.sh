#!/bin/zsh
# GPU 단계: 엔진 수명 M2(커넥톰, 무작위 망) -> 챗봇 Stage 2(커넥톰, 무작위 망)
cd "$(dirname $0)/.."
PY=${PY:-.venv/bin/python}
for v in connectome random; do
  echo "=== $(date '+%H:%M:%S') M2 $v"
  $PY -u cmapss/m2_train.py --variant $v --stride 10 --epochs 8 --patience 3 2>&1 | grep --line-buffered -vE "Warning|sparse_coo_tensor|detach|gain.:"
done
for v in connectome random; do
  echo "=== $(date '+%H:%M:%S') regqa stage2 $v"
  $PY -u regqa/stage2_train.py --variant $v --epochs 6 2>&1 | grep --line-buffered -vE "Warning|sparse_coo_tensor|detach"
done
echo "=== M2 DONE $(date '+%H:%M:%S')"
