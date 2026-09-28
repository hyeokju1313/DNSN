#!/bin/zsh
# Phase B: 커넥톰 최적 설정에서 부호 변경, 입력 대응, 부위 절제, 부분 커넥톰, 무작위 망 추가 시드.
# Phase A가 끝날 때까지 기다린 뒤, 각 망의 최적 설정(교차검증 RMSE 최소)을 readout JSON에서 읽는다.
cd "$(dirname $0)/.."
PY=${PY:-.venv/bin/python}
L=cmapss/results/logs/phaseA.log
until grep -q "PHASE A DONE" $L; do sleep 30; done
# M2 warm-start(M1 창 버전에서 출발)를 CPU 작업보다 먼저 GPU에서 돌린다
for v in connectome random; do
  echo "=== $(date '+%H:%M:%S') M2 warm $v"
  $PY -u cmapss/m2_train.py --variant $v --stride 10 --epochs 8 --patience 3 --lr 1e-5 --lr_dyn 1e-6 --warm 2>&1 \
    | grep --line-buffered -vE "Warning|sparse_coo_tensor|detach|gain.:"
done
best() {  # $1 = 망 종류 -> "rho leak"
  $PY - "$1" <<'PYEOF'
import json, glob, sys
v = sys.argv[1]
rows = []
for f in glob.glob(f"cmapss/results/readout/FD001_{v}_glu-1_rho*_leak*_mapsemantic_s0.json"):
    r = json.load(open(f)); k = r["key"]
    rho = k.split("_rho")[1].split("_")[0]; leak = k.split("_leak")[1].split("_")[0]
    rows.append((r["cv_rmse"], rho, leak))
print(*min(rows)[1:])
PYEOF
}
read CRHO CLEAK <<< "$(best connectome)"
read RRHO RLEAK <<< "$(best random)"
read DRHO DLEAK <<< "$(best degree_shuffle)"
read WRHO WLEAK <<< "$(best weight_shuffle)"
echo "=== best connectome rho=$CRHO leak=$CLEAK / random $RRHO $RLEAK / degree $DRHO $DLEAK / weight $WRHO $WLEAK"
run() {  # 인자 = reservoir_states 인자, 마지막에 key
  local key=$1; shift
  echo "=== $(date '+%H:%M:%S') $key"
  $PY -u cmapss/reservoir_states.py "$@" || echo "FAILED states $key"
  $PY -u cmapss/readout.py $key || echo "FAILED readout $key"
}
C="--variant connectome --rho $CRHO --leak $CLEAK"
# 시드 반복: 무작위·섞은 망은 망 자체와 입력 가중치가, 커넥톰은 입력 가중치만 바뀐다
for s in 1 2; do
  run FD001_connectome_glu-1_rho${CRHO}_leak${CLEAK}_mapsemantic_s${s} $=C --seed $s
  run FD001_random_glu-1_rho${RRHO}_leak${RLEAK}_mapsemantic_s${s} --variant random --rho $RRHO --leak $RLEAK --seed $s
  run FD001_degree_shuffle_glu-1_rho${DRHO}_leak${DLEAK}_mapsemantic_s${s} --variant degree_shuffle --rho $DRHO --leak $DLEAK --seed $s
  run FD001_weight_shuffle_glu-1_rho${WRHO}_leak${WLEAK}_mapsemantic_s${s} --variant weight_shuffle --rho $WRHO --leak $WLEAK --seed $s
done
run FD001_connectome_glu+1_rho${CRHO}_leak${CLEAK}_mapsemantic_s0 $=C --glu 1
run FD001_connectome_glu-1_rho${CRHO}_leak${CLEAK}_maprandom_s0 $=C --mapping random
for ab in optic_lobe mushroom_body central_complex vnc neck; do
  run FD001_connectome_glu-1_rho${CRHO}_leak${CLEAK}_mapsemantic_s0_abl${ab} $=C --ablate $ab
done
for top in 50 100 700 1500; do
  run FD001_connectome_glu-1_rho${CRHO}_leak${CLEAK}_mapsemantic_s0_top${top} $=C --top $top
done
echo "=== PHASE B DONE $(date '+%H:%M:%S')"
