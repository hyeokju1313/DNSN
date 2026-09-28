#!/bin/zsh
# 순서: Phase A와 챗봇 _ix 대조군이 끝나면 -> M2(커넥톰, 무작위 망)를 혼자 GPU에서 -> Phase B
cd "$(dirname $0)/.."
until grep -q "PHASE A DONE" cmapss/results/logs/phaseA.log && ! pgrep -f "regqa/rerank.py" >/dev/null; do sleep 30; done
./cmapss/run_m2.sh > cmapss/results/logs/m2.log 2>&1
./cmapss/run_phaseB.sh > cmapss/results/logs/phaseB.log 2>&1
echo "=== ALL DONE $(date '+%H:%M:%S')"
