# 항공엔진 잔여수명 예측 (cmapss)

수컷 초파리 전체 CNS 커넥톰을 저장소(reservoir)로 써서 NASA C-MAPSS FD001 엔진의 잔여수명을 예측한다.

- 기획서: [`proposal.md`](proposal.md)
- 결과 보고서: [`REPORT.md`](REPORT.md)

## 시연

```
.venv/bin/python -u cmapss/simulate.py --engines 34 49 93
```

- 시험 엔진의 센서 기록을 사이클마다 뉴런 165,122개에 한 스텝씩 넣고, 하행·운동 뉴런 2,129개 상태로 잔여수명을 읽는다.
- 커넥톰, 무작위 망, 배선 섞은 망 세 가지를 돌린다. 엔진 한 대에 망 하나당 3~5초 걸린다.
- 새로 계산한 뉴런 상태가 실험 때 저장한 상태(`results/states/`)와 같은지 함께 확인한다.
- 엔진 번호는 1~100이다. 고장에 가까운 엔진은 34, 68, 31, 81번이다.
- 결과는 `results/simulate/`에, 그래프는 `results/figures/simulate_engines_*.png`에 저장한다.
- 필요한 것: 공통 커넥톰 데이터, C-MAPSS 원본, `results/states/`의 저장소 상태. 저장소 상태는 레포에 없으므로 아래 Phase A를 먼저 돌려야 한다.

## 실행 순서

레포 루트에서 실행한다. 공통 커넥톰 데이터([`../data/README.md`](../data/README.md))와 C-MAPSS 원본([`data/README.md`](data/README.md))이 필요하다.

```
.venv/bin/python -u cmapss/baselines.py      # 선형 2종, LSTM, GRU, 1D-CNN
./cmapss/run_phaseA.sh                       # 망 4종 × spectral radius 2 × 누설률 2
./cmapss/run_phaseB.sh                       # 부호, 입력 대응, 절제, 부분 커넥톰, 무작위 망 시드
./cmapss/run_m2.sh                           # M2 처음부터 학습(Apple GPU) + 챗봇 Stage 2
.venv/bin/python cmapss/report.py            # 요약 JSON과 그래프
```

- 실행 스크립트는 레포 루트의 `.venv/bin/python`을 쓴다. 다른 파이썬을 쓰려면 `PY=python3 ./cmapss/run_phaseA.sh`처럼 준다.
- M2를 M1 창 버전에서 출발시키는 옵션은 `--warm --lr 1e-5 --lr_dyn 1e-6`이다(`run_phaseB.sh` 맨 앞에 들어 있음).
- M2를 M1 창 버전에서 출발시킬 때 학습률을 크게 주면 발산한다. 뇌 상태의 변동 폭이 0.001 수준이라서다(`results/logs/phaseB_attempt1_warm_diverged.log`).

## 파일

| 경로 | 내용 |
|---|---|
| `dataset.py` | C-MAPSS 불러오기, 정규화, 평가 지표 |
| `reservoir_states.py` | 엔진 시계열을 저장소에 넣어 readout 뉴런 상태 저장 |
| `readout.py` | 저장소 상태 → ridge 출력층, 데이터 양 실험 |
| `m2_train.py` | 세포 유형별 값 역전파 학습(M2) |
| `simulate.py` | 시연 |
| `results/states/` | 저장소 상태(설정별 npz, 약 270MB씩). 올리지 않음. 출력층을 다시 맞출 때 재사용 |
| `results/readout/`, `baselines/`, `m2/` | 저장소 설정별 지표, 기준 모델 지표, M2 지표와 가중치 |
| `results/summary.json`, `figures/`, `logs/` | 요약, 그래프, 실행 로그(중단한 시도 포함) |
