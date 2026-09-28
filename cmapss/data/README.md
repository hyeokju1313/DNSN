# C-MAPSS 데이터

원본은 레포에 올리지 않는다(`.gitignore`). 아래에서 받아 압축을 풀고 `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`를 이 폴더에 바로 둔다.

- 출처: NASA PCoE Turbofan Engine Degradation Simulation Data Set
- 원본 주소: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
- 파일: train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt(x = 1~4), readme.txt, Damage Propagation Modeling.pdf(Saxena 등 2008, 센서 표와 NASA 점수 공식)
- 각 행은 공백으로 구분된 26개 열이다: 엔진 번호, 사이클, 운전 조건 3개, 센서 21개
- 주의: FD004는 readme에 학습 248대, 시험 249대로 적혀 있지만 실제 파일은 학습 249대, 시험 248대다.

## derived/ (자동 생성, 올리지 않음)

- `inputs/win_<설정>.npz`: 입력 채널 → 감각 뉴런 가중치. `reservoir_states.py`가 만든다.
