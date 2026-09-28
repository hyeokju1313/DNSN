# DNSN

Drosophila Central Nervous System Network. 수컷 초파리 중추신경계 전체 커넥톰(Janelia male CNS v1.0)을 신경망 구조로 그대로 쓰는 AI+X(2026년 2학기) 수업 프로젝트 모음이다.

## 프로젝트

| 폴더 | 과제 | 상태 | 요약 |
|---|---|---|---|
| [`regqa/`](regqa/) | 한국공학대학교 학사규정 질의응답 챗봇 | v0 | 한국어 검색 인코더(KURE-v1) dense 검색이 R@1 0.85. 커넥톰 재순위 모델은 무작위 망보다 낫지 않았다 |
| [`cmapss/`](cmapss/) | 항공엔진 잔여수명 예측(NASA C-MAPSS FD001) | v0 | 커넥톰 저장소 시험 RMSE 17.07로 무작위 망과 같았고, 배선 섞은 망(13.36)이 더 좋았다 |

프로젝트 폴더마다 기획서(`proposal.md`), 결과 보고서(`REPORT.md`), 실행 방법(`README.md`)이 있다.

## 폴더 구조

```
DNSN/
├── flycns/           공통 라이브러리
│   ├── connectome.py   커넥톰 불러오기, 신경전달물질 부호, 입력 비율 정규화, 대조군(무작위·배선 섞기·가중치 섞기), 부위 절제, 부분 커넥톰
│   ├── reservoir.py    저장소 동역학을 여러 프로세스로 나눠 돌리기
│   └── plotstyle.py    그래프 색
├── data/             공통 커넥톰 데이터(올리지 않음, data/README.md)
├── regqa/            프로젝트 1: 학사규정 챗봇
│   ├── *.py
│   ├── data/           규정 수집본, 평가 질문(derived/는 캐시라 올리지 않음)
│   └── results/        지표, 그래프, 로그, 학습 가중치
└── cmapss/           프로젝트 2: 항공엔진 잔여수명
    ├── *.py, run_*.sh
    ├── data/           C-MAPSS 원본(올리지 않음, cmapss/data/README.md)
    └── results/        지표, 그래프, 로그, 학습 가중치(states/*.npz는 올리지 않음)
```

## 환경

```
git clone https://github.com/hyeokju1313/DNSN.git
cd DNSN
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- 모든 명령은 레포 루트에서 실행한다. 스크립트가 스스로 레포 루트를 import 경로에 넣는다.
- 개발 기기: Apple M5 Pro, 메모리 24GB, Python 3.14.3.
  - 이 기기에서는 시스템 파이썬 패키지를 공유(`python3 -m venv --system-site-packages .venv`)하고, kiwipiepy와 matplotlib만 venv에 설치했다.
  - 저장소 계산은 CPU 12개 프로세스로, 역전파 학습은 Apple GPU(MPS)로 돌렸다.
- 데이터 준비:
  - 커넥톰: [`data/README.md`](data/README.md)
  - C-MAPSS: [`cmapss/data/README.md`](cmapss/data/README.md)
  - 학사규정 수집본은 레포에 들어 있다.

## 새 프로젝트를 추가할 때

- 레포 최상위에 프로젝트 폴더를 만들고, 그 안에 코드, `data/`, `results/`, `proposal.md`, `REPORT.md`, `README.md`를 둔다.
- 스크립트 맨 위에 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`를 넣으면 `from flycns.connectome import build_matrix`로 커넥톰을 쓸 수 있다.
- 경로는 프로젝트 폴더 기준으로 잡는다(`Path(__file__).resolve().parent / "data"`).
- 프로젝트 폴더 안에 `data.py`를 만들지 않는다. `data/` 폴더와 이름이 겹친다(엔진 프로젝트는 `dataset.py`로 이름을 바꿨다).
- 큰 파일(수십 MB 이상)은 `.gitignore`에 넣고, 받는 방법이나 다시 만드는 방법을 README에 적는다.

## 주의

- 로그를 grep으로 거를 때는 `grep --line-buffered`를 쓴다. 안 그러면 진행 줄이 끝날 때까지 로그에 안 보인다.
- 백그라운드 실험을 강제로 멈추면 저장소 작업자 프로세스가 부모 없이 남을 수 있다. `ps -o pid,ppid,command -ax | awk '$2==1 && /multiprocessing-fork/'`로 확인해 정리한다.
