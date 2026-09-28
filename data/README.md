# 공통 커넥톰 데이터

이 폴더의 파일은 용량이 커서 레포에 올리지 않는다(`.gitignore`). 아래 원본을 `male_cns_v1.0/`에 받으면 `derived/`는 코드가 처음 실행될 때 만든다.

## male_cns_v1.0/

- 출처: Janelia FlyEM Male CNS Connectome v1.0 (https://male-cns.janelia.org/download/)
- 원본 주소: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<파일 이름>` (로그인 불필요)
- 라이선스: CC-BY 4.0
- 논문: Sexual dimorphism in the complete Drosophila male central nervous system connectome, Cell, 2026-09-03
- 2026-09-27에 받았고 내용은 수정하지 않았다.

| 파일 | 크기 | 내용 |
|---|---|---|
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1,051,241,946 바이트 | 열: body_pre, body_post, weight(시냅스 수). 151,856,684행 |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14,483,314 바이트 | 뉴런별 유형, 부위(superclass, class), 추적 상태(status). 211,577행 |
| body-neurotransmitters-male-cns-v1.0.feather | 43,282,834 바이트 | 뉴런별 신경전달물질 예측(consensus_nt 등) |

```
mkdir -p data/male_cns_v1.0
cd data/male_cns_v1.0
for f in connectome-weights-male-cns-v1.0-minconf-0.5.feather \
         body-annotations-male-cns-v1.0-minconf-0.5.feather \
         body-neurotransmitters-male-cns-v1.0.feather; do
  curl -O "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/$f"
done
```

추적이 끝난 뉴런은 `status == "Traced"`인 165,122개다. 기획서의 수치는 이 뉴런끼리의 연결만 센 값이다.

## derived/ (자동 생성)

| 경로 | 만드는 함수 | 내용 |
|---|---|---|
| neurons.parquet | `flycns.connectome.load_neurons` | 추적 뉴런 표. 행 순서가 행렬 인덱스 |
| edges_traced.npz | `flycns.connectome.load_edges` | 추적 뉴런끼리의 연결(pre, post, 시냅스 수) |
| matrices/ | `flycns.connectome.build_matrix` | 망 종류·spectral radius·절제별 가중치 행렬(CSR). 전체 커넥톰 행렬 하나에 약 100MB |
