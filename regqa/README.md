# 학사규정 질의응답 챗봇 (regqa)

수컷 초파리 전체 CNS 커넥톰을 재순위 모델로 쓰는 한국공학대학교 학사규정 질의응답 챗봇이다.

- 기획서: [`proposal.md`](proposal.md)
- 결과 보고서: [`REPORT.md`](REPORT.md)

## 시연

```
.venv/bin/python regqa/ask.py "25학번 학사경고자는 몇 학점까지 들을 수 있어?"   # 질문 하나
.venv/bin/python regqa/ask.py                                                  # 질문을 계속 입력, q로 종료
```

- 답을 생성하지 않고 조문에서 뽑아 보여준다. 근거 조문, 핵심 문장, 학번별 표, 함께 봐야 할 조문, 다른 후보 조문을 함께 보여준다.
- 확신도는 질문과 1위 조문의 코사인 유사도다. 이 값이 기준값(개발 질문 20개로 정한 0.592)보다 낮으면 "학사규정에서 확인할 수 없음"으로 답한다.
- 시연은 dense 검색만 쓴다. 커넥톰 재순위를 합쳐도 dense 단독(R@1 0.85)을 넘지 못했다(REPORT.md 3.2, 3.4절).

## 실행 순서

레포 루트에서 실행한다. 재순위 실험(`rerank.py`, `stage2_train.py`)에는 공통 커넥톰 데이터가 필요하다([`../data/README.md`](../data/README.md)).

```
.venv/bin/python -u regqa/crawl.py          # 규정 64개 원문 HTML 수집(요청 간격 0.5초). 수집본이 레포에 있어 다시 할 필요는 없다
.venv/bin/python -u regqa/to_markdown.py    # markdown + 조문 JSONL
.venv/bin/python regqa/make_eval.py         # 평가 질문 102개(Claude 작성)
.venv/bin/python -u regqa/embed_all.py      # KURE-v1 임베딩
.venv/bin/python -u regqa/pairs.py          # 후보·학습 쌍, 개발 질문 20개
.venv/bin/python regqa/eval_retrieval.py    # BM25, dense, hybrid
.venv/bin/python -u regqa/rerank.py --feats lr_pair randfeat res_connectome_ix res_random_ix res_degree_shuffle_ix
.venv/bin/python -u regqa/stage2_train.py --variant connectome   # Stage 2(역전파, Apple GPU). random도 같은 방식
.venv/bin/python regqa/compare.py           # 재순위 모델 쌍체 부트스트랩
.venv/bin/python regqa/compare_stage2.py    # Stage 2 커넥톰 vs 무작위 망
.venv/bin/python regqa/report.py            # 요약 JSON과 그래프
```

- 한국어 인코더 `nlpai-lab/KURE-v1`(MIT)은 `~/.cache/aix_models/KURE-v1/`에 있으면 그것을 쓰고, 없으면 모델 ID로 받는다. huggingface_hub 기본 다운로더가 멈춰서 개발 기기에서는 파일을 curl로 받았다(`corpus.py`).
- Stage 2는 엔진 프로젝트의 `cmapss/run_m2.sh`에서도 함께 돌린다.

## 파일

| 경로 | 내용 |
|---|---|
| `corpus.py` | 검색 대상 조문(학부 현행 본문 698개), 임베딩, BM25 |
| `rerank.py` | 재순위 특징(로지스틱 회귀, 커넥톰·무작위 망 저장소)과 평가 |
| `stage2_train.py` | 세포 유형별 값 역전파 학습 |
| `data/regulations/` | 규정 수집본과 평가 질문([`data/README.md`](data/README.md)) |
| `data/derived/` | 임베딩·특징 캐시(올리지 않음, 위 순서대로 실행하면 만들어진다) |
| `results/` | 지표 JSON, 그래프, 로그, Stage 2 가중치 |
