# 학사규정 데이터

## regulations/

- 출처: 한국공학대학교 규정관리시스템 (https://rule.tukorea.ac.kr/lmxsrv/main/main.do), 2026-09-27 수집
- 범위: 학사 관련 규정 64개(제2편 학칙, 제3편 제4장 학사행정, 제3편 제6장 학생행정). '지침' 메뉴는 로그인이 필요해 제외
- 수집 코드: `regqa/crawl.py`(요청 간격 0.5초), 변환 코드: `regqa/to_markdown.py`

| 경로 | 내용 |
|---|---|
| tree.json | 전체 규정 트리(353개, 폐지 포함) |
| manifest.json | 수집한 64개 규정의 번호, 이력 번호, 담당 부서, 원문 주소 |
| raw/ | 본문 HTML 원본(수정하지 않음) |
| md/ | 규정별 markdown. 장·절·조·항·호 구조, 표, 규정 간 링크 보존 |
| articles.jsonl | 조문 단위 2,032개(본문 1,244, 부칙 788). 참조 496개 |
| qa/eval_v0_claude.jsonl | 평가 질문 102개. Claude가 원문을 읽고 작성(사람 작성 시험셋 아님) |
| qa/dev_v0_claude.jsonl | 답변 불가 기준값을 정하는 개발 질문 20개(범위 안 10, 밖 10) |

별표·별지(HWP 첨부)는 변환하지 않았다. md 파일 끝에 이름만 적어 두었다.

## derived/ (올리지 않음)

임베딩(`emb_*.npy`), 재순위 특징(`feats_*.npz`, 하나에 약 120MB), 후보·학습 쌍(`pairs.json`), 답변 불가 기준값(`tau_dense.json`), 저장소 입력 가중치(`inputs/`)다. `regqa/README.md`의 실행 순서대로 돌리면 다시 만들어진다.
