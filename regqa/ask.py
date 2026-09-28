"""학사규정 질의응답 명령줄 시연. 문장을 생성하지 않고 조문에서 뽑아 근거와 함께 보여준다.

1. KURE-v1 dense 검색으로 조문 후보를 찾는다
2. 1위 코사인 점수가 기준값 τ보다 낮으면 '학사규정에서 확인할 수 없음'으로 답한다(τ는 개발 질문 20개로 정함)
3. 1위 조문에서 질문과 가장 가까운 줄과 표를 보여준다
4. 1위 조문이 가리키는 조문, 1위 조문을 가리키는 조문을 함께 보여준다(여러 규정을 이어야 하는 질문 대응)
사용 예:
  python regqa/ask.py "25학번인데 학사경고 받고 지원 프로그램 안 들으면 몇 학점까지 들을 수 있어?"
  python regqa/ask.py        # 질문을 계속 입력하는 대화형 모드
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.corpus import DER, REG, doc_text, embed, load_corpus  # noqa: E402

NOTICE = "이 답변은 규정 원문에서 찾은 내용입니다. 최종 확인은 학사운영팀에 문의하세요."


def article_label(a):
    return f"{a['law']} {a['article']}" + (f"({a['title']})" if a["title"] else "")


def lines_of(a):
    return [ln for ln in a["text_clean"].split("\n") if ln.strip()]


def tau_from_dev(C, E):
    p = DER / "tau_dense.json"
    if p.exists():
        return json.loads(p.read_text())["tau"]
    dev = [json.loads(l) for l in open(REG / "qa" / "dev_v0_claude.jsonl", encoding="utf-8")]
    DE = embed([d["question"] for d in dev], "dev_v0")
    conf = (DE @ E.T).max(1)
    ins = np.array([d["in_scope"] for d in dev])
    tau = float(max(sorted(set(conf)), key=lambda t: (conf[ins] >= t).mean() + (conf[~ins] < t).mean()))
    p.write_text(json.dumps({"tau": tau, "note": "개발 질문 20개(범위 안 10, 밖 10)에서 균형 정확도 최대"}))
    return tau


def main(question: str, k: int = 3):
    C = load_corpus()
    E = embed([doc_text(a) for a in C], "corpus")
    tau = tau_from_dev(C, E)
    by_id = {a["id"]: a for a in C}
    q = embed([question])[0]
    s = E @ q
    order = np.argsort(-s)[:k]
    print(f"\n질문: {question}")
    if s[order[0]] < tau:
        print(f"\n답변: 학사규정에서 확인할 수 없는 내용입니다. (확신도 {s[order[0]]:.3f} < 기준 {tau:.3f})")
        print("학사일정, 시설, 생활 정보는 학교 홈페이지나 해당 부서에 문의하세요.")
        return
    top = C[order[0]]
    lines = lines_of(top)
    text_lines = [ln for ln in lines if not ln.startswith("|")]
    table = [ln for ln in lines if ln.startswith("|")]
    print(f"\n근거 조문: {article_label(top)}  (확신도 {s[order[0]]:.3f})")
    if text_lines:
        LE = embed(text_lines)
        best = np.argsort(-(LE @ q))[:2]
        print("핵심 문장:")
        for i in sorted(best):
            print(f"  - {text_lines[i].strip()}")
    if table:
        print("관련 표:")
        for ln in table:
            print("  " + ln)
    # 참조 관계(1위 조문 -> 다른 조문, 다른 조문 -> 1위 조문)
    out_refs = [f"{r['seq']}:{r['article']}" for r in top["refs"] if r.get("article")]
    in_refs = [a["id"] for a in C if any(f"{r['seq']}:{r.get('article')}" == top["id"] for r in a["refs"])]
    linked = [x for x in dict.fromkeys(out_refs + in_refs) if x in by_id and x != top["id"]]
    if linked:
        print("함께 봐야 할 조문:")
        for x in linked[:4]:
            a = by_id[x]
            first = lines_of(a)[0][:120] if lines_of(a) else ""
            print(f"  - {article_label(a)}: {first}")
    print("다른 후보 조문:")
    for i in order[1:]:
        print(f"  - {article_label(C[i])} (확신도 {s[i]:.3f})")
    print(f"원문: {top['url']}")
    print(NOTICE)


def interactive():
    """질문을 계속 받는다. 빈 줄, q, 종료, Ctrl+D로 끝낸다."""
    print("모델을 불러오는 중...", flush=True)
    embed(["준비"])
    print("학사규정 질문을 입력하세요. 끝내려면 q 또는 Ctrl+D.")
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in ("", "q", "quit", "exit", "종료"):
            break
        main(q)


if __name__ == "__main__":
    if sys.argv[1:]:
        main(" ".join(sys.argv[1:]))
    else:
        interactive()
