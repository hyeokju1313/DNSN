"""한국공학대 규정관리시스템에서 학사 관련 규정 원문(HTML)을 수집한다.

- 규정 트리: /lmxsrv/law/lawTreeNodes.do (JSON)
- 본문: /lmxsrv/law/lawFullView.do 에서 이력 번호를 읽고 /lmxsrv/law/lawFullContent.do 를 받는다
- 서버 부담을 줄이려고 요청 사이에 0.5초 이상 쉰다
사용 예:
  python -u regqa/crawl.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import ssl
import time
import urllib.request
from pathlib import Path

BASE = "https://rule.tukorea.ac.kr"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "regulations"
SCOPE_PREFIX = ("제2편 학칙", "제3편 행정 > 제4장 학사행정", "제3편 행정 > 제6장 학생행정")
DELAY = 0.5

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE   # 학교 서버 인증서 체인 문제로 검증을 끈다(읽기 전용 공개 문서)


def get(path: str) -> str:
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0 (AI+X course project)"})
    time.sleep(DELAY)
    return urllib.request.urlopen(req, context=_ctx, timeout=30).read().decode("utf-8", "ignore")


def walk_tree(group=1, root=1):
    out = []

    def rec(parent, depth, path):
        txt = get(f"/lmxsrv/law/lawTreeNodes.do?LAWGROUP={group}&PARENT={parent}&TREEDEPTH={depth}").strip()
        for n in (json.loads(txt) if txt.startswith("[") else []):
            if n.get("folder") == "1":
                rec(n["key"], depth + 1, path + [n["title"]])
            else:
                out.append({"seq": n["key"], "title": n["title"], "path": " > ".join(path)})

    rec(root, 1, [])
    return out


def main():
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    tree = walk_tree()
    (OUT / "tree.json").write_text(json.dumps({"collected": today, "laws": tree}, ensure_ascii=False, indent=1))
    scope = [t for t in tree if t["path"].startswith(SCOPE_PREFIX)]
    print(f"tree: {len(tree)} laws, in scope: {len(scope)}", flush=True)
    manifest = []
    t0 = time.time()
    for i, t in enumerate(scope):
        view = get(f"/lmxsrv/law/lawFullView.do?SEQ={t['seq']}")
        m = re.search(r"lawFullContent\.do\?SEQ=\d+&SEQ_HISTORY=(\d+)", view)
        if not m:
            print(f"  [warn] no content link for {t['seq']} {t['title']}", flush=True)
            continue
        hseq = m.group(1)
        dept = re.search(r"담당부서\s*:\s*([^<\n]+)", view)
        content = get(f"/lmxsrv/law/lawFullContent.do?SEQ={t['seq']}&SEQ_HISTORY={hseq}")
        (OUT / "raw" / f"{t['seq']}_{hseq}.html").write_text(content, encoding="utf-8")
        manifest.append({**t, "hseq": int(hseq), "dept": dept.group(1).strip() if dept else None,
                         "url": f"{BASE}/lmxsrv/law/lawFullView.do?SEQ={t['seq']}", "collected": today})
        el = time.time() - t0
        print(f"  {i + 1}/{len(scope)} {t['title']} ({el:.0f}s, ETA {el / (i + 1) * (len(scope) - i - 1):.0f}s)", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"saved {len(manifest)} regulations to {OUT / 'raw'}", flush=True)


if __name__ == "__main__":
    main()
