"""수집한 규정 HTML을 markdown과 조문 단위 JSONL로 바꾼다.

규칙(proposal_1 4.1절)
1. 장·절·조·항·호 구조를 보존하고 조문마다 고유 ID를 붙인다 (예: 32:제22조의1, 부칙은 32:부칙3:제1조)
2. 표는 markdown 표로 보존한다(colspan, rowspan을 펼쳐 격자로 만든다)
3. 다른 규정·조문을 가리키는 링크를 참조로 남긴다(golink('1', 규정, 'c022001000') = 제22조의1)
4. 개정 이력, 담당 부서, 수집일을 메타데이터로 남긴다
사용 예:
  python -u regqa/to_markdown.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent
REG = ROOT / "data" / "regulations"
BASE = "https://rule.tukorea.ac.kr/lmxsrv/law/lawFullView.do?SEQ="

NOTE_RE = re.compile(r"<\s*(?:신설|개정|삭제|전문개정|본조신설)[^<>]*>|\[\s*(?:본조신설|본조개정|신설|개정|삭제|전문개정|제목개정)[^\]]*\]")
ART_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의?\s*(\d+))?\s*(?:\((.*)\))?")  # 원문에 "제 25 조 2"처럼 '의'가 빠진 표기도 있다


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def art_label(jo: int, ui: int | None) -> str:
    return f"제{jo}조" + (f"의{ui}" if ui else "")


def anchor(label: str) -> str:
    m = re.match(r"제(\d+)조(?:의(\d+))?", label)
    return f"a{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "") if m else re.sub(r"\W", "", label)


def safe(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", name).strip("_")


class Converter:
    def __init__(self, manifest):
        self.by_seq = {m["seq"]: m for m in manifest}
        self._seen: dict = {}

    def md_file(self, seq: int) -> str | None:
        m = self.by_seq.get(seq)
        return f"{seq}_{safe(m['title'])}.md" if m else None

    # ---- 인라인 텍스트 ----
    def inline(self, node, refs: list, clean: bool) -> str:
        out = []
        for ch in node.children if isinstance(node, Tag) else []:
            if isinstance(ch, NavigableString):
                out.append(str(ch))
                continue
            if not isinstance(ch, Tag):
                continue
            if ch.name == "br":
                out.append(" ")
                continue
            if "doc_btn" in (ch.get("class") or []):
                continue
            style = (ch.get("style") or "").replace(" ", "").lower()
            if clean and "color:#0000ff" in style:
                continue                                  # <개정 2016.3.1.> 같은 개정 표시
            if ch.name == "img":
                out.append("[그림]")
                continue
            if ch.name == "a":
                out.append(self.link(ch, refs, clean))
                continue
            out.append(self.inline(ch, refs, clean))
        return "".join(out)

    def link(self, a: Tag, refs: list, clean: bool) -> str:
        text = norm_space(a.get_text(" "))
        href = a.get("href") or ""
        m = re.match(r"javascript:golink\('(\d)',\s*'(\d+)',\s*'([^']*)'", href)
        if m:
            kind, seq, code = m.group(1), int(m.group(2)), m.group(3)
            label = None
            mc = re.match(r"c(\d{3})(\d{3})", code)
            if kind == "1" and mc:
                label = art_label(int(mc.group(1)), int(mc.group(2)) or None)
            refs.append({"seq": seq, "article": label, "text": text})
            if clean:
                return text
            f = self.md_file(seq)
            target = (f if f else BASE + str(seq)) + (f"#{anchor(label)}" if label and f else "")
            return f"[{text}]({target})"
        m = re.match(r"javascript:showPopup\('(http[^']+)'", href)
        if m and not clean:
            return f"[{text}]({m.group(1).replace(' ', '%20')})"
        return text

    # ---- 표 ----
    def table(self, t: Tag, refs: list, clean: bool) -> str:
        grid: dict[tuple[int, int], str] = {}
        rows = t.find_all("tr")
        for r, tr in enumerate(rows):
            c = 0
            for td in tr.find_all(["td", "th"], recursive=False):
                while (r, c) in grid:
                    c += 1
                txt = norm_space(" ".join(self.inline(p, refs, clean) for p in (td.find_all("p") or [td])))
                txt = txt.replace("|", "\\|")
                rs, cs = int(td.get("rowspan", 1) or 1), int(td.get("colspan", 1) or 1)
                for dr in range(rs):
                    for dc in range(cs):
                        grid[(r + dr, c + dc)] = txt
                c += cs
        if not grid:
            return ""
        nr = max(k[0] for k in grid) + 1
        nc = max(k[1] for k in grid) + 1
        lines = []
        for r in range(nr):
            lines.append("| " + " | ".join(grid.get((r, c), "") for c in range(nc)) + " |")
            if r == 0:
                lines.append("|" + "---|" * nc)
        return "\n".join(lines)

    # ---- 블록 ----
    def block(self, el: Tag, refs: list, clean: bool) -> list[str]:
        """조 안의 블록 하나(항, 호, 목, 표 등)를 줄 목록으로."""
        cls = el.get("class") or []
        if el.find("table") is not None:
            out = []
            for child in el.children:
                if isinstance(child, Tag) and child.name == "table":
                    out.append(self.table(child, refs, clean))
                elif isinstance(child, Tag) and child.find("table") is not None:
                    out += self.block(child, refs, clean)
                else:
                    s = norm_space(self.inline(child, refs, clean) if isinstance(child, Tag) else str(child))
                    if s:
                        out.append(s)
            return out
        s = norm_space(self.inline(el, refs, clean))
        if not s:
            return []
        indent = "    " if "mok" in cls or "level3" in cls else "  " if "ho" in cls or "level2" in cls else ""
        return [indent + s]

    def convert(self, seq: int, html: str):
        meta = self.by_seq[seq]
        soup = BeautifulSoup(html, "lxml")
        lc = soup.select_one("#lawcontent")
        title = norm_space(lc.select_one(".lawname2").get_text(" ")) if lc.select_one(".lawname2") else meta["title"]
        history = [norm_space(h.get_text(" ")) for h in lc.select(".history.historyContent, .history.historyContentLast")]
        attach = [norm_space(a.get_text(" ")) for a in lc.select(".attach_file")]

        md = [f"# {meta['title']}", "",
              f"- 규정 번호(SEQ): {seq}, 이력 번호: {meta['hseq']}",
              f"- 분류: {meta['path']}",
              f"- 담당 부서: {meta.get('dept') or '표시 없음'}",
              f"- 제정·개정: {history[0] if history else '표시 없음'} ... {history[-1] if len(history) > 1 else ''} (총 {len(history)}건)",
              f"- 원문: {meta['url']}",
              f"- 수집일: {meta['collected']}", ""]
        articles = []
        chapter = section = None
        addenda_k = 0
        in_addenda = False

        for jo in lc.select("div.JO"):
            if jo.find_parent("div", class_="JO") is not None:
                continue
            ch = jo.select_one("div.chapter")
            se = jo.select_one("div.section")
            if ch is not None and jo.select_one("span.article2") is None:
                chapter, section = norm_space(ch.get_text(" ")), None
                md += [f"## {chapter}", ""]
                continue
            if se is not None and jo.select_one("span.article2") is None:
                section = norm_space(se.get_text(" "))
                md += [f"### {section}", ""]
                continue
            # 부칙 시작 여부: 이 JO 앞에 addenda2가 나왔는지
            prev_add = jo.find_previous("div", class_="addenda2")
            if prev_add is not None and not in_addenda:
                in_addenda = True
            if in_addenda:
                cur_add = prev_add
                if getattr(self, "_last_add", None) is not cur_add:
                    self._last_add = cur_add
                    addenda_k += 1
                    md += [f"## 부칙 {addenda_k}", ""]
            head = jo.select_one("span.article2") or jo.select_one("div.article")
            head_txt = norm_space(self.inline(head, [], True)) if head is not None else ""
            m = ART_RE.search(head_txt)
            if m:
                label = art_label(int(m.group(1)), int(m.group(2)) if m.group(2) else None)
                atitle = (m.group(3) or "").strip()
            else:
                label, atitle = (head_txt or "본문"), ""
            refs: list = []
            lines, clean_lines = [], []
            for child in jo.children:
                if not isinstance(child, Tag) or child is head or (head is not None and child in head.parents):
                    continue
                if child.name == "span" and "article2" in (child.get("class") or []):
                    continue
                if "article" in (child.get("class") or []):
                    continue
                lines += self.block(child, refs, False)
                clean_lines += self.block(child, [], True)
            if head is None and not any(ln.strip() for ln in lines):
                continue                                  # 내용 없는 빈 JO
            if head is None:
                label = "전문"
            aid = f"{seq}:부칙{addenda_k}:{label}" if in_addenda else f"{seq}:{label}"
            seen = self._seen.setdefault(seq, {})
            seen[aid] = seen.get(aid, 0) + 1
            if seen[aid] > 1:
                aid += f"#{seen[aid]}"                    # 원문에 번호가 중복된 경우
            if in_addenda:
                md += [f"#### {label}" + (f" ({atitle})" if atitle else ""), ""]
            else:
                md += [f'<a id="{anchor(label)}"></a>', f"#### {label}" + (f" ({atitle})" if atitle else ""), ""]
            body = []
            for i, ln in enumerate(lines):
                is_tab = ln.startswith("|")
                prev_tab = i > 0 and lines[i - 1].startswith("|")
                if is_tab and not prev_tab and body:
                    body.append("")                      # 표 앞에는 빈 줄이 있어야 markdown 표로 보인다
                if not is_tab and prev_tab:
                    body.append("")
                body.append(ln if is_tab else ln + "  ")
            md += body + [""]
            text_clean = "\n".join(clean_lines)
            articles.append({
                "id": aid, "law_seq": seq, "law": meta["title"], "path": meta["path"],
                "chapter": None if in_addenda else chapter, "section": None if in_addenda else section,
                "article": label, "title": atitle, "addenda": addenda_k if in_addenda else None,
                "text": "\n".join(lines), "text_clean": NOTE_RE.sub("", text_clean).strip(),
                "refs": refs, "url": meta["url"],
            })
        self._last_add = None
        if attach:
            md += ["## 별표·별지 (HWP 첨부, 변환하지 않음)", ""] + [f"- {a}" for a in attach] + [""]
        return title, "\n".join(md), articles


def main():
    manifest = json.loads((REG / "manifest.json").read_text())
    conv = Converter(manifest)
    (REG / "md").mkdir(exist_ok=True)
    allart = []
    for m in manifest:
        html = (REG / "raw" / f"{m['seq']}_{m['hseq']}.html").read_text(encoding="utf-8")
        _, md, arts = conv.convert(m["seq"], html)
        (REG / "md" / conv.md_file(m["seq"])).write_text(md, encoding="utf-8")
        allart += arts
    with open(REG / "articles.jsonl", "w", encoding="utf-8") as f:
        for a in allart:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    n_body = sum(a["addenda"] is None for a in allart)
    n_ref = sum(len(a["refs"]) for a in allart)
    print(f"{len(manifest)} regulations -> {len(allart)} articles (body {n_body}, addenda {len(allart) - n_body}), refs {n_ref}")


if __name__ == "__main__":
    main()
