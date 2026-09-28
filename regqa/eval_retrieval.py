"""재순위 없는 검색 성능: BM25, dense(KURE-v1), hybrid(RRF)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.corpus import DER
from regqa.metrics import score_one, summarize, fmt

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    P = json.load(open(DER / "pairs.json"))
    ids = P["corpus_ids"]; ev = P["eval"]
    qs = [q for q in ev if q["level"] != "L5b"]
    out = {}
    for name, key in [("bm25", "bm25_rank"), ("dense", "dense_rank"), ("hybrid", "cands")]:
        per = [score_one([ids[i] for i in q[key]], q) for q in qs]
        out[name] = summarize(per, qs)
        print(f"[{name}] {fmt(out[name])}")
    o = ROOT / "results" / "retrieval_baselines.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=1))
