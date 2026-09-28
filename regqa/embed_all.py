"""코퍼스와 평가 질문을 KURE-v1로 임베딩해 캐시한다."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.corpus import load_corpus, load_eval, doc_text, embed

if __name__ == "__main__":
    t0 = time.time()
    C = load_corpus()
    E = embed([doc_text(a) for a in C], "corpus")
    print(f"corpus {E.shape} ({time.time()-t0:.0f}s)", flush=True)
    Q = load_eval()
    EQ = embed([q["question"] for q in Q], "eval_v0")
    print(f"eval {EQ.shape} ({time.time()-t0:.0f}s)", flush=True)
