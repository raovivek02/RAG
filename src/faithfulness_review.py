"""Write each RAG answer next to the numbered passages it was given, for manual faithfulness review."""
import json
from retriever import DenseRetriever
from reranker import Reranker, fuse

ret, rr = DenseRetriever(256), Reranker()
answers = {o["q"]: o["answer"] for o in json.load(open("results/generation.json")) if o["mode"] == "RAG"}
out = []
for q, a in answers.items():
    top = fuse(q, ret.search(q, 20), rr, 5)
    out.append(f"##### Q: {q}\nANSWER: {a}\n")
    for i, h in enumerate(top, 1):
        out.append(f"  [{i}] {h['drug']} | {h['section']}\n      {h['text'].split(chr(10), 1)[1][:700]}")
    out.append("\nSUPPORTED? (y / n) ______\n")
open("results/faithfulness_review.txt", "w").write("\n".join(out))
print("wrote results/faithfulness_review.txt for", len(answers), "answers")
