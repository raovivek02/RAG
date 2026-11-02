"""Print the questions whose gold chunk was not ranked first, with what was retrieved instead."""
import json
from retriever import DenseRetriever
from reranker import Reranker

qa = [json.loads(l) for l in open("eval/qa.jsonl") if l.strip()]
qa = [x for x in qa if x["answer"] != "NOT_FOUND"]
K, size = 5, 256
ret, rr = DenseRetriever(size=size), Reranker()

def rank_of(hits, x):
    for i, h in enumerate(hits[:K], 1):
        if x["drug"].lower() in h["drug"] and h["section"] == x["section"]:
            return i

modes = {"dense": lambda q: ret.search(q, K, use_filter=False),
         "+filter+rerank": lambda q: rr.rerank(q, ret.search(q, 20), K)}
for name, fn in modes.items():
    print(f"\n===== chunk={size} {name} =====")
    for x in qa:
        hits = fn(x["q"])
        r = rank_of(hits, x)
        if r != 1:
            print(f"rank={r}  Q: {x['q']}")
            print("   want:", x["drug"], "|", x["section"])
            print("   got :", [f"{h['drug'][:12]}|{h['section'][:18]}" for h in hits[:3]])
