"""Compute recall@5 and MRR for each chunk size and retrieval mode on eval/qa.jsonl."""
import json, csv
from retriever import DenseRetriever
from reranker import Reranker, fuse

qa = [json.loads(l) for l in open("eval/qa.jsonl") if l.strip()]
qa = [x for x in qa if x["answer"] != "NOT_FOUND"]
K = 5
rr = Reranker()

def rank_of(hits, x):
    for i, h in enumerate(hits[:K], 1):
        if x["drug"].lower() in h["drug"] and h["section"] == x["section"]:
            return i
    return None

rows = []
for size in (256, 512):
    ret = DenseRetriever(size=size)
    modes = {
        "dense": lambda q: ret.search(q, K, use_filter=False),
        "+filter": lambda q: ret.search(q, K),
        "+filter+rerank": lambda q: rr.rerank(q, ret.search(q, 20), K),
        "+fusion": lambda q: fuse(q, ret.search(q, 20), rr, K),
    }
    for name, fn in modes.items():
        ranks = [rank_of(fn(x["q"]), x) for x in qa]
        recall = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        rows.append((size, name, round(recall, 3), round(mrr, 3)))
        print(f"chunk={size:<4} {name:<15} recall@{K}={recall:.2f}  MRR={mrr:.2f}")

with open("results/retrieval.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chunk_size", "mode", f"recall@{K}", "MRR"]); w.writerows(rows)
