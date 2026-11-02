import sys
from sentence_transformers import CrossEncoder
from retriever import DenseRetriever

class Reranker:
    def __init__(self, device="cpu"):
        self.model = CrossEncoder("BAAI/bge-reranker-base", device=device, max_length=512)

    def rerank(self, query, hits, top_n=5):
        scores = self.model.predict([(query, h["text"]) for h in hits])
        ranked = sorted(zip(scores, hits), key=lambda x: -x[0])
        return [{**h, "rerank_score": float(s)} for s, h in ranked[:top_n]]

def fuse(query, hits, rr, top_n=5, k0=60):
    """Reciprocal-rank fusion of dense order (given) and reranker order."""
    rr_scores = rr.model.predict([(query, h["text"]) for h in hits])
    rr_order = sorted(range(len(hits)), key=lambda i: -rr_scores[i])
    fused = {}
    for rank, i in enumerate(range(len(hits))):
        fused[i] = fused.get(i, 0) + 1 / (k0 + rank + 1)
    for rank, i in enumerate(rr_order):
        fused[i] = fused.get(i, 0) + 1 / (k0 + rank + 1)
    best = sorted(fused, key=lambda i: -fused[i])[:top_n]
    return [{**hits[i], "rerank_score": float(rr_scores[i])} for i in best]

if __name__ == "__main__":
    size = int(sys.argv[1])
    ret, rr = DenseRetriever(size=size), Reranker()
    queries = ["What are the contraindications of warfarin?",
               "Can sertraline be taken during pregnancy?",
               "What is the recommended starting dose of metformin?",
               "Does warfarin interact with NSAIDs like ibuprofen?"]
    for q in queries:
        cands = ret.search(q, 20)
        print("\nQ:", q)
        print(" dense top3:   ", [f"{h['drug'][:12]}|{h['section'][:14]}" for h in cands[:3]])
        top = rr.rerank(q, cands, 3)
        print(" reranked top3:", [f"{h['drug'][:12]}|{h['section'][:14]}" for h in top])
        print(" rerank scores:", [round(h['rerank_score'], 2) for h in top])
