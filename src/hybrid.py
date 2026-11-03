"""Hybrid retrieval: BM25 (exact words) + dense (meaning), merged by reciprocal rank fusion."""
import re
from rank_bm25 import BM25Okapi
from retriever import DenseRetriever

tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())

class HybridRetriever(DenseRetriever):
    def __init__(self, size=256, device="cpu"):
        super().__init__(size=size, device=device)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search_hybrid(self, query, k=20, pool=50, k0=60):
        drugs = self.detect_drugs(query)
        scores = self.bm25.get_scores(tokenize(query))
        bm_order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if drugs:                                   # same drug filter as the dense side
            bm_order = [i for i in bm_order if self.chunks[i]["drug"] in drugs]
        bm_ids = bm_order[:pool]
        id_of = {c["id"]: i for i, c in enumerate(self.chunks)}
        dense_ids = [id_of[h["id"]] for h in self.search(query, pool)]
        fused = {}
        for ids in (dense_ids, bm_ids):
            for rank, i in enumerate(ids):
                fused[i] = fused.get(i, 0) + 1 / (k0 + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.chunks[i] for i in best]
