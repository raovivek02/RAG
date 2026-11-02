"""Compare dense, filtered and reranked retrieval on a few sample questions."""
import sys
from retriever import DenseRetriever
from reranker import Reranker

size = int(sys.argv[1])
ret, rr = DenseRetriever(size=size), Reranker()
tag = lambda hs: [f"{h['drug'][:10]}|{h['section'][:14]}" for h in hs[:3]]

for q in ["What are the contraindications of warfarin?",
          "Does warfarin interact with NSAIDs like ibuprofen?",
          "Can sertraline be taken during pregnancy?",
          "Is there a boxed warning for sertraline?",
          "Which drugs cause serious liver injury?"]:
    print("\nQ:", q)
    print(" dense          :", tag(ret.search(q, 3, use_filter=False)))
    print(" +filter        :", tag(ret.search(q, 3)))
    print(" +filter+rerank :", tag(rr.rerank(q, ret.search(q, 20), 3)))
