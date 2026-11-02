"""Dense retrieval over the FAISS index with a drug-name filter."""
import json, re, sys, faiss
from sentence_transformers import SentenceTransformer

QPREFIX = "Represent this sentence for searching relevant passages: "
STOP = {"sodium", "hydrochloride", "potassium", "besylate", "bisulfate", "calcium",
        "oral", "injection", "sulfate", "acid", "tablets", "extended", "release"}

class DenseRetriever:
    def __init__(self, size=256, device="cpu"):
        self.chunks = [json.loads(l) for l in open(f"data/processed/chunks_{size}.jsonl")]
        self.index = faiss.read_index(f"index/faiss_{size}.index")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
        # word in a generic name -> set of full drug names (e.g. "warfarin" -> {"warfarin sodium"})
        self.word2drugs = {}
        for d in {c["drug"] for c in self.chunks}:
            for w in re.findall(r"[a-z]+", d):
                if len(w) >= 5 and w not in STOP:
                    self.word2drugs.setdefault(w, set()).add(d)

    def detect_drugs(self, query):
        drugs = set()
        for w in re.findall(r"[a-z]+", query.lower()):
            drugs |= self.word2drugs.get(w, set())
        return drugs

    def search(self, query, k=5, use_filter=True):
        q = self.model.encode([QPREFIX + query], normalize_embeddings=True).astype("float32")
        drugs = self.detect_drugs(query) if use_filter else set()
        n = self.index.ntotal if drugs else k      # exact index is tiny, so scan all and filter
        scores, ids = self.index.search(q, n)
        hits = [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]
        if drugs:
            hits = [h for h in hits if h["drug"] in drugs]
        return hits[:k]
