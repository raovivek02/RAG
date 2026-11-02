import json, argparse, torch, faiss
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=256)
size = ap.parse_args().size

chunks = [json.loads(l) for l in open(f"data/processed/chunks_{size}.jsonl")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)

emb = model.encode([c["text"] for c in chunks], batch_size=64,
                   normalize_embeddings=True, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine on normalized vectors
index.add(emb.astype("float32"))
faiss.write_index(index, f"index/faiss_{size}.index")
print(f"indexed {index.ntotal} chunks, dim {emb.shape[1]}, device {dev}")
